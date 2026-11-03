"""내 쪽 포켓몬의 세트 사후분포: FP가 상대 세트 추정에 쓰는 완결 세트 표(pokemon_full_sets.json, 종별 "테라|특성|도구|성격|노력치|기술..." -> 횟수)를
로그에서 드러난 증거(기술 부분집합, 도구, 특성, 테라)로 걸러 횟수 비례 가중으로 둔다. 맞는 세트가 없으면 증거를 하나씩 풀어 간다(RELAX 순서).
진단(python tools/set_posterior.py): 사람 로그/풀 팀에서 증거와 맞는 세트가 있는 비율, 유효 세트 수, 정답을 아는 풀 팀에서의 사후 정확도,
사람 로그에서 드러난 값 하나를 빼고 맞히는 정확도(사용률 최빈과 비교). FP 불필요."""
import argparse
import json
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import pyarrow.parquet as pq

from fp_fidelity import degrade, load_recs, requests, species_of
from human_to_fp import CHOICE, PAD, _dex, _norm, _top, _usage, calc_stats
from proxy_accuracy import RECS, parse
from src.training.replay_dataset import _hindsight

SETS_PATH = "C:/Users/lsh/Desktop/orcl_server/foul-play/fp/data/pkmn_sets_cache/gen9ou/pokemon_full_sets.json"
RELAX = ((), ("tera",), ("tera", "item"), ("tera", "item", "ability"), ("tera", "item", "ability", "moves"))   # 맞는 세트가 없으면 앞에서부터 풀어 가는 증거
_SETS = None


def sets():
    """종 id -> [(테라, 특성, 도구, 성격, 노력치 튜플, 기술 집합, 횟수)]"""
    global _SETS
    if _SETS is None:
        _SETS = {}
        for sp, d in json.load(open(SETS_PATH, encoding="utf-8")).items():
            rows = []
            for s, n in d.items():
                p = s.split("|")
                rows.append((p[0], p[1], p[2], p[3], tuple(int(x) for x in p[4].split(",")), frozenset(p[5:]), n))
            _SETS[sp] = rows
    return _SETS


def posterior(sp, hind, drop=()):
    """(세트 목록, 정규화 가중치, 푼 단계 수). 세트 목록이 비면 표에 종이 없는 것"""
    rows = sets().get(sp)
    if not rows:
        return [], np.zeros(0), None
    for level, relaxed in enumerate(RELAX):
        off = set(drop) | set(relaxed)
        keep = [r for r in rows if (hind.get("item") is None or "item" in off or r[2] == hind["item"]) and
                (hind.get("ability") is None or "ability" in off or r[1] == hind["ability"]) and
                (hind.get("tera") is None or "tera" in off or r[0] == hind["tera"]) and
                not (hind.get("no_choice") and r[2] in CHOICE) and
                ("moves" in off or set(hind.get("moves") or []) <= r[5])]
        if keep:
            w = np.array([r[6] for r in keep], dtype=float)
            return keep, w / w.sum(), level
    return [], np.zeros(0), None


def make_world(tag, K, mode):
    """한 판의 세트 선택 함수 K개 (월드마다 하나). mode "map" = 사후 최빈 세트, "sample" = 사후에서 K개를 가능하면 겹치지 않게 뽑아 월드마다 하나씩
    (종마다 뽑기를 모든 월드가 공유). 반환 dict는 human_to_fp.convert의 오라클 덮어쓰기 형식"""
    rng, cache = np.random.RandomState(zlib.crc32(tag.encode())), {}

    def fn_for(j):
        def fn(sp, hind, e):
            keep, w, _ = posterior(sp, hind)
            if not keep or sp not in _dex():
                return {}
            if mode == "map":
                i = int(w.argmax())
            else:
                if sp not in cache:
                    cache[sp] = rng.choice(len(keep), size=K, replace=len(keep) < K, p=w)
                i = int(cache[sp][j])
            tera, ability, item, nature, evs, moves, _ = keep[i]
            rev = list(hind.get("moves") or [])
            full = (rev + [m for m in sorted(moves) if m not in rev])[:4]
            stats = calc_stats(_dex()[sp]["baseStats"], nature, evs, e["level"])
            maxhp = stats.pop("hp")
            return {"moves": full + [PAD] * (4 - len(full)), "item": "" if item in ("none", "nothing") else item, "ability": ability,
                    "tera": tera if tera and tera != "typeless" else e["tera"], "stats": stats, "maxhp": maxhp, "nature": nature, "evs": list(evs), "sampled": True}
        return fn
    return [fn_for(j) for j in range(K)]


def mons(lines, side):
    hind = _hindsight(lines, side)
    for s in lines:
        if s[1] == "poke" and len(s) > 3 and s[2] == side:
            sp = _norm(s[3].split(",")[0])
            yield sp, hind.get(sp, {})


def stats(rows):
    """[(sp, hind)] -> 종 포함률, 단계별 비율, 유효 세트 수 분포"""
    c, neff, nsets = Counter(), [], []
    for sp, h in rows:
        keep, w, level = posterior(sp, h)
        c["mons"] += 1
        if not keep:
            c["species_missing" if sp not in sets() else "none"] += 1
            continue
        c[f"level{level}"] += 1
        nsets.append(len(keep)); neff.append(1 / float((w ** 2).sum()))
    return c, np.array(neff), np.array(nsets)


FIELD_IDX = {"item": 2, "ability": 1, "tera": 0}


def accuracy(sp, h, truth, usage):
    """증거를 모두 쓴 사후의 MAP 세트/사후 확률 정확도와 현재 변환기 추정(드러난 값, 없으면 사용률 최빈) 정확도. truth = {item, ability, tera}"""
    keep, w, _ = posterior(sp, h)
    out = {}
    for f, i in FIELD_IDX.items():
        if not keep:
            continue
        mp = keep[int(w.argmax())][i]
        p_true = float(sum(wi for r, wi in zip(keep, w) if r[i] == truth[f]))
        col = {"item": "Items", "ability": "Abilities", "tera": "Tera Types"}[f]
        est = h.get(f) or _top((usage.get(sp) or {}).get(col))
        out[f] = (mp == truth[f], p_true, est == truth[f])
    return out


def loo(sp, h, usage):
    """드러난 값 하나(f)를 증거에서 빼고 사후가 맞히는가: (f, MAP 정확, 사후 확률, 사용률 최빈 정확)"""
    out = []
    for f, i in FIELD_IDX.items():
        if not h.get(f):
            continue
        keep, w, level = posterior(sp, h, drop=(f,))
        if not keep:
            continue
        col = {"item": "Items", "ability": "Abilities", "tera": "Tera Types"}[f]
        top = _top((usage.get(sp) or {}).get(col))
        out.append((f, keep[int(w.argmax())][i] == h[f], float(sum(wi for r, wi in zip(keep, w) if r[i] == h[f])), top == h[f], level))
    return out


def report_stats(name, rows):
    c, neff, nsets = stats(rows)
    n = c["mons"]
    lv = " ".join(f"단계{k}:{c[f'level{k}'] / n:.3f}" for k in range(len(RELAX)))
    print(f"{name}: 포켓몬 {n}마리 | 종 없음 {c['species_missing'] / n:.3f} | 맞는 세트 없음 {c['none'] / n:.3f} | {lv}")
    print(f"   맞는 세트 수 중앙 {np.median(nsets):.0f} 평균 {nsets.mean():.1f} | 유효 세트 수(1/Σw²) 중앙 {np.median(neff):.2f} 평균 {neff.mean():.2f} p90 {np.percentile(neff, 90):.2f} | 유효 세트 수 < 1.5인 비율 {(neff < 1.5).mean():.3f}")


def main(args):
    usage = _usage()
    pool_rows, truth_rows = [], []
    for name in RECS:
        for rec in load_recs(f"data/dagger/raw/d7/{name}.jsonl.gz"):
            log, side = degrade(rec)
            truth = {species_of(p["details"]): {"item": _norm(p["item"]), "ability": _norm(p["baseAbility"]), "tera": _norm(p["teraType"])}
                     for p in requests(rec)[0][1]["side"]["pokemon"]}
            for sp, h in mons(parse(log), side):
                pool_rows.append((sp, h))
                if sp in truth:
                    truth_rows.append((sp, h, truth[sp]))
    t = pq.read_table("data/metamon_raw/train-00000-of-00046.parquet", columns=["rating", "log"]).to_pydict()
    human_rows, used = [], 0
    for rating, lg in zip(t["rating"], t["log"]):
        if not str(rating).isdigit() or int(rating) < args.min_rating:
            continue
        lines = parse(lg)
        for side in ("p1", "p2"):
            human_rows += list(mons(lines, side))
        used += 1
        if used >= args.games:
            break
    report_stats("풀 팀", pool_rows)
    report_stats(f"사람 r{args.min_rating}+ {used}판", human_rows)
    acc = defaultdict(list)
    for sp, h, tr in truth_rows:
        for f, v in accuracy(sp, h, tr, usage).items():
            acc[f].append(v)
    print("--- 풀 팀 (정답 앎): 증거를 다 쓴 사후 vs 지금 변환기 추정")
    for f in FIELD_IDX:
        a = np.array(acc[f], dtype=float)
        print(f"{f:8s} MAP 정확도 {a[:, 0].mean():.3f} | 사후 질량(진짜 값) {a[:, 1].mean():.3f} | 지금 추정 {a[:, 2].mean():.3f}  (n={len(a)})")
    lo = defaultdict(list)
    for sp, h in human_rows:
        for f, mp, pt, tp, level in loo(sp, h, usage):
            lo[f].append((mp, pt, tp, level))
    print("--- 사람: 드러난 값 하나를 증거에서 빼고 맞히기 (사용률 최빈과 같은 표본)")
    for f in FIELD_IDX:
        a = np.array(lo[f], dtype=float)
        print(f"{f:8s} MAP 정확도 {a[:, 0].mean():.3f} | 사후 질량(드러난 값) {a[:, 1].mean():.3f} | 사용률 최빈 {a[:, 2].mean():.3f} | 증거를 푼 비율 {(a[:, 3] > 0).mean():.3f}  (n={len(a)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--min-rating", type=int, default=1400)
    main(ap.parse_args())
