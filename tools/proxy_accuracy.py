"""사람 데이터에서 내 쪽 세트 추정(사용률 최빈값)의 대리 정확도: 로그에 드러난 도구/특성/테라를 정답으로 보고, tools/human_to_fp.py가 쓰는 사용률 순위와 비교한다.
드러난 것만 정답이 되는 편향(예: 남은 도구는 터져야 보임)을 보정하려고, 정답을 아는 풀 팀(d7 rec의 실제 request)에도 같은 방식으로 재서 나란히 놓는다:
풀 팀의 "드러난 부분집합" 정확도와 사람 데이터의 "드러난 부분집합" 정확도가 비슷하면 사람 팀의 추정 정확도도 풀 팀과 비슷하다고 볼 수 있다.
사용: python tools/proxy_accuracy.py [--games 4000] [--min-rating 1400]"""
import argparse
import sys
from collections import Counter

import pyarrow.parquet as pq

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from fp_fidelity import degrade, load_recs, requests, species_of
from human_to_fp import _norm, _usage
from src.training.replay_dataset import _hindsight

FIELDS = (("item", "Items", "item"), ("ability", "Abilities", "ability"), ("tera", "Tera Types", "tera"))
RECS = ["rec_fp_h0", "rec_fp_h1", "rec_fp_h2", "rec_mr_h0_a", "rec_mr_h0_b", "rec_mr_h1_a", "rec_mr_h1_b", "rec_mr_h2_a", "rec_mr_h2_b"]


def ranked(d):
    d = {k: v for k, v in (d or {}).items() if k != "other"}
    return sorted(d, key=d.get, reverse=True)


def parse(log):
    return [s for s in (l.split("|") for l in log.split("\n") if l.startswith("|") and l != "|") if len(s) > 1]


def team_rows(lines, side):
    """내 쪽 미리보기 포켓몬마다 (종, 드러난 정보, 사용률 순위 3종, 드러난 기술 수)"""
    hind, usage, out = _hindsight(lines, side), _usage(), []
    for s in lines:
        if s[1] == "poke" and len(s) > 3 and s[2] == side:
            sp = _norm(s[3].split(",")[0])
            h, u = hind.get(sp, {}), usage.get(sp)
            out.append({"sp": sp, "known": u is not None, "nrev": len(h.get("moves") or []),
                        **{f: {"rev": h.get(k) or None, "rank": ranked(u.get(col)) if u else []} for f, col, k in FIELDS}})
    return out


def pool_rows():
    rows = []
    for name in RECS:
        for rec in load_recs(f"data/dagger/raw/d7/{name}.jsonl.gz"):
            log, side = degrade(rec)
            truth = {species_of(p["details"]): p for p in requests(rec)[0][1]["side"]["pokemon"]}
            for r in team_rows(parse(log), side):
                p = truth.get(r["sp"])
                if p is None:
                    continue
                r["truth"] = {"item": _norm(p["item"]), "ability": _norm(p["baseAbility"]), "tera": _norm(p["teraType"])}
                rows.append(r)
    return rows


def human_rows(n_games, min_rating):
    t = pq.read_table("data/metamon_raw/train-00000-of-00046.parquet", columns=["rating", "log"]).to_pydict()
    rows, used = [], 0
    for rating, lg in zip(t["rating"], t["log"]):
        if not str(rating).isdigit() or int(rating) < min_rating:
            continue
        lines = parse(lg)
        for side in ("p1", "p2"):
            rows += team_rows(lines, side)
        used += 1
        if used >= n_games:
            break
    return rows, used


def rate(num, den):
    return f"{num / den:.3f}" if den else " n/a "


def summarize(rows, truth_key):
    """필드별: 드러난 비율, 드러난 부분집합의 top1/top3 정확도 (정답=드러난 값), 풀이면 안 드러난 부분집합과 전체의 정확도 (정답=진짜 값)"""
    out = {}
    for f, _, _ in FIELDS:
        c = Counter()
        for r in rows:
            if not r["known"]:
                continue
            d, rk = r[f], r[f]["rank"]
            c["n"] += 1
            if d["rev"]:
                c["rev"] += 1
                c["rev1"] += bool(rk) and rk[0] == d["rev"]
                c["rev3"] += d["rev"] in rk[:3]
                if truth_key:
                    c["rev_eq_truth"] += d["rev"] == r["truth"][f]
            if truth_key:
                t = r["truth"][f]
                c["all1"] += bool(rk) and rk[0] == t
                if not d["rev"]:
                    c["unrev"] += 1
                    c["unrev1"] += bool(rk) and rk[0] == t
                    c["unrev3"] += t in rk[:3]
        out[f] = c
    return out


def main(args):
    pool = pool_rows()
    human, n_games = human_rows(args.games, args.min_rating)
    for name, rows in (("풀 팀 (d7, 정답 앎)", pool), ("사람 (Metamon r%d+, %d판)" % (args.min_rating, n_games), human)):
        known = sum(r["known"] for r in rows)
        print(f"{name}: 포켓몬 {len(rows)}마리, 사용률 표에 있음 {rate(known, len(rows))}, 드러난 기술 수 평균 {sum(r['nrev'] for r in rows) / len(rows):.2f}/4")
    sp, sh = summarize(pool, True), summarize(human, False)
    print(f"{'필드':8s} | {'풀: 드러남':>10s} {'정확도 top1':>11s} {'top3':>6s} {'정답=드러난값':>12s} | {'풀: 안드러남 top1':>16s} {'top3':>6s} {'전체 top1':>9s} | {'사람: 드러남':>11s} {'top1':>6s} {'top3':>6s}")
    for f, _, _ in FIELDS:
        p, h = sp[f], sh[f]
        print(f"{f:8s} | {rate(p['rev'], p['n']):>10s} {rate(p['rev1'], p['rev']):>11s} {rate(p['rev3'], p['rev']):>6s} {rate(p['rev_eq_truth'], p['rev']):>12s} | "
              f"{rate(p['unrev1'], p['unrev']):>16s} {rate(p['unrev3'], p['unrev']):>6s} {rate(p['all1'], p['n']):>9s} | {rate(h['rev'], h['n']):>11s} {rate(h['rev1'], h['rev']):>6s} {rate(h['rev3'], h['rev']):>6s}")
    print(f"표본 수(드러남): 풀 {[sp[f]['rev'] for f, _, _ in FIELDS]}, 사람 {[sh[f]['rev'] for f, _, _ in FIELDS]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--min-rating", type=int, default=1400)
    main(ap.parse_args())
