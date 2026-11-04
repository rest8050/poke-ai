"""교사(Foul Play)의 믿음과 학생 세트 추정의 정렬 진단: 같은 결정 시점에서 상대 포켓몬의 숨겨진 특성/도구/기술 분포를 비교한다.
교사 쪽: fp/belief_probe.py가 결정마다 FP의 월드 샘플러를 N번 돌려 남긴 주변분포 (logs/belief_probe_run.sh가 생성, bel_*.jsonl).
학생 쪽: 같은 결정 상태를 build_fp_dataset과 같은 재생으로 만들어 세트 추정 모듈에 넣은 출력 (사전학습 belief_27.pt, 그리고 모델 체크포인트에 든 미세조정본).
비교 대상은 그 시점에 값이 안 드러난 (슬롯, 필드)만: 특성/도구가 숨김이거나 기술 칸이 비어 있는 경우. 기술은 "안 드러난 칸 하나에 들어가는 기술의 분포"로 맞춤(교사: 월드에 든 미공개 기술 횟수, 학생: 공개 기술을 뺀 softmax).
지표: KL(교사 || 학생), 총변동거리(TV), 교사 최빈값과 학생 최빈값 일치, 학생이 교사 최빈값에 준 확률, 교사 표본 N 때문에 생기는 TV 잡음 하한(이론 근사).
사용: python tools/belief_align.py --probe "data/dagger/raw/d7b/bel_rec_*_h*.jsonl" --models v6_full   (--models 의 이름은 checkpoints/supervised_v2_fp_<이름>.pt, 사전학습 belief_27.pt는 항상 같이 비교)"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.belief import SetBelief
from src.core.model import model_from_ckpt
from src.training.build_fp_dataset import _states, load_jsonl
from src.training.replay_dataset import _norm

EPS = 1e-8
CHOICE = ("choiceband", "choicespecs", "choicescarf")


def collect(pattern, workers):
    """배틀마다 (교사 믿음 기록, 같은 결정의 학생 관측)을 맞춘다. 반환: 항목 목록 [(opp_cat [6,11], opp_num [6,NUM_DIM], 교사 기록 mons)]"""
    by_tag = defaultdict(lambda: {"bel": []})
    for r in load_jsonl(pattern):
        d = by_tag[(r["_src"], r["tag"])]
        if r["type"] == "belief":
            d["bel"].append(r)
        elif r["type"] in ("start", "protocol"):
            d[r["type"]] = r
    jobs, metas = [], []
    for (src, tag), d in by_tag.items():
        if "protocol" not in d or "start" not in d or not d["bel"]:
            continue
        protocol, user = d["protocol"]["lines"], d["start"]["user"]
        side = next((s[2] for s in (l.split("|") for l in protocol) if len(s) > 4 and s[1] == "player" and s[3] == user), None)
        if side is None:
            continue
        jobs.append((protocol, side, user, d["start"]["team"]))
        metas.append(d["bel"])
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            all_states = pool.map(_states, jobs, chunksize=4)
    else:
        all_states = [_states(j) for j in jobs]
    items = []
    for bel, states in zip(metas, all_states):
        if states is None:
            continue
        queues = defaultdict(list)
        for st, turn, kind in states:
            queues[(turn, kind)].append(st)
        for b in bel:
            q = queues[(b["turn"], int(b["force_switch"]))]
            if q:
                st = q.pop(0)
                items.append((st["obs"][3], st["obs"][4], b["mons"]))
    return items


def student_probs(logits, unk, drop):
    """softmax에서 빈칸(0), <unk> 이상, drop(이미 공개된 id)을 뺌"""
    lg = logits.clone()
    lg[0] = -1e9
    lg[unk:] = -1e9
    for i in drop:
        lg[i] = -1e9
    return torch.softmax(lg, -1).numpy().astype(np.float64)


def fp_vector(counts, name_to_id, size, total, drop=()):
    """교사 횟수 -> 학생 어휘 위의 분포. 어휘에 없는 이름과 drop은 뺀 뒤 정규화. 반환 (분포, 어휘에 있던 질량 비율)"""
    v = np.zeros(size)
    for name, c in counts.items():
        i = name_to_id.get(_norm(name))
        if i is not None and i not in drop:
            v[i] += c
    s = v.sum()
    return (v / s if s > 0 else v), (s / max(total, 1e-9))


def metrics(p, q, n):
    kl = float(np.sum(np.where(p > 0, p * (np.log(np.clip(p, EPS, 1)) - np.log(np.clip(q, EPS, 1))), 0.0)))
    tv = 0.5 * float(np.abs(p - q).sum())
    floor = 0.5 * 0.798 * float(np.sqrt(np.clip(p * (1 - p), 0, None) / n).sum())     # 표본 n개 경험분포의 기대 TV 오차 (플러그인 근사)
    return kl, tv, floor, float(p.argmax() == q.argmax()), float(p.max()), float(q[p.argmax()]), float(-(p[p > 0] * np.log(p[p > 0])).sum()), float(-(q[q > 0] * np.log(q[q > 0])).sum())


def evaluate(belief, items, V):
    ab, it, mv, sp = V["ability"], V["item"], V["move"], V["species"]
    unk = {k: V[k]["<unk>"] for k in ("ability", "item", "move")}
    rows = {"특성": [], "도구": [], "기술": []}
    choice, miss = {"fp": [], "st": []}, defaultdict(list)
    opp_cat = torch.tensor(np.stack([x[0] for x in items])).long()
    opp_num = torch.tensor(np.stack([x[1] for x in items])).float()
    with torch.no_grad():
        out = {k: [] for k in ("ability", "item", "move")}
        for i in range(0, len(items), 512):
            o = belief(opp_cat[i:i + 512], opp_num[i:i + 512])
            for k in out:
                out[k].append(o[k].float().cpu())
        out = {k: torch.cat(v) for k, v in out.items()}
    for n_i, (cat, _, mons) in enumerate(items):
        for slot in range(6):
            row = cat[slot]
            if row[10] == 0:
                continue
            name = next((k for k in mons if sp.get(_norm(k)) == int(row[10])), None)
            if name is None:
                continue
            m = mons[name]
            n = m["n"]
            if int(row[1]) == unk["ability"] + 1:
                p, cov = fp_vector(m["ability"], ab, out["ability"].size(-1), n)
                miss["특성"].append(cov)
                rows["특성"].append(metrics(p, student_probs(out["ability"][n_i, slot], unk["ability"], ()), n))
            if int(row[0]) == unk["item"] + 1:
                p, cov = fp_vector(m["item"], it, out["item"].size(-1), n)
                miss["도구"].append(cov)
                q = student_probs(out["item"][n_i, slot], unk["item"], ())
                rows["도구"].append(metrics(p, q, n))
                ci = [it[c] for c in CHOICE if c in it]
                choice["fp"].append(float(p[ci].sum())); choice["st"].append(float(q[ci].sum()))
            known = [int(x) for x in row[5:9] if 1 <= int(x) < unk["move"]]
            if len(known) < 4:
                counts = {c: v for c, v in m["moves"].items() if mv.get(_norm(c)) not in known}
                p, cov = fp_vector(counts, mv, out["move"].size(-1), sum(counts.values()))
                miss["기술"].append(cov)
                if p.sum() > 0:
                    rows["기술"].append(metrics(p, student_probs(out["move"][n_i, slot], unk["move"], known), max(sum(counts.values()), 1)))
    return rows, choice, miss


def report(title, rows, choice, miss):
    print(f"\n===== {title} =====")
    print(f"{'필드':6s} {'쌍':>6s} | {'KL':>6s} {'TV':>6s} {'TV 잡음 하한':>11s} | {'최빈값 일치':>10s} {'교사 최빈 확률':>12s} {'학생이 그 값에 준 확률':>20s} | {'교사 엔트로피':>10s} {'학생 엔트로피':>10s} | {'교사 질량 중 학생 어휘':>14s}")
    for f, r in rows.items():
        if not r:
            continue
        a = np.array(r)
        print(f"{f:6s} {len(a):6d} | {a[:, 0].mean():6.3f} {a[:, 1].mean():6.3f} {a[:, 2].mean():11.3f} | {a[:, 3].mean():10.3f} {a[:, 4].mean():12.3f} {a[:, 5].mean():20.3f} | {a[:, 6].mean():10.3f} {a[:, 7].mean():10.3f} | {np.mean(miss[f]):14.3f}")
    if choice["fp"]:
        fp, st = np.array(choice["fp"]), np.array(choice["st"])
        print(f"구애 계열 도구 확률(숨겨진 도구 {len(fp)}쌍): 교사 평균 {fp.mean():.3f} / 학생 평균 {st.mean():.3f} | 상관 {np.corrcoef(fp, st)[0, 1]:.3f} | 교사>0.5인 쌍 {int((fp > 0.5).sum())}개 중 학생이 0.5 넘긴 쌍 {int(((fp > 0.5) & (st > 0.5)).sum())}개")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--models", default="v6_full")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    V = json.load(open("data/vocab.json", encoding="utf-8"))
    items = collect(args.probe, args.workers)
    n_pairs = sum(1 for x in items for s in range(6) if x[0][s][10] != 0)
    print(f"교사 믿음 기록과 맞춘 결정 {len(items)}개 (상대 포켓몬 슬롯 {n_pairs}개)")
    ck = torch.load("checkpoints/belief_27.pt", map_location="cpu")
    pre = SetBelief(**ck["cfg"]).eval()
    pre.load_state_dict(ck["state"])
    report("학생: 팀 풀 사전학습 belief_27.pt", *evaluate(pre, items, V))
    for name in args.models.split(","):
        b = model_from_ckpt(f"checkpoints/supervised_v2_fp_{name}.pt").belief.eval()
        report(f"학생: {name}의 세트 추정 (배틀 미세조정 후)", *evaluate(b, items, V))


if __name__ == "__main__":
    main()
