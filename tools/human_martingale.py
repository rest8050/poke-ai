"""FP가 사람 행동을 과소평가하는가: 결정 t에서 둔 행동 a의 FP 가치 Q_t(a)와, 그 행동 뒤 같은 편의 다음 결정에서 FP가 매긴 가치 V_{t+1}의 차 Δ = V_{t+1} − Q_t(a).
FP의 Q가 보정돼 있으면 Δ의 평균은 0 (마팅게일). 사람이 FP가 모르는 걸 알아서 FP가 낮게 본 행동이 실제로는 좋다면, regret이 큰 구간에서 Δ > 0.
Q가 낮은 행동을 골라 보는 선택 편향(평균으로의 회귀)과 깊은 탐색의 차이를 상쇄하려고, 같은 구간 분할을 기준선(est2 변환 판에서 학생이 실제로 둔 행동, 같은 변환 잡음)에도 적용해 사람 − 기준선을 읽는다.
Q는 방문 >= 100, 선택지는 드러난 기술 + 교체로 제한 (tools/human_vs_fp.py의 measure와 같은 정의). 마지막 결정(다음 결정 없음)은 뺌.
사용: python tools/human_martingale.py --dir data/human_vs_fp"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from human_vs_fp import human_actions, kind, load_labels   # noqa: E402
from regret_eval import boot_ci                            # noqa: E402
from teacher_ceiling import qvals                          # noqa: E402

BINS = (("regret<0.02", 0, 0.02), ("0.02~0.05", 0.02, 0.05), ("0.05~0.15", 0.05, 0.15), (">=0.15", 0.15, 9))


def rows_of(rec_path, lab_path, band):
    """한 파일의 (판 id, 구간, 행동 종류, regret, Δ) 행들"""
    L = load_labels(lab_path)
    order = defaultdict(list)
    for k in L:
        order[k[0]].append(k)
    out = []
    for line in open(rec_path, encoding="utf-8"):
        rec = json.loads(line)
        ha, rev, tag = human_actions(rec), set(rec["meta"]["revealed"]), rec["tag"]
        ok = lambda c: c.startswith("switch ") or c.removesuffix("-tera") in rev
        ks = order.get(tag, [])
        for i, k in enumerate(ks[:-1]):
            act, nxt = ha.get(k[1:]), L[ks[i + 1]].get("value")
            if act is None or nxt is None:
                continue
            q = {c: v for c, v in qvals(L[k], 100).items() if ok(c)}
            if len(q) < 2 or act not in q:
                continue
            out.append(((rec_path, tag), band(rec), kind(act), max(q.values()) - q[act], nxt - q[act]))
    return out


def show(title, rows, rng):
    gid = {g: i for i, g in enumerate(sorted({r[0] for r in rows}))}
    battle = np.array([gid[r[0]] for r in rows]); reg = np.array([r[3] for r in rows]); d = np.array([r[4] for r in rows])
    cells = []
    for nm, lo, hi in BINS:
        m = (reg >= lo) & (reg < hi)
        cells.append(f"{nm} n={int(m.sum()):4d} Δ {('{:+.3f} [{:+.3f},{:+.3f}]'.format(*boot_ci(d[m], battle[m], rng)) if m.sum() >= 30 else 'n/a'):28s}")
    print(f"{title:18s} " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default="data/fp_fidelity_est2")
    args = ap.parse_args()
    H = []
    for p in sorted(glob.glob(f"{args.dir}/rec_hv_h*.jsonl")):
        lab = f"{args.dir}/lab_{os.path.basename(p)}"
        if os.path.exists(lab):
            H += rows_of(p, lab, lambda rec: rec["meta"]["band"])
    B = []
    for p in sorted(glob.glob(f"{args.baseline}/rec_*_h*.jsonl")):
        lab = f"{args.baseline}/lab_{os.path.basename(p)}"
        if os.path.exists(lab):
            B += rows_of(p, lab, lambda rec: "base")
    rng = np.random.RandomState(0)
    print(f"Δ = V_(t+1) − Q_t(둔 행동) (FP 단위, 우리 편 승률 쪽). 사람 {len(H)}결정, 기준선(학생이 둔 행동) {len(B)}결정")
    show("기준선(학생 행동)", B, rng)
    for b in ("top", "mid"):
        show(f"사람 [{b}]", [r for r in H if r[1] == b], rng)
    print("\n[행동 종류별]")
    for kn in ("기술", "테라 기술", "교체"):
        show(f"기준선 {kn}", [r for r in B if r[2] == kn], rng)
        show(f"사람 top {kn}", [r for r in H if r[1] == "top" and r[2] == kn], rng)
    print("\n[전체 평균 Δ (보정 확인)]")
    for nm, R in (("기준선", B), ("사람 top", [r for r in H if r[1] == "top"]), ("사람 mid", [r for r in H if r[1] == "mid"])):
        gid = {g: i for i, g in enumerate(sorted({r[0] for r in R}))}
        print(f"{nm:10s} {np.mean([r[4] for r in R]):+.4f} [{'{:+.4f},{:+.4f}'.format(*boot_ci(np.array([r[4] for r in R]), np.array([gid[r[0]] for r in R]), rng)[1:])}]")


if __name__ == "__main__":
    main()
