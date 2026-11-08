"""교사 확신도별 평가: 학생의 KL/일치가 어디에 몰려 있는가 (교사가 확신하는 결정 vs 아슬아슬한 결정, 교사 선택 종류별).
교사 최상위 확률 구간으로 나눠 구간별 결정 수, KL, 전체 KL 중 차지하는 몫, 일치율, 학생이 교사 최상위 행동에 준 확률을 낸다 (구간 평균은 판 단위 부트스트랩 95%)
참고: 교사 자체 재현율(같은 상태를 다시 라벨링해도 같은 선택이 나오는 비율)은 전체 0.77~0.80이고 아슬아슬한 구간일수록 낮아, 그 구간의 낮은 일치는 일부 피할 수 없음
사용: python tools/entropy_eval.py --models v9 --sets data/dagger/dg5.npz,data/dagger/d6_ho.npz"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

BINS = [("확신 (교사 최상위 >= 0.9)", 0.9, 1.01), ("0.7 ~ 0.9", 0.7, 0.9), ("0.5 ~ 0.7", 0.5, 0.7), ("아슬아슬 (< 0.5)", 0.0, 0.5)]


def mean_ci(x, mask, battle, rng, n_boot=1000):
    u, inv = np.unique(battle, return_inverse=True)
    s, n = np.bincount(inv, weights=x * mask, minlength=len(u)), np.bincount(inv, weights=mask.astype(float), minlength=len(u))
    ds = []
    for _ in range(n_boot):
        p = rng.randint(0, len(u), len(u))
        if n[p].sum() > 0:
            ds.append(s[p].sum() / n[p].sum())
    return s.sum() / max(1, n.sum()), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v9")
    ap.add_argument("--sets", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(0)
    sets = [L.load_set(p) for p in args.sets.split(",")]
    battle, off = [], 0
    for d in sets:
        battle.append(np.repeat(np.arange(len(d["lens"])), d["lens"]) + off)
        off += len(d["lens"])
    battle, T = np.concatenate(battle), np.concatenate([d["target"] for d in sets]).astype(np.float64)
    ok = np.concatenate([d["action_mask"] for d in sets]).sum(1) >= 2
    t_top, t_arg = T.max(1), T.argmax(1)
    kind = np.where(t_arg >= 8, 2, np.where(t_arg >= 4, 1, 0))                       # 0 기술, 1 테라 기술, 2 교체
    for name in args.models.split(","):
        net = model_from_ckpt(f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
        P = np.concatenate([L.infer(net, d, dev) for d in sets])
        pd = L.per_decision(P, T)
        p_on_teacher = P[np.arange(len(P)), t_arg]
        tot = pd["total"][ok].sum()
        print(f"\n===== {name}: 결정 {int(ok.sum())}, 판 {len(np.unique(battle[ok]))} | 평균 KL {pd['total'][ok].mean():.4f} | 일치 {pd['agree'][ok].mean():.3f} =====")
        hdr = f"{'구간':26s} {'결정':>6s} {'비율':>6s} | {'KL [95%]':>24s} {'KL 몫':>6s} | {'일치 [95%]':>22s} | {'학생 최상위':>9s} {'교사 최상위':>9s} {'교사선택에 준 확률':>16s}"
        print(hdr)
        for label, lo, hi in BINS:
            m = ok & (t_top >= lo) & (t_top < hi)
            if m.sum() < 50:
                continue
            k, a = mean_ci(pd["total"], m, battle, rng), mean_ci(pd["agree"], m, battle, rng)
            print(f"{label:26s} {int(m.sum()):6d} {m.sum() / ok.sum():6.1%} | {k[0]:7.4f} [{k[1]:.4f}, {k[2]:.4f}] {pd['total'][m].sum() / tot:6.1%} | "
                  f"{a[0]:6.3f} [{a[1]:.3f}, {a[2]:.3f}] | {P[m].max(1).mean():9.3f} {t_top[m].mean():9.3f} {p_on_teacher[m].mean():16.3f}")
        print(f"\n{'교사 선택 종류':26s} {'결정':>6s} {'비율':>6s} | {'KL [95%]':>24s} {'KL 몫':>6s} | {'일치 [95%]':>22s} | {'교사선택에 준 확률':>16s}")
        for label, kv in (("기술 (테라 아님)", 0), ("테라 기술", 1), ("교체", 2)):
            m = ok & (kind == kv)
            if m.sum() < 50:
                continue
            k, a = mean_ci(pd["total"], m, battle, rng), mean_ci(pd["agree"], m, battle, rng)
            print(f"{label:26s} {int(m.sum()):6d} {m.sum() / ok.sum():6.1%} | {k[0]:7.4f} [{k[1]:.4f}, {k[2]:.4f}] {pd['total'][m].sum() / tot:6.1%} | "
                  f"{a[0]:6.3f} [{a[1]:.3f}, {a[2]:.3f}] | {p_on_teacher[m].mean():16.3f}")


if __name__ == "__main__":
    main()
