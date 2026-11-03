"""디코딩이 교체 약점을 만드는가: 같은 학생 분포에서 (a) 칸 argmax, (b) 종류 우선(기술/테라/교체 질량 합이 최대인 종류를 고른 뒤 그 안에서 argmax), (c) 정책에서 표집한 평균 regret을 FP Q로 잰다.
교체 질량은 최대 6칸으로 쪼개지고 기술은 4칸이라 칸 argmax가 교체에 불리할 수 있음. 교사 argmax 종류별로도 낸다.
사용: python tools/decode_eval.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import boot_ci, regrets
from src.core.model import model_from_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="v8_full")
ap.add_argument("--sets", required=True)
args = ap.parse_args()
paths = args.sets.split(",")
sets = [L.load_set(p) for p in paths]
zs = [np.load(p) for p in paths]
q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
T, M = np.concatenate([s["target"] for s in sets]).astype(np.float64), np.concatenate([s["action_mask"] for s in sets])
rel = (vis >= 100) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)
net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
P = np.concatenate([L.infer(net, s, torch.device("cpu")) for s in sets])
reg_arg, reg_t, reg_pol, *_ = regrets(P, q, rel, qstar, T)
gap = np.where(rel, qstar[:, None] - q, np.nan)
n, ar = len(P), np.arange(len(P))
KIND = np.array([0] * 4 + [1] * 4 + [2] * 6)                                   # 기술 / 테라 / 교체 (14~21번 칸은 안 씀)
mass = np.stack([P[:, :14][:, KIND == k].sum(1) for k in range(3)], 1)
kind = mass.argmax(1)
dec = np.array([np.where(KIND == kind[i], P[i, :14], -1).argmax() for i in range(n)])      # 종류 우선 디코딩
reg_kind = np.where(rel[ar, dec], gap[ar, dec], np.nan)
battle = np.concatenate([np.full(len(s["target"]), i) for i, s in enumerate(sets)])
rng = np.random.RandomState(0)
f = lambda x, m: "{:.4f} [{:.4f},{:.4f}]".format(*boot_ci(x[m], np.arange(n)[m] // 50, rng))
a_t = T.argmax(1)
print(f"모델 {args.models} | 비교 가능 {int(valid.sum())}결정")
print(f"{'구분':14s} {'결정':>6s} | {'칸 argmax':>24s} | {'종류 우선':>24s} | {'표집(평균 정책)':>24s} | 교사 argmax")
for nm, m in (("전체", valid), ("교사 교체", valid & (a_t >= 8)), ("교사 기술", valid & (a_t < 4)), ("교사 테라", valid & (a_t >= 4) & (a_t < 8))):
    ok = m & ~np.isnan(reg_arg) & ~np.isnan(reg_kind)
    print(f"{nm:14s} {int(ok.sum()):6d} | {f(reg_arg, ok):>24s} | {f(reg_kind, ok):>24s} | {f(reg_pol, ok):>24s} | {np.nanmean(reg_t[ok]):.4f}")
sw_t = valid & (a_t >= 8)
print(f"\n교사 교체 결정에서 학생 종류 선택 일치: 칸 argmax {np.mean((P.argmax(1) >= 8)[sw_t]):.3f} → 종류 우선 {np.mean((kind == 2)[sw_t]):.3f} | 학생 교체 총질량 평균 {P[sw_t][:, 8:14].sum(1).mean():.3f} (교사 {T[sw_t][:, 8:14].sum(1).mean():.3f})")
print(f"교사 교체 결정에서 (종류 우선으로 교체를 골랐을 때) 대상 일치 {np.mean((dec == a_t)[sw_t & (kind == 2)]):.3f}")

# 위임 곡선: 학생이 불확실한 결정 상위 x%를 교사(탐색)로 바꾸면 평균 regret이 얼마나 줄어드는가 (오프라인 추정: 바꾼 결정의 regret = 교사 argmax regret)
H = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
swm = P[:, 8:14].sum(1)
ok = valid & ~np.isnan(reg_arg) & ~np.isnan(reg_t)
base = np.nanmean(reg_arg[ok])
print(f"\n[위임 곡선] 학생만 {base:.4f}, 전부 교사 {np.nanmean(reg_t[ok]):.4f}. 위임 비율별 평균 regret (괄호 = 제거한 regret 비율)")
for nm, score in (("정책 엔트로피 큰 순", H), ("1 - 최대 확률 큰 순", 1 - P.max(1)), ("교체 질량 0.5에 가까운 순", -np.abs(swm - 0.5))):
    idx = np.where(ok)[0][np.argsort(-score[ok])]
    out = []
    for frac in (0.05, 0.1, 0.2, 0.3, 0.5):
        k = int(frac * len(idx)); r = reg_arg[ok].copy()
        r_all = np.where(np.isin(np.arange(n), idx[:k]), reg_t, reg_arg)[ok]
        out.append(f"{frac:.0%}: {r_all.mean():.4f} ({1 - r_all.mean() / base:.0%})")
    print(f"  {nm:22s} " + " | ".join(out))
