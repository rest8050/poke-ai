"""상대 행동을 강제하는 기술(앙코르/도발/트집/사슬묶기)과 위협(Intimidate) 교체 이득을 학생이 교사(FP)만큼 쓰는가.
(A) 내 활성이 해당 기술을 가진 결정: 교사/학생 정책 질량(테라 칸 포함), 교사 argmax가 그 기술일 때 학생 일치/regret/전체 regret 몫, 학생 질량의 AUC (정답 = 교사 질량 >= 0.5), 대조 = 그 밖의 모든 교사 argmax 기술.
(B) 벤치에 살아 있는 위협 특성 포켓몬이 있는 결정: 그 포켓몬으로의 교체에 교사/학생이 둔 질량, 교사 확신(>= 0.5) 위협 교체에서 학생 일치/regret, 대조 = 교사 확신 다른 교체.
사용: python tools/forcing_eval.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import regrets
from src.core.model import model_from_ckpt

V = json.load(open("data/vocab.json", encoding="utf-8"))
FORCING = ("encore", "taunt", "torment", "disable")
INTIM = V["ability"]["intimidate"]

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="v8_full")
ap.add_argument("--sets", required=True)
ap.add_argument("--min-vis", type=float, default=100)
args = ap.parse_args()
paths = args.sets.split(",")
sets = [L.load_set(p) for p in paths]
zs = [np.load(p) for p in paths]
q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
cat = lambda k: np.concatenate([s[k] for s in sets])
T, M = cat("target").astype(np.float64), cat("action_mask")
rel = (vis >= args.min_vis) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)
net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
P = np.concatenate([L.infer(net, s, torch.device("cpu")) for s in sets])
reg, *_ = regrets(P, q, rel, qstar, T)
reg = np.where(valid, reg, np.nan)
tot = np.nansum(reg)
mc, mn = cat("my_team_cat"), cat("my_team_num")
n, ar = len(P), np.arange(len(P))
ma = mn[:, :, 0].argmax(1)
mv = mc[ar, ma, 5:9]                                           # 내 활성의 기술 id 4칸
a_t, a_s = T.argmax(1), P.argmax(1)
vol = (M[:, :4].sum(1) > 0) & valid
print(f"모델 {args.models} | 비교 가능 {int(valid.sum())}결정 | 전체 regret 합 기준 몫으로 표기")

print("\n[A] 상대 행동 강제 기술 (내 활성이 보유, 그 칸이 선택 가능)")
print(f"{'기술':9s} {'보유 결정':>8s} | {'교사 질량':>8s} {'학생 질량':>8s} {'AUC':>6s} | {'교사 argmax':>10s} {'학생 일치':>8s} {'평균 regret':>11s} {'regret 몫':>9s}")
allc = np.zeros(n, bool)
for nm in FORCING:
    mid = V["move"][nm]
    slot = (mv == mid).argmax(1)
    has = (mv == mid).any(1) & valid & M[ar, slot]
    tm = T[ar, slot] + np.where(M[ar, slot + 4], T[ar, slot + 4], 0)          # 일반 + 테라 칸
    sm = P[ar, slot] + np.where(M[ar, slot + 4], P[ar, slot + 4], 0)
    is_t = has & ((a_t == slot) | (a_t == slot + 4))
    allc |= is_t
    if has.sum() < 30:
        continue
    hit = is_t & ((a_s == slot) | (a_s == slot + 4))
    print(f"{nm:9s} {int(has.sum()):8d} | {tm[has].mean():8.3f} {sm[has].mean():8.3f} {L.auc(sm[has], tm[has] >= 0.5):6.3f} | {int(is_t.sum()):10d} {hit[is_t].mean() if is_t.sum() else np.nan:8.3f} {np.nanmean(reg[is_t]) if is_t.sum() else np.nan:11.4f} {np.nansum(reg[is_t]) / tot:9.4f}")
other = vol & (a_t < 8) & ~allc
print(f"{'(대조) 그 밖의 교사 argmax 기술':20s} 교사 argmax {int(other.sum())}결정 | 학생 일치 {np.mean((a_s == a_t)[other]):.3f} 평균 regret {np.nanmean(reg[other]):.4f}")
print(f"강제 기술이 교사 argmax인 결정 합계 {int(allc.sum())}건 ({allc.sum() / valid.sum():.2%}), 학생 일치 {np.mean(((a_s % 4 == a_t % 4) & (a_s < 8))[allc]):.3f} (칸 기준), regret 몫 {np.nansum(reg[allc]) / tot:.4f}")

print("\n[B] 위협(Intimidate) 포켓몬 교체")
ab = mc[:, :, 1]
alive = (mn[:, :, 1] < 0.5) & (mc[:, :, 10] > 2)
sw_ok = M[:, 8:14]
intim = (ab == INTIM) & alive & sw_ok & (np.arange(6)[None] != ma[:, None])      # 벤치의 위협 포켓몬 중 교체 가능한 칸
has = intim.any(1) & valid
tm = (T[:, 8:14] * intim).sum(1); sm = (P[:, 8:14] * intim).sum(1)
print(f"벤치에 위협 포켓몬 있는 결정 {int(has.sum())} ({has.mean():.1%}) | 그 포켓몬으로 교체: 교사 질량 평균 {tm[has].mean():.3f} 학생 {sm[has].mean():.3f} | AUC(학생 질량 -> 교사 >= 0.5) {L.auc(sm[has], tm[has] >= 0.5):.3f}")
tgt_is_i = np.zeros(n, bool)
sw_t = (a_t >= 8) & (a_t < 14)
tgt_is_i[sw_t] = (ab[ar, (a_t - 8).clip(0, 5)] == INTIM)[sw_t]
conf_o = vol & sw_t & (T.max(1) >= 0.5) & ~tgt_is_i
conf_i2 = vol & sw_t & (T.max(1) >= 0.5) & tgt_is_i
for nm, m in (("교사 확신 위협 교체", conf_i2), ("(대조) 교사 확신 다른 교체", conf_o)):
    print(f"{nm:22s} {int(m.sum()):6d}결정 | 학생 argmax 일치 {np.mean((a_s == a_t)[m]):.3f} | 학생이 교체 자체는 선택 {np.mean(((a_s >= 8) & (a_s < 14))[m]):.3f} | 평균 regret {np.nanmean(reg[m]):.4f} | regret 몫 {np.nansum(reg[m]) / tot:.4f}")
