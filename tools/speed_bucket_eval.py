"""선공 불확실 구간별 regret: 매치업이 준 스피드 특징(spd_my = 내 활성/상대 활성 스피드 비 log2/2, fm_my = 기술칸 선공 기대값 -1~1)으로 결정을 나누고 구간마다 regret을 비교.
스피드가 비슷할수록(|spd_my| 작음), 교사가 고른 기술의 선공이 불확실할수록(|fm| 작음) 학생 regret이 높으면 "선공 의존 판단이 약하다"는 근거.
사용: python tools/speed_bucket_eval.py --model v1 --sets data/base_v1/eval/d7_ho.npz,data/base_v1/eval/d6_ho.npz,data/base_v1/eval/dg5.npz"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import HARM, regrets
from src.core.model import model_from_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="v1")
ap.add_argument("--sets", required=True)
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = args.sets.split(",")
sets = [L.load_set(p) for p in paths]
z = [np.load(p) for p in paths]
q, vis = np.concatenate([a["q_mean"] for a in z]), np.concatenate([a["q_vis"] for a in z])
T = np.concatenate([d["target"] for d in sets]).astype(np.float64)
M = np.concatenate([d["action_mask"] for d in sets])
mt = np.concatenate([d["my_team_num"] for d in sets])
rel = (vis >= 100) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)
net = model_from_ckpt(f"checkpoints/supervised_v2_fp_{args.model}.pt").to(dev).eval()
cap = []
net.matchup.register_forward_hook(lambda m, i, o: cap.append((o["spd_my"].cpu().numpy(), o["fm_my"].cpu().numpy())))
P = np.concatenate([L.infer(net, d, dev) for d in sets])
spd, fm = np.concatenate([c[0] for c in cap]), np.concatenate([c[1] for c in cap])
assert len(spd) == len(P), (len(spd), len(P))
ar = np.arange(len(P))
act = mt[..., 0].argmax(1)
reg_s, reg_t, _, _, a_s, a_t = regrets(P, q, rel, qstar, T)
ok = valid & ~np.isnan(reg_s)
spd_a = np.abs(spd[ar, act])                                                       # 내 활성의 스피드 근접도 (0 = 거의 같음, 1 = 4배 이상)
mv = a_t < 8                                                                       # 교사가 기술(테라 포함)을 고른 결정
fm_t = np.abs(fm[ar, act, a_t % 4])                                                # 교사가 고른 기술칸의 |선공 기대값| (1 = 확실, 0 = 불확실/동속)
print(f"{args.model}: 결정 {ok.sum()} | 전체 argmax regret {reg_s[ok].mean():.4f} | 해로움 {np.mean(reg_s[ok] >= HARM):.3f}")


def table(title, score, edges, mask):
    print(f"\n{title}\n{'구간':>14s} {'결정':>7s} {'학생 regret':>11s} {'교사 regret':>11s} {'해로움률':>8s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = mask & ok & (score >= lo) & (score < hi)
        if s.sum():
            print(f"{f'[{lo:.2f},{hi:.2f})':>14s} {s.sum():7d} {reg_s[s].mean():11.4f} {np.nanmean(reg_t[s]):11.4f} {np.mean(reg_s[s] >= HARM):8.3f}")


table("(1) 스피드 근접도 |spd_my| (내 활성 vs 상대 활성, 전체 결정)", spd_a, [0, 0.05, 0.15, 0.3, 0.6, 1.01], np.ones(len(P), bool))
table("(2) 교사가 고른 기술의 |선공 기대값| |fm| (기술 결정만)", fm_t, [0, 0.2, 0.5, 0.8, 0.999, 1.01], mv)
