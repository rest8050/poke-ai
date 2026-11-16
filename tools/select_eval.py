"""선별 라벨링 사전 점검: 학생 정책만으로 '해로운 결정(regret>=0.05)'을 얼마나 골라낼 수 있나.
결정마다 점수(엔트로피, 1-최상위확률, 1등-2등 격차의 반대)를 매겨 상위 k%만 라벨링한다고 할 때, 그 안에 든 해로운 결정/총 regret 비율.
무작위 선택이면 k% 그대로(리프트 1.0). 사용: python tools/select_eval.py --model v1 --sets data/base_v1/eval/d7_ho.npz,data/base_v1/eval/d6_ho.npz,data/base_v1/eval/dg5.npz"""
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
sets = [L.load_set(p) for p in args.sets.split(",")]
z = [np.load(p) for p in args.sets.split(",")]
q, vis = np.concatenate([a["q_mean"] for a in z]), np.concatenate([a["q_vis"] for a in z])
T = np.concatenate([d["target"] for d in sets]).astype(np.float64)
M = np.concatenate([d["action_mask"] for d in sets])
rel = (vis >= 100) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)
net = model_from_ckpt(f"checkpoints/supervised_v2_fp_{args.model}.pt").to(dev).eval()
P = np.concatenate([L.infer(net, d, dev) for d in sets])
reg = regrets(P, q, rel, qstar, T)[0]
ok = valid & ~np.isnan(reg)
P, reg = P[ok], reg[ok]
ent = -(P * np.log(np.clip(P, 1e-9, 1))).sum(1)
srt = np.sort(P, 1)
scores = {"엔트로피": ent, "1-최상위확률": 1 - srt[:, -1], "1·2등 격차 작음": -(srt[:, -1] - srt[:, -2])}
harm = reg >= HARM
print(f"결정 {len(reg)} | 해로움 비율 {harm.mean():.3f} | 총 regret {reg.sum():.1f}")
print(f"{'점수':14s} " + " ".join(f"상위{k:>2d}%(해로움/regret)" for k in (10, 20, 30, 50)))
for name, s in scores.items():
    order = np.argsort(-s)
    cells = []
    for k in (10, 20, 30, 50):
        top = order[: int(len(s) * k / 100)]
        cells.append(f"{harm[top].sum() / harm.sum():.2f}/{reg[top].sum() / reg.sum():.2f}".rjust(19))
    print(f"{name:14s} " + " ".join(cells))

# 이진 특징: (선택 비율, 포착한 해로움 비율, 선택 안의 해로움 비율)
turn = np.concatenate([a["turn"] for a in z])[ok] if "turn" in z[0].files else None
am = P.argmax(1)
feats = {"학생 argmax=테라": (am >= 4) & (am < 8), "학생 argmax=교체": am >= 8, "테라 확률>0.1": P[:, 4:8].sum(1) > 0.1, "교체 확률>0.3": P[:, 8:14].sum(1) > 0.3}
if turn is not None:
    feats.update({"턴>=20": turn >= 20, "턴>=30": turn >= 30})
print(f"{'특징':16s} 선택비율  해로움포착  선택내 해로움률(전체 {harm.mean():.3f})")
for name, f in feats.items():
    print(f"{name:16s} {f.mean():7.3f}  {harm[f].sum() / harm.sum():9.3f}  {harm[f].mean():.3f}")
