"""기술/교체 종류 결정이 후보별 정보를 요약 벡터로만 받아서 손해를 보는가 (동결 프로브).
type_head 입력 = latent + 활성의 cand + (합법 후보 중 최고 공격 상성, 최저 위협) 두 값 + 전역 요약. 후보별 점수(switch_score)는 type_head가 못 봄.
정답 = 교사 교체 질량 >= 0.5 (자발 결정: 기술과 교체 둘 다 가능). 시험 = 판 번호 %5 < 2. 로지스틱(표준화)으로 AUC 비교:
 (1) 학생 자신의 교체 질량  (2) type_head 입력 전체  (3) (2) + 후보별 switch_score의 최댓값/logsumexp/평균  (4) (2) + 후보별 switch_score 은닉(192)의 합법 후보 max-pool.
(3), (4)가 (2)보다 AUC가 의미 있게 오르면 후보별 정보가 type_head에서 손실되는 중.
사용: python tools/kind_probe.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="v8_full")
ap.add_argument("--sets", required=True)
args = ap.parse_args()
sets = [L.load_set(p) for p in args.sets.split(",")]
net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
cap = {"sc": [], "hid": [], "tin": []}
hs = [net.switch_score.register_forward_hook(lambda m, i, o: cap["sc"].append(o.detach()[..., 0])),
      net.switch_score[1].register_forward_hook(lambda m, i, o: cap["hid"].append(o.detach())),
      net.type_head.register_forward_hook(lambda m, i, o: cap["tin"].append(i[0].detach()))]
P = np.concatenate([L.infer(net, s, torch.device("cpu")) for s in sets])
for h in hs:
    h.remove()
# switch_score는 배치마다 [B*6?] 모양일 수 있어 [B,6,...]로 맞춤
sc = torch.cat([c.reshape(-1, 6) for c in cap["sc"]]).numpy()
hid = torch.cat([c.reshape(-1, 6, c.shape[-1]) for c in cap["hid"]]).numpy()
tin = torch.cat(cap["tin"]).numpy()
T = np.concatenate([s["target"] for s in sets]).astype(np.float64)
M = np.concatenate([s["action_mask"] for s in sets])
game = np.concatenate([np.repeat(np.arange(len(s["lens"])) + 1000 * i, s["lens"]) for i, s in enumerate(sets)])
vol = (M[:, :4].sum(1) > 0) & (M[:, 8:14].sum(1) > 0)
ok = M[:, 8:14]
neg = np.where(ok, sc, -1e9)
pooled = np.stack([neg.max(1), np.log(np.exp(np.where(ok, sc, -50.0) - neg.max(1, keepdims=True)).sum(1)) + neg.max(1), np.where(ok, sc, 0).sum(1) / np.maximum(ok.sum(1), 1)], 1)
hmax = np.where(ok[..., None], hid, -1e9).max(1)
hmax = np.where(ok.any(1)[:, None], hmax, 0)
pos = T[:, 8:14].sum(1) >= 0.5
test = (game % 5) < 2
print(f"모델 {args.models} | 자발 결정 {int(vol.sum())} (시험 {int((vol & test).sum())}) | 정답 비율 {pos[vol].mean():.3f} | type_head 입력 {tin.shape[1]}차원")
def fit(X, name):
    sc_ = StandardScaler().fit(X[vol & ~test])
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc_.transform(X[vol & ~test]), pos[vol & ~test])
    s = clf.predict_proba(sc_.transform(X[vol & test]))[:, 1]
    print(f"  {name:48s} AUC {roc_auc_score(pos[vol & test], s):.4f}")
print(f"  {'(1) 학생 자신의 교체 질량 (학습된 헤드)':48s} AUC {roc_auc_score(pos[vol & test], P[vol & test][:, 8:14].sum(1)):.4f}")
fit(tin, "(2) type_head 입력 전체 (로지스틱)")
fit(np.concatenate([tin, pooled], 1), "(3) (2) + 후보 점수 max/logsumexp/평균")
fit(np.concatenate([tin, pooled, hmax], 1), "(4) (3) + 후보 은닉 max-pool")
fit(np.concatenate([pooled], 1), "(5) 후보 점수 요약만 (3개)")
