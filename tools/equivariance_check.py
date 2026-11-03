"""슬롯 순서 불변성 점검: 내 팀 6칸(또는 상대 6칸)의 순서를 판 단위로 섞고(같은 판은 모든 턴에 같은 순열), 교체 칸(8~13)의 행동 마스크/확률도 같이 섞었을 때 정책이 같은지.
내 칸: 섞은 입력의 정책 P'[8+j] 가 원래 P[8+pi[j]] 와 같아야 함 (등변), 상대 칸: P' = P (불변). 어긋나면 모델이 칸 순서(팀 입력 순서)를 쓰고 있다는 뜻.
사용: python tools/equivariance_check.py --models v8_full --set data/dagger/d7_ho.npz --n 300"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="v8_full")
ap.add_argument("--set", required=True)
ap.add_argument("--n", type=int, default=300)
args = ap.parse_args()
d = L.load_set(args.set)
lens = d["lens"][:args.n]
rows = int(lens.sum())
net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
rng = np.random.RandomState(0)


def run(perm_my, perm_opp):
    """perm_*: [시퀀스, 6]"""
    out, off = [], np.concatenate([[0], np.cumsum(lens)])
    mv_in = net.embeddings.move_encoder.fc[0].in_features - 48
    with torch.no_grad():
        for i in range(0, len(lens), 32):
            sl = slice(off[i], off[min(i + 32, len(lens))])
            ch = lens[i:i + 32]
            pm = np.repeat(perm_my[i:i + 32], ch, 0); po = np.repeat(perm_opp[i:i + 32], ch, 0)          # [행, 6]
            ar = np.arange(len(pm))[:, None]
            obs = {k: d[k][sl] for k in L.OBS}
            obs["my_team_cat"], obs["my_team_num"], obs["my_move_num"] = (obs[k][ar, pm] for k in ("my_team_cat", "my_team_num", "my_move_num"))
            obs["opp_team_cat"], obs["opp_team_num"], obs["opp_move_num"] = (obs[k][ar, po] for k in ("opp_team_cat", "opp_team_num", "opp_move_num"))
            mask = d["action_mask"][sl].copy()
            mask[:, 8:14] = mask[:, 8:14][ar, pm]
            ts = [torch.tensor(obs[k][..., :mv_in] if k in ("my_move_num", "opp_move_num") else obs[k]) for k in L.OBS]
            si = torch.cat([torch.full((n,), j) for j, n in enumerate(ch)]); ti = torch.cat([torch.arange(n) for n in ch])
            lg = net.forward_sequences(ts, si, ti, len(ch), action_mask=torch.tensor(mask))["policy_logits"]
            out.append(torch.softmax(lg.double(), -1).numpy())
    return np.concatenate(out)


ident = np.tile(np.arange(6), (len(lens), 1))
P0 = run(ident, ident)
pm = np.stack([rng.permutation(6) for _ in lens]); po = np.stack([rng.permutation(6) for _ in lens])
Pm = run(pm, ident)
Po = run(ident, po)
seq = np.repeat(np.arange(len(lens)), lens)
back = P0.copy()
back[:, 8:14] = P0[np.arange(rows)[:, None], 8 + pm[seq]]                      # 등변이면 Pm[8+j] == P0[8+pm[j]]
M = d["action_mask"][:rows]
live = M.sum(1) >= 2
tv = lambda a, b: 0.5 * np.abs(a - b).sum(1)
print(f"모델 {args.models} | 결정 {rows}개 ({len(lens)}판), 합법 행동 2개 이상 {int(live.sum())}개")
print(f"내 칸 섞기: 정책 총변동거리 평균 {tv(Pm, back)[live].mean():.4f} | 최대 {tv(Pm, back)[live].max():.3f} | argmax가 달라진 비율 {np.mean((Pm.argmax(1) != back.argmax(1))[live]):.4f}")
print(f"상대 칸 섞기: 정책 총변동거리 평균 {tv(Po, P0)[live].mean():.4f} | 최대 {tv(Po, P0)[live].max():.3f} | argmax가 달라진 비율 {np.mean((Po.argmax(1) != P0.argmax(1))[live]):.4f}")
print(f"(참고) 같은 입력 두 번: {tv(P0, run(ident, ident))[live].mean():.6f}")
