"""모델 구조별 학습 한 스텝(전진+역전파) 시간/메모리 측정. 사용: python tools/bench_train_step.py '<arch JSON>' [스텝 수=8] [배틀 수=32] [amp=0|1]
데이터는 dg2_ho 앞쪽 배틀을 씀 (학습 루프와 같은 형태의 배치). 다른 프로세스가 GPU를 쓰고 있으면 값이 부풀려지니 단독으로 돌릴 것"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.model import build_model

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
arch = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
nb = int(sys.argv[3]) if len(sys.argv) > 3 else 32
amp = (sys.argv[4] == "1") if len(sys.argv) > 4 else False
dev = "cuda"
z = np.load("data/dagger/dg2_ho.npz")
lens = z["lens"][:nb]
n = int(lens.sum())
obs = [torch.tensor(z[k][:n]).to(dev) for k in KEYS]
mv_in = 46
for i in (2, 5):
    if obs[i].size(-1) < mv_in:
        obs[i] = torch.cat([obs[i], obs[i].new_zeros(*obs[i].shape[:-1], mv_in - obs[i].size(-1))], -1)
seq = torch.cat([torch.full((int(L),), b) for b, L in enumerate(lens)]).to(dev)
tix = torch.cat([torch.arange(int(L)) for L in lens]).to(dev)
mask = torch.tensor(z["action_mask"][:n]).to(dev)
tgt = torch.tensor(z["target"][:n]).to(dev)
m = build_model(arch).to(dev).train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
print(f"구조 {arch or 'v1 기본'} | 파라미터 {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M | 배치 = {nb}판 / {n}턴 | amp={amp}")
torch.cuda.reset_peak_memory_stats()
ts = []
for i in range(steps + 2):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = m.forward_sequences(obs, seq, tix, nb, mask)
        loss = -(tgt * torch.log_softmax(out["policy_logits"].float(), -1)).sum(-1).mean() + out["value"].float().pow(2).mean() * 0.25
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    if i >= 2:
        ts.append(time.time() - t0)
print(f"스텝당 {np.mean(ts) * 1000:.0f} ms (중앙값 {np.median(ts) * 1000:.0f}) | 최대 GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
