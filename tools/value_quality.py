"""가치 헤드 품질: 학생 상태 홀드아웃(d7_ho,d6_ho,dg5)에서 모델 가치 v와 (a) 교사(Foul Play 루트) 가치 teacher_value, (b) 실제 결과 목표 value_target의 상관/MSE.
사용: python tools/value_quality.py <모델 이름|경로> ...   (teacher_value는 0~1 승률 -> 비교는 +-1 척도로 환산, 없는 세트는 건너뜀)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

paths = "data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz".split(",")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = [L.load_set(p) for p in paths]; zs = [np.load(p) for p in paths]
tv = np.concatenate([z["teacher_value"] * 2 - 1 if "teacher_value" in z.files else np.full(len(z["value_target"]), np.nan) for z in zs])
vt = np.concatenate([z["value_target"] for z in zs])
for name in sys.argv[1:]:
    net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
    mv_in = net.embeddings.move_encoder.fc[0].in_features - 48
    vs = []
    with torch.no_grad():
        for s in sets:
            lens, off = s["lens"], np.concatenate([[0], np.cumsum(s["lens"])])
            for i in range(0, len(lens), 32):
                sl = slice(off[i], off[min(i + 32, len(lens))])
                obs = [torch.tensor(s[k][sl][..., :mv_in] if k in ("my_move_num", "opp_move_num") else s[k][sl]).to(dev) for k in L.OBS]
                ch = lens[i:i + 32]
                si = torch.cat([torch.full((n,), j) for j, n in enumerate(ch)]).to(dev); ti = torch.cat([torch.arange(n) for n in ch]).to(dev)
                vs.append(net.forward_sequences(obs, si, ti, len(ch), action_mask=torch.tensor(s["action_mask"][sl]).to(dev))["value"].float().view(-1).cpu().numpy())
    v = np.concatenate(vs); ok = ~np.isnan(tv)
    print(f"{os.path.basename(name):32s} n={len(v)} | 결과 목표: 상관 {np.corrcoef(v, vt)[0, 1]:.3f} MSE {np.mean((v - vt) ** 2):.3f} | 교사 가치({int(ok.sum())}): 상관 {np.corrcoef(v[ok], tv[ok])[0, 1]:.3f} MSE {np.mean((v[ok] - tv[ok]) ** 2):.3f}")
