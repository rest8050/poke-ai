"""반복 추론 블록(model_v8 loop_n) 검사: 출력 투영이 0으로 시작해 처음엔 v8과 출력이 같고(= v8 가중치에서 이어 학습 가능), 기울기가 반복 층에 닿고, 파라미터가 예상만큼 늘어남.
사용: PYTHONPATH=. python tests/test_loop_block.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import legal_eval as L
from src.core.model import build_model

torch.manual_seed(0)
d = L.load_set("data/dagger/d7_ho.npz")
n = int(d["lens"][:2].sum())
d = {k: (v[:n] if k != "lens" else v[:2]) for k, v in d.items()}
base = build_model({"model": "v8", "my_pos": False}).eval()
loop = build_model({"model": "v8", "my_pos": False, "loop_n": 4, "loop_layers": 2}).eval()
missing, unexpected = loop.load_state_dict(base.state_dict(), strict=False)
assert not unexpected and all(k.startswith(("loop.", "iter_emb", "bias_")) for k in missing), missing
for i, kind in enumerate(("bias_def", "bias_off", "bias_hid")):          # 반복 층의 편향 모듈은 새 것(기본 층 몫은 위에서 복사됨)
    assert len(getattr(loop, kind)) == len(getattr(base, kind)) + 2
a, b = L.infer(base, d, "cpu"), L.infer(loop, d, "cpu")
assert np.abs(a - b).max() < 1e-5, np.abs(a - b).max()
extra = sum(p.numel() for p in loop.parameters()) - sum(p.numel() for p in base.parameters())
print(f"처음 출력 차이 {np.abs(a - b).max():.2e}, 추가 파라미터 {extra / 1e6:.2f}M")
assert 1.5e6 < extra < 1.8e6, extra

loop.train()                                                       # 기울기: 출력 투영이 0이라 첫 역전파는 out_proj/linear2/iter_emb에만 닿음
obs = [torch.tensor(d[k]) for k in L.OBS]
lens = d["lens"]
si = torch.cat([torch.full((m,), j) for j, m in enumerate(lens)]); ti = torch.cat([torch.arange(m) for m in lens])
out = loop.forward_sequences(obs, si, ti, len(lens), action_mask=torch.tensor(d["action_mask"]))
out["policy_logits"].clamp_min(-30).sum().backward()
g = {k: p.grad.abs().sum().item() for k, p in loop.named_parameters() if k.startswith(("loop.0.linear2", "loop.0.self_attn.out_proj", "iter_emb"))}
assert all(v > 0 for v in g.values()), g
print("loop block OK")
