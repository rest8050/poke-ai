"""cand_attn(후보 교차 어텐션 헤드) / tera_split(칸별 테라 게이트) self-check: PYTHONPATH=. python tests/test_cand_attn.py
1) 꺼져 있으면 cfg/파라미터 그대로, 2) 합법 칸 확률 합 1 + 불법 칸 0, 3) 테라 게이트가 확률을 일반/테라로 나눔(테라 불법이면 0, 일반 불법이면 테라로 몰림),
4) 체크포인트 왕복, 5) 후보 어텐션/게이트까지 기울기가 닿음"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from test_pair_glu import random_obs

torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
small = {"model": "v8", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "head_width": 32, "my_pos": False}
off = build_model(small).eval()
assert "cand_attn" not in off.cfg and hasattr(off, "move_score")
L = 4
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True      # 기술 0-3, 테라는 슬롯 1만, 교체 9-11
mask[1, 5] = False                                                                              # 1행: 테라 불법
mask[2, 0] = False                                                                              # 2행: 슬롯 0은 일반 불법
mask[2, 4] = True                                                                               # (테라만 합법)
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m: m.forward_sequences(obs, Z, T, 1, mask)["policy_logits"]

for flags in ({"cand_attn": True}, {"cand_attn": True, "tera_split": True}):
    m = build_model({**small, **flags}).eval()
    assert not hasattr(m, "move_score") and all(m.cfg[k] for k in flags)
    with torch.no_grad():
        p = fw(m).exp()
    assert torch.allclose(p.sum(-1), torch.ones(L), atol=1e-4), p.sum(-1)
    assert (p[~mask] < 1e-6).all()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt"); save_ckpt(m, path)
        y = model_from_ckpt(path).eval()
        with torch.no_grad():
            assert torch.allclose(fw(y), fw(m), atol=1e-6)
    m.train(); fw(m)[mask].sum().backward()
    assert m.c_enc.layers[0].linear1.weight.grad.abs().sum() > 0
    if "tera_split" in flags:
        assert m.tera_net[0].weight.grad.abs().sum() > 0
        m.eval()
        with torch.no_grad():
            m.tera_net[-1].weight.zero_(); m.tera_net[-1].bias.fill_(30.0); q = fw(m).exp()          # 게이트 g=1: 테라가 합법인 슬롯 1은 일반 칸 0 + 테라 칸 양수, 테라 불법 행은 테라 칸 0
            assert q[0, 1] < 1e-6 and q[0, 5] > 0 and (q[1, 4:8] < 1e-6).all()
            m.tera_net[-1].bias.fill_(-30.0); q = fw(m).exp()                                          # g=0: 테라 0, 단 일반 불법(2행 슬롯 0)은 테라로 몰림
            assert (q[0, 4:8] < 1e-6).all() and q[2, 4] > 0 and q[2, 0] < 1e-6
print("ok")
