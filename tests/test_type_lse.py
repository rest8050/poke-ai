"""v8 type_lse / value_detach self-check: PYTHONPATH=. python tests/test_type_lse.py
1) 꺼져 있으면 cfg/파라미터 그대로 2) type_lse는 alpha 0 시작이라 같은 가중치면 꺼진 모델과 출력 동일, alpha를 키우면 P(교체)가 변하고 종류 손실이 switch_score/move_score에 기울기를 줌
3) 합법 교체가 없는 결정도 유한 출력 4) value_detach는 출력 동일, 가치 손실이 몸통/융합층에 기울기를 안 주고 value_head에만 줌 5) 체크포인트 왕복"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):   # tests/test_mon_moves.py와 같은 입력 생성기
    my_cat = torch.randint(1, 8, (n, 6, 11), generator=gen); opp_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    my_num = torch.rand(n, 6, NUM_DIM, generator=gen); my_num[:, :, 0] = 0; my_num[:, 0, 0] = 1; my_num[:, :, 1] = 0
    opp_num = torch.rand(n, 6, NUM_DIM, generator=gen); opp_num[:, :, 0] = 0; opp_num[:, 1, 0] = 1; opp_num[:, :, 1] = 0
    my_mv, opp_mv = torch.rand(n, 6, 4, 46, generator=gen), torch.rand(n, 6, 4, 46, generator=gen)
    opp_num[:, 4:, 11:19] = 0; opp_mv[:, 4:] = 0; opp_mv[:, :4, 2:] = 0
    return [my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, torch.rand(n, FIELD_DIM, generator=gen)]


torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
small = {"model": "v8", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False}
L = 5
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
mask[4, 8:14] = False            # 교체 불가 결정
mask[3, :8] = False              # 기술 불가(강제 교체) 결정
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m: m.forward_sequences(obs, Z, T, 1, mask)

off = build_model(small).eval()
assert "type_lse" not in off.cfg and "value_detach" not in off.cfg and not hasattr(off, "lse_alpha")
on = build_model({**small, "type_lse": True, "value_detach": True}).eval()
assert on.cfg["type_lse"] is True and on.cfg["value_detach"] is True and on.lse_alpha.shape == (2,) and (on.lse_alpha == 0).all()
on.load_state_dict({k: v for k, v in off.state_dict().items()}, strict=False)
with torch.no_grad():
    a, b = fw(off), fw(on)
    assert torch.allclose(a["policy_logits"], b["policy_logits"], atol=1e-6) and torch.allclose(a["value"], b["value"], atol=1e-6)
    on.lse_alpha.copy_(torch.tensor([0.7, 1.3]))
    c = fw(on)["policy_logits"]
    assert torch.isfinite(c[mask]).all() and not torch.allclose(c[:3], a["policy_logits"][:3])
    pe = lambda x: x[:, 8:14].exp().sum(-1)
    assert pe(c)[4] == 0 or pe(c)[4] < 1e-6 and pe(c)[3] > 0.99          # 교체 불가면 교체 확률 0, 기술 불가면 교체 확률 1

# 기울기: 종류 손실이 후보 점수 헤드에 닿음 (alpha != 0), value_detach면 가치 손실이 몸통에 안 닿음
m = build_model({**small, "type_lse": True, "value_detach": True}); m.train()
with torch.no_grad():
    m.lse_alpha.copy_(torch.tensor([0.5, 0.5]))
out = fw(m)
out["policy_logits"][mask].sum().backward(retain_graph=True)
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.switch_score.parameters())
assert m.lse_alpha.grad is not None and m.lse_alpha.grad.abs().sum() > 0
m.zero_grad()
out["value"].square().sum().backward()
assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in m.named_parameters() if not n.startswith("value_head")), "가치 손실이 몸통에 닿음"
assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.value_head.parameters())
nd = build_model({**small}); nd.train(); fw(nd)["value"].square().sum().backward()
assert nd.fusion[0].weight.grad.abs().sum() > 0          # detach 없으면 닿음 (대조)

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "m.pt"); save_ckpt(on, p)
    y = model_from_ckpt(p).eval()
    assert y.cfg["type_lse"] is True and y.cfg["value_detach"] is True and torch.allclose(fw(y)["policy_logits"][mask], fw(on)["policy_logits"][mask], atol=1e-6)
print("test_type_lse OK")
