"""v8 token_head(정책을 토큰 출력에서 읽는 22칸 단일 softmax) self-check: PYTHONPATH=. python tests/test_token_head.py
1) 헤드/히스토리/가치 모듈이 없고 cfg에 기록 2) 22칸 확률 합 1, 마스크된 칸 0 3) 체크포인트 왕복 4) 내 벤치 슬롯 순서를 바꾸면 교체 점수도 따라 바뀜(my_pos=False)
5) 교체 후보 특징(cand)과 전역 요약(leg_g)이 토큰을 거쳐 출력에 기울기를 보냄 6) get_action: 가치 0, history_state 없음"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):   # tests/test_pair_glu.py와 같은 입력 생성기
    my_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    opp_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    my_num = torch.rand(n, 6, NUM_DIM, generator=gen); my_num[:, :, 0] = 0; my_num[:, 0, 0] = 1; my_num[:, :, 1] = 0
    opp_num = torch.rand(n, 6, NUM_DIM, generator=gen); opp_num[:, :, 0] = 0; opp_num[:, 1, 0] = 1; opp_num[:, :, 1] = 0
    my_mv, opp_mv = torch.rand(n, 6, 4, 46, generator=gen), torch.rand(n, 6, 4, 46, generator=gen)
    opp_num[:, 4:, 11:19] = 0
    opp_mv[:, 4:] = 0
    opp_mv[:, :4, 2:] = 0
    return [my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, torch.rand(n, FIELD_DIM, generator=gen)]


torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
small = {"model": "v8", "d_model": 64, "n_layers": 2, "n_heads": 4, "head_width": 32, "my_pos": False, "mon_moves": True, "pair_glu": 16, "token_head": True}
m = build_model(small).eval()
keys = list(m.state_dict())
assert m.cfg["token_head"] is True
assert not any(k.startswith(("history", "turn_", "fusion", "move_score", "switch_score", "type_head", "value_head")) for k in keys), keys
assert any(k.startswith("readout") for k in keys) and any(k.startswith("cand_proj") for k in keys)

L = 6
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
mask[4] = False; mask[4, 8:14] = True                           # 강제 교체
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda mod, o, mk=mask: mod.forward_sequences(o, Z, T, 1, mk)

with torch.no_grad():
    out = fw(m, obs)
    assert "value" not in out and "latent" not in out and out["policy_logits"].shape == (L, 22)
    p = out["policy_logits"].exp()
    assert torch.allclose(p.sum(-1), torch.ones(L), atol=1e-5) and torch.isfinite(out["policy_logits"]).all()
    assert (p[~mask] < 1e-6).all() and (p[mask] > 0).all()
    assert (p[4, 8:14].sum() - 1).abs() < 1e-5                       # 강제 교체는 교체 칸에만 확률
    # forward(한 턴)과 forward_sequences가 같고 history_state는 없음
    o1 = m(*[x[:1] for x in obs], None, mask[:1])
    assert torch.allclose(o1["policy_logits"], out["policy_logits"][:1], atol=1e-5) and "history_state" not in o1
    # 체크포인트 왕복
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "m.pt"); save_ckpt(m, pth)
        y = model_from_ckpt(pth).eval()
        assert y.cfg["token_head"] is True and torch.allclose(fw(y, obs)["policy_logits"], out["policy_logits"], atol=1e-6)
    # 내 벤치 슬롯 순서를 바꾸면 교체 점수가 따라 바뀜
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    ob2 = [obs[0][:, perm], obs[1][:, perm], obs[2][:, perm]] + obs[3:]
    m2 = mask[:, list(range(8)) + [8 + int(i) for i in perm] + list(range(14, 22))]
    a1, a2 = out["policy_logits"], fw(m, ob2, m2)["policy_logits"]
    ok = mask[:, 8:14][:, perm]
    assert torch.allclose(a2[:, 8:14][ok], a1[:, 8:14][:, perm][ok], atol=1e-4), (a2[:, 8:14][ok] - a1[:, 8:14][:, perm][ok]).abs().max()
    # get_action
    idx, probs, val, hist = m.get_action(*[x[:1] for x in obs], None, mask[:1], mask[:1])
    assert mask[0, idx] and val == 0.0 and hist is None

# 교체 후보 특징/전역 요약이 토큰을 거쳐 출력에 기울기를 줌
m.train()
(fw(m, obs)["policy_logits"][mask] * torch.randn(int(mask.sum()))).sum().backward()
for name in ("cand_proj.weight", "leg_proj.weight", "kind_emb.weight", "readout.0.weight"):
    g = dict(m.named_parameters())[name].grad
    assert g is not None and g.abs().sum() > 0, name
print("test_token_head OK")
