"""v8 tera_def / tera_gate self-check: PYTHONPATH=. python tests/test_tera_def.py
1) 꺼져 있으면 cfg/파라미터 그대로 2) tera_def는 기술칸마다 붙는 수치를 9 -> 37열로 늘리되 앞 9열은 그대로, 4칸에 같은 방어 수치
3) 내 활성이 이미 테라 중이면 테라 전/후 방어 수치가 같음, 테라 전인 활성은 달라짐 4) tera_gate는 마지막 층 0 시작이라 같은 가중치면 꺼진 모델과 출력 동일, 값을 키우면 테라 확률만 변함
5) 게이트에 기울기가 닿음 6) tera_def 단독은 거부 7) 체크포인트 왕복"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.model_v8 import TERA_DEF_N, TERA_N
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):   # tests/test_type_lse.py와 같은 입력 생성기
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
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 6, 9, 10, 11]] = True
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m, o=obs: m.forward_sequences(o, Z, T, 1, mask)
enc = lambda m, o=obs: m._encode_turn(*o, legal=mask)

off = build_model(small).eval()
assert not any(k in off.cfg for k in ("tera_feat", "tera_def", "tera_gate")) and not hasattr(off, "tera_gate_net") and off.need_tera is False
f1 = build_model({**small, "tera_feat": True}).eval()
f2 = build_model({**small, "tera_feat": True, "tera_def": True}).eval()
assert f1.cfg["tera_feat"] is True and f2.cfg["tera_def"] is True and f2.tera_dim == TERA_N + TERA_DEF_N == 37
f2.load_state_dict({k: v for k, v in f1.state_dict().items() if not k.startswith("move_score.0")}, strict=False)
with torch.no_grad():
    a, b = enc(f1)["tera_feat"], enc(f2)["tera_feat"]
    assert a.shape == (L, 4, TERA_N) and b.shape == (L, 4, TERA_N + TERA_DEF_N) and torch.isfinite(b).all()
    assert torch.allclose(a, b[..., :TERA_N]), "앞 9열이 달라짐"
    assert (b[:, :1, TERA_N:] == b[:, :, TERA_N:]).all(), "방어 수치가 기술칸마다 다름"
    assert torch.isfinite(fw(f2)["policy_logits"][mask]).all()

    # 내 활성이 이미 테라 중이면 테라 전/후가 같고, 테라 전이면 (타입이 다르면) 달라짐
    tera_on = [o.clone() for o in obs]; tera_on[1][:, 0, 9] = 1.0
    tera_off = [o.clone() for o in obs]; tera_off[1][:, 0, 9] = 0.0
    for o in (tera_on, tera_off):
        o[0][:, 0, 9] = torch.randint(1, 8, (L,), generator=gen)       # 테라 타입
        o[0][:, 0, 2] = torch.randint(1, 8, (L,), generator=gen)       # 원래 타입
    d_on = enc(f2, tera_on)["tera_feat"][:, 0, TERA_N:]
    d_off = enc(f2, tera_off)["tera_feat"][:, 0, TERA_N:]
    mv_on, mv_off = d_on[:, :24].reshape(L, 4, 6), d_off[:, :24].reshape(L, 4, 6)
    assert torch.allclose(mv_on[..., :3], mv_on[..., 3:], atol=1e-5), "이미 테라 중인데 방어 수치가 전/후로 다름"
    assert torch.allclose(d_on[:, 24:26], d_on[:, 26:28], atol=1e-5)
    assert not torch.allclose(mv_off[..., :3], mv_off[..., 3:], atol=1e-5), "테라 전인데 전/후 수치가 같음"

# tera_gate: 0 시작
g = build_model({**small, "tera_gate": True}).eval()
assert g.cfg["tera_gate"] is True and "tera_feat" not in g.cfg and g.need_tera is True
g.load_state_dict({k: v for k, v in off.state_dict().items()}, strict=False)
with torch.no_grad():
    p0, p1 = fw(off)["policy_logits"], fw(g)["policy_logits"]
    assert torch.allclose(p0, p1, atol=1e-6), "게이트 0 시작인데 출력이 다름"
    g.tera_gate_net[-1].bias.fill_(1.5)
    p2 = fw(g)["policy_logits"]
    assert torch.isfinite(p2[mask]).all() and not torch.allclose(p2[mask], p0[mask])
    pm = lambda x: x[:, :4].exp().sum(-1)
    assert torch.allclose(x := p2[:, :8].exp().sum(-1), p0[:, :8].exp().sum(-1), atol=1e-5)      # 기술 총확률은 그대로 (종류 헤드는 안 바뀜)
    assert (p2[:, 4:8].exp().sum(-1) > p0[:, 4:8].exp().sum(-1))[mask[:, 4:8].any(1)].all()      # 게이트 +이면 테라 쪽으로 이동

# 게이트와 방어 입력까지 켠 모델: 기울기, 단독 거부, 체크포인트 왕복
m = build_model({**small, "tera_feat": True, "tera_def": True, "tera_gate": True}); m.train()
with torch.no_grad():
    m.tera_gate_net[-1].weight.normal_(0, 0.1)
fw(m)["policy_logits"][mask].sum().backward()
assert m.tera_gate_net[0].weight.grad.abs().sum() > 0 and m.tera_gate_net[-1].weight.grad.abs().sum() > 0
try:
    build_model({**small, "tera_def": True}); raise SystemExit("tera_def 단독이 거부되지 않음")
except AssertionError:
    pass
m.eval()
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.pt"); save_ckpt(m, path)
    r = model_from_ckpt(path).eval()
    assert r.cfg == m.cfg and r.tera_gate and r.tera_def
    with torch.no_grad():
        assert torch.allclose(fw(m)["policy_logits"], fw(r)["policy_logits"], atol=1e-6)
    del r
print("tera_def/tera_gate OK")
