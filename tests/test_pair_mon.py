"""v8 pair_mon(포켓몬 x 포켓몬 쌍 모듈) self-check: PYTHONPATH=. python tests/test_pair_mon.py
1) 꺼져 있으면 cfg/파라미터 그대로, 2) 투영 0 시작이라 같은 가중치면 꺼진 모델과 출력 동일, 3) 체크포인트 왕복,
4) 투영을 키우면 출력이 변하고 구조 관계가 서로 다름(자기/같은 편/상대 편), 5) pair_glu와 함께 써도 my_pos=False 순열 대칭 유지"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.pair_glu import PairMon
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):   # tests/test_mon_moves.py와 같은 입력 생성기
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
small = {"model": "v8", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False}
off = build_model(small).eval()
on = build_model({**small, "pair_mon": 16}).eval()
assert "pair_mon" not in off.cfg and not any(k.startswith("pair_m.") for k in off.state_dict())
assert on.cfg["pair_mon"] == 16 and any(k.startswith("pair_m.0.") for k in on.state_dict())

r = PairMon(8, 4, 2).rel_idx
assert r.shape == (12, 12) and r[0, 0] == 0 and r[0, 1] == 1 and r[0, 6] == 2 and r[7, 6] == 1 and r[7, 2] == 2 and r[11, 11] == 0

L = 4
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m, o, mk=mask: m.forward_sequences(o, Z, T, 1, mk)["policy_logits"]

with torch.no_grad():
    # 2) 같은 가중치, 투영 0이면 동일
    off.load_state_dict({k: v for k, v in on.state_dict().items() if not k.startswith("pair_m.")})
    assert torch.allclose(fw(on, obs), fw(off, obs), atol=1e-5)
    # 3) 왕복
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt"); save_ckpt(on, p)
        y = model_from_ckpt(p).eval()
        assert y.cfg["pair_mon"] == 16 and torch.allclose(fw(y, obs), fw(on, obs), atol=1e-6)
    # 4) 투영을 키우면 출력이 변함, 관계 임베딩을 바꾸면 결과가 달라짐
    for mod in on.pair_m:
        torch.nn.init.normal_(mod.to_bias.weight, std=0.5); torch.nn.init.normal_(mod.to_tok.weight, std=0.5)
    assert (fw(on, obs) - fw(off, obs)).abs().max() > 1e-6
    a = fw(on, obs)
    on.pair_m[0].rel.weight[2].add_(1.0)
    assert (fw(on, obs) - a).abs().max() > 1e-6
    on.pair_m[0].rel.weight[2].sub_(1.0)
    # 5) pair_glu와 함께, 순열 대칭
    both = build_model({**small, "pair_mon": 16, "pair_glu": 16}).eval()
    for mod in list(both.pair_m) + list(both.pair):
        torch.nn.init.normal_(mod.to_bias.weight, std=0.3); torch.nn.init.normal_(mod.to_tok.weight, std=0.3)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    ob2 = [obs[0][:, perm], obs[1][:, perm], obs[2][:, perm]] + obs[3:]
    m2 = mask[:, list(range(8)) + [8 + int(i) for i in perm] + list(range(14, 22))]
    a1, a2 = fw(both, obs), fw(both, ob2, m2)
    ok = a1[:, 8:14][:, perm] > -1e8
    assert torch.allclose(a2[:, 8:14][ok], a1[:, 8:14][:, perm][ok], atol=1e-4), (a2[:, 8:14][ok] - a1[:, 8:14][:, perm][ok]).abs().max()
    assert torch.allclose(a2[:, :8][a2[:, :8] > -1e8], a1[:, :8][a1[:, :8] > -1e8], atol=1e-4)
print("test_pair_mon OK")
