"""v8 mon_moves(포켓몬 토큰의 기술 가방) self-check: PYTHONPATH=. python tests/test_mon_moves.py
1) 꺼져 있으면 cfg/파라미터가 그대로(기존 체크포인트 호환), 켜면 cfg에 기록, 2) 체크포인트 왕복,
3) 켜면 내 비활성 포켓몬의 기술이 그 포켓몬 토큰을 거쳐 출력에 닿음(비피해 기술 id만 바꿔도), 4) my_pos=False 순열 대칭 유지,
5) 상대 쪽 공개된 기술이 늘면(숨은 칸 기대값은 그만큼 줄어듦) 출력이 변함"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):   # tests/test_model_v8.py와 같은 입력 생성기
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
off, on = build_model(small).eval(), build_model({**small, "mon_moves": True}).eval()
assert "mon_moves" not in off.cfg and on.cfg["mon_moves"] is True
assert not any(k.startswith("bag_proj") for k in off.state_dict()) and any(k.startswith("bag_proj") for k in on.state_dict())

L = 4
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m, o, mk=mask: m.forward_sequences(o, Z, T, 1, mk)["policy_logits"]

with torch.no_grad():
    # 2) 왕복
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt"); save_ckpt(on, p)
        assert torch.allclose(fw(model_from_ckpt(p).eval(), obs), fw(on, obs), atol=1e-6)
    # 3) 내 비활성 포켓몬(2번)의 기술 id를 바꾸면 켠 모델은 출력이 변함 (기술 토큰은 그대로 두고 포켓몬 토큰 경로만 확인하려고 두 모델의 차이를 비교)
    ch = [t.clone() for t in obs]
    ch[0][:, 2, 5:9] = (ch[0][:, 2, 5:9] % 40) + 40
    d_on = (fw(on, ch) - fw(on, obs)).abs().max()
    assert d_on > 1e-6, "기술이 출력에 안 닿음"
    # 가방 경로만 끄고 같은 변경을 하면 변화량이 달라져야 함(가방이 실제로 기여): bag_proj를 0으로 만든 사본
    z = build_model({**small, "mon_moves": True}).eval(); z.load_state_dict(on.state_dict())
    torch.nn.init.zeros_(z.bag_proj.weight); torch.nn.init.zeros_(z.bag_proj.bias)
    assert (fw(z, ch) - fw(z, obs)).abs().max() != d_on
    # 4) 순열 대칭
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    ob2 = [obs[0][:, perm], obs[1][:, perm], obs[2][:, perm]] + obs[3:]
    m2 = mask[:, list(range(8)) + [8 + int(i) for i in perm] + list(range(14, 22))]
    a1, a2 = fw(on, obs), fw(on, ob2, m2)
    ok = a1[:, 8:14][:, perm] > -1e8
    assert torch.allclose(a2[:, 8:14][ok], a1[:, 8:14][:, perm][ok], atol=1e-4), (a2[:, 8:14][ok] - a1[:, 8:14][:, perm][ok]).abs().max()
    assert torch.allclose(a2[:, :8][a2[:, :8] > -1e8], a1[:, :8][a1[:, :8] > -1e8], atol=1e-4)
    # 5) 상대 쪽 공개 기술 수가 바뀌면 변함: 상대 1번 포켓몬의 기술 3번째 칸을 공개/비공개로
    ch = [t.clone() for t in obs]
    ch[5][:, 0, 2, 2:] = 0.0
    ch[5][:, 0, 2, 0] = 1.0
    assert (fw(on, ch) - fw(on, obs)).abs().max() > 1e-7
print("test_mon_moves OK")
