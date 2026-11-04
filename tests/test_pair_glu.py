"""v8 pair_glu(기술 x 포켓몬 쌍 모듈) self-check: PYTHONPATH=. python tests/test_pair_glu.py
1) 꺼져 있으면 cfg/파라미터 그대로(기존 체크포인트 호환), 2) (모드 비교는 bilin/add 제거로 삭제) 3) 출력 투영 0 시작이라 켜도 처음엔 꺼진 모델과 출력 동일(같은 가중치),
4) 체크포인트 왕복, 5) 투영을 키우면 내 예비 포켓몬의 특성/기술이 출력에 닿고 구조 관계(소유자/같은 편/상대 편)가 서로 다름, 6) my_pos=False 순열 대칭 유지"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.pair_glu import PairGLU
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
ms = {"glu": build_model({**small, "pair_glu": 16, "pair_mode": "glu"}).eval()}
assert "pair_glu" not in off.cfg and not any(k.startswith("pair.") for k in off.state_dict())
assert sum(p.numel() for p in ms["glu"].parameters()) > sum(p.numel() for p in off.parameters())

L = 4
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m, o, mk=mask: m.forward_sequences(o, Z, T, 1, mk)["policy_logits"]

# 구조 관계 행렬: 소유자 0, 같은 편 1, 상대 편 2
r = PairGLU(8, 4, 2).rel_idx
assert r.shape == (48, 12) and (r == 0).sum() == 48 and r[0, 0] == 0 and r[0, 1] == 1 and r[0, 6] == 2 and r[47, 11] == 0 and r[47, 0] == 2

with torch.no_grad():
    for m, x in ms.items():
        # 3) 같은 가중치로 투영이 0이면 꺼진 모델과 동일
        base = build_model(small).eval()
        sd = {k: v for k, v in x.state_dict().items() if not k.startswith("pair.")}
        base.load_state_dict(sd)
        assert torch.allclose(fw(x, obs), fw(base, obs), atol=1e-5), m
        # 4) 왕복
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt"); save_ckpt(x, p)
            y = model_from_ckpt(p).eval()
            assert y.cfg["pair_glu"] == 16 and y.cfg["pair_mode"] == m
            assert torch.allclose(fw(y, obs), fw(x, obs), atol=1e-6)
        # 5) 투영을 키우면 내 예비 포켓몬(2번)의 기술 id 변경이 출력에 닿고, 변화량이 꺼진 모델과 다름
        for mod in x.pair:
            torch.nn.init.normal_(mod.to_bias.weight, std=0.5); torch.nn.init.normal_(mod.to_tok.weight, std=0.5)
        ch = [t.clone() for t in obs]
        ch[0][:, 2, 5:9] = (ch[0][:, 2, 5:9] % 40) + 40
        d_on, d_off = (fw(x, ch) - fw(x, obs)).abs().max(), (fw(base, ch) - fw(base, obs)).abs().max()
        assert d_on > 1e-6 and abs(d_on - d_off) > 1e-6, (m, d_on, d_off)
        # 6) 순열 대칭
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        ob2 = [obs[0][:, perm], obs[1][:, perm], obs[2][:, perm]] + obs[3:]
        m2 = mask[:, list(range(8)) + [8 + int(i) for i in perm] + list(range(14, 22))]
        a1, a2 = fw(x, obs), fw(x, ob2, m2)
        ok = a1[:, 8:14][:, perm] > -1e8
        assert torch.allclose(a2[:, 8:14][ok], a1[:, 8:14][:, perm][ok], atol=1e-4), (m, (a2[:, 8:14][ok] - a1[:, 8:14][:, perm][ok]).abs().max())
        assert torch.allclose(a2[:, :8][a2[:, :8] > -1e8], a1[:, :8][a1[:, :8] > -1e8], atol=1e-4), m

print("test_pair_glu OK")
