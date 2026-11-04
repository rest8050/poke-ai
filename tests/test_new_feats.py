"""v8 tera_feat(테라 가정 매치업을 기술 점수 입력에) self-check: PYTHONPATH=. python tests/test_new_feats.py
1) 꺼져 있으면 cfg/파라미터 그대로, 켜면 cfg 기록 2) 특징 모양/유한/의미(테라 전후 차가 실제로 생김) 3) 켠 모델의 체크포인트 왕복과 my_pos=False 순열 대칭
(bench_off, pair_feat는 이득이 없어 제거됨)"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.model_v8 import TERA_N
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
small = {"model": "v8", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False, "mon_moves": True, "pair_glu": 16}
L = 6
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
Z, T = torch.zeros(L, dtype=torch.long), torch.arange(L)
fw = lambda m, o, mk=mask: m.forward_sequences(o, Z, T, 1, mk)["policy_logits"]
enc = lambda m, o=obs: m._encode_turn(*o, legal=mask)

base = build_model(small).eval()
assert "tera_feat" not in base.cfg
with torch.no_grad():
    # 2) 개별 옵션
    t = build_model({**small, "tera_feat": True}).eval()
    assert t.cfg["tera_feat"] is True and enc(t)["tera_feat"].shape == (L, 4, TERA_N) and torch.isfinite(enc(t)["tera_feat"]).all()
    tf = enc(t)["tera_feat"]
    g2 = torch.Generator().manual_seed(100)
    o2 = random_obs(32, g2); o2[1][:, 0, 9] = 0.0                              # 내 활성은 아직 테라 전
    mk2 = mask[:1].expand(32, -1)
    tf0 = t._encode_turn(*o2, legal=mk2)["tera_feat"]
    assert tf0[..., 3:5].abs().max() > 0, "테라 전후 차이가 전혀 없음"            # 테라 후 데미지/KO가 달라지는 경우가 있어야 함
    o2[1][:, 0, 9] = 1.0                                                       # 이미 테라 중이면 가정과 현실이 같아 차이가 정확히 0
    tf1 = t._encode_turn(*o2, legal=mk2)["tera_feat"]
    assert tf1[..., 3:5].abs().max() == 0 and torch.equal(tf1[..., 5:7], tf1[..., 7:9])
    assert enc(base)["tera_feat"] is None
    # 3) 켠 모델
    allm = t
    for mod in allm.pair:
        torch.nn.init.normal_(mod.to_bias.weight, std=0.3); torch.nn.init.normal_(mod.to_tok.weight, std=0.3)
    assert torch.isfinite(fw(allm, obs)[mask]).all()
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "m.pt"); save_ckpt(allm, pth)
        y = model_from_ckpt(pth).eval()
        assert y.cfg["tera_feat"] is True
        assert torch.allclose(fw(y, obs), fw(allm, obs), atol=1e-6)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    ob2 = [obs[0][:, perm], obs[1][:, perm], obs[2][:, perm]] + obs[3:]
    m2 = mask[:, list(range(8)) + [8 + int(i) for i in perm] + list(range(14, 22))]
    a1, a2 = fw(allm, obs), fw(allm, ob2, m2)
    ok = a1[:, 8:14][:, perm] > -1e8
    assert torch.allclose(a2[:, 8:14][ok], a1[:, 8:14][:, perm][ok], atol=1e-4), (a2[:, 8:14][ok] - a1[:, 8:14][:, perm][ok]).abs().max()
    assert torch.allclose(a2[:, :8][a2[:, :8] > -1e8], a1[:, :8][a1[:, :8] > -1e8], atol=1e-4)
print("test_new_feats OK")
