"""모델 v5(합법성 입력) self-check (v4 항목은 test_model_v4.py): PYTHONPATH=. python tests/test_model_v5.py
1) 롤아웃 == 학습(forward_sequences), 2) legal_mask를 안 주면 action_mask를 씀, 3) 합법 비트가 출력에 영향 (기술 하나만 가능 / 교체 봉쇄),
4) 합법성 요약(기술 수·한 개만 가능·고정 의심·잃은 화력) 값, 5) 가지치기 마스크와 순수 합법 마스크를 따로 줄 수 있음, 6) 체크포인트 왕복"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):
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
gen = torch.Generator().manual_seed(0)
cfg = {"model": "v5", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False}
model = build_model(cfg).eval()
assert type(model).__name__ == "EntityPokemonNetV5" and model.uses_legal
lengths = [5, 3]
obs = [random_obs(L, gen) for L in lengths]
masks = []
for L in lengths:
    m = torch.zeros(L, 22, dtype=torch.bool); m[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True; masks.append(m)
flat = [torch.cat([o[k] for o in obs]) for k in range(7)]
seq_index = torch.cat([torch.full((L,), b) for b, L in enumerate(lengths)])
time_index = torch.cat([torch.arange(L) for L in lengths])
mask = torch.cat(masks)
fs = lambda **kw: model.forward_sequences(flat, seq_index, time_index, len(lengths), **kw)

with torch.no_grad():
    # 1) 롤아웃 == 시퀀스
    step = []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = model(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1], legal_mask=masks[b][t:t + 1])
            hs = out["history_state"]; step.append(out["policy_logits"][0])
    base = fs(action_mask=mask, legal_mask=mask)["policy_logits"]
    assert torch.allclose(base, torch.stack(step), atol=1e-4)
    # 2) legal_mask 생략 == action_mask와 같은 값을 준 것
    assert torch.allclose(fs(action_mask=mask)["policy_logits"], base)
    # 3) 합법 비트가 출력에 영향: 기술 하나만 가능 / 교체 봉쇄
    only1 = mask.clone(); only1[:, 1:4] = False
    pol = lambda **kw: fs(**kw)["policy_logits"]
    assert not torch.allclose(pol(action_mask=mask, legal_mask=only1)[:, 8:14], base[:, 8:14]), "기술 한 개만 가능해도 교체 점수가 안 변함"
    trapped = mask.clone(); trapped[:, 8:14] = False
    ps = lambda lg: lg.exp()[:, 8:14].sum(-1)
    assert not torch.allclose(ps(pol(action_mask=mask, legal_mask=trapped)), ps(base)), "교체 봉쇄 정보가 종류 판단에 안 닿음"
    # 5) 가지치기(action_mask)와 합법(legal_mask)을 따로 줘도 동작하고, 로짓 마스킹은 action_mask를 따름
    pruned = mask.clone(); pruned[:, 2] = False
    lg = pol(action_mask=pruned, legal_mask=mask)
    assert (lg[:, 2] < -1e8).all() and (lg[:, 1] > -1e8).all()
    # 4) 합법성 요약 값
    leg = torch.ones(2, 22); leg[1, 1:4] = 0; leg[1, 4:8] = 0; leg[1, 8:14] = 0
    off = torch.zeros(2, 4, 4); off[..., 3] = 1; off[:, :, 1] = torch.tensor([0.2, 0.9, 0.3, 0.4]); off[:, :, 2] = torch.tensor([0., 1., 0., 0.])
    g = model._legal_summary(leg, off, torch.tensor([4., 4.]))
    assert g.shape == (2, 25)
    n_normal, n_attack, n_sw, tera, one, lock = g[:, 14:20].unbind(-1)
    assert torch.allclose(n_normal, torch.tensor([1.0, 0.25])) and torch.allclose(n_sw, torch.tensor([1.2, 0.0])) and torch.allclose(tera, torch.tensor([1.0, 0.0]))
    assert torch.allclose(one, torch.tensor([0.0, 1.0])) and torch.allclose(lock, torch.tensor([0.0, 0.75]))
    assert torch.allclose(g[:, 23], torch.tensor([0.0, 0.7])) and torch.allclose(g[:, 24], torch.tensor([0.0, 1.0]))      # 고정으로 잃은 화력(데미지 0.9 -> 0.2, KO 1 -> 0)

# 6) 체크포인트 왕복
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.pt")
    save_ckpt(model, path)
    m2 = model_from_ckpt(path).eval()
    assert type(m2).__name__ == "EntityPokemonNetV5" and m2.cfg == model.cfg
    with torch.no_grad():
        assert torch.allclose(m2.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask)["policy_logits"], base, atol=1e-5)
print("model v5 OK")
