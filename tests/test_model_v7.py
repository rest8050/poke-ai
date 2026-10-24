"""모델 v7(특성 EXEC 규칙 + 합법 요약 슬롯 순서 통로 제거) self-check: PYTHONPATH=. python tests/test_model_v7.py
1) 롤아웃 == 학습(forward_sequences), 2) 폭이 다른 field_vec 호환 (v7은 옛 짧은 데이터를 0으로 채움 / v4 legacy는 새 열을 버림),
3) 효과 열과 새 규칙 입력(특성/스카프 확률)이 출력에 영향, 4) my_pos=False 순열 대칭이 부분 마스크에서도 정확히 성립 (v6는 교체 비트 순서 통로로 깨졌음),
5) 신념 특성/스카프 확률의 형태와 공개 특성의 원핫, 6) 체크포인트 왕복"""
import json
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import EFFECT_START, FIELD_DIM, NUM_DIM

V = json.load(open("data/vocab.json"))


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
small = {"d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False}
model = build_model({"model": "v7", **small}).eval()
assert type(model).__name__ == "EntityPokemonNetV7" and model.uses_legal
lengths = [5, 3]
obs = [random_obs(L, gen) for L in lengths]
masks = []
for L in lengths:
    m = torch.zeros(L, 22, dtype=torch.bool); m[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True; masks.append(m)
flat = [torch.cat([o[k] for o in obs]) for k in range(7)]
seq_index = torch.cat([torch.full((L,), b) for b, L in enumerate(lengths)])
time_index = torch.cat([torch.arange(L) for L in lengths])
mask = torch.cat(masks)
fs = lambda m, obs_=flat, **kw: m.forward_sequences(obs_, seq_index, time_index, len(lengths), action_mask=mask, **kw)["policy_logits"]

with torch.no_grad():
    # 1) 롤아웃 == 시퀀스
    step = []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = model(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1], legal_mask=masks[b][t:t + 1])
            hs = out["history_state"]; step.append(out["policy_logits"][0])
    base = fs(model)
    assert torch.isfinite(base[mask]).all()
    assert torch.allclose(base, torch.stack(step), atol=1e-4)
    # 2) 폭 호환
    old_w = EFFECT_START
    zero_eff = flat[:6] + [torch.cat([flat[6][:, :old_w], torch.zeros(len(mask), FIELD_DIM - old_w)], 1)]
    short = flat[:6] + [flat[6][:, :old_w]]
    assert torch.allclose(fs(model, short), fs(model, zero_eff), atol=1e-5)
    v4 = build_model({"model": "v4", **small}).eval()
    assert torch.allclose(fs(v4, short), fs(v4, flat), atol=1e-5), "v4가 field_vec 뒤쪽 새 열의 영향을 받음"
    # 3) 새 입력이 출력에 닿음: 효과 열, 내 활성 특성(정밀기술/장난꾸러기 등), 순풍/날씨 플래그(스피드 규칙)
    assert not torch.allclose(fs(model), fs(model, zero_eff)), "효과 열이 출력에 안 닿음"
    ab = V["ability"]
    chg = [t.clone() for t in flat]
    chg[0][:, :, 1] = ab["prankster"]
    assert (fs(model, chg) - base).abs().max() > 1e-5, "특성이 출력에 안 닿음"
    chg = [t.clone() for t in flat]
    chg[6][:, 26] = 1.0 - (chg[6][:, 26] > 0.5).float()                                        # 내 쪽 순풍 켜기/끄기
    assert (fs(model, chg) - base).abs().max() > 1e-5, "순풍(스피드 규칙)이 출력에 안 닿음"

    # 4) 순열 대칭: my_pos=False이면 내 팀 순서를 바꿔도 교체 점수가 같은 순열로 바뀌어야 함 (마스크가 일부만 켜져 있어도)
    Z = lambda L: torch.zeros(L, dtype=torch.long)
    oo = obs[0]
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    ob2 = [oo[0][:, perm], oo[1][:, perm], oo[2][:, perm]] + oo[3:]
    m2 = masks[0][:, list(range(8)) + [8 + int(i) for i in perm] + list(range(14, 22))]
    a1 = model.forward_sequences(oo, Z(5), torch.arange(5), 1, masks[0])["policy_logits"]
    a2 = model.forward_sequences(ob2, Z(5), torch.arange(5), 1, m2)["policy_logits"]
    ok = a1[:, 8:14][:, perm] > -1e8
    assert torch.allclose(a2[:, 8:14][ok], a1[:, 8:14][:, perm][ok], atol=1e-4), (a2[:, 8:14][ok] - a1[:, 8:14][:, perm][ok]).abs().max()

    # 5) 신념 확률: 형태, 합 1, 공개된 특성은 원핫, 특수 칸은 0
    opp_cat = flat[3]
    bl = model.belief(opp_cat, flat[4])
    q = model._ability_prob(opp_cat, bl)
    A = model.matchup.f_def.size(0)
    assert q.shape == (len(mask), 6, A) and torch.allclose(q.sum(-1), torch.ones(len(mask), 6), atol=1e-5)
    hid = opp_cat.clone(); hid[:, :, 1] = A - 1                                                  # 숨김 토큰
    qh = model._ability_prob(hid, bl)
    assert (qh[..., 0] == 0).all() and (qh[..., A - 2:] == 0).all(), "특수 칸에 확률이 샘"
    assert torch.equal(q.argmax(-1), opp_cat[..., 1]), "공개된 특성이 원핫이 아님"
    ps = model._scarf_prob(opp_cat, bl)
    assert ps.shape == (len(mask), 6) and ((ps >= 0) & (ps <= 1)).all()

# 6) 체크포인트 왕복
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.pt")
    save_ckpt(model, path)
    m2 = model_from_ckpt(path).eval()
    assert type(m2).__name__ == "EntityPokemonNetV7" and m2.cfg == model.cfg
    with torch.no_grad():
        assert torch.allclose(fs(m2), base, atol=1e-5)
print("model v7 OK")
