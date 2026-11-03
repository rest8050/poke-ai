"""모델 v8(v7 규칙 + 숨은 특성/도구 기대 임베딩 + 융합층 활성 토큰 입력) self-check: PYTHONPATH=. python tests/test_model_v8.py
1) 롤아웃 == 학습(forward_sequences), 2) 폭이 다른 field_vec 호환 (v8은 옛 짧은 데이터를 0으로 채움),
3) 효과 열과 새 규칙 입력(특성/스카프 확률)이 출력에 영향, 4) my_pos=False 순열 대칭이 부분 마스크에서도 정확히 성립 (v6는 교체 비트 순서 통로로 깨졌음),
5) 세트 추정 특성/도구 확률의 형태와 공개 값의 원핫, 6) 체크포인트 왕복,
7) 기대 임베딩: 상대 쪽 숨겨진 값만 q @ 표로 바뀌고 공개/내 쪽/빈 칸은 원래 임베딩 그대로, 숨겨진 값은 세트 추정 분포에 따라 출력이 달라짐, 8) 융합층이 활성 토큰 2개를 받음"""
import json
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.model_v8 import G_DIM
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
model = build_model({"model": "v8", **small}).eval()
assert type(model).__name__ == "EntityPokemonNetV8" and model.uses_legal
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

    # 5) 세트 추정 확률: 형태, 합 1, 공개된 특성은 원핫, 특수 칸은 0
    opp_cat = flat[3]
    bl = model.belief(opp_cat, flat[4])
    q = model._prob(opp_cat[..., 1], bl["ability"])
    A = model.matchup.f_def.size(0)
    assert q.shape == (len(mask), 6, A) and torch.allclose(q.sum(-1), torch.ones(len(mask), 6), atol=1e-5)
    hid = opp_cat.clone(); hid[:, :, 1] = A - 1                                                  # 숨김 토큰
    qh = model._prob(hid[..., 1], bl["ability"])
    assert (qh[..., 0] == 0).all() and (qh[..., A - 2:] == 0).all(), "특수 칸에 확률이 샘"
    assert torch.equal(q.argmax(-1), opp_cat[..., 1]), "공개된 특성이 원핫이 아님"
    qi = model._prob(opp_cat[..., 0], bl["item"])
    ps = qi[..., model.matchup.scarf]
    assert torch.allclose(qi.sum(-1), torch.ones(len(mask), 6), atol=1e-5) and ps.shape == (len(mask), 6) and ((ps >= 0) & (ps <= 1)).all()

    # 7) 기대 임베딩: 상대 쪽 숨겨진 특성(앞 3칸)만 q @ 표, 공개/내 쪽/나머지 칸은 원래 임베딩 그대로
    B, E = len(mask), model.embeddings
    oc = flat[3].clone(); oc[:, :3, 1] = A - 1
    ids = torch.cat([flat[0][..., 1], oc[..., 1]], 1).reshape(B * 12)
    qo = model._prob(oc[..., 1], model.belief(oc, flat[4])["ability"])
    out_e = model._expected_emb("ability", ids, qo, B).view(B, 12, -1)
    plain = E._emb("ability", ids).view(B, 12, -1)
    w = E.ability_embed.weight + (E.ability_proj(E.ability_feat) if E.has_feat else 0)
    assert torch.allclose(out_e[:, :6], plain[:, :6]) and torch.allclose(out_e[:, 9:], plain[:, 9:]), "공개/내 쪽 임베딩이 바뀜"
    assert torch.allclose(out_e[:, 6:9], qo[:, :3] @ w[:A], atol=1e-5), "숨겨진 값이 q @ 표가 아님"
    assert not torch.allclose(out_e[:, 6:9], plain[:, 6:9]), "숨겨진 값이 숨김 임베딩 그대로"
    # 8) 융합층은 CLS + 히스토리 + 요약 + 활성 토큰 2개를 받음
    assert model.fusion[0].in_features == 3 * model.d + 64 + G_DIM

# 9) 히스토리 보조 헤드(hist_aux): 처음엔 세트 추정과 같은 로짓, 기울기는 히스토리 모듈과 이 헤드로만 (트렁크/세트 추정 불변), 켜지 않은 모델의 cfg는 그대로
aux = build_model({"model": "v8", **small, "hist_aux": True}).eval()
assert aux.cfg["hist_aux"] is True and "hist_aux" not in model.cfg and not hasattr(model, "hist_hid")
out = aux.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask)
for k in ("move", "item", "ability"):
    assert out[f"hidh_{k}"].shape == out[f"hid_{k}"].shape and torch.allclose(out[f"hidh_{k}"], out[f"hid_{k}"], atol=1e-6)
for m in aux.hist_hid.values():
    torch.nn.init.normal_(m.weight, std=0.02)
out = aux.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask)
out["hidh_move"].square().sum().backward()
assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in aux.history_seq.parameters()), "히스토리 모듈에 기울기가 안 옴"
assert all(p.grad is None for n, p in aux.named_parameters() if n.startswith(("layers.", "mon_proj", "move_proj", "belief.", "fusion", "move_score"))), "보조 손실이 트렁크/세트 추정에 닿음"

# 10) 다중 γ 분류형 가치 헤드: 켜지 않은 모델의 cfg/출력은 그대로, 켜면 [N,G,K] 로짓과 기본 γ(0.995) 기대값, two-hot 손실이 목표값 쪽으로 학습됨
from src.training.train_fp_distill import two_hot_ce
vg = build_model({"model": "v8", **small, "value_gammas": [0.9, 0.995], "value_bins": 11}).eval()
assert "value_gammas" not in model.cfg and not hasattr(model, "value_support") and vg.cfg["value_gammas"] == [0.9, 0.995] and vg.value_primary == 1
out = vg.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask)
assert out["value_logits"].shape == (len(mask), 2, 11) and out["value"].shape == (len(mask), 1) and (out["value"].abs() <= 1).all()
assert torch.allclose(out["value"][:, 0], (out["value_logits"][:, 1].softmax(-1) * vg.value_support).sum(-1), atol=1e-6)
tgt = torch.tensor([[0.3, -1.0], [1.0, 0.55]])
lg = torch.zeros(2, 2, 11, requires_grad=True)
assert abs(two_hot_ce(torch.zeros(2, 2, 11), tgt, vg.value_support).item() - torch.log(torch.tensor(11.0)).item()) < 1e-5    # 균등 분포의 손실 = ln(K) (two-hot 가중치 합이 1)
opt = torch.optim.SGD([lg], lr=5.0)
for _ in range(200):
    opt.zero_grad(); two_hot_ce(lg, tgt, vg.value_support).backward(); opt.step()
assert torch.allclose((lg.softmax(-1) * vg.value_support).sum(-1), tgt, atol=1e-2), (lg.softmax(-1) * vg.value_support).sum(-1)
with tempfile.TemporaryDirectory() as td:
    save_ckpt(vg, os.path.join(td, "v.pt"))
    assert model_from_ckpt(os.path.join(td, "v.pt")).cfg == vg.cfg

# 11) 마지막 기술 입력(prev_act): 켜지 않은 모델은 그대로, 켠 모델은 prev_move가 출력에 영향, 인과적(미래 턴의 prev_move는 앞 턴 출력을 안 바꿈), None = 0, 체크포인트 왕복
pa = build_model({"model": "v8", **small, "prev_act": True}).eval()
assert "prev_act" not in model.cfg and model.pm_emb is None and pa.cfg["prev_act"] is True
with torch.no_grad():
    pm = torch.randint(1, pa.n_mv, (len(mask), 2))
    ref = pa.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask)["policy_logits"]
    assert torch.equal(ref, pa.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask, prev_move=torch.zeros_like(pm))["policy_logits"]), "None != 0"
    with_pm = pa.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask, prev_move=pm)["policy_logits"]
    assert not torch.allclose(ref, with_pm), "prev_move가 출력에 영향 없음"
    pm2 = pm.clone(); last = (time_index == time_index[seq_index == 0].max()) & (seq_index == 0)       # 시퀀스 0의 마지막 턴만 바꿈
    pm2[last] = (pm2[last] % (pa.n_mv - 1)) + 1
    alt = pa.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask, prev_move=pm2)["policy_logits"]
    assert torch.allclose(alt[~last], with_pm[~last], atol=1e-5) and not torch.allclose(alt[last], with_pm[last]), "인과성 위반"
with tempfile.TemporaryDirectory() as td:
    save_ckpt(pa, os.path.join(td, "p.pt"))
    assert model_from_ckpt(os.path.join(td, "p.pt")).cfg == pa.cfg

# 12) pm_fusion(마지막 기술을 융합층에도 직접)과 hist_only(융합층이 현재 턴 토큰을 직접 안 받음): 기본 모델 불변, 켜면 융합 입력 폭이 맞고 학습 경로 == 롤아웃, 체크포인트 왕복
assert model.cfg.get("pm_fusion") is None and model.cfg.get("hist_only") is None
for flags in ({"pm_fusion": True}, {"hist_only": True}, {"pm_fusion": True, "hist_only": True}):
    v = build_model({"model": "v8", **small, "prev_act": True, **flags}).eval()
    want = 64 + G_DIM + (0 if flags.get("hist_only") else 3 * v.d) + (32 if flags.get("pm_fusion") else 0)
    assert v.fusion[0].in_features == want, (flags, v.fusion[0].in_features, want)
    with torch.no_grad():
        pm = torch.randint(1, v.n_mv, (len(mask), 2))
        a1 = v.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask, prev_move=pm)["policy_logits"]
        a0 = v.forward_sequences(flat, seq_index, time_index, len(lengths), action_mask=mask)["policy_logits"]
    assert not torch.allclose(a1, a0), flags
    with tempfile.TemporaryDirectory() as td:
        save_ckpt(v, os.path.join(td, "f.pt"))
        assert model_from_ckpt(os.path.join(td, "f.pt")).cfg == v.cfg
try:
    build_model({"model": "v8", **small, "pm_fusion": True})
    raise SystemExit("prev_act 없이 pm_fusion이 통과함")
except AssertionError:
    pass

# 6) 체크포인트 왕복
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.pt")
    save_ckpt(model, path)
    m2 = model_from_ckpt(path).eval()
    assert type(m2).__name__ == "EntityPokemonNetV8" and m2.cfg == model.cfg
    with torch.no_grad():
        assert torch.allclose(fs(m2), base, atol=1e-5)
print("model v8 OK")
