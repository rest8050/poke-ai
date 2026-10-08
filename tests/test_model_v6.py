"""모델 v6(천진·장판 진입 피해 매치업 + 행동 제약 효과 입력) self-check: PYTHONPATH=. python tests/test_model_v6.py
1) 롤아웃 == 학습, 2) 폭이 다른 field_vec 호환 (v5는 새 열을 버림 / v6는 옛 짧은 데이터를 0으로 채움), 3) 효과 열이 v6 출력에만 영향,
4) 천진: 공격/방어 랭크 무시 (방어측·공격측 양방향), 5) 장판 진입 피해 값 + 후보 KO가 늘기만 함, 6) 인코더 효과 열 (poke-env 카운터 -> 남은 턴), 7) 체크포인트 왕복"""
import json
import os
import tempfile

import torch
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon import Pokemon

from src.core.matchup import Matchup
from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import EFFECT_SIDE, EFFECT_START, FIELD_DIM, NUM_DIM, BattleTensorEncoder, DummyBattle, active_effect_vec

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
model = build_model({"model": "v6", **small}).eval()
assert type(model).__name__ == "EntityPokemonNetV6" and model.uses_legal and model.matchup.entry
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
    assert torch.allclose(base, torch.stack(step), atol=1e-4)
    # 2) 폭 호환: v6에 옛 폭(58) 데이터 == 효과 열이 0인 새 폭 데이터 / v5는 새 열을 버려 값이 같음
    old_w = EFFECT_START
    zero_eff = flat[:6] + [torch.cat([flat[6][:, :old_w], torch.zeros(len(mask), FIELD_DIM - old_w)], 1)]
    short = flat[:6] + [flat[6][:, :old_w]]
    assert torch.allclose(fs(model, short), fs(model, zero_eff), atol=1e-5)
    v5 = build_model({"model": "v5", **small}).eval()
    assert torch.allclose(fs(v5, short), fs(v5, flat), atol=1e-5), "v5가 field_vec 뒤쪽 새 열의 영향을 받음"
    # 3) 효과 열은 v6 출력에 영향
    no_eff = flat[:6] + [torch.cat([flat[6][:, :old_w], torch.zeros(len(mask), FIELD_DIM - old_w)], 1)]
    assert not torch.allclose(fs(model), fs(model, no_eff)), "효과 열이 v6 출력에 안 닿음"

    # 4) 천진 (방어측): 상대 +2 공격 랭크를 천진 후보는 무시 / (공격측): 천진 공격수는 방어 랭크를 무시
    o = random_obs(1, gen)
    my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, _ = o
    my_cat[..., 1] = 0; opp_cat[..., 1] = 0
    my_num[..., 2] = 1.0; my_num[..., 3:8] = 0; opp_num[..., 3:8] = 0
    opp_mv[:, :4, :, 0], opp_mv[:, :4, :, 4], opp_mv[:, :4, :, 6] = 0.4, 0.33, 0.5       # 상대 물리 공격기
    my_mv[..., 0], my_mv[..., 4], my_mv[..., 6] = 0.4, 0.33, 0.5
    M, M0 = Matchup(entry=True), Matchup()
    UN = V["ability"]["unaware"]
    dmg_to_me = lambda m, mc, on: m._pair(opp_cat, on, opp_mv, False, mc, my_num, True)[..., 1]
    rank2 = opp_num.clone(); rank2[:, :, 3] = 2 / 6
    d0, d2 = dmg_to_me(M, my_cat, opp_num), dmg_to_me(M, my_cat, rank2)
    assert (d2 > d0).any() and torch.all(d2 >= d0), "공격 랭크가 데미지를 안 올림"
    un_cat = my_cat.clone(); un_cat[..., 1] = UN
    assert torch.allclose(dmg_to_me(M, un_cat, rank2), d0), "천진 방어측이 상대 공격 랭크를 못 무시함"
    assert torch.allclose(dmg_to_me(M0, un_cat, rank2), d2), "entry=False면 옛 계산이어야 함"
    dmg_to_opp = lambda mc, on: M._pair(mc, my_num, my_mv, True, opp_cat, on, False)[..., 1]
    rank_d = opp_num.clone(); rank_d[:, :, 4] = 2 / 6                                    # 상대 방어 +2
    a0, a2 = dmg_to_opp(my_cat, opp_num), dmg_to_opp(my_cat, rank_d)
    assert (a2 < a0).any(), "방어 랭크가 데미지를 안 내림"
    assert torch.allclose(dmg_to_opp(un_cat, rank_d), a0), "천진 공격측이 상대 방어 랭크를 못 무시함"

    # 5) 장판 진입 피해: 불 단일 = 바위 2배(0.25) + 압정 3겹(0.25), 불/비행 = 바위 4배(0.5) + 접지 아님, 부츠 = 0, 활성 = 0
    fire, flying, boots = V["type"]["fire"], V["type"]["flying"], V["item"]["heavydutyboots"]
    cat = torch.zeros(1, 6, 11, dtype=torch.long); cat[0, :, 2] = fire; cat[0, 1, 3] = flying; cat[0, 2, 0] = boots
    num = torch.zeros(1, 6, NUM_DIM); num[0, 3, 0] = 1
    fv = torch.zeros(1, FIELD_DIM); fv[0, 28], fv[0, 29] = 1.0, 1.0
    ent = M.entry_frac(cat, num, fv)
    assert torch.allclose(ent[0], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.5, 0.5])), ent       # 0번: 불 -> 바위 2배(0.25) + 압정(0.25), 1번: 불/비행 -> 0.5 + 0
    assert M0.entry_frac(cat, num, fv) is None and M.entry_frac(cat, num, None) is None
    fv[0, 29] = 1 / 3                                                                      # 압정 1겹 = 1/8
    assert abs(M.entry_frac(cat, num, fv)[0, 4].item() - (0.25 + 0.125)) < 1e-6
    my_num2 = my_num.clone(); my_num2[..., 2] = 1.0
    ent2 = torch.full((1, 6), 0.6)
    ko_base = M._pair(opp_cat, opp_num, opp_mv, False, my_cat, my_num2, True)[..., 2]
    ko_ent = M._pair(opp_cat, opp_num, opp_mv, False, my_cat, my_num2, True, entry=ent2)[..., 2]
    assert torch.all(ko_ent >= ko_base) and (ko_ent > ko_base).any(), "장판 피해가 후보 KO 판정에 안 반영됨"

# 6) 인코더 효과 열: 시작 직후 카운터 0 -> 턴이 끝날 때마다 +1
mon = Pokemon(gen=9, species="clefable")
assert active_effect_vec(mon) == [0.0] * EFFECT_SIDE and active_effect_vec(None) == [0.0] * EFFECT_SIDE
mon.start_effect("move: Taunt"); mon.start_effect("Encore")
mon.end_turn()
v = active_effect_vec(mon)
assert v[0] == 1.0 and v[1] == 1.0 and v[2] == 0.0                                          # 도발, 앵콜 플래그 (사슬묶기 아님)
assert abs(v[4] - 2 / 3) < 1e-6 and abs(v[5] - 2 / 3) < 1e-6 and v[6] == 0.0                # 남은 턴 비율: 도발 (3-1)/3, 앵콜 (3-1)/3
mon.end_turn(); mon.end_turn()
assert active_effect_vec(mon)[4] == 0.0 and active_effect_vec(mon)[0] == 1.0                # 카운터가 지속 턴을 넘으면 남은 턴 0 (플래그는 그대로)
mon._effects[Effect.TRAPPED] = 0
assert active_effect_vec(mon)[3] == 1.0
t = BattleTensorEncoder(vocab_path="data/vocab.json").encode_battle(DummyBattle())
assert t[6].shape == (1, FIELD_DIM) and float(t[6][0, EFFECT_START:].abs().sum()) == 0.0

# 7) 체크포인트 왕복
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.pt")
    save_ckpt(model, path)
    m2 = model_from_ckpt(path).eval()
    assert type(m2).__name__ == "EntityPokemonNetV6" and m2.cfg == model.cfg and m2.matchup.entry
    with torch.no_grad():
        assert torch.allclose(fs(m2), base, atol=1e-5)
print("model v6 OK")
