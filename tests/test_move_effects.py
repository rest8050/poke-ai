"""move_effect_features 대표 기술 self-check: python src/core/test_move_effects.py"""
from src.core.tensor_encoder import MOVE_EFFECT_DIM, MOVE_EFFECT_NAMES, move_effect_features


def f(move_id):
    v = move_effect_features(move_id)
    assert len(v) == MOVE_EFFECT_DIM
    return {name: round(x, 3) for name, x in zip(MOVE_EFFECT_NAMES, v) if x}


assert f("earthquake") == {}                                                        # 효과 없음
assert f("uturn") == {"self_switch": 1, "contact": 1}
assert f("knockoff")["item_manipulation"] == 1
assert f("trickroom") == {"trick_room": 1}
assert f("reflect")["screen"] == 1 and f("lightscreen")["screen"] == 1 and f("auroraveil")["screen"] == 1
assert f("stealthrock")["hazard_set"] == 1 and f("ceaselessedge")["hazard_set"] == 1
assert f("rapidspin")["hazard_removal"] == 1 and f("rapidspin")["self_speed"] == 0.5    # 스피드 +1 = 0.5
assert f("defog")["hazard_removal"] == 1
assert f("closecombat")["self_defense"] == -1                                       # 방어 -1, 특방 -1
assert f("swordsdance")["self_offense"] == 1 and "tgt_offense" not in f("swordsdance")   # 대상=자신
assert f("charm")["tgt_offense"] == -1                                              # 상대 공격 -2
assert f("moonblast")["tgt_offense"] == -0.15                                       # 30% 특공 -1
assert f("shellsmash")["self_offense"] == 1 and f("shellsmash")["self_defense"] == -1 and f("shellsmash")["self_speed"] == 1
assert f("scald")["tgt_brn"] == 0.3
assert f("willowisp")["tgt_brn"] == 1                                               # 변화기 status 필드
assert f("fakeout")["tgt_flinch"] == 1
assert f("yawn")["tgt_slp"] == 1
assert f("recover")["heal"] == 0.5 and f("rest")["heal"] == 1 and f("wish")["heal"] == 0.5
assert "tgt_slp" not in f("rest")                                                   # 자기 수면은 상대 상태이상 아님
assert f("drainingkiss")["drain"] == 0.75
assert f("flareblitz")["recoil"] > 0 and f("flareblitz")["tgt_brn"] == 0.1
assert f("whirlwind")["force_switch"] == 1
assert f("protect")["protect"] == 1
assert f("taunt")["disrupt"] == 1
assert f("leechseed")["residual_on_target"] == 1 and f("saltcure")["residual_on_target"] == 1
assert f("substitute")["substitute"] == 1 and f("shedtail")["self_switch"] == 1
assert f("bodypress")["alt_offense_stat"] == 1 and f("foulplay")["alt_offense_stat"] == 1
assert f("sacredsword")["ignore_target_boosts"] == 1
assert f("explosion")["self_sacrifice"] == 1
assert f("hyperbeam")["charge_or_recharge"] == 1 and f("solarbeam")["charge_or_recharge"] == 1
assert f("raindance")["weather_terrain_tailwind"] == 1 and f("tailwind")["weather_terrain_tailwind"] == 1
assert f("bulletseed")["extra_hits"] == round((19 / 6 - 1) / 4, 3)                  # 2-5타 기대 3.17
assert f("stoneedge")["crit_stage"] == 0.5
assert f("seismictoss")["fixed_damage"] == 1
assert f("moonlight")["heal"] == 0.5 and f("painsplit")["heal"] == 0.5 and f("strengthsap")["heal"] == 0.5
assert f("curse") == {"self_offense": 0.5, "self_defense": 0.5, "self_speed": -0.5}
assert f("bellydrum")["self_offense"] == 1                                          # +6은 1로 클램프
assert f("notarealmove") == {}                                                      # 모르는 기술 = 0벡터
assert f("thunderclap")["priority_conditional"] == 1                                # 상대가 공격기 아니면 우선도 0으로 처리됨
assert f("suckerpunch")["priority_conditional"] == 1                                # 상대가 공격기 아니면 기술 자체가 실패
assert f("upperhand")["priority_conditional"] == 1                                  # 상대가 우선기 아니면 기술 자체가 실패
assert "priority_conditional" not in f("quickattack")                               # 조건 없는 일반 선공기는 해당 없음
assert "priority_conditional" not in f("extremespeed")
print(f"move effects OK ({MOVE_EFFECT_DIM}칸)")
