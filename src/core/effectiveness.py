"""
타입 상성 계산 (장판/특성/도구로 바뀌는 무효·반감 포함): 인코더, 브리지, 플레이어, 전황 평가가 같은 계산을 씀.

원본: Radical Cobblemon Trainers API 0.15.2-beta (서버 jar와 같은 버전)
      com/gitlab/srcmc/rctapi/api/ai/utils/TypeChart.java, PokeMath.java
      Copyright (c) 2026, HDainester. Mod Community License (MCOML).
      PokeMath은 davo899의 CobblemonTrainers Gen5 AI에서 파생.

변경 사항 (MCOML에 따라 명시):
- Cobblemon 배틀 객체 대신 poke-env Battle/Pokemon 객체를 읽음.
- 상대 특성은 공개됐거나 후보가 1개일 때만 사용.
- 원본의 전투 AI(RCTBattleAI) 이식판은 삭제 (git 이력에 있음, 현행 코드는 타입 상성 계산만 씀).
"""
import math


# ---------- poke-env 객체 읽기 ----------

# TypeChart: 방어 타입 -> {공격 타입: 배율}
CHART = {
    "normal": {"fighting": 2, "ghost": 0},
    "fighting": {"flying": 2, "rock": .5, "bug": .5, "psychic": 2, "dark": .5, "fairy": 2},
    "flying": {"fighting": .5, "ground": 0, "rock": 2, "bug": .5, "grass": .5, "electric": 2, "ice": 2},
    "poison": {"fighting": .5, "poison": .5, "ground": 2, "bug": .5, "grass": .5, "psychic": 2, "fairy": .5},
    "ground": {"poison": .5, "rock": .5, "water": 2, "grass": 2, "electric": 0, "ice": 2},
    "rock": {"normal": .5, "fighting": 2, "flying": .5, "poison": .5, "ground": 2, "steel": 2, "fire": .5,
             "water": 2, "grass": 2},
    "bug": {"fighting": .5, "flying": 2, "ground": .5, "rock": 2, "fire": 2, "grass": .5},
    "ghost": {"normal": 0, "fighting": 0, "poison": .5, "bug": .5, "ghost": 2, "dark": 2},
    "steel": {"normal": .5, "fighting": 2, "flying": .5, "poison": 0, "ground": 2, "rock": .5, "bug": .5,
              "steel": .5, "fire": 2, "grass": .5, "psychic": .5, "ice": .5, "dragon": .5, "fairy": .5},
    "fire": {"ground": 2, "rock": 2, "bug": .5, "steel": .5, "fire": .5, "water": 2, "grass": .5, "ice": .5,
             "fairy": .5},
    "water": {"steel": .5, "fire": .5, "water": .5, "grass": 2, "electric": 2, "ice": .5},
    "grass": {"flying": 2, "poison": 2, "ground": .5, "bug": 2, "fire": 2, "water": .5, "grass": .5,
              "electric": .5, "ice": 2},
    "electric": {"flying": .5, "ground": 2, "steel": .5, "electric": .5},
    "psychic": {"fighting": .5, "bug": 2, "ghost": 2, "psychic": .5, "dark": 2},
    "ice": {"fighting": 2, "rock": 2, "steel": 2, "fire": 2, "ice": .5},
    "dragon": {"fire": .5, "water": .5, "grass": .5, "electric": .5, "ice": 2, "dragon": 2, "fairy": 2},
    "dark": {"fighting": 2, "bug": 2, "ghost": .5, "psychic": 0, "dark": .5, "fairy": 2},
    "fairy": {"fighting": .5, "poison": 2, "bug": .5, "steel": 2, "dragon": 0, "dark": .5},
}


def _id(x):
    """Enum/문자열 -> 소문자 id ('LIGHT_SCREEN' -> 'lightscreen')"""
    if x is None:
        return ""
    return str(getattr(x, "name", x)).lower().replace("_", "").replace(" ", "")


def types(mon):
    return [t for t in (_id(mon.type_1), _id(mon.type_2)) if t]


def ability(mon):
    a = mon.ability
    if not a:
        possible = list(mon.possible_abilities or [])
        a = possible[0] if len(possible) == 1 else None
    return _id(a)


def has_eff(mon, name):
    return any(_id(e) == name for e in (mon.effects or {}))


def field(battle, name):
    return any(_id(f) == name for f in (battle.fields or {}))


def raised(battle, mon):
    if field(battle, "gravity") or has_eff(mon, "ingrain") or has_eff(mon, "smackdown"):
        return False
    item = _id(mon.item)
    if item == "ironball":
        return False
    if item == "airballoon" or ability(mon) == "levitate" or "flying" in types(mon):
        return True
    return has_eff(mon, "magnetrise") or has_eff(mon, "telekinesis")


def eff_single(battle, atk_type, def_type, defender, ab=None):
    """ab: 방어측 특성을 직접 지정 (미공개 특성을 추론값으로 판단할 때). None이면 공개 정보 사용"""
    ab = ab if ab is not None else ability(defender)
    if atk_type == "water" and ab in ("stormdrain", "waterabsorb", "dryskin"):
        return 0
    if atk_type == "electric" and ab in ("voltabsorb", "lightningrod", "motordrive"):
        return 0
    if atk_type == "ground":
        if ab == "eartheater" or raised(battle, defender):
            return 0
        if def_type == "flying":
            return 1.0
    if atk_type == "fire" and ab in ("wellbakedbody", "flashfire"):
        return 0
    if atk_type == "grass" and ab == "sapsipper":
        return 0
    return CHART.get(def_type, {}).get(atk_type, 1.0)


def eff_move(battle, move_type, defender, ab=None):
    e = math.prod(eff_single(battle, move_type, dt, defender, ab) for dt in types(defender))
    return 0 if e < 2.0 and (ab if ab is not None else ability(defender)) == "wonderguard" else e



if __name__ == "__main__":
    # poke-env 없이 계산부만 확인하는 자체 점검
    from types import SimpleNamespace as NS

    def mon(species, t1, t2=None, ab=None):
        return NS(species=species, type_1=t1, type_2=t2, ability=ab, possible_abilities=[], item="", effects={})

    garchomp, rotom, skarmory = mon("garchomp", "dragon", "ground"), mon("rotomwash", "electric", "water", ab="levitate"), mon("skarmory", "steel", "flying")
    battle = NS(fields={})
    assert eff_move(battle, "ground", rotom) == 0            # 부유
    assert eff_move(battle, "ice", garchomp) == 4.0
    assert eff_move(battle, "ground", skarmory) == 0         # 비행 타입 = 떠 있음
    battle.fields = {"GRAVITY": 1}
    assert eff_move(battle, "ground", skarmory) == 2.0       # 중력이면 비행 1배 × 강철 2배
    assert _id("LIGHT_SCREEN") == "lightscreen"
    print("effectiveness self-check OK")
