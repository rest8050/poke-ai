"""게임 규칙 계산(torch 없음): 타입 상성/특성/도구 조회, 면역 특성 표, 스칼라 데미지 계산, 상태 평가식(보상 정형화).
"""
import math


# ===== effectiveness: 타입 상성 계산 (장판/특성/도구로 바뀌는 무효·반감 포함): 인코더, 브리지, 플레이어, 전황 평가가 같은 계산을 씀. =====
#
# 원본: Radical Cobblemon Trainers API 0.15.2-beta (서버 jar와 같은 버전)
#       com/gitlab/srcmc/rctapi/api/ai/utils/TypeChart.java, PokeMath.java
#       Copyright (c) 2026, HDainester. Mod Community License (MCOML).
#       PokeMath은 davo899의 CobblemonTrainers Gen5 AI에서 파생.
#
# 변경 사항 (MCOML에 따라 명시):
# - Cobblemon 배틀 객체 대신 poke-env Battle/Pokemon 객체를 읽음.
# - 상대 특성은 공개됐거나 후보가 1개일 때만 사용.
# - 원본의 전투 AI(RCTBattleAI) 이식판은 삭제 (git 이력에 있음, 현행 코드는 타입 상성 계산만 씀).
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


# ===== mechanics: torch 없이 쓰는 게임 메커니즘 상수 (matchup.py의 텐서 계산과 damage_calc.py의 스칼라 기준 계산이 같은 표를 씀). =====
# 이 특성을 가진 방어측은 해당 타입 공격기에 무효 (공중부양 계열 + 흡수/마중물 계열)
IMMUNE_ABILITIES = {"levitate": "ground", "eartheater": "ground", "flashfire": "fire", "wellbakedbody": "fire", "waterabsorb": "water",
                    "dryskin": "water", "stormdrain": "water", "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
                    "sapsipper": "grass"}


# ===== damage_calc: 모델의 손계산(src/core/matchup.py의 Matchup + 인코더의 능력치/위력 보정)을 torch 없이 한 쌍씩 계산하는 기준 구현. =====
# 사건 단위 진단(tools/damage_residual_report.py)이 "모델이 쓰는 손계산이 실제 피해와 얼마나 어긋나는가"를 특성/도구/기술별로 재는 데 씀.
#
# Matchup과 같은 근사: 레벨 100 공식(42 x 위력 x A/D / 50 + 2), 자속 1.5(원래 타입 + 테라 중이면 테라 타입), 화상이면 물리 0.5,
#   타입 상성(방어측이 테라 중이면 테라 타입 하나), 면역 특성 11종(mechanics.IMMUNE_ABILITIES) + 풍선의 땅 무효, 공/방 랭크 배율,
#   평균 난수 0.925(KO 판정은 최댓값 1.0)
# 인코더와 같은 보정: 비/쾌청의 물·불 1.5/0.5, 웨더볼 위력 2배 + 날씨 타입(effective_move_power/effective_move_type),
#   능력치에 미리 곱하는 도구 배율(구애머리띠/안경/스카프, 생명의구슬 1.3, 돌격조끼, 진화의휘석, 부스트에너지 — tensor_encoder 내 쪽 인코딩과 동일)
# 반영 안 함(= 진단이 찾아낼 대상): 급소, 벽, 특성의 위력/피해/랭크 보정(천진, 멀티스케일, 천하장사 등), 그 밖의 도구 위력 보정, 필드, 가변 위력 기술
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
# 성격: (올리는 능력, 내리는 능력). 무보정 성격은 표에 없음
NATURES = {
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
}
RAIN, SUN = {"raindance", "primordialsea"}, {"sunnyday", "desolateland"}
WEATHER_BALL_TYPE = {"raindance": "water", "primordialsea": "water", "sunnyday": "fire", "desolateland": "fire",
                     "sandstorm": "rock", "snow": "ice", "snowscape": "ice", "hail": "ice"}
MEAN_ROLL = 0.925
# 위력 수치 자체가 의미 없거나(무게/HP/랭크/횟수로 계산) 다른 능력치·타입·분류로 바뀌는 기술 → 진단에서 비교하지 않고 따로 셈.
# 조건부 배율 기술(탁쳐내기, 병상첨병, 객기 등)은 고정 위력으로 비교 대상에 남김 — 손계산이 놓치는 보정으로 기술별 표에 드러나게
VARIABLE_POWER = {
    "foulplay", "bodypress", "psyshock", "psystrike", "secretsword", "terablast", "terastarstorm", "ivycudgel", "photongeyser",
    "shellsidearm", "ragefist", "lastrespects", "storedpower", "powertrip", "gyroball", "electroball", "heavyslam", "heatcrash",
    "lowkick", "grassknot", "eruption", "waterspout", "dragonenergy", "flail", "reversal", "wringout", "crushgrip", "hardpress",
    "punishment", "beatup", "present", "magnitude", "naturalgift", "spitup", "tripleaxel", "triplekick", "trumpcard",
}


def calc_stats(base, evs=None, ivs=None, level=100, nature=None):
    """종족값 dict + 노력치/개체값 [hp,atk,def,spa,spd,spe] + 레벨 + 성격 → 실제 능력치 dict (게임 공식 그대로)"""
    evs = evs or [0] * 6
    ivs = ivs or [31] * 6
    up, down = NATURES.get(str(nature or "").lower(), (None, None))
    out = {}
    for k, ev, iv in zip(STAT_KEYS, evs, ivs):
        core = (2 * base[k] + iv + ev // 4) * level // 100
        if k == "hp":
            out[k] = core + level + 10
        else:
            mult = 1.1 if k == up else 0.9 if k == down else 1.0
            out[k] = int((core + 5) * mult)
    return out


def standard_stats(base):
    """상대 세트를 모를 때 Matchup이 쓰는 가정: 레벨 100, 개체값 31, 노력치 252, 무보정 (2b+99, HP 2b+204)"""
    return {k: 2 * base[k] + (204 if k == "hp" else 99) for k in STAT_KEYS}


def item_stat_mult(stats, item, ability=None):
    """인코더가 능력치에 미리 곱하는 도구 배율 (tensor_encoder.encode_battle의 내 쪽 인코딩과 같은 규칙)"""
    s = dict(stats)
    if item == "choiceband":
        s["atk"] *= 1.5
    elif item == "choicespecs":
        s["spa"] *= 1.5
    elif item == "choicescarf":
        s["spe"] *= 1.5
    elif item == "lifeorb":
        s["atk"] *= 1.3
        s["spa"] *= 1.3
    elif item == "assaultvest":
        s["spd"] *= 1.5
    elif item == "eviolite":
        s["def"] *= 1.5
        s["spd"] *= 1.5
    elif item == "boosterenergy" and ability in ("protosynthesis", "quarkdrive"):
        best = max(("atk", "def", "spa", "spd", "spe"), key=lambda k: s[k])
        s[best] *= 1.5 if best == "spe" else 1.3
    return s


def stage_mult(stage):
    stage = max(-6, min(6, int(stage)))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def type_eff(chart, move_type, def_types):
    """chart[공격 타입][방어 타입] (data/type_chart.json 형식, 소문자) — 표에 없는 조합은 등배"""
    eff = 1.0
    for t in def_types:
        if t:
            eff *= chart.get(move_type, {}).get(t, 1.0)
    return eff


def supported(move):
    """이 손계산으로 피해를 비교할 수 있는 기술인가: 공격기 + 고정 위력 + 고정 데미지/일격기/능력치 대체가 아님"""
    if move.get("category") not in ("Physical", "Special") or not move.get("basePower"):
        return False
    if move.get("damage") or move.get("ohko") or move.get("id") in VARIABLE_POWER:
        return False
    return not any(move.get(k) for k in ("overrideOffensiveStat", "overrideOffensivePokemon", "overrideDefensiveStat"))


def damage(att, dfn, move, chart, weather=None):
    """한 번 맞을 때의 피해(절대 HP). att/dfn: {"stats": 실능력치(도구 배율 적용 후), "types": [..], "tera": 타입|None(테라 중일 때만),
    "boosts": {"atk":..}, "status": "brn"|..|None, "ability": .., "item": ..}, move: poke-env 기술 dict(+ "id"), chart: 상성표.
    반환 {"eff": 상성 배율(면역 0), "mean": 평균 난수 피해, "max": 최대 난수 피해(KO 판정용)}"""
    mtype = str(move.get("type", "")).lower()
    bp = float(move.get("basePower") or 0)
    if move.get("id") == "weatherball" and weather in WEATHER_BALL_TYPE:
        mtype, bp = WEATHER_BALL_TYPE[weather], bp * 2.0
    elif weather in RAIN:
        bp *= 1.5 if mtype == "water" else 0.5 if mtype == "fire" else 1.0
    elif weather in SUN:
        bp *= 1.5 if mtype == "fire" else 0.5 if mtype == "water" else 1.0

    phys = move.get("category") == "Physical"
    a_key, d_key = ("atk", "def") if phys else ("spa", "spd")
    A = att["stats"][a_key] * stage_mult(att.get("boosts", {}).get(a_key, 0))
    D = max(1.0, dfn["stats"][d_key] * stage_mult(dfn.get("boosts", {}).get(d_key, 0)))

    stab = 1.5 if mtype in att.get("types", []) or (att.get("tera") and mtype == att["tera"]) else 1.0
    def_types = [dfn["tera"]] if dfn.get("tera") else dfn.get("types", [])
    eff = type_eff(chart, mtype, def_types)
    if IMMUNE_ABILITIES.get(dfn.get("ability")) == mtype or (dfn.get("item") == "airballoon" and mtype == "ground"):
        eff = 0.0
    burn = 0.5 if phys and att.get("status") == "brn" else 1.0

    dmg = (42.0 * bp * A / D / 50.0 + 2.0) * stab * eff * burn
    return {"eff": eff, "mean": dmg * MEAN_ROLL, "max": dmg}


# ===== battle_eval: Foul Play(poke-engine src/genx/evaluate.rs) 방식의 국면 평가 → 보상 설계용 잠재 함수 Φ. =====
#
# evaluate(battle) = 내 진영 점수 - 상대 진영 점수 (Foul Play 단위: 포켓몬 1마리 만피 생존 도구 보유 = 140)
# potential(battle) = SHAPING_SCALE × evaluate / 100
# terms(battle) = 항목별 (내 점수 - 상대 점수): hp / alive / item / status / boost / volatile / hazard / screen / tera (검증과 가중치 보정용)
#
# 보상에는 잠재 함수 형태로만 사용: r = γ·Φ(s') - Φ(s), 종료 상태 Φ = 0
# → 한 판 합이 (승패 - Φ(시작))으로 상쇄되어 최적 정책은 이론상 그대로, 장기 자산(장판/벽/랭크/테라)에 대한 신용 할당만 빨라짐.
#
# 원본과 같게 맞춘 것: 설치 기술(스텔스록/압정/독압정/끈적네트)은 그 진영의 **살아 있는 포켓몬마다** 적용 (두꺼운부츠·매직가드·비접지 예외 포함),
# 화상은 물리기 수 × -25 (특공이 높으면 절반, 근성/매직스킨/속보는 +50), 도구 +10을 더한 뒤 0 아래로 내려가지 않게 자르고 생존 +30.
#
# 원본과 다른 점 (ponytail 근사):
# - 접지 판정: 비행 타입, 부유, 풍선, 전자부유만 봄 (중력 등 미반영)
# - 화상 물리기 수: 공개된 기술 기준 (상대가 기술을 하나도 안 보였으면 공격 종족값 >= 특공이면 2개로 가정), 공격/특공 비교는 종족값
# - 치유소원: poke-env가 추적 안 해서 미반영
# - 상대의 공개 안 된 포켓몬은 만피·생존·도구 보유로 계산하고 설치 기술은 접지·부츠 없음으로 가정
SHAPING_SCALE = 0.2

ALIVE, HP, ITEM, USED_TERA = 30.0, 100.0, 10.0, -75.0
BOOST_WEIGHT = {"atk": 30.0, "def": 15.0, "spa": 30.0, "spd": 15.0, "spe": 30.0}
BOOST_MULT = {6: 3.3, 5: 3.15, 4: 3.0, 3: 2.5, 2: 2.0, 1: 1.0, 0: 0.0,
              -1: -1.0, -2: -2.0, -3: -2.5, -4: -3.0, -5: -3.15, -6: -3.3}
STATUS = {"frz": -40.0, "slp": -25.0, "par": -25.0, "tox": -30.0, "psn": -10.0, "brn": -25.0}
VOLATILE = {"leechseed": -30.0, "substitute": 75.0, "confusion": -20.0}
SCREEN = {"reflect": 20.0, "lightscreen": 20.0, "auroraveil": 40.0, "safeguard": 5.0, "tailwind": 7.0}
HAZARD = {"stealthrock": -10.0, "spikes": -7.0, "toxicspikes": -7.0, "stickyweb": -25.0}
LAYERED = {"spikes", "toxicspikes"}
UNSEEN_MON = HP + ALIVE + ITEM
TERMS = ("hp", "alive", "item", "status", "boost", "volatile", "hazard", "screen", "tera")


def _burn_multiplier(mon) -> float:
    st = mon.base_stats
    known = [m for m in mon.moves.values()]
    n = sum(_id(m.category) == "physical" for m in known) if known else (2 if st["atk"] >= st["spa"] else 0)
    return n / 2 if st["spa"] > st["atk"] else float(n)


def _status_score(mon) -> float:
    status = _id(mon.status)
    ab = ability(mon)
    if status in ("psn", "tox"):
        if ab == "poisonheal":
            return 15.0
        if ab in ("guts", "marvelscale", "quickfeet", "toxicboost", "magicguard"):
            return 10.0
    if status == "brn":
        if ab in ("guts", "marvelscale", "quickfeet"):
            return -2 * STATUS["brn"]
        return _burn_multiplier(mon) * STATUS["brn"]
    return STATUS.get(status, 0.0)


def _grounded(mon) -> bool:
    return not ("flying" in types(mon) or ability(mon) == "levitate" or _id(mon.item) == "airballoon"
                or "magnetrise" in {_id(e) for e in (mon.effects or {})})


def _hazard_score(mon, layers) -> float:
    """이 진영에 깔린 설치 기술이 이 포켓몬 하나에게 주는 손해 (mon=None: 공개 안 된 상대 = 접지, 부츠·매직가드 없음)"""
    if mon is not None and _id(mon.item) == "heavydutyboots":
        return 0.0
    grounded = mon is None or _grounded(mon)
    score = 0.0
    if mon is None or ability(mon) != "magicguard":
        score += layers.get("stealthrock", 0) * HAZARD["stealthrock"]
        if grounded:
            score += layers.get("spikes", 0) * HAZARD["spikes"] + layers.get("toxicspikes", 0) * HAZARD["toxicspikes"]
    if grounded:
        score += layers.get("stickyweb", 0) * HAZARD["stickyweb"]
    return score


def _layers(conditions) -> dict:
    out = {}
    for cond, val in (conditions or {}).items():
        name = _id(cond)
        out[name] = val if name in LAYERED and isinstance(val, int) and not isinstance(val, bool) else 1
    return out


def _side_terms(mons, conditions, used_tera, unseen=0) -> dict:
    t = dict.fromkeys(TERMS, 0.0)
    layers = _layers(conditions)
    for m in mons:
        if m.fainted:
            continue
        hp, item = HP * m.current_hp_fraction, (ITEM if m.item else 0.0)
        t["hp"] += hp
        t["item"] += item
        t["status"] += max(_status_score(m), -(hp + item))      # 원본은 체력+상태+도구가 음수면 0으로 자름 (그 몫을 상태 항목이 받음)
        t["alive"] += ALIVE
        t["hazard"] += _hazard_score(m, layers)
        if m.active:
            t["boost"] += sum(BOOST_MULT[max(-6, min(6, m.boosts.get(k, 0)))] * w for k, w in BOOST_WEIGHT.items())
            t["volatile"] += sum(VOLATILE.get(_id(e), 0.0) for e in (m.effects or {}))
    t["hp"] += unseen * HP
    t["item"] += unseen * ITEM
    t["alive"] += unseen * ALIVE
    t["hazard"] += unseen * _hazard_score(None, layers)
    t["screen"] = sum(v for name, v in SCREEN.items() if name in layers)
    t["tera"] = USED_TERA if used_tera else 0.0
    return t


def terms(battle) -> dict:
    mine = _side_terms(battle.team.values(), battle.side_conditions, getattr(battle, "used_tera", False))
    theirs = _side_terms(battle.opponent_team.values(), battle.opponent_side_conditions, getattr(battle, "opponent_used_tera", False),
                         unseen=max(0, 6 - len(battle.opponent_team)))
    return {k: mine[k] - theirs[k] for k in TERMS}


def evaluate(battle) -> float:
    return sum(terms(battle).values())


def potential(battle, drop=()) -> float:
    """drop: 포텐셜에서 뺄 항목 이름들 (TERMS 중). RL advantage로 쓸 때 tera는 뺌: 테라스탈 사용 자체에 -75를 매겨서 정책이 테라를 안 쓰는 쪽으로 감 (2026-10 pilot_phi)"""
    if not getattr(battle, "team", None):
        return 0.0
    return SHAPING_SCALE * sum(v for k, v in terms(battle).items() if k not in drop) / 100.0
