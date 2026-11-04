"""모델의 손계산(src/core/matchup.py의 Matchup + 인코더의 능력치/위력 보정)을 torch 없이 한 쌍씩 계산하는 기준 구현.
사건 단위 진단(tools/damage_residual_report.py)이 "모델이 쓰는 손계산이 실제 피해와 얼마나 어긋나는가"를 특성/도구/기술별로 재는 데 씀.

Matchup과 같은 근사: 레벨 100 공식(42 x 위력 x A/D / 50 + 2), 자속 1.5(원래 타입 + 테라 중이면 테라 타입), 화상이면 물리 0.5,
  타입 상성(방어측이 테라 중이면 테라 타입 하나), 면역 특성 11종(mechanics.IMMUNE_ABILITIES) + 풍선의 땅 무효, 공/방 랭크 배율,
  평균 난수 0.925(KO 판정은 최댓값 1.0)
인코더와 같은 보정: 비/쾌청의 물·불 1.5/0.5, 웨더볼 위력 2배 + 날씨 타입(effective_move_power/effective_move_type),
  능력치에 미리 곱하는 도구 배율(구애머리띠/안경/스카프, 생명의구슬 1.3, 돌격조끼, 진화의휘석, 부스트에너지 — tensor_encoder 내 쪽 인코딩과 동일)
반영 안 함(= 진단이 찾아낼 대상): 급소, 벽, 특성의 위력/피해/랭크 보정(천진, 멀티스케일, 천하장사 등), 그 밖의 도구 위력 보정, 필드, 가변 위력 기술
"""
from src.core.mechanics import IMMUNE_ABILITIES

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
