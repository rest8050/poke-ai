"""
Foul Play(poke-engine genx/evaluate.rs) 방식의 국면 평가 → 보상 설계용 잠재 함수 Φ.

evaluate(battle) = 내 진영 점수 - 상대 진영 점수 (Foul Play 단위: 포켓몬 1마리 만피 생존 ≈ 140)
potential(battle) = SHAPING_SCALE × evaluate / 100

보상에는 잠재 함수 형태로만 사용: r = γ·Φ(s') - Φ(s), 종료 상태 Φ = 0
→ 한 판 합이 (승패 - Φ(시작))으로 상쇄되어 최적 정책은 이론상 그대로, 장기 자산(장판/벽/랭크/테라)에 대한 신용 할당만 빨라짐.

원본과 다른 점 (ponytail 근사):
- 화상: 원본은 물리기 수 비례. 여기선 물리기 보유 시 -25, 없으면 -10
- 두꺼운부츠 장판 면역, 치유소원 등 slot condition 미반영
- 상대의 공개 안 된 포켓몬은 만피·생존·도구 보유로 계산
"""
from src.core.rct_player import _id, ability

SHAPING_SCALE = 0.2

ALIVE, HP, ITEM, USED_TERA = 30.0, 100.0, 10.0, -75.0
BOOST_WEIGHT = {"atk": 30.0, "def": 15.0, "spa": 30.0, "spd": 15.0, "spe": 30.0}
BOOST_MULT = {6: 3.3, 5: 3.15, 4: 3.0, 3: 2.5, 2: 2.0, 1: 1.0, 0: 0.0,
              -1: -1.0, -2: -2.0, -3: -2.5, -4: -3.0, -5: -3.15, -6: -3.3}
STATUS = {"frz": -40.0, "slp": -25.0, "par": -25.0, "tox": -30.0, "psn": -10.0, "brn": -25.0}
VOLATILE = {"leechseed": -30.0, "substitute": 75.0, "confusion": -20.0}
SIDE = {"reflect": 20.0, "lightscreen": 20.0, "auroraveil": 40.0, "safeguard": 5.0, "tailwind": 7.0,
        "stealthrock": -10.0, "spikes": -7.0, "toxicspikes": -7.0, "stickyweb": -25.0}
LAYERED = {"spikes", "toxicspikes"}
UNSEEN_MON = HP + ALIVE + ITEM


def _status_score(mon) -> float:
    status = _id(mon.status)
    ab = ability(mon)
    if status in ("psn", "tox"):
        if ab == "poisonheal":
            return 15.0
        if ab in ("guts", "marvelscale", "quickfeet", "toxicboost", "magicguard"):
            return 10.0
    if status == "brn":
        if ab in ("guts", "marvelscale", "quickfeet", "flareboost", "magicguard"):
            return 10.0
        physical = any(_id(m.category) == "physical" for m in mon.moves.values())
        return STATUS["brn"] if physical else -10.0
    return STATUS.get(status, 0.0)


def pokemon_score(mon, active: bool) -> float:
    if mon.fainted:
        return 0.0
    score = max(0.0, HP * mon.current_hp_fraction + _status_score(mon)) + ALIVE
    if mon.item:  # 'unknown_item'(상대 미공개)도 보유로 봄
        score += ITEM
    if active:
        score += sum(BOOST_MULT[max(-6, min(6, mon.boosts.get(k, 0)))] * w for k, w in BOOST_WEIGHT.items())
        score += sum(VOLATILE.get(_id(e), 0.0) for e in (mon.effects or {}))
    return score


def side_score(conditions) -> float:
    score = 0.0
    for cond, val in (conditions or {}).items():
        name = _id(cond)
        if name in SIDE:
            layers = val if name in LAYERED and isinstance(val, int) and not isinstance(val, bool) else 1
            score += SIDE[name] * layers
    return score


def evaluate(battle) -> float:
    mine = sum(pokemon_score(m, m.active) for m in battle.team.values())
    theirs = sum(pokemon_score(m, m.active) for m in battle.opponent_team.values())
    theirs += UNSEEN_MON * max(0, 6 - len(battle.opponent_team))
    mine += side_score(battle.side_conditions) + (USED_TERA if getattr(battle, "used_tera", False) else 0.0)
    theirs += side_score(battle.opponent_side_conditions) + (
        USED_TERA if getattr(battle, "opponent_used_tera", False) else 0.0)
    return mine - theirs


def potential(battle) -> float:
    if not getattr(battle, "team", None):
        return 0.0
    return SHAPING_SCALE * evaluate(battle) / 100.0
