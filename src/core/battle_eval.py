"""
Foul Play(poke-engine src/genx/evaluate.rs) 방식의 국면 평가 → 보상 설계용 잠재 함수 Φ.

evaluate(battle) = 내 진영 점수 - 상대 진영 점수 (Foul Play 단위: 포켓몬 1마리 만피 생존 도구 보유 = 140)
potential(battle) = SHAPING_SCALE × evaluate / 100
terms(battle) = 항목별 (내 점수 - 상대 점수): hp / alive / item / status / boost / volatile / hazard / screen / tera (검증과 가중치 보정용)

보상에는 잠재 함수 형태로만 사용: r = γ·Φ(s') - Φ(s), 종료 상태 Φ = 0
→ 한 판 합이 (승패 - Φ(시작))으로 상쇄되어 최적 정책은 이론상 그대로, 장기 자산(장판/벽/랭크/테라)에 대한 신용 할당만 빨라짐.

원본과 같게 맞춘 것: 설치 기술(스텔스록/압정/독압정/끈적네트)은 그 진영의 **살아 있는 포켓몬마다** 적용 (두꺼운부츠·매직가드·비접지 예외 포함),
화상은 물리기 수 × -25 (특공이 높으면 절반, 근성/매직스킨/속보는 +50), 도구 +10을 더한 뒤 0 아래로 내려가지 않게 자르고 생존 +30.

원본과 다른 점 (ponytail 근사):
- 접지 판정: 비행 타입, 부유, 풍선, 전자부유만 봄 (중력 등 미반영)
- 화상 물리기 수: 공개된 기술 기준 (상대가 기술을 하나도 안 보였으면 공격 종족값 >= 특공이면 2개로 가정), 공격/특공 비교는 종족값
- 치유소원: poke-env가 추적 안 해서 미반영
- 상대의 공개 안 된 포켓몬은 만피·생존·도구 보유로 계산하고 설치 기술은 접지·부츠 없음으로 가정
"""
from src.core.effectiveness import _id, ability, types

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
