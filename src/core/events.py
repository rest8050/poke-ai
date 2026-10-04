"""
턴 사건 + 상대 포켓몬별 증거 추적 (엔진 없이 신경망이 상대 세트를 추론하게 하는 입력 특징).

battle.parse_message를 한 번 감싸서 리플레이 재생 / 실전 / Foul Play 데이터 변환이 모두 같은 경로로 채움.
- 턴 사건 (TURN_DIM=10, field_vec 뒤에 붙음): 직전 턴에 누가 움직였고 누가 먼저였는지, 서로 준 피해(% HP), 급소/상성
- 상대 슬롯별 증거 (SLOT_DIM=8, 상대 팀 수치 뒤에 붙음):
    공격 잔차 / 방어 잔차 = log(관측 피해 / 표준 세트 예상 피해)의 평균 → 구애/생명의구슬/노력치/내구 도구 단서
    속도 구간 [하한, 상한] = 행동 순서에서 얻은 "표준 대비 속도 배율" 범위 → 스카프/속도 노력치 단서
표준 세트 가정(공격 노력치 252, 방어/HP 84, 성격 보정 없음)은 양쪽 모두 같은 함수라서 학습/실전이 같음.

ponytail: 날씨/벽/특성/아이템 보정과 다단히트/고정 피해기는 예상 피해에서 제외(잔차에 섞임 → 신경망이 분포로 학습).
"""
import math
import re

from poke_env.battle import Field, Move, MoveCategory, SideCondition, Status
from poke_env.battle.abstract_battle import AbstractBattle

TURN_DIM, SLOT_DIM = 10, 8
_FLUSH = {"move", "turn", "switch", "drag", "replace", "upkeep", "faint", "cant", "-terastallize"}
# 위력/스탯이 특수하게 정해지는 기술은 예상 피해를 못 구함
_SKIP = {"foulplay", "bodypress", "seismictoss", "nightshade", "superfang", "naturesmadness", "ruination", "gyroball",
         "electroball", "heavyslam", "heatcrash", "grassknot", "lowkick", "storedpower", "powertrip", "punishment",
         "acrobatics", "facade", "hex", "venoshock", "brine", "knockoff", "psyshock", "psystrike", "secretsword",
         "photongeyser", "terablast", "tera starstorm", "terastarstorm", "weatherball", "rage fist", "ragefist",
         "lastrespects", "beatup", "counter", "mirrorcoat", "metalburst", "endeavor", "finalgambit", "bide"}


def _n(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _stat(base, level, ev):
    return ((2 * base + 31 + ev // 4) * level) // 100 + 5


def _boost(b):
    return (2 + b) / 2 if b >= 0 else 2 / (2 - b)


def _mon(battle, side):
    return battle.active_pokemon if side == battle.player_role else battle.opponent_active_pokemon


def expected_frac(A, D, mv, crit):
    """표준 세트 가정의 예상 피해 (방어자 최대 HP 대비 비율). 못 구하면 None"""
    if mv.id in _SKIP or not mv.base_power or mv.damage or mv.n_hit != (1, 1) or mv.category not in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
        return None
    phys = mv.category == MoveCategory.PHYSICAL
    ak, dk = ("atk", "def") if phys else ("spa", "spd")
    if not A.base_stats or not D.base_stats:
        return None
    ab, db = A.boosts.get(ak, 0), D.boosts.get(dk, 0)
    if crit:
        ab, db = max(0, ab), min(0, db)
    lvl = A.level or 100
    a = _stat(A.base_stats[ak], lvl, 252) * _boost(ab)
    d = _stat(D.base_stats[dk], D.level or 100, 84) * _boost(db)
    hp = ((2 * D.base_stats["hp"] + 31 + 21) * (D.level or 100)) // 100 + (D.level or 100) + 10
    dmg = ((2 * lvl // 5 + 2) * mv.base_power * a / d) / 50 + 2
    mod = 0.925 * (1.5 if crit else 1.0) * (1.5 if mv.type in A.types else 1.0) * D.damage_multiplier(mv)
    if A.status == Status.BRN and phys:
        mod *= 0.5
    return dmg * mod / hp if mod > 0 else None


class EventTracker:
    def __init__(self):
        self.last = [0.0] * TURN_DIM
        self.slot = {}  # 상대 종 → {atk:[n,sum], defn:[n,sum], lo, hi}
        self._reset_turn()

    def _reset_turn(self):
        self.moves = []  # (side, 준비된 속도 추정, 우선도, 트릭룸)
        self.moved = {"me": 0.0, "opp": 0.0}
        self.dealt = {"me": 0.0, "opp": 0.0}
        self.crit = {"me": 0.0, "opp": 0.0}
        self.eff = {"me": 0.0, "opp": 0.0}
        self.attack = None

    def _s(self, sp):
        return self.slot.setdefault(sp, {"atk": [0, 0.0], "defn": [0, 0.0], "lo": None, "hi": None})

    def flush_attack(self):
        a, self.attack = self.attack, None
        if not a or a["n"] != 1 or a["ko"] or a["lost"] <= 0:
            return
        exp = a["exp"]
        if not exp:
            return
        r = max(-1.5, min(1.5, math.log(a["lost"] / exp)))
        if a["side"] == "opp":  # 상대가 나를 때림 → 그 공격자의 공격 잔차
            t = self._s(_n(a["A"].species))["atk"]
        else:  # 내가 상대를 때림 → 그 방어자의 방어 잔차 (클수록 무름)
            t = self._s(_n(a["D"].species))["defn"]
        t[0] += 1
        t[1] += r

    def roll_turn(self):
        self.flush_attack()
        first = {}
        for m in self.moves:
            first.setdefault(m[0], m)
        order, inf = 0.0, 0.0
        if "me" in first and "opp" in first and first["me"][2] == first["opp"][2]:
            inf = 1.0
            me_first = self.moves.index(first["me"]) < self.moves.index(first["opp"])
            order = 1.0 if me_first else -1.0
            tr = first["me"][3]
            faster_opp_moved_first = (not me_first) != tr  # 트릭룸이면 느린 쪽이 먼저
            r = math.log(first["me"][1] / first["opp"][1])  # 표준 속도 대비 내 속도 / 상대 속도
            s = self._s(first["opp"][4])
            if faster_opp_moved_first:
                s["lo"] = r if s["lo"] is None else max(s["lo"], r)
            else:
                s["hi"] = r if s["hi"] is None else min(s["hi"], r)
        self.last = [self.moved["me"], self.moved["opp"], order, inf,
                     min(1.0, self.dealt["me"]), min(1.0, self.dealt["opp"]), self.crit["me"], self.crit["opp"],
                     self.eff["me"], self.eff["opp"]]
        self._reset_turn()

    def slot_vec(self, species):
        s = self.slot.get(species)
        if not s:
            return [0.0] * SLOT_DIM
        c = lambda x: max(-1.0, min(1.0, x))
        return [min(s["atk"][0], 4) / 4, c(s["atk"][1] / max(1, s["atk"][0])), min(s["defn"][0], 4) / 4,
                c(s["defn"][1] / max(1, s["defn"][0])), float(s["lo"] is not None), c(s["lo"] or 0.0),
                float(s["hi"] is not None), c(s["hi"] or 0.0)]


def _speed(battle, side, mon):
    """표준 세트 가정의 유효 속도 (능력 단계, 마비, 순풍 반영)"""
    v = _stat(mon.base_stats["spe"], mon.level or 100, 84) * _boost(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        v *= 0.5
    conds = battle.side_conditions if side == "me" else battle.opponent_side_conditions
    return v * (2.0 if SideCondition.TAILWIND in conds else 1.0)


def _hook(orig):
    def parse_message(self, split_message):
        try:
            ev = self.__dict__.get("_ev") or self.__dict__.setdefault("_ev", EventTracker())
            kind = split_message[1] if len(split_message) > 1 else ""
            tags = [x for x in split_message[4:] if x.startswith("[from]")] if len(split_message) > 4 else []
            if kind in _FLUSH:
                ev.flush_attack()
            pre = None
            if kind == "-damage" and not tags and ev.attack:
                t = _mon(self, split_message[2][:2])
                pre = t.current_hp_fraction if t else None
        except Exception:
            ev = kind = pre = None
        orig(self, split_message)
        if ev is None:
            return
        try:
            if kind == "move" and not tags:
                side = "me" if split_message[2][:2] == self.player_role else "opp"
                mon = _mon(self, split_message[2][:2])
                mv = Move(_n(split_message[3]), gen=9)
                ev.moved[side] = 1.0
                ev.moves.append((side, _speed(self, side, mon), mv.priority, Field.TRICK_ROOM in self.fields, _n(mon.species)))
                ev.attack = {"side": side, "A": mon, "mv": mv, "D": None, "n": 0, "lost": 0.0, "ko": False, "crit": False, "exp": None}
            elif kind == "-crit" and ev.attack:
                ev.attack["crit"] = True
                ev.crit[ev.attack["side"]] = 1.0
            elif kind in ("-supereffective", "-resisted") and ev.attack:
                ev.eff[ev.attack["side"]] = 1.0 if kind == "-supereffective" else -1.0
            elif kind == "-damage" and not tags and ev.attack and pre is not None:
                t = _mon(self, split_message[2][:2])
                a = ev.attack
                a["D"], a["n"] = t, a["n"] + 1
                if a["n"] == 1:  # 부스트 등이 바뀌기 전인 지금 예상 피해를 계산
                    a["exp"] = expected_frac(a["A"], t, a["mv"], a["crit"])
                lost = max(0.0, pre - t.current_hp_fraction)
                a["lost"] += lost
                a["ko"] = a["ko"] or t.current_hp_fraction <= 0
                ev.dealt[a["side"]] += lost
            elif kind == "turn":
                ev.roll_turn()
        except Exception:
            pass
    return parse_message


if not getattr(AbstractBattle.parse_message, "_ev_hooked", False):
    AbstractBattle.parse_message = _hook(AbstractBattle.parse_message)
    AbstractBattle.parse_message._ev_hooked = True
