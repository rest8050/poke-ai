"""events.py: 내 쪽은 실제 능력치/도구로 계산 (잔차와 스피드 구간에 내 세트가 섞이지 않게). 실행: PYTHONPATH=. python tests/test_events_own.py"""
from types import SimpleNamespace

from poke_env.battle import Move, Pokemon, Status

from src.core.events import _speed, _stat, expected_frac

mon = lambda sp: Pokemon(details=sp, gen=9)
A, D, eq = mon("Garchomp"), mon("Clefable"), Move("earthquake", gen=9)
std = expected_frac(A, D, eq, False)
assert abs(expected_frac(A, D, eq, False, a_mine=True) - std) < 1e-9, "내 능력치를 모르면(stats 없음) 표준 가정 그대로"
A._stats = {"hp": 357, "atk": 296, "def": 226, "spa": 176, "spd": 206, "spe": 333}          # 공격 무보정 0노력치 = 296, 표준(252)은 359
own = expected_frac(A, D, eq, False, a_mine=True)
assert 0.80 < own / std < 0.85, f"내 실제 공격(296/359)이 반영됨: {own / std}"
A._item = "choiceband"
assert 1.45 < expected_frac(A, D, eq, False, a_mine=True) / own <= 1.5 and expected_frac(A, D, eq, False) == std, "구애머리띠는 내 쪽일 때만"
D._stats, D._item = {"hp": 394, "atk": 176, "def": 245, "spa": 226, "spd": 216, "spe": 156}, "assaultvest"
assert abs(expected_frac(A, D, eq, False, d_mine=True) / std - (_stat(73, 100, 84) / 245) * (((2 * 95 + 52) + 110) / 394)) < 0.02, "내가 맞을 때는 내 실제 방어/HP (물리라 조끼는 무관)"

b = SimpleNamespace(side_conditions={}, opponent_side_conditions={}, weather={})
S = mon("Garchomp")
assert _speed(b, "opp", S) == 2 * 102 + 99, "상대는 표준 가정 2b+99 (매치업과 같음)"
S._stats = {"spe": 333}
assert _speed(b, "me", S) == 333
S._item = "choicescarf"
assert _speed(b, "me", S) == 333 * 1.5 and _speed(b, "opp", S) == 2 * 102 + 99, "스카프는 내 쪽만 (상대 스카프는 구간이 알려 줄 대상)"
S._status = Status.PAR
assert _speed(b, "me", S) == 333 * 1.5 * 0.5


# 선제공격손톱/퀵드로/후공열매가 발동한 턴은 행동 순서를 속도 증거로 쓰지 않음
from src.core.events import EventTracker, _order_jump
assert _order_jump("|-activate|p2a: Slowbro|item: Quick Claw".split("|")) and _order_jump("|-activate|p2a: Slowbro|ability: Quick Draw".split("|"))
assert _order_jump("|-enditem|p2a: Skarmory|Custap Berry|[eat]".split("|")) and not _order_jump("|-activate|p2a: Slowbro|move: Protect".split("|"))
def turn(jump):
    ev = EventTracker()
    ev.moves, ev.jump = [("opp", 100.0, 0, False, "slowbro", False), ("me", 300.0, 0, False, "garchomp", False)], jump
    ev.roll_turn()
    return ev.slot_vec("slowbro")[4], ev.last[3]
assert turn(False) == (1.0, 1.0) and turn(True) == (0.0, 0.0), "발동이 있던 턴은 하한/같은 우선도 표시를 안 남김"
print("events jump OK")
