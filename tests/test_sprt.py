"""SPRT / 장부 / Bradley–Terry self-check: PYTHONPATH=. python tests/test_sprt.py"""
import math
import random

from tools.bt_report import fit_bt, win_table
from tools.ledger import balanced, pair_counts
from tools.sprt import decide

# 1) 경계: 처음엔 계속, 압도적이면 낫다/나쁘다, 오래 반반이면 차이 없음
assert decide(0, 0)["decision"] == "continue"
assert decide(80, 20)["decision"] == "better" and decide(20, 80)["decision"] == "worse"
assert decide(60, 20)["decision"] == "continue"   # 로그우도비 2.26 < 경계 2.94
assert decide(1000, 1000, delta=0.03)["decision"] == "equal" and decide(10, 10)["decision"] == "continue"

# 2) 시뮬레이션: 진짜 승률 53%면 대부분 '낫다', 50%면 대부분 '차이 없음'이고 오탐은 소수
rng = random.Random(0)


def run(p, delta=0.03, cap=20000):
    w = l = 0
    while w + l < cap:
        w, l = (w + 1, l) if rng.random() < p else (w, l + 1)
        d = decide(w, l, delta)["decision"]
        if d != "continue":
            return d
    return "cap"


r53 = [run(0.53) for _ in range(300)]
r50 = [run(0.50) for _ in range(300)]
assert r53.count("better") / 300 > 0.9, r53.count("better") / 300
assert r50.count("equal") / 300 > 0.85 and (r50.count("better") + r50.count("worse")) / 300 < 0.15, (r50.count("equal"), len(r50))

# 3) 장부: 방향 뒤집힌 기록 합산, 풀 균형, void 제외
rows = [{"kind": "game", "a": "x", "b": "y", "winner": "a", "pool": "P", "t": 1}, {"kind": "game", "a": "y", "b": "x", "winner": "a", "pool": "P", "t": 2},
        {"kind": "game", "a": "x", "b": "y", "winner": "void", "pool": "P", "t": 3}, {"kind": "agg", "a": "y", "b": "x", "wins_a": 5, "wins_b": 2, "draws": 1}]
assert pair_counts(rows, "x", "y") == (1 + 2, 1 + 5, 1, 1), pair_counts(rows, "x", "y")
games = [{"winner": "a", "pool": "P", "t": i} for i in range(10)] + [{"winner": "b", "pool": "Q", "t": i} for i in range(4)] + \
        [{"winner": "void", "pool": "Q", "t": 99}]
w, l, per = balanced(games)
assert (w, l) == (4, 4) and per == {"P": (4, 4), "Q": (0, 4)}, (w, l, per)   # 풀마다 앞 4판만

# 4) Bradley–Terry: 알려진 Elo(0, +50, +100)를 되찾음
elo = {"m0": 0.0, "m1": 50.0, "m2": 100.0}
wins = {}
for i, a in enumerate(elo):
    for b in list(elo)[i + 1:]:
        p = 1 / (1 + 10 ** (-(elo[a] - elo[b]) / 400))
        wa = sum(rng.random() < p for _ in range(4000))
        wins[(a, b)], wins[(b, a)] = wa, 4000 - wa
res, _, _ = fit_bt(wins, ref="m0", boot=40)
assert abs(res["m1"][0] - 50) < 15 and abs(res["m2"][0] - 100) < 15, res
assert res["m2"][1] < 100 < res["m2"][2], res["m2"]
print("all passed")

