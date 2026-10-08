"""기준 손계산 + 사건 평가 self-check (표준 라이브러리만): PYTHONPATH=. python tests/test_damage_calc.py
능력치 공식(실제 게임 값), Matchup과 같은 피해 근사(자속/상성/면역/화상/테라/날씨/도구 배율), 진단의 사건 평가(피해 비율, KO, 제외 사유)"""
import math

from src.core.damage_calc import calc_stats, damage, item_stat_mult, stage_mult, standard_stats, supported
from tools.damage_residual_report import evaluate

CHART = {"ground": {"fire": 2.0, "flying": 0.0, "steel": 2.0, "grass": 0.5}, "water": {"fire": 2.0, "grass": 0.5}}
GARCHOMP = {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102}


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def mon(atk=300, df=200, types=("ground",), **kw):
    return {"stats": {"hp": 300, "atk": atk, "def": df, "spa": atk, "spd": df, "spe": 100}, "types": list(types), **kw}


def test_stats():
    s = calc_stats(GARCHOMP, [0, 252, 4, 0, 0, 252], [31] * 6, 100, "jolly")
    assert s == {"hp": 357, "atk": 359, "def": 227, "spa": 176, "spd": 206, "spe": 333}, s   # 실제 게임의 최속 한카리아스
    assert standard_stats(GARCHOMP)["atk"] == 359 and standard_stats(GARCHOMP)["hp"] == 420
    assert item_stat_mult({"atk": 100, "def": 1, "spa": 1, "spd": 1, "spe": 1}, "choiceband")["atk"] == 150
    assert stage_mult(2) == 2.0 and close(stage_mult(-1), 2 / 3)


def test_damage():
    eq = {"type": "Ground", "category": "Physical", "basePower": 100}
    base = 42 * 100 * 300 / 200 / 50 + 2                                    # 128
    r = damage(mon(), mon(types=("fire",)), eq, CHART)
    assert close(r["max"], base * 1.5 * 2) and close(r["mean"], base * 1.5 * 2 * 0.925) and r["eff"] == 2.0
    assert damage(mon(), mon(types=("flying",)), eq, CHART)["eff"] == 0.0                    # 타입 무효
    assert damage(mon(), mon(types=("fire",), ability="levitate"), eq, CHART)["eff"] == 0.0  # 특성 무효
    assert damage(mon(), mon(types=("fire",), item="airballoon"), eq, CHART)["eff"] == 0.0   # 풍선
    burned = damage(mon(status="brn"), mon(types=("fire",)), eq, CHART)
    assert close(burned["max"], r["max"] / 2)
    assert close(damage(mon(types=("water",), tera="ground"), mon(types=("fire",)), eq, CHART)["max"], r["max"])  # 테라 자속
    assert damage(mon(), mon(types=("fire",), tera="grass"), eq, CHART)["eff"] == 0.5        # 방어 테라는 테라 타입 하나
    boosted = damage(mon(boosts={"atk": 2}), mon(types=("fire",)), eq, CHART)
    assert close(boosted["max"], (42 * 100 * 600 / 200 / 50 + 2) * 3)
    surf = {"type": "Water", "category": "Special", "basePower": 90}
    dry = damage(mon(types=("water",)), mon(types=("fire",)), surf, CHART)
    rain = damage(mon(types=("water",)), mon(types=("fire",)), surf, CHART, weather="raindance")
    assert close((rain["max"] / 3 - 2) / (dry["max"] / 3 - 2), 1.5)                           # 비: 물 위력 1.5배
    assert not supported({"category": "Status"}) and not supported({"id": "foulplay", "category": "Physical", "basePower": 95})
    assert supported({"id": "knockoff", "category": "Physical", "basePower": 65}) and not supported({"category": "Special", "basePower": 0})


SPECIES = {"attackmon": {"baseStats": dict.fromkeys(GARCHOMP, 100), "types": ["Ground"]},
           "defmon": {"baseStats": dict.fromkeys(GARCHOMP, 100), "types": ["Normal"]}}
MOVES = {"earthquake": {"type": "Ground", "category": "Physical", "basePower": 100},
         "swordsdance": {"type": "Normal", "category": "Status", "basePower": 0}}
HEADER = {"sets": {"p1": [{"sp": "attackmon", "evs": [0, 252, 0, 0, 0, 0], "ivs": [31] * 6, "level": 100, "nature": "serious"}],
                   "p2": [{"sp": "defmon", "evs": [0] * 6, "ivs": [31] * 6, "level": 100, "nature": "serious"}]}}


def event(out, **t):
    return {"type": "move", "move": "earthquake", "via": None, "us": "p1", "ts": "p2", "field": {}, "sc": {}, "fnt": {}, "out": out,
            "u": {"sp": "attackmon", "hp": 1.0, "b": {}, "st": None, "tera": None, "item": None, "ab": None},
            "t": {"sp": "defmon", "hp": 1.0, "b": {}, "st": None, "tera": None, "item": None, "ab": None, **t}}


def test_evaluate():
    # 공격 299(252 노력치), 방어 236, HP 341, 자속 1.5, 등배 → 평균 피해 비율
    pred = (42 * 100 * 299 / 236 / 50 + 2) * 1.5 * 0.925 / 341
    r = evaluate(event({"dmg": 0.5}), HEADER, SPECIES, MOVES, CHART)
    assert r["kind"] == "dmg" and close(r["pred"], pred) and close(r["r"], math.log(0.5 / pred))
    assert evaluate(event({"dmg": 0.5, "hits": 2}), HEADER, SPECIES, MOVES, CHART)["pred"] > 2 * pred - 1e-9
    assert evaluate(event({"dmg": 0.4, "ko": True}, hp=0.4), HEADER, SPECIES, MOVES, CHART)["pred_ko"] is True
    assert evaluate(event({"dmg": 0.9, "ko": True}, hp=0.9), HEADER, SPECIES, MOVES, CHART)["pred_ko"] is False
    assert evaluate(event({"crit": True, "dmg": 0.6}), HEADER, SPECIES, MOVES, CHART)["kind"] == "crit"
    assert evaluate(event({"dmg": 0.0, "immune": True}), HEADER, SPECIES, MOVES, CHART)["kind"] == "immune_extra"
    assert evaluate(dict(event({}), move="swordsdance"), HEADER, SPECIES, MOVES, CHART)["kind"] == "status"
    assert evaluate(dict(event({"dmg": 0.5}), via="ability:magicbounce"), HEADER, SPECIES, MOVES, CHART)["kind"] == "via"


if __name__ == "__main__":
    test_stats()
    test_damage()
    test_evaluate()
    print("ok")
