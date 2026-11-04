"""행동 사건 추출 self-check (표준 라이브러리만): PYTHONPATH=. python tests/test_battle_events.py
위협(동시 등장 후 발동), 울퉁불퉁멧 반동, 잔여 회복 미부착, 스텔스록 등장 피해, 매직미러 반사, split 비공개 줄, 심술꾸러기 랭크, 급소 KO"""
from src.training.battle_events import _hp, _tag, parse_battle

TEAMS = {
    "alice": [
        {"species": "Garchomp", "item": "Leftovers", "ability": "Rough Skin", "moves": ["Earthquake", "Stealth Rock"], "tera_type": "Steel",
         "nature": "Jolly", "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252}, "level": 100},
        {"species": "Serperior", "item": "Leftovers", "ability": "Contrary", "moves": ["Leaf Storm"], "nature": "Timid",
         "evs": {"hp": 4, "spa": 252, "spe": 252}},
    ],
    "bob": [
        {"species": "Gyarados", "item": "Heavy-Duty Boots", "ability": "Intimidate", "moves": ["Waterfall"], "nature": "Adamant",
         "evs": {"hp": 252, "atk": 252, "spe": 4}},
        {"species": "Hatterene", "item": "Leftovers", "ability": "Magic Bounce", "moves": ["Psychic"], "nature": "Quiet",
         "evs": {"hp": 252, "spa": 252}},
    ],
}
LINES = [
    "|player|p1|alice|1", "|player|p2|bob|2", "|", "|t:|100",
    "|switch|p1a: Garchomp|Garchomp, L100, M|357/357",
    "|switch|p2a: Gyarados|Gyarados, L100, F|100/100",
    "|-ability|p2a: Gyarados|Intimidate|boost",
    "|-unboost|p1a: Garchomp|atk|1",
    "|turn|1", "|", "|t:|101",
    "|move|p1a: Garchomp|Stealth Rock|p2a: Gyarados",
    "|-sidestart|p2: bob|move: Stealth Rock",
    "|move|p2a: Gyarados|Waterfall|p1a: Garchomp",
    "|-damage|p1a: Garchomp|200/357",
    "|-damage|p2a: Gyarados|90/100|[from] ability: Rough Skin|[of] p1a: Garchomp",
    "|", "|-heal|p1a: Garchomp|222/357|[from] item: Leftovers", "|upkeep",
    "|turn|2", "|", "|t:|102",
    "|switch|p2a: Hatterene|Hatterene, L100, F|100/100",
    "|-damage|p2a: Hatterene|88/100|[from] Stealth Rock",
    "|move|p1a: Garchomp|Stealth Rock|p2a: Hatterene",
    "|move|p2a: Hatterene|Stealth Rock|p1a: Garchomp|[from]ability: Magic Bounce",
    "|-sidestart|p1: alice|move: Stealth Rock",
    "|", "|upkeep", "|turn|3", "|",
    "|switch|p1a: Serperior|Serperior, L100, M|292/292",
    "|-damage|p1a: Serperior|256/292|[from] Stealth Rock",
    "|move|p2a: Hatterene|Psychic|p1a: Serperior",
    "|split|p1", "|-damage|p1a: Serperior|200/292", "|-damage|p1a: Serperior|50/100",
    "|move|p1a: Serperior|Leaf Storm|p2a: Hatterene",
    "|-crit|p2a: Hatterene", "|-damage|p2a: Hatterene|0 fnt", "|-boost|p1a: Serperior|spa|2",
    "|faint|p2a: Hatterene",
]


def close(a, b, tol=1e-3):
    return abs(a - b) < tol


def test_helpers():
    assert _hp("254/357 par") == (254 / 357, "par") and _hp("0 fnt") == (0.0, "fnt")
    assert _tag(["[from] item: Life Orb"]) == ("item", "lifeorb") and _tag(["[from] Stealth Rock"]) == ("", "stealthrock")
    assert _tag(["[from]ability: Magic Bounce"]) == ("ability", "magicbounce") and _tag(["[still]"]) is None


def test_events():
    header, evs = parse_battle(LINES, TEAMS)
    assert header["users"] == {"p1": "alice", "p2": "bob"} and header["known"] == {"p1": True, "p2": True}
    garchomp = next(s for s in header["sets"]["p1"] if s["sp"] == "garchomp")
    assert garchomp["evs"] == [0, 252, 4, 0, 0, 252] and garchomp["nature"] == "jolly" and garchomp["ab"] == "roughskin"
    kinds = [(e["type"], e.get("move") or e["mon"]["sp"]) for e in evs]
    assert kinds == [("switch", "garchomp"), ("switch", "gyarados"), ("move", "stealthrock"), ("move", "waterfall"),
                     ("switch", "hatterene"), ("move", "stealthrock"), ("move", "stealthrock"), ("switch", "serperior"),
                     ("move", "psychic"), ("move", "leafstorm")], kinds

    # 위협: 가랴도스 등장 사건이 상대(한카리아스) 공격 -1을 받음, 한카리아스 등장 사건에는 안 붙음
    assert evs[1]["out"]["opp_boosts"] == {"atk": -1} and ["p2", "intimidate"] in evs[1]["out"]["abilities"]
    assert "opp_boosts" not in evs[0]["out"]
    # 폭포오르기: 대상(한카리아스) 직전 랭크 -1, 피해 비율, 사용자 울퉁불퉁멧 반동, 잔여 회복(먹다남은음식)은 안 붙음
    wf = evs[3]
    assert wf["turn"] == 1 and wf["ord"] == 1 and wf["t"]["b"] == {"atk": -1} and wf["t"]["hp"] == 1.0
    assert close(wf["out"]["dmg"], 1 - 200 / 357) and close(wf["out"]["user_dmg"], 0.1)
    assert ["p1", "roughskin"] in wf["out"]["abilities"] and "user_heal" not in wf["out"]
    # 스텔스록 등장 피해는 그 등장 사건으로
    assert close(evs[4]["out"]["hazard"], 0.12) and evs[4]["sc"]["p2"] == {"stealthrock": 1}
    # 매직미러: 원래 기술은 reflected, 되돌린 기술은 via로 따로 (선택 순서 ord는 안 늘어남)
    assert evs[5]["out"]["reflected"] is True
    assert evs[6]["via"] == "ability:magicbounce" and evs[6]["out"]["side_start"] == [["p1", "stealthrock"]] and evs[6]["ord"] == 1
    assert close(evs[7]["out"]["hazard"], 36 / 292) and evs[7]["sc"]["p1"] == {"stealthrock": 1}
    # split: 비공개 줄(200/292)만 씀 — 공개 줄(50/100)을 썼다면 0.38
    assert close(evs[8]["out"]["dmg"], 56 / 292)
    # 리프스톰: 급소, KO, 심술꾸러기로 특공 +2가 사용자 랭크 변화로
    ls = evs[9]["out"]
    assert ls["crit"] and ls["ko"] and ls["boosts"] == {"u": {"spa": 2}} and ls["hp_after"] == 0.0


if __name__ == "__main__":
    test_helpers()
    test_events()
    print("ok")
