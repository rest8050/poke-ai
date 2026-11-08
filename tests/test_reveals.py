"""events.py self-check (가짜 전투 객체로 규칙 하나씩): PYTHONPATH=. python tests/test_reveals.py
공개 (item:/ability:/폴터가이스트), 돌격 부츠 추론, 구애 스카프 추론과 취소, 재생력 추론, 마지막 행동/등장 후 쓴 기술, 합성 휘발성(unburden, 충전, 반동), 수면 나이"""
from types import SimpleNamespace as NS

from poke_env.battle import PokemonType, SideCondition

from src.core.events import RevealTracker


def mon(species, hp=1.0, abilities=(), ability=None, item="unknown_item", status=None, spe=100, types=("NORMAL",)):
    return NS(species=species, current_hp_fraction=hp, possible_abilities=list(abilities), ability=ability, item=item, status=status, level=100,
              boosts={}, effects={}, base_stats={"spe": spe}, stats={"spe": spe}, types=[PokemonType[t] for t in types])


def battle(me, opp, opp_sc=None, turn=1):
    return NS(player_role="p1", active_pokemon=me, opponent_active_pokemon=opp, opponent_side_conditions=opp_sc or {}, side_conditions={}, fields={}, weather={}, turn=turn)


def feed(rev, b, *lines):
    for l in lines:
        rev.on_message(b, l.split("|"))


# 1) 공개: [from] item / ability / 폴터가이스트
r, b = RevealTracker(), battle(mon("garchomp"), mon("heatran"))
feed(r, b, "|-heal|p2a: Heatran|50/100|[from] item: Leftovers", "|-activate|p2a: Heatran|ability: Flash Fire")
assert r.known_item("opp", "heatran") == "leftovers" and r.known_ability("opp", "heatran") == "flashfire"
feed(r, b, "|-activate|p2a: Heatran|move: Poltergeist|Life Orb")
assert r.known_item("opp", "heatran") == "lifeorb"
feed(r, b, "|-damage|p1a: Garchomp|90/100|[from] item: Rocky Helmet|[of] p2a: Heatran")        # [of] 가 주인
assert r.known_item("opp", "heatran") == "rockyhelmet" and r.known_item("me", "garchomp") is None
feed(r, b, "|-enditem|p2a: Heatran|Rocky Helmet|[from] move: Knock Off")                        # 사라진 도구는 None
assert r.known_item("opp", "heatran") is None

# 2) 돌격 부츠: 스텔스록이 있는데 교체 후 피해가 없으면 추론, 피해가 있으면 불가능
SR = {SideCondition.STEALTH_ROCK: 1}
r, b = RevealTracker(), battle(mon("garchomp"), mon("rillaboom"), SR)
feed(r, b, "|switch|p2a: Rillaboom|Rillaboom, M|100/100", "|upkeep")
assert r.known_item("opp", "rillaboom") == "heavydutyboots" and r.inferred["opp"]["rillaboom"]
r, b = RevealTracker(), battle(mon("garchomp"), mon("rillaboom"), SR)
feed(r, b, "|switch|p2a: Rillaboom|Rillaboom, M|100/100", "|-damage|p2a: Rillaboom|88/100|[from] Stealth Rock", "|upkeep")
assert r.known_item("opp", "rillaboom") is None and "rillaboom" in r.impossible_boots

# 3) 구애 스카프: 같은 우선도, 상대가 먼저 움직였는데 최대 속도 가정으로도 내(spe 300)가 더 빠르면 스카프. 서로 다른 기술을 쓰면 취소
slow = mon("slowpoke", spe=15)
r, b = RevealTracker(), battle(mon("garchomp", spe=300), slow)
feed(r, b, "|move|p2a: Slowpoke|Tackle|p1a: Garchomp", "|move|p1a: Garchomp|Tackle|p2a: Slowpoke", "|turn|2")
assert r.known_item("opp", "slowpoke") == "choicescarf"
feed(r, b, "|move|p2a: Slowpoke|Surf|p1a: Garchomp")
assert r.known_item("opp", "slowpoke") is None and r.can_choice["slowpoke"] is False
r, b = RevealTracker(), battle(mon("garchomp", spe=100), mon("slowpoke", spe=100))                # 내가 더 빠르지 않으면 추론 안 함
feed(r, b, "|move|p2a: Slowpoke|Tackle|p1a: Garchomp", "|move|p1a: Garchomp|Tackle|p2a: Slowpoke", "|turn|2")
assert r.known_item("opp", "slowpoke") is None

# 4) 재생력: 나갔다 들어왔더니 HP가 회복돼 있고 가능한 특성에 재생력이 있으면
reg = mon("slowking", hp=0.5, abilities=["Oblivious", "Regenerator"])
r, b = RevealTracker(), battle(mon("garchomp"), reg)
feed(r, b, "|-damage|p2a: Slowking|50/100")                       # HP 비율 0.5를 봄
reg.current_hp_fraction = 0.83
feed(r, b, "|switch|p2a: Slowking|Slowking, M|83/100")
assert r.known_ability("opp", "slowking") == "regenerator"

# 5) 마지막 행동 / 등장 후 쓴 기술 / 합성 효과
r, b = RevealTracker(), battle(mon("garchomp"), mon("kingambit"))
feed(r, b, "|switch|p1a: Garchomp|Garchomp, M|100/100", "|move|p1a: Garchomp|Earthquake|p2a: Kingambit", "|move|p1a: Garchomp|Outrage|p2a: Kingambit")
assert r.last_action["me"] == ("move", "outrage") and r.used["me"]["garchomp"] == {"earthquake", "outrage"}
assert "lockedmove" in r.synth["me"]
feed(r, b, "|move|p1a: Garchomp|Earthquake|p2a: Kingambit", "|move|p1a: Garchomp|Sleep Talk|p1a: Garchomp|[from]Sleep Talk")
assert "lockedmove" not in r.synth["me"] and r.last_action["me"] == ("move", "earthquake")      # [from] 기술은 기록 안 함
feed(r, b, "|switch|p1a: Garchomp|Garchomp, M|100/100")
assert r.last_action["me"] == ("switch", "garchomp") and r.used["me"]["garchomp"] == set()
feed(r, b, "|-prepare|p1a: Garchomp|Solar Beam|p2a: Kingambit")
assert "solarbeam" in r.synth["me"]
feed(r, b, "|move|p1a: Garchomp|Solar Beam|p2a: Kingambit")
assert "solarbeam" not in r.synth["me"]
feed(r, b, "|-mustrecharge|p1a: Garchomp")
assert "mustrecharge" in r.synth["me"]
feed(r, b, "|cant|p1a: Garchomp|recharge")
assert "mustrecharge" not in r.synth["me"]

# 6) unburden: 도구를 잃으면 가능한 특성에 unburden이 있는 포켓몬에 효과
hawk = mon("hawlucha", abilities=["Limber", "Unburden", "Mold Breaker"])
r, b = RevealTracker(), battle(mon("garchomp"), hawk)
feed(r, b, "|-enditem|p2a: Hawlucha|Sitrus Berry|[eat]")
assert "unburden" in r.synth["opp"]

# 7) 수면/맹독 나이: 걸린 턴을 0으로 세고 매 턴 늘어남, 맹독은 교체하면 처음부터
r, b = RevealTracker(), battle(mon("garchomp"), mon("kingambit"))
feed(r, b, "|-status|p2a: Kingambit|slp")
assert r.status_age("opp", "kingambit", "slp") == 0
feed(r, b, "|turn|2", "|turn|3")
assert r.status_age("opp", "kingambit", "slp") == 2 and r.status_age("opp", "kingambit", "tox") is None
feed(r, b, "|-curestatus|p2a: Kingambit|slp")
assert r.status_age("opp", "kingambit", "slp") is None

# 8) 소원과 미래예지
r, b = RevealTracker(), battle(mon("garchomp"), mon("clefable"))
feed(r, b, "|move|p2a: Clefable|Wish|p2a: Clefable")
assert r.wish["opp"] == 2
feed(r, b, "|upkeep", "|upkeep")
assert r.wish["opp"] == 0
feed(r, b, "|-start|p1a: Garchomp|move: Future Sight")
assert r.future["me"] == 3
print("reveals OK")
