"""battle_eval self-check: PYTHONPATH=. python tests/test_battle_eval.py"""
import random
from types import SimpleNamespace

from poke_env.battle import Effect, Move, Pokemon, SideCondition, Status

from src.core.rules import SHAPING_SCALE, TERMS, UNSEEN_MON, evaluate, potential, terms


def mon(species, active=False, item="leftovers", moves=(), hp=1.0, ability=None):
    p = Pokemon(gen=9, species=species)
    p._max_hp, p._current_hp = 100, int(100 * hp)
    p._active, p._item = active, item
    if ability:
        p.ability = ability
    for m in moves:
        p._moves[m] = Move(m, gen=9)
    return p


def battle(my, opp, side=None, opp_side=None, used_tera=False, opp_used_tera=False):
    return SimpleNamespace(team={f"p1: {m.species}": m for m in my}, opponent_team={f"p2: {m.species}": m for m in opp},
                           side_conditions=side or {}, opponent_side_conditions=opp_side or {},
                           used_tera=used_tera, opponent_used_tera=opp_used_tera)


def mirror(**kw):
    # 접지: 한카리아스·내룡(독수리)·... 중 비행(철갑무사)과 부유(로토무)는 접지 아님 → 접지 4마리
    return [mon("garchomp", True, **kw)] + [mon(s, **kw) for s in ("toxapex", "corviknight", "clefable", "heatran", "rotomwash")]


base = battle(mirror(), mirror())
assert evaluate(base) == 0 and potential(base) == 0 and set(terms(base)) == set(TERMS)   # 대칭 시작 = 0
assert evaluate(battle(mirror(), mirror()[:3])) == 0                                      # 미공개 상대는 만피 생존 도구 보유로 계산

# 설치 기술은 살아 있는 포켓몬마다: 스텔스록 6마리 x -10 + 압정 2겹은 접지 4마리 x -14 → 상대가 깔려 있으면 내가 +116
assert evaluate(battle(mirror(), mirror(), opp_side={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2})) == 116
assert terms(battle(mirror(), mirror(), opp_side={SideCondition.STEALTH_ROCK: 1}))["hazard"] == 60
# 공개 안 된 상대 3마리도 접지·부츠 없음으로 맞음: 3마리 공개 + 3마리 미공개 → 스텔스록 6 x -10
assert evaluate(battle(mirror(), mirror()[:3], opp_side={SideCondition.STEALTH_ROCK: 1})) == 60
# 기절한 포켓몬은 설치 기술을 안 받음
opp = mirror(); opp[5]._status = Status.FNT
assert terms(battle(mirror(), opp, opp_side={SideCondition.STEALTH_ROCK: 1}))["hazard"] == 50
# 두꺼운부츠와 매직가드는 면제
assert evaluate(battle(mirror(), mirror(item="heavydutyboots"), opp_side={SideCondition.STEALTH_ROCK: 1})) == 0
opp = mirror(); opp[3] = mon("clefable", ability="magicguard")
assert terms(battle(mirror(), opp, opp_side={SideCondition.STEALTH_ROCK: 1}))["hazard"] == 50
# 끈적끈적네트는 접지에만 -25 (4마리)
assert terms(battle(mirror(), mirror(), opp_side={SideCondition.STICKY_WEB: 1}))["hazard"] == 100

assert evaluate(battle(mirror(), mirror(), side={SideCondition.REFLECT: 3})) == 20
assert evaluate(battle(mirror(), mirror(), used_tera=True)) == -75                         # 테라 사용 비용
tb = battle(mirror(), mirror(), used_tera=True)
assert potential(tb, drop=("tera",)) == 0 and abs(potential(tb) + SHAPING_SCALE * 0.75) < 1e-12   # RL에서는 tera 항목을 뺄 수 있음

my = mirror(); my[0]._boosts["atk"] = 2                                                    # 액티브 공격 +2 → 2.0×30
assert evaluate(battle(my, mirror())) == 60
my = mirror(); my[1]._boosts["atk"] = 2                                                    # 벤치 랭크는 무시 (원본과 동일)
assert evaluate(battle(my, mirror())) == 0
my = mirror(); my[0]._effects[Effect.SUBSTITUTE] = 1
assert evaluate(battle(my, mirror())) == 75

# 화상: 물리기 수 x -25 (원본), 특공이 높은 쪽은 절반, 근성/매직스킨/속보는 +50
two = mon("garchomp", moves=["earthquake", "outrage"]); two._status = Status.BRN              # 물리 2 → -50
spec = mon("heatran", moves=["magmastorm"]); spec._status = Status.BRN                         # 물리 0 → 0
guts = mon("ursaluna", moves=["facade"]); guts._status = Status.BRN; guts.ability = "guts"     # +50
assert terms(battle([two] + mirror()[1:], mirror()))["status"] == -50
assert terms(battle([spec] + mirror()[1:], mirror()))["status"] == 0
assert terms(battle([guts] + mirror()[1:], mirror()))["status"] == 50
# 체력이 낮은 동결: 체력+도구+상태가 음수면 0으로 자르고 생존 30만 남음
low = mon("garchomp", hp=0.05); low._status = Status.FRZ
t = terms(battle([low] + mirror()[1:], mirror()))
assert sum(t.values()) == 30 - UNSEEN_MON and t["status"] == -15       # 합 = 생존 30만 남음 (상대 140 대비), 5 + 10 = 15점이 상태로 깎임
dead = mon("garchomp"); dead._status = Status.FNT
assert evaluate(battle([dead] + mirror()[1:], mirror())) == -UNSEEN_MON

# 잠재 함수 보상의 상쇄 (학습 코드와 같은 순서):
# 관측 s_0..s_{T-1}, 스텝 t<T-1 보상 γΦ(s_{t+1})-Φ(s_t), 마지막 스텝 보상 (승패 - Φ(s_{T-1}))
# → 할인 반환 = γ^{T-1}·승패 - Φ(s_0): 중간 국면 점수와 무관 = 최적 정책 불변
GAMMA = 0.995
random.seed(0)
for outcome in (1.0, -1.0):
    phis = [random.uniform(-1, 1) for _ in range(30)]
    rewards = [GAMMA * phis[t + 1] - phis[t] for t in range(len(phis) - 1)] + [outcome - phis[-1]]
    ret = sum(GAMMA ** t * r for t, r in enumerate(rewards))
    assert abs(ret - (GAMMA ** (len(phis) - 1) * outcome - phis[0])) < 1e-9, ret
print(f"battle_eval OK (포켓몬 1마리 가치 = {SHAPING_SCALE * UNSEEN_MON / 100:.2f}, 승패 ±1)")
