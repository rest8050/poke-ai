"""battle_eval self-check: python src/core/test_battle_eval.py"""
import random
from types import SimpleNamespace

from poke_env.battle import Effect, Move, Pokemon, SideCondition, Status

from src.core.battle_eval import SHAPING_SCALE, UNSEEN_MON, evaluate, pokemon_score, potential


def mon(species, active=False, item="leftovers", moves=(), hp=1.0):
    p = Pokemon(gen=9, species=species)
    p._max_hp, p._current_hp = 100, int(100 * hp)
    p._active, p._item = active, item
    for m in moves:
        p._moves[m] = Move(m, gen=9)
    return p


def battle(my, opp, side=None, opp_side=None, used_tera=False, opp_used_tera=False):
    return SimpleNamespace(team={f"p1: {m.species}": m for m in my}, opponent_team={f"p2: {m.species}": m for m in opp},
                           side_conditions=side or {}, opponent_side_conditions=opp_side or {},
                           used_tera=used_tera, opponent_used_tera=opp_used_tera)


def mirror():
    return [mon("garchomp", True)] + [mon(s) for s in ("toxapex", "corviknight", "clefable", "heatran", "rotomwash")]


base = battle(mirror(), mirror())
assert evaluate(base) == 0 and potential(base) == 0                        # 대칭 시작 = 0
assert pokemon_score(mon("garchomp"), False) == UNSEEN_MON                 # 만피 생존 도구 = 140
assert evaluate(battle(mirror(), mirror()[:3])) == 0                       # 미공개 상대는 만피로 계산

# 상대 진영 스텔스록 + 압정 2겹 → 내가 유리 (상대 점수 -10 -14)
assert evaluate(battle(mirror(), mirror(), opp_side={SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2})) == 24
assert evaluate(battle(mirror(), mirror(), side={SideCondition.REFLECT: 3})) == 20
assert evaluate(battle(mirror(), mirror(), used_tera=True)) == -75         # 테라 사용 비용

my = mirror(); my[0]._boosts["atk"] = 2                                    # 액티브 공격 +2 → 2.0×30
assert evaluate(battle(my, mirror())) == 60
my = mirror(); my[1]._boosts["atk"] = 2                                    # 벤치 랭크는 무시 (원본과 동일)
assert evaluate(battle(my, mirror())) == 0
my = mirror(); my[0]._effects[Effect.SUBSTITUTE] = 1
assert evaluate(battle(my, mirror())) == 75

phys = mon("garchomp", moves=["earthquake"]); phys._status = Status.BRN
spec = mon("heatran", moves=["magmastorm"]); spec._status = Status.BRN
guts = mon("ursaluna", moves=["facade"]); guts._status = Status.BRN; guts.ability = "guts"
assert pokemon_score(phys, False) == UNSEEN_MON - 25 and pokemon_score(spec, False) == UNSEEN_MON - 10
assert pokemon_score(guts, False) == UNSEEN_MON + 10
dead = mon("garchomp"); dead._status = Status.FNT
assert pokemon_score(dead, False) == 0
assert pokemon_score(mon("garchomp", hp=0.5, item=""), False) == 50 + 30

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
