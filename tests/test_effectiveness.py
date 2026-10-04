"""move_effectiveness 대표 사례 self-check: python src/core/test_effectiveness.py"""
from types import SimpleNamespace

from poke_env.battle import Move, Pokemon

from src.core.tensor_encoder import defender_ability, move_effectiveness


def mon(species, item=None, ability=None, tera=None):
    p = Pokemon(gen=9, species=species)
    if item:
        p._item = item
    if ability:
        p.ability = ability
    if tera:
        p.terastallize(tera)
    return p


def eff(move_id, defender):
    battle = SimpleNamespace(fields={}, weather={})
    return move_effectiveness(battle, Move(move_id, gen=9), defender, defender_ability(defender)) * 4


assert eff("icebeam", mon("garchomp")) == 4                           # 타입 4배
assert eff("dragonclaw", mon("sylveon")) == 0                         # 타입 무효
assert eff("earthquake", mon("rotomwash")) == 0                       # 부유 (특성 후보 1개)
assert eff("earthquake", mon("gholdengo")) == 2                       # 강철 2배 × 고스트 1배
assert eff("earthquake", mon("gholdengo", item="airballoon")) == 0    # 풍선
assert eff("surf", mon("gastrodon")) == 1                             # 특성 미공개(후보 여러 개) → 추정 없이 타입 상성만
assert eff("surf", mon("gastrodon", ability="stickyhold")) == 1       # 공개된 특성이 추론보다 우선
assert eff("dragonclaw", mon("garchomp", tera="fairy")) == 0          # 테라 후 타입
assert eff("swordsdance", mon("garchomp")) == 1                       # 변화기 = 등배 처리
print("effectiveness OK")
