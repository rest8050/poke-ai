"""도구 배율 한 번만 적용 self-check: PYTHONPATH=. python tests/test_item_once.py  (인코더는 능력치에 구애/조끼 등을 곱하지 않고 matchup_v7이 곱함)"""
from src.core.tensor_encoder import BattleTensorEncoder, DummyBattle

E = BattleTensorEncoder(vocab_path="data/vocab.json")
def spe(item):
    b = DummyBattle(); p = b.active_pokemon; p.item = item; p.base_stats = dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 100)
    return float(E.encode_battle(b)[1][0, 0, 16]) * 255
assert abs(spe("choicescarf") - spe("leftovers")) < 1e-3, "인코더가 스카프를 미리 곱함 (matchup_v7과 이중 적용)"
print("item once OK")
