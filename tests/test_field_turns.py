"""지속 턴 self-check: PYTHONPATH=. python tests/test_field_turns.py  (지형은 'terrain' 안의 'rain' 때문에 날씨 분기로 새면 안 됨: 젖은바위가 지형을 8턴으로 만들던 버그)"""
from src.core.tensor_encoder import get_dynamic_turns

P = type("P", (), {"__init__": lambda s, i: setattr(s, "item", i)})
B = lambda *items: type("B", (), {"team": {str(i): P(x) for i, x in enumerate(items)}, "opponent_team": {}})()
left = lambda name, t, b: round(get_dynamic_turns(10, name, t, b)[1], 2)
assert left("electricterrain", 12, B("damprock")) == left("electricterrain", 12, B()) == 0.6          # 5턴, 젖은바위 무관
assert left("electricterrain", 12, B("terrainextender")) == 0.75                                      # 8턴
assert left("raindance", 12, B("damprock")) == 0.75                                                   # 날씨는 그대로 8턴
print("field turns OK")
