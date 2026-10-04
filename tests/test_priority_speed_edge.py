"""priority_speed_edge 검증: 우선도 등급이 다르면 스피드 무관하게 등급으로 결정, 같으면 종족값 스피드로, 트릭룸이면 반전.
실행: python -m tests.test_priority_speed_edge"""
from types import SimpleNamespace

from src.core.tensor_encoder import priority_speed_edge


def mon(spe, moves_priority=()):
    return SimpleNamespace(base_stats={"spe": spe}, moves={f"m{i}": SimpleNamespace(priority=p) for i, p in enumerate(moves_priority)})


def move(priority):
    return SimpleNamespace(priority=priority)


battle_no_tr = SimpleNamespace(fields={})
battle_tr = SimpleNamespace(fields={"trickroom": 5})

# 상대가 우선기를 하나도 안 드러냄(=0으로 취급) → 내 기술이 우선도 1이면 무조건 먼저
assert priority_speed_edge(battle_no_tr, move(1), mon(50), mon(200)) == 1.0
# 상대가 더 높은 우선도 기술을 드러냄 → 내 스피드가 훨씬 빨라도 무조건 나중 (등급이 스피드를 이김)
assert priority_speed_edge(battle_no_tr, move(1), mon(300), mon(50, moves_priority=[2])) == -1.0
# 등급이 같으면(둘 다 0) 종족값 스피드로 비교
assert priority_speed_edge(battle_no_tr, move(0), mon(100), mon(80, moves_priority=[0])) == 1.0
assert priority_speed_edge(battle_no_tr, move(0), mon(80), mon(100, moves_priority=[0])) == -1.0
# 트릭룸이면 등급 동률일 때 스피드 비교가 반전
assert priority_speed_edge(battle_tr, move(0), mon(100), mon(80, moves_priority=[0])) == -1.0
# 등급이 다르면 트릭룸도 무관 (등급이 항상 우선)
assert priority_speed_edge(battle_tr, move(1), mon(50), mon(200)) == 1.0
# 스피드 동률(등급도 동률) → 0(불명)
assert priority_speed_edge(battle_no_tr, move(0), mon(100), mon(100, moves_priority=[0])) == 0.0
# 상대 정보 없음 → 0
assert priority_speed_edge(battle_no_tr, move(1), mon(100), None) == 0.0
print("priority_speed_edge OK")
