"""bridge_server.avoid_bad_weatherball: 날씨가 없거나(위력 반토막) 지금 날씨가 만드는 타입이 상대한테
반감(0.5배) 이하면 웨더볼 대신 다음 합법 행동으로 대체하는지 확인. 실전 재현 사례(비→모래로 덮임, 하마돈 상대
바위타입 웨더볼) 기반."""
from src.core.bridge_server import avoid_bad_weatherball
import torch


class FakeMove:
    def __init__(self, id_):
        self.id = id_


class FakePokemon:
    """rct_player.types()/ability()가 읽는 poke-env Pokemon 속성만 최소로 흉내"""
    def __init__(self, type_1, type_2=None):
        self.type_1 = type_1
        self.type_2 = type_2
        self.ability = None
        self.possible_abilities = []
        self.item = None


class FakeBattle:
    def __init__(self, weather_key, type_1, type_2=None):
        self.weather = {weather_key: 1} if weather_key else {}
        self.opponent_active_pokemon = FakePokemon(type_1, type_2)


def decoded_of(kind, id_=None):
    return (kind, FakeMove(id_)) if kind == "move" else (kind, object())


def decode_fn_factory(order):
    return lambda idx: decoded_of(*order[idx])


def test_sandstorm_vs_rock_ground_replaced():
    # 모래바람 → 웨더볼 바위타입, 상대(하마돈: 땅/바위)한테 반감 → 대체돼야 함
    battle = FakeBattle("sandstorm", "ground", "rock")
    probs = torch.tensor([[0.6, 0.4]])
    mask = torch.tensor([[True, True]])
    order = {0: ("move", "weatherball"), 1: ("move", "earthquake")}
    idx, decoded = avoid_bad_weatherball(0, probs, mask, decoded_of("move", "weatherball"),
                                          decode_fn_factory(order), battle)
    assert idx == 1 and decoded[1].id == "earthquake"


def test_rain_vs_water_type_target_kept():
    # 비 상태 → 웨더볼 물타입, 상대(불꽃)한테 효과적 → 그대로 둬야 함
    battle = FakeBattle("raindance", "fire")
    probs = torch.tensor([[0.7, 0.3]])
    mask = torch.tensor([[True, True]])
    order = {0: ("move", "weatherball"), 1: ("move", "hurricane")}
    idx, decoded = avoid_bad_weatherball(0, probs, mask, decoded_of("move", "weatherball"),
                                          decode_fn_factory(order), battle)
    assert idx == 0 and decoded[1].id == "weatherball"


def test_no_weather_always_replaced():
    battle = FakeBattle(None, "normal")
    probs = torch.tensor([[0.6, 0.4]])
    mask = torch.tensor([[True, True]])
    order = {0: ("move", "weatherball"), 1: ("move", "hurricane")}
    idx, decoded = avoid_bad_weatherball(0, probs, mask, decoded_of("move", "weatherball"),
                                          decode_fn_factory(order), battle)
    assert idx == 1


def test_non_weatherball_choice_untouched():
    battle = FakeBattle("sandstorm", "ground", "rock")
    probs = torch.tensor([[0.6, 0.4]])
    mask = torch.tensor([[True, True]])
    order = {0: ("move", "earthquake"), 1: ("move", "weatherball")}
    idx, decoded = avoid_bad_weatherball(0, probs, mask, decoded_of("move", "earthquake"),
                                          decode_fn_factory(order), battle)
    assert idx == 0 and decoded[1].id == "earthquake"


if __name__ == "__main__":
    test_sandstorm_vs_rock_ground_replaced()
    test_rain_vs_water_type_target_kept()
    test_no_weather_always_replaced()
    test_non_weatherball_choice_untouched()
    print("ok")
