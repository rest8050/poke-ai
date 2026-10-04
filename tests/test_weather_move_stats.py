"""tensor_encoder의 날씨 의존 기술 보정: 웨더볼 타입(effective_move_type), 물/불타입 위력 1.5배/0.5배 +
웨더볼 위력 2배(effective_move_power), 허리케인/천둥 명중률 필중/50%(effective_move_accuracy).
poke-env가 이런 동적 보정을 안 해줘서(고정 dex값만 줌) 직접 계산해야 했던 실전 버그들."""
from src.core.tensor_encoder import effective_move_accuracy, effective_move_power, effective_move_type


class FakeType:
    def __init__(self, name):
        self.name = name.upper()


class FakeMove:
    def __init__(self, id_, type_):
        self.id = id_
        self.type = FakeType(type_)


class FakeBattle:
    def __init__(self, weather_key=None):
        self.weather = {weather_key: 1} if weather_key else {}


def test_weatherball_power_doubles_in_any_weather():
    assert effective_move_power(FakeBattle("sandstorm"), FakeMove("weatherball", "normal"), 50) == 100.0
    assert effective_move_power(FakeBattle(None), FakeMove("weatherball", "normal"), 50) == 50


def test_water_move_boosted_in_rain_and_weakened_in_sun():
    surf = FakeMove("surf", "water")
    assert effective_move_power(FakeBattle("raindance"), surf, 60) == 90.0
    assert effective_move_power(FakeBattle("sunnyday"), surf, 60) == 30.0
    assert effective_move_power(FakeBattle("sandstorm"), surf, 60) == 60  # 다른 날씨는 영향 없음


def test_fire_move_boosted_in_sun_and_weakened_in_rain():
    flamethrower = FakeMove("flamethrower", "fire")
    assert effective_move_power(FakeBattle("sunnyday"), flamethrower, 90) == 135.0
    assert effective_move_power(FakeBattle("raindance"), flamethrower, 90) == 45.0


def test_unrelated_type_power_untouched():
    eq = FakeMove("earthquake", "ground")
    assert effective_move_power(FakeBattle("raindance"), eq, 100) == 100


def test_hurricane_thunder_accuracy_by_weather():
    for mid in ("hurricane", "thunder"):
        mv = FakeMove(mid, "flying")
        assert effective_move_accuracy(FakeBattle("raindance"), mv, 0.7) == 1.0
        assert effective_move_accuracy(FakeBattle("sunnyday"), mv, 0.7) == 0.5
        assert effective_move_accuracy(FakeBattle("sandstorm"), mv, 0.7) == 0.7
        assert effective_move_accuracy(FakeBattle(None), mv, 0.7) == 0.7


def test_other_moves_accuracy_untouched():
    mv = FakeMove("discharge", "electric")
    assert effective_move_accuracy(FakeBattle("raindance"), mv, 1.0) == 1.0


def test_weatherball_type_still_correct():
    assert effective_move_type(FakeBattle("sandstorm"), FakeMove("weatherball", "normal")) == "rock"
    assert effective_move_type(FakeBattle("raindance"), FakeMove("earthquake", "ground")) == "ground"


if __name__ == "__main__":
    test_weatherball_power_doubles_in_any_weather()
    test_water_move_boosted_in_rain_and_weakened_in_sun()
    test_fire_move_boosted_in_sun_and_weakened_in_rain()
    test_unrelated_type_power_untouched()
    test_hurricane_thunder_accuracy_by_weather()
    test_other_moves_accuracy_untouched()
    test_weatherball_type_still_correct()
    print("ok")
