"""bridge_server.avoid_redundant_trickroom: 필드에 트릭룸이 이미 켜져 있을 때 모델이 또 트릭룸을 고르면
다음 합법 행동으로 대체하는지 확인. 실전 버그(교사도 100ms에선 가끔 틀림, 300ms에선 안 그럼) 재현 사례 기반."""
import torch

from src.core.bridge_server import avoid_redundant_trickroom

MOVE, SWITCH = "move", "switch"


class FakeMove:
    def __init__(self, id_):
        self.id = id_


def decoded_of(kind, id_=None):
    return (kind, FakeMove(id_)) if kind == MOVE else (kind, object())


def decode_fn(order):
    """idx -> decoded. order[idx]가 (kind, id) 튜플."""
    return lambda idx: decoded_of(*order[idx])


def test_replaces_trickroom_when_already_active():
    # idx0 = trickroom(모델이 고른 것, 1등), idx1 = trickroom-tera(역시 트릭룸이라 제외), idx2 = 다른 기술(정답)
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    mask = torch.tensor([[True, True, True]])
    order = {0: (MOVE, "trickroom"), 1: (MOVE, "trickroom"), 2: (MOVE, "lightscreen")}
    idx, decoded = avoid_redundant_trickroom(0, probs, mask, decoded_of(MOVE, "trickroom"), decode_fn(order), True)
    assert idx == 2 and decoded[1].id == "lightscreen"


def test_skips_illegal_alternatives():
    probs = torch.tensor([[0.6, 0.3, 0.1]])
    mask = torch.tensor([[True, False, True]])  # idx1은 불법
    order = {0: (MOVE, "trickroom"), 1: (MOVE, "icebeam"), 2: (SWITCH,)}
    idx, decoded = avoid_redundant_trickroom(0, probs, mask, decoded_of(MOVE, "trickroom"), decode_fn(order), True)
    assert idx == 2 and decoded[0] == SWITCH


def test_leaves_non_trickroom_choice_alone():
    probs = torch.tensor([[0.7, 0.3]])
    mask = torch.tensor([[True, True]])
    order = {0: (MOVE, "earthquake"), 1: (MOVE, "trickroom")}
    idx, decoded = avoid_redundant_trickroom(0, probs, mask, decoded_of(MOVE, "earthquake"), decode_fn(order), True)
    assert idx == 0 and decoded[1].id == "earthquake"  # 트릭룸을 고른 게 아니면 건드리지 않음


def test_leaves_trickroom_alone_when_field_not_active():
    probs = torch.tensor([[0.5, 0.5]])
    mask = torch.tensor([[True, True]])
    order = {0: (MOVE, "trickroom"), 1: (MOVE, "icebeam")}
    idx, decoded = avoid_redundant_trickroom(0, probs, mask, decoded_of(MOVE, "trickroom"), decode_fn(order), False)
    assert idx == 0 and decoded[1].id == "trickroom"  # 필드에 트릭룸이 없으면 정상적으로 써야 함


if __name__ == "__main__":
    test_replaces_trickroom_when_already_active()
    test_skips_illegal_alternatives()
    test_leaves_non_trickroom_choice_alone()
    test_leaves_trickroom_alone_when_field_not_active()
    print("ok")
