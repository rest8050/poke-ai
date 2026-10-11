"""순수 합법 마스크 self-check: PYTHONPATH=. python tests/test_legal_mask.py
1) 서버 요청 -> 22칸 마스크 (정상 / 구애 고정 / 기술 한 개만 남는 고정 / 교체 봉쇄 / PP 소진 / 강제 교체 / 폼 이름 접두사), 2) 인코더가 순수 합법 마스크(legal_mask)와 휴리스틱 가지치기 마스크(action_mask)를 따로 주고 후자가 전자의 부분집합"""
import numpy as np

from src.core.tensor_encoder import BattleTensorEncoder, DummyBattle
from src.training.replay_dataset import legal_from_request

SIDE = [{"details": "Great Tusk", "condition": "300/300", "active": True}, {"details": "Hoopa-Unbound", "condition": "200/200", "active": False},
        {"details": "Slowking, F", "condition": "0 fnt", "active": False}, {"details": "Ogerpon-Teal", "condition": "250/250", "active": False}]
TEAM = ["greattusk", "hoopaunbound", "slowking", "ogerpontealtera"]
MOVES = ["closecombat", "knockoff", "stealthrock", "earthquake"]


def req(moves, tera="Ground", trapped=False):
    act = {"moves": [{"id": m, "disabled": d, "pp": pp} for m, d, pp in moves]}
    if tera:
        act["canTerastallize"] = tera
    if trapped:
        act["trapped"] = True
    return {"active": [act], "side": {"pokemon": SIDE}}


bits = lambda m: "".join("1" if x else "0" for x in m[:14])
free = [(m, False, 8) for m in MOVES]
# 1) 정상: 기술 4개 + 테라, 교체는 생존한 비활성 포켓몬만 (3번 슬롯 기절, 4번 슬롯은 폼 이름이 달라도 접두사로 매칭)
assert bits(legal_from_request(req(free), MOVES, TEAM, "turn")) == "11111111" + "01" + "0" + "1" + "00", bits(legal_from_request(req(free), MOVES, TEAM, "turn"))
# 2) 구애 고정: 지진만 가능, 나머지는 disabled
locked = [(m, m != "earthquake", 8) for m in MOVES]
assert bits(legal_from_request(req(locked), MOVES, TEAM, "turn"))[:8] == "00010001"
# 3) 기술이 한 개만 목록에 남는 고정(난동 등): 목록에 없는 기술은 불법
assert bits(legal_from_request(req([("closecombat", False, 8)]), MOVES, TEAM, "turn"))[:8] == "10001000"
# 4) 교체 봉쇄 / 테라 불가 / PP 소진
assert bits(legal_from_request(req(free, trapped=True), MOVES, TEAM, "turn"))[8:] == "000000"
assert bits(legal_from_request(req(free, tera=None), MOVES, TEAM, "turn"))[:8] == "11110000"
assert bits(legal_from_request(req([("closecombat", False, 0)] + free[1:]), MOVES, TEAM, "turn"))[:8] == "01110111"
# 5) 강제 교체: 기술 없음, 기절한 활성 자리에서 생존자만 (활성 표시가 있어도 기절이면 제외, 기절 안 한 활성(유턴 등)도 제외)
fs = {"forceSwitch": [True], "side": {"pokemon": [dict(SIDE[0], condition="0 fnt")] + SIDE[1:]}}
assert bits(legal_from_request(fs, [], TEAM, "forced")) == "00000000" + "01" + "0" + "1" + "00"
# 6) 인코더: 두 마스크가 따로 나오고 휴리스틱 마스크는 순수 합법의 부분집합
t = BattleTensorEncoder(vocab_path="data/vocab.json").encode_battle(DummyBattle())
assert len(t) == 9 and t[7].shape == t[8].shape == (1, 22)
assert not (t[7].numpy() & ~t[8].numpy()).any(), "가지치기 마스크가 순수 합법 마스크를 벗어남"
print("legal mask OK")
