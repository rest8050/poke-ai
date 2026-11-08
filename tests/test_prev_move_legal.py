"""직전 기술 입력과 합법성 요약 self-check: PYTHONPATH=. python tests/test_prev_move_legal.py
- encoder.prev_move: events.py가 기록한 양쪽 마지막 사용 기술 -> 기술 어휘 번호 [1,2], 기록이 없으면 0
- v8 _legal_summary(forced_zero): 합법 기술이 0개(강제 교체)일 때만 고정·잃은 화력 칸이 0, 기술이 있는 결정은 예전과 같음"""
import json

import torch

from src.core.model import LEG_G, EntityPokemonNetV8
from src.core.tensor_encoder import BattleTensorEncoder

V = json.load(open("data/vocab.json", encoding="utf-8"))


class _Rev:
    last_move_sp = {"me": ("garchomp", "earthquake"), "opp": None}


class _Ev:
    rev = _Rev()


class _Battle:
    pass


def test_encoder_prev_move():
    enc = BattleTensorEncoder(vocab_path="data/vocab.json")
    b = _Battle()
    b.__dict__["_ev"] = _Ev()
    assert enc.prev_move(b).tolist() == [[V["move"]["earthquake"], 0]]
    assert enc.prev_move(_Battle()).tolist() == [[0, 0]]          # 추적기가 없으면 둘 다 "없음"


def test_legal_summary_forced_zero():
    leg = torch.zeros(2, 22)
    leg[0, :2] = 1; leg[0, 8:10] = 1                                     # 결정 0: 기술 2개만 합법 (나머지 2개는 막힘)
    leg[1, 8:10] = 1                                                     # 결정 1: 강제 교체 (기술 0개)
    off = torch.zeros(2, 4, 4)
    off[..., 1] = torch.tensor([0.2, 0.3, 0.6, 0.1]); off[..., 3] = 1
    n_rev = torch.tensor([4.0, 4.0])
    old = EntityPokemonNetV8._legal_summary(leg, off, n_rev)
    new = EntityPokemonNetV8._legal_summary(leg, off, n_rev, forced_zero=True)
    LOCK, LOST = 13, slice(17, 19)                                       # 비트 8 + [기술 수, 공격기 수, 교체 수, 테라, 한 개만, 고정] + 가능 공격 3 + 잃은 화력 2
    assert old.shape == (2, LEG_G) and new.shape == (2, LEG_G)
    assert old[1, LOCK] == 1.0 and old[1, 17] > 0                        # 예전 계산: 강제 교체를 "전부 막힘, 화력 다 잃음"으로 봄
    assert new[1, LOCK] == 0 and (new[1, LOST] == 0).all()
    assert torch.equal(new[0], old[0]) and abs(float(old[0, LOCK]) - 0.5) < 1e-6 and abs(float(old[0, 17]) - 0.3) < 1e-6


if __name__ == "__main__":
    test_encoder_prev_move()
    test_legal_summary_forced_zero()
    print("ok")
