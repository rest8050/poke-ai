"""매치업 특징 self-check: PYTHONPATH=. python tests/test_matchup.py
1) 손으로 만든 상황(땅기술 vs 비행/공중부양, 반감, 데미지 공식)  2) 실제 데이터에서 인코더의 배율(my_mv[...,7])과 99% 이상 일치"""
import os

import numpy as np
import torch

from src.core.matchup import Matchup
from src.core.tensor_encoder import MOVE_NUM_DIM, NUM_DIM

m = Matchup()
T = {"normal": 1, "fire": 2, "water": 3, "grass": 4, "electric": 5, "ground": 9, "flying": 10, "steel": 16}


def blank():
    return (torch.zeros(1, 6, 11, dtype=torch.long), torch.zeros(1, 6, NUM_DIM), torch.zeros(1, 6, 4, MOVE_NUM_DIM))


# 상대(공격) 0번이 지진(땅, 위력100, 물리)을 내 0번(비행)/1번(강철)/2번(불꽃)/3번(전기)에게
my_cat, my_num, my_mv = blank()
opp_cat, opp_num, opp_mv = blank()
for i, ty in enumerate([T["flying"], T["steel"], T["fire"], T["electric"]]):
    my_cat[0, i, 2] = ty
    my_num[0, i, 11:17] = torch.tensor([300, 200, 200, 200, 200, 200]) / 255.0
    my_num[0, i, 2] = 1.0
my_num[0, 0, 0] = 1.0
opp_cat[0, 0, 2] = T["ground"]
opp_num[0, 0, 11:17] = torch.tensor([100, 130, 100, 100, 100, 100]) / 255.0       # 종족값 → 추정 능력치 (공격 2*130+99 = 359)
opp_num[0, 0, 0] = 1.0
opp_num[0, 0, 2] = 1.0
opp_mv[0, 0, 0, 0] = 100 / 200; opp_mv[0, 0, 0, 4] = 0.33; opp_mv[0, 0, 0, 6] = T["ground"] / 20.0; opp_mv[0, 0, 0, 1] = 1.0
d = m(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv)["def"][0]                    # [6, 24, 4]
assert d[0, 0, 0] == 0.0 and d[0, 0, 1] == 0.0 and d[0, 0, 3] == 1.0                # 비행: 땅 무효, 공격기 표시
assert abs(d[1, 0, 0] * 4 - 2.0) < 1e-6 and abs(d[3, 0, 0] * 4 - 2.0) < 1e-6          # 강철/불꽃: 땅이 효과 굉장 → 2배, 전기도 2배
assert d[2, 0, 0] * 4 == 2.0
base = (42 * 100 * 359 / 200) / 50 + 2                                              # 강철 (방어 200, 스탯 그대로)
expect = base * 1.5 * 2.0 * 0.925 / 300                                             # 자속(땅) 1.5, 효과 굉장 2배, 평균 편차 0.925, 최대 HP 300
assert abs(d[1, 0, 1] * 1.5 - expect) < 1e-3 * expect, (float(d[1, 0, 1] * 1.5), expect)
# 공중부양 특성이면 무효
my_cat[0, 1, 1] = m.ab_immune.nonzero()[0, 0]  # 면역 특성 중 첫 번째 id (아래에서 땅 면역인지 확인)
lev = int(m.ab_immune.nonzero()[(m.ab_immune[m.ab_immune.nonzero()[:, 0]] == T["ground"]).nonzero()[0, 0], 0])
my_cat[0, 1, 1] = lev
assert m(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv)["def"][0][1, 0, 0] == 0.0
# 기절한 방어측은 0
my_num[0, 3, 1] = 1.0
assert m(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv)["def"][0][3, 0].abs().sum() == 0
print("손계산 OK")

# 실제 데이터에서 인코더와 대조
path = "data/dagger/dg2_ho.npz"
if os.path.exists(path):
    z = np.load(path); n = 3000
    g = lambda k: torch.tensor(z[k][:n]).float() if ("num" in k) else torch.tensor(z[k][:n]).long()
    mc, mn, mm, oc, on, om = [g(k) for k in ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num"]]
    out = m(mc, mn, mm, oc, on, om)
    ar, oa = torch.arange(n), on[..., 0].argmax(1)
    O = out["off"][ar, oa]
    enc = (mm[..., 7] * 4).reshape(n, 24)
    sel = (O[..., 3] > 0) & (mm[..., 0].reshape(n, 24) > 0) & (mn[..., 11] > 0).repeat_interleave(4, 1)
    agree = (((O[..., 0] * 4) - enc).abs() < 0.01)[sel].float().mean().item()
    assert agree > 0.99, agree
    print(f"인코더 배율 일치율 {agree:.4f} (표본 {int(sel.sum())})")
print("matchup OK")
