"""chain_targets: 합성 시퀀스로 응수 클래스와 Δ 정답이 기대값인지 확인 (실행: PYTHONPATH=. python tests/test_chain_targets.py)"""
import numpy as np
from src.training.chain_targets import build, move_class_table, SWITCH_CLS

table = move_class_table()
V = __import__("json").load(open("data/vocab.json", encoding="utf-8"))["move"]
sp, tb, ap = V["suckerpunch"], V["toxic"], V["swordsdance"]
assert table[sp] == 1 and table[ap] == 3 and table[V["earthquake"]] == 0 and table[V["recover"]] == 4, "클래스 표"

N = 5                                        # 시퀀스 2개: 길이 3, 2
lens = np.array([3, 2])
mine = np.zeros((N, 6, 43), np.float32); opp = np.zeros((N, 6, 43), np.float32)
mine[:, 0, 0] = 1; opp[:, 0, 0] = 1; mine[:, :, 2] = 1; opp[:, :, 2] = 1
mine[1, 0, 2] = 0.4; mine[1, 0, 1] = 0.0; opp[1, 0, 2] = 0.7                 # 행0 -> 행1: 내 HP -0.6, 상대 -0.3
opp[1, 0, 3] = 2 / 6; mine[2, 0, 1] = 1; mine[2, 0, 2] = 0; mine[2, :, 0] = 0   # 행2는 내 활성이 기절하고 활성 표시가 없음
cat = np.zeros((N, 6, 11), np.int64); cat[:, :, 10] = np.arange(1, 7)        # 상대 슬롯 j의 종 번호 = j+1
cat[1] = cat[1][[1, 0, 2, 3, 4, 5]]; opp[1] = opp[1][[1, 0, 2, 3, 4, 5]]      # 행1에서 상대 슬롯 0/1이 서로 바뀜 (공개 순서 변화)
data = dict(opp_action=np.array([0, 2, 1, 0, -2]), opp_move=np.array([sp, ap, 0, tb, 0]), kind=np.array([0, 0, 0, 0, 1]),
            action_taken=np.array([0, 9, 8, 3, 8]), my_team_num=mine, opp_team_num=opp, opp_team_cat=cat)
t = build(data, lens, table)
assert t["opp_cls"].tolist() == [1, 3, SWITCH_CLS, 7, -1], t["opp_cls"]
assert t["opp_tera"].tolist() == [0, 1, -1, 0, -1], t["opp_tera"]
assert t["r_valid"].tolist() == [True, True, False, True, False], t["r_valid"]     # 시퀀스 마지막 행(2, 4)과 응수를 모르는 행은 제외, 행3은 다음 행이 강제 교체 행이어도 유효
assert np.allclose(t["dhp"][0, 0], -0.6) and np.allclose(t["dhp"][0, 6], -0.3), t["dhp"][0]
assert t["act_my"][0] == 0 and t["act_opp"][0] == 0 and t["act_my"][1] == -1, (t["act_my"], t["act_opp"])
assert np.allclose(t["dhp"][1, 0], -0.4) and t["faint"][1, 0] == 1 and t["faint"][0, 0] == 0
assert np.allclose(t["boost"][0, 5], 2 / 6), t["boost"][0]        # 행1의 상대 활성 부스트 atk: 열 5 + 0
print("chain targets OK")
