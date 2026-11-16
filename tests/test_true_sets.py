"""상대 실제 세트 라벨 (hidden_labels.true_sets + hidden_loss): 배틀에서 끝내 안 드러난 도구/기술도 정답이 되고, 능력치 배율 정답이 실제 능력치에서 나옴.
실행: PYTHONPATH=. python tests/test_true_sets.py"""
import numpy as np
import torch

from src.core.model import STAT_SCALE
from src.core.tensor_encoder import NUM_DIM
from src.training.hidden_labels import hidden_loss, true_sets, vocab_sizes

UNK = vocab_sizes()
H = {k: UNK[k] + 1 for k in UNK}                      # 숨김 id
sp = lambda a: np.arange(a, a + 6)                    # 종 6마리
def side(my_sp, opp_sp, n=2):                         # 한 시점 n결정: 내 팀은 도구 7/특성 3/기술 11~14, 상대는 전부 숨김 (끝까지 안 드러남)
    mc, oc = np.zeros((n, 6, 11), np.int16), np.zeros((n, 6, 11), np.int16)
    mc[..., 10], mc[..., 0], mc[..., 1], mc[..., 5:9] = my_sp, 7, 3, [11, 12, 13, 14]
    oc[..., 10], oc[..., 0], oc[..., 1] = opp_sp[::-1], H["item"], H["ability"]          # 상대 슬롯 순서는 내 팀 순서와 다름
    mn, on = np.zeros((n, 6, NUM_DIM), np.float32), np.zeros((n, 6, NUM_DIM), np.float32)
    on[..., 11:17] = 100 / 255                                                           # 종족값 100 -> 표준 hp 404, 나머지 299
    mn[..., 11:17] = np.array([404, 299, 299, 236, 299, 328]) / 255                      # 실제: 특공 노력치 0(236), 스피드 +성격(328)
    return mc, mn, oc, on
A, B, C = side(sp(10), sp(20)), side(sp(20), sp(10)), side(sp(40), sp(50))               # A와 B는 같은 판의 양쪽(승패 반대), C의 상대(50~55)는 데이터에 없음
data = {k: np.concatenate([x[i] for x in (A, B, C)]) for i, k in enumerate(("my_team_cat", "my_team_num", "opp_team_cat", "opp_team_num"))}
data["outcome"] = np.array([1.0, -1.0, 1.0])
ts, ok = true_sets(data, np.array([0, 2, 4, 6]))
assert ok.tolist() == [True, True, False] and (ts[0][:, 10] == sp(20)).all() and ts[0][0, 0] == 7 and ts[0][0, 11:17].tolist() == [404, 299, 299, 236, 299, 328]
data["opp_team_cat"][0:2, 5, 5] = 99                                                     # A가 본 상대(종 20)의 공개 기술이 그 팀 세트에 없음 -> 반대편으로 안 침
assert true_sets(data, np.array([0, 2, 4, 6]))[1].tolist() == [False, True, False]
data["opp_team_cat"][0:2, 5, 5] = 0

b = {"opp_team_cat": torch.tensor(data["opp_team_cat"]).long(), "opp_team_num": torch.tensor(data["opp_team_num"]), "seq_index": torch.tensor([0, 0, 1, 1, 2, 2])}
lg = {"move": torch.zeros(6, 6, UNK["move"] + 2), "item": torch.zeros(6, 6, UNK["item"] + 2), "ability": torch.zeros(6, 6, UNK["ability"] + 2), "stat": torch.zeros(6, 6, 6)}
_, st = hidden_loss({k: v for k, v in lg.items() if k != "stat"}, b, UNK)
assert st["item_n"] == 0 and st["move_n"] == 0, "실제 세트 없이는 안 드러난 것에 정답이 없음"
b["true_set"], b["true_ok"] = torch.tensor(ts).long(), torch.tensor(ok)
lg["item"][..., 7] = 5.0; lg["stat"][..., 3] = np.log(236 / 299) / STAT_SCALE; lg["stat"][..., 5] = np.log(328 / 299) / STAT_SCALE
loss, st = hidden_loss(lg, b, UNK)
assert st["item_n"] == 24 and st["ability_n"] == 24 and st["move_n"] == 96, st       # A, B 두 시점 x 2결정 x 6마리 (C는 팀을 못 찾아 빠짐)
assert st["item_acc"] == 1.0 and st["stat_mae"] < 1e-4, st
print("true sets OK")

# 팀 프리뷰가 폼을 숨긴 내 포켓몬 (Greninja-*): 팀 파일의 폼 이름(greninjabond)으로도 세트를 찾아야 함
from src.training.replay_dataset import _preview_mon
_info = {"greninjabond": {"moves": ["surf"], "item": "lifeorb", "ability": "battlebond", "tera": None}, "greattusk": {"moves": ["earthquake"], "item": None, "ability": None, "tera": None}}
_m = _preview_mon("Greninja-*, M", _info)
assert list(_m.moves) == ["surf"] and _m.item == "lifeorb" and _m.ability == "battlebond", (_m.moves, _m.item)
assert not _preview_mon("Urshifu-*", _info).moves, "팀에 없는 종은 그대로 비어 있음"
print("preview form OK")
