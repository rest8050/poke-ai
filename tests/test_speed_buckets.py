"""상대 스피드 세트 구간 분포 + 같은 우선도 안의 순서 도구/특성 self-check: PYTHONPATH=. python tests/test_speed_buckets.py
1) 구간 배율 값, 2) 구간 확률로 선공 기대값을 합침 (기대 스피드 하나로 비교하지 않음), 3) 선제공격손톱 20% / 퀵드로 30%(공격기만) / 후공열매(HP 1/4 이하) / 뒤처지는꼬리,
4) 상대의 숨은 도구는 확률로, 5) 스피드 구간 손실의 정답 = 실제 스피드에 가장 가까운 구간"""
import torch
import torch.nn.functional as F

from src.core.matchup_v7 import speed_bucket_ratio
from src.core.tensor_encoder import FIELD_DIM
from src.training.hidden_labels import hidden_loss, vocab_sizes
from tests.test_matchup_v7 import A, IT, M, active, team

HID = IT["<unk>"] + 1
atk, status = [(40, 0, "phys", "dragon")], [(0, 0, "status", "normal")]


def run(my, opp, spe_p=None, order_p=None, hide_opp_item=False):
    mc, mn, mm = team(my, True)
    oc, on, om = team(opp, False)
    if hide_opp_item:
        oc[0, 0, 0] = HID
    with torch.no_grad():
        return M(mc, mn, mm, oc, on, om, torch.zeros(1, FIELD_DIM), F.one_hot(oc[..., 1], A).float(), torch.zeros(1, 6),
                 opp_spe_p=None if spe_p is None else torch.tensor(spe_p).view(1, 1, 5).expand(1, 6, 5),
                 opp_order_p=None if order_p is None else torch.tensor(order_p).view(1, 1, 4).expand(1, 6, 4))


r = speed_bucket_ratio(torch.tensor([100.0]))[0]
assert torch.allclose(r, torch.tensor([205 * 0.9, 236 * 0.9, 236.0, 299.0, 299 * 1.1]) / 299), r        # 종족값 100: 184.5 / 212.4 / 236 / 299 / 328.9

me = lambda spe, moves=atk, **kw: [active(stats=(100, 100, 100, 100, 100, spe), moves=moves, **kw)]
fm = lambda o: float(o["fm_my"][0, 0, 0])
near = lambda a, b: abs(a - b) < 1e-5
# 내 스피드 250 대 상대 종족값 100: 무보정(236)이면 내가 먼저, 풀보정(299)이면 나중
assert fm(run(me(250), [active()])) == -1.0, "구간 확률이 없으면 예전처럼 표준(299) 하나와 비교"
assert near(fm(run(me(250), [active()], spe_p=[0, 0, 0.5, 0.5, 0])), 0.0), "반반이면 0"
assert near(fm(run(me(250), [active()], spe_p=[0, 0.1, 0.6, 0.2, 0.1])), 0.4), "먼저일 확률 0.7 -> 0.7 - 0.3"

fast, slow = [active()], [active(stats=(100, 100, 100, 100, 100, 20))]          # 상대: 종족값 100(299) / 20(139)
assert near(fm(run(me(50, item="quickclaw"), fast)), 0.2 - 0.8), "선제공격손톱: 느려도 20%는 먼저"
assert near(fm(run(me(50, ab="quickdraw"), fast)), 0.3 - 0.7) and fm(run(me(50, status, ab="quickdraw"), fast)) == -1.0, "퀵드로 30%는 공격기만"
assert fm(run(me(50, item="custapberry", hp=0.2), fast)) == 1.0 and fm(run(me(50, item="custapberry", hp=0.5), fast)) == -1.0, "후공열매는 HP 1/4 이하일 때"
assert fm(run(me(400, item="laggingtail"), fast)) == -1.0 and fm(run(me(400, ab="stall"), fast)) == -1.0, "뒤처지는꼬리/시간벌기: 빨라도 나중"
assert fm(run(me(400, status, ab="myceliummight"), fast)) == -1.0 and fm(run(me(400, ab="myceliummight"), fast)) == 1.0, "균사의힘은 변화기만 나중"
assert near(fm(run(me(400), fast, order_p=[1.0, 0, 0, 0], hide_opp_item=True)), 0.8 - 0.2), "상대의 숨은 선제공격손톱 확률"
assert fm(run(me(400), fast, order_p=[1.0, 0, 0, 0])) == 1.0, "상대 도구가 숨김이 아니면 확률을 안 씀"
assert near(fm(run(me(400, item="quickclaw"), fast, order_p=[1.0, 0, 0, 0], hide_opp_item=True)), 0.16 - 0.16 + 0.68), "둘 다 손톱: 한쪽만 발동한 경우만 뒤집힘"
prio = [(40, 1, "phys", "dragon")]
assert fm(run(me(50, prio, item="laggingtail"), fast)) == 1.0, "우선도가 다르면 순서 도구는 무관"
o = run(me(400), [active(moves=atk, item="quickclaw")])
assert near(float(o["fm_opp"][0, 0, 0]), 0.2 - 0.8), "상대 기술칸 쪽도 같은 규칙"

# 스피드 구간 손실: 실제 스피드 328(풀보정 +성격) -> 구간 4, 236(무보정) -> 구간 2
UNK = vocab_sizes()
cat = torch.zeros(2, 6, 11, dtype=torch.long)
cat[..., 10] = torch.arange(1, 7)
num = torch.zeros(2, 6, 43)
num[..., 11:17] = 100 / 255
true = torch.zeros(1, 6, 17, dtype=torch.long)
true[..., 10], true[..., 11:17] = torch.arange(1, 7), torch.tensor([404, 299, 299, 299, 299, 328])
true[0, 1, 16] = 236
b = {"opp_team_cat": cat, "opp_team_num": num, "seq_index": torch.zeros(2, dtype=torch.long), "true_set": true, "true_ok": torch.ones(1, dtype=torch.bool)}
lg = {"move": torch.zeros(2, 6, UNK["move"] + 2), "item": torch.zeros(2, 6, UNK["item"] + 2), "ability": torch.zeros(2, 6, UNK["ability"] + 2), "stat": torch.zeros(2, 6, 6), "spe": torch.zeros(2, 6, 5)}
lg["spe"][..., 4] = 5.0
lg["spe"][:, 1] = torch.tensor([0, 0, 5.0, 0, 0])
_, st = hidden_loss(lg, b, UNK)
assert st["spe_n"] == 12 and st["spe_acc"] == 1.0, st
print("speed buckets OK")
