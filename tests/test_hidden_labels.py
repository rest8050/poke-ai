"""숨겨진 정보 보조 손실 self-check: PYTHONPATH=. python tests/test_hidden_labels.py
정답 = 같은 배틀에서 나중에 드러난 기술/도구/특성 (이미 드러난 것 제외), 종이 다른 슬롯/끝내 안 드러난 것은 제외, 모델이 hid_* 출력을 냄"""
import torch

from src.core.model import build_model
from src.training.hidden_labels import hidden_loss, vocab_sizes

U = vocab_sizes()                                         # <unk> id: move 967, item 588, ability 317 (숨김 id = unk+1)
HM, HI, HA = U["move"] + 1, U["item"] + 1, U["ability"] + 1
N, Vm, Vi, Va = 5, U["move"] + 2, U["item"] + 2, U["ability"] + 2
cat = torch.zeros(N, 6, 11, dtype=torch.long)
cat[..., 10] = 7                                          # 슬롯0만 종 7로 채움 (나머지는 종 0 = 건너뜀)
cat[:, 1:, 10] = 0
cat[:, 0, 0], cat[:, 0, 1] = HI, HA                       # 도구/특성은 숨겨짐
# 배틀 0 (결정 0~2): 기술 100 공개 -> 200 -> 300 / 마지막 턴에 도구 50, 특성 20 공개. 배틀 1 (결정 3~4): 기술 100만 계속 (나중에 드러난 게 없음)
mv = [[100, HM, HM, HM], [100, 200, HM, HM], [100, 200, 300, HM], [100, HM, HM, HM], [100, HM, HM, HM]]
for i, m in enumerate(mv):
    cat[i, 0, 5:9] = torch.tensor(m)
cat[2, 0, 0], cat[2, 0, 1] = 50, 20
b = {"opp_team_cat": cat, "seq_index": torch.tensor([0, 0, 0, 1, 1])}

def logits(good_move, good_item, good_ab):
    lm, li, la = torch.zeros(N, 6, Vm), torch.zeros(N, 6, Vi), torch.zeros(N, 6, Va)
    if good_move:
        lm[0:3, 0, 200], lm[0:3, 0, 300] = 20.0, 20.0
    if good_item:
        li[0:3, 0, 50] = 20.0
    if good_ab:
        la[0:3, 0, 20] = 20.0
    return {"move": lm, "item": li, "ability": la}

bad, st_bad = hidden_loss(logits(False, False, False), b, U)
good, st_good = hidden_loss(logits(True, True, True), b, U)
assert good < 0.2 * bad, (float(good), float(bad))
# 기술 정답 수: 결정0은 {200,300}, 결정1은 {300}, 결정2는 없음(이미 모두 공개) -> 3개 / 도구, 특성: 결정0,1 (결정2는 이미 공개) 각 2개
assert st_good["move_n"] == 3 and st_good["item_n"] == 2 and st_good["ability_n"] == 2, st_good
assert st_good["move_top4"] > 0.99 and st_good["item_acc"] == 1.0 and st_good["ability_acc"] == 1.0, st_good
# 같은 슬롯의 종이 마지막 턴과 다르면 건너뜀
cat2 = cat.clone(); cat2[2, 0, 10] = 9
l2, st2 = hidden_loss(logits(True, True, True), {"opp_team_cat": cat2, "seq_index": b["seq_index"]}, U)
assert st2["move_n"] == 0 or st2["move_n"] < st_good["move_n"] + 1
# 모델이 hid_* 출력을 내고 손실이 신념 모듈로 역전파
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM
g = torch.Generator().manual_seed(0)
def rnd(n):
    my_num = torch.rand(n, 6, NUM_DIM, generator=g); my_num[:, :, 0] = 0; my_num[:, 0, 0] = 1
    opp_num = torch.rand(n, 6, NUM_DIM, generator=g); opp_num[:, :, 0] = 0; opp_num[:, 1, 0] = 1
    return [torch.randint(1, 8, (n, 6, 11), generator=g), my_num, torch.rand(n, 6, 4, 46, generator=g),
            torch.randint(1, 8, (n, 6, 11), generator=g), opp_num, torch.rand(n, 6, 4, 46, generator=g), torch.rand(n, FIELD_DIM, generator=g)]
o = rnd(4)
mask = torch.zeros(4, 22, dtype=torch.bool); mask[:, [0, 2, 8]] = True
m = build_model({"model": "v3", "version": 2, "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64}).eval()
out = m.forward_sequences(o, torch.zeros(4, dtype=torch.long), torch.arange(4), 1, mask)
assert out["hid_move"].shape == (4, 6, Vm) and out["hid_item"].shape == (4, 6, Vi) and out["hid_ability"].shape == (4, 6, Va)
loss, _ = hidden_loss({"move": out["hid_move"], "item": out["hid_item"], "ability": out["hid_ability"]},
                      {"opp_team_cat": o[3], "seq_index": torch.zeros(4, dtype=torch.long)}, U)
loss.backward()
print("hidden labels OK")
