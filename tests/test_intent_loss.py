"""상대 의도 손실(train_fp_distill.intent_loss) self-check: PYTHONPATH=. python tests/test_intent_loss.py
라벨 없는 결정(NaN)은 무시되고, 기술 질량이 모델 칸에 정렬되며(칸에 없으면 '기타'), 정답 분포가 균등 분포보다 손실이 낮다"""
import numpy as np
import torch

from src.core.model_v5 import N_SLOT
from src.training.train_fp_distill import intent_loss

n = 3
ids = torch.zeros(n, N_SLOT, dtype=torch.long); ids[:, 2] = 7                    # 칸 2 = 기술 7
b = {"opp_sw": torch.tensor([float("nan"), 0.3, 0.0]), "opp_tera": torch.tensor([0.0, 0.2, 0.0]),
     "opp_mv_id": torch.tensor([[0] * 5, [7, 9, 0, 0, 0], [7, 0, 0, 0, 0]], dtype=torch.int32),     # 기술 9는 칸에 없음 → 기타
     "opp_mv_p": torch.tensor([[0.0] * 5, [0.5, 0.2, 0, 0, 0], [1.0, 0, 0, 0, 0]]),
     "opp_tgt": torch.tensor([[0.0] * 6, [0.3, 0, 0, 0, 0, 0], [0.0] * 6])}
ok = torch.zeros(n, N_SLOT, dtype=torch.bool); ok[:, 2] = True; ok[:, 5] = True
tgt_ok = torch.ones(n, 6, dtype=torch.bool)


def make(move_probs, kind=(0.0, 0.0)):
    lp = torch.log(torch.tensor(move_probs).clamp_min(1e-9)).expand(n, -1).clone()
    return {"kind": torch.tensor(kind).expand(n, 2).clone(), "move_lp": lp, "tgt_lp": torch.log(torch.full((n, 6), 1 / 6)),
            "slot_ids": ids, "slot_ok": ok, "tgt_ok": tgt_ok}


good = [0.0] * (N_SLOT + 1); good[2], good[-1] = 0.9, 0.1
uni = [1 / (N_SLOT + 1)] * (N_SLOT + 1)
l_good, st, p_sw, y_sw = intent_loss(make(good), b)
l_uni, *_ = intent_loss(make(uni), b)
assert torch.isfinite(l_good) and st["n"] == 2 and len(p_sw) == 2 and y_sw.tolist() == [False, False]
assert l_good < l_uni, (l_good, l_uni)
# 라벨이 전부 없으면 0
l0, st0, _, _ = intent_loss(make(good), {**b, "opp_sw": torch.full((n,), float("nan"))})
assert l0.item() == 0.0 and st0 == {}
print("ok")
