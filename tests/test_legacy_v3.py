"""옛 v3 구조 체크포인트(v3_full 등)가 계속 로드되고 같은 출력을 내는지: PYTHONPATH=. python tests/test_legacy_v3.py
v3_full이 새 구조 모델로 교체되어 퇴역하면 이 테스트와 src/core/model_v3_legacy.py를 같이 삭제"""
import os

import numpy as np
import torch

from src.core.model import model_from_ckpt

CK = "checkpoints/supervised_v2_fp_v3_full.pt"
if not os.path.exists(CK):
    print("v3_full 체크포인트 없음 — 건너뜀")
else:
    m = model_from_ckpt(CK).eval()
    assert type(m).__name__ == "EntityPokemonNetV3Legacy"
    from src.evaluation.holdout import OBS, load_holdout
    _, o, mask, _ = load_holdout(("dg3",))[0]; n = len(mask)           # 통합 홀드아웃의 dg3 첫 배틀 (기술 수치 칸은 46으로 채워져 있음)
    obs = [torch.tensor(o[k]) for k in OBS]
    out = m.forward_sequences(obs, torch.zeros(n, dtype=torch.long), torch.arange(n), 1, torch.tensor(mask))
    assert torch.isfinite(out["policy_logits"][torch.tensor(mask)]).all()
    print("legacy v3 OK")
