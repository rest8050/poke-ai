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
    z = np.load("data/dagger/dg3_ho.npz"); n = int(z["lens"][0])
    keys = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
    obs = [torch.tensor(z[k][:n]) for k in keys]
    for i in (2, 5):                                         # 옛 npz는 기술 수치 칸이 44개 -> 뒤를 0으로 채움
        obs[i] = torch.cat([obs[i], obs[i].new_zeros(*obs[i].shape[:-1], 46 - obs[i].size(-1))], -1)
    out = m.forward_sequences(obs, torch.zeros(n, dtype=torch.long), torch.arange(n), 1, torch.tensor(z["action_mask"][:n]))
    assert torch.isfinite(out["policy_logits"][torch.tensor(z["action_mask"][:n])]).all()
    print("legacy v3 OK")
