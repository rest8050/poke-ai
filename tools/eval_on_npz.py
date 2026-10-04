"""체크포인트가 npz(교사 라벨) 위에서 교사와 얼마나 일치하는지 측정. DAgger 진단용: 학생 방문 상태 vs 교사 방문 상태 비교.
사용: eval_on_npz.py <ckpt> <npz> [<npz> ...]   (npz의 배틀을 시퀀스 단위로 통째로 넣음 — 히스토리 포함)"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.model import model_from_ckpt

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]

model = model_from_ckpt(sys.argv[1])
model.eval()
move_in = model.embeddings.move_encoder.fc[0].in_features - 48  # 모델이 기대하는 기술 수치 칸 수 (조건부 선공권 등 새 피처 추가로 늘어날 수 있음)
for path in sys.argv[2:]:
    z0 = np.load(path)
    z = {k: z0[k] for k in KEYS + ["action_mask", "target", "lens"]}  # NpzFile은 접근할 때마다 파일을 다시 읽으므로 한 번에 메모리로 올림 (큰 파일에서 수십 배 느려짐)
    for k in ("my_move_num", "opp_move_num"):
        # ponytail: 이 npz가 옛 인코더(칸 수 적음)로 만들어졌으면 뒤를 0으로 채워 맞춤 (재빌드 전 임시방편)
        if z[k].shape[-1] < move_in:
            pad = np.zeros(z[k].shape[:-1] + (move_in - z[k].shape[-1],), dtype=z[k].dtype)
            z[k] = np.concatenate([z[k], pad], axis=-1)
    lens = z["lens"]
    off = np.concatenate([[0], np.cumsum(lens)])
    agree = n = ce = ent = top = 0
    with torch.no_grad():
        for b in range(len(lens)):
            s, e = off[b], off[b + 1]
            T = e - s
            out = model.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(T, dtype=torch.long),
                                          torch.arange(T), 1, torch.tensor(z["action_mask"][s:e]))
            lp = torch.log_softmax(out["policy_logits"], -1)
            t = torch.tensor(z["target"][s:e])
            agree += (lp.argmax(-1) == t.argmax(-1)).sum().item()
            ce += -(t * lp.clamp(min=-30)).sum().item()
            ent += -(t * torch.log(t.clamp(min=1e-9))).sum().item()
            top += t.max(-1).values.sum().item()
            n += T
    print(f"{path}: 배틀 {len(lens)} 결정 {n} | 교사와 일치 {agree / n:.3f} | 소프트CE {ce / n:.4f} | 교사 엔트로피 {ent / n:.3f} 최상위확률 {top / n:.3f}")
