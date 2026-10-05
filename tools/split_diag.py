import sys
import os
import numpy as np, torch
sys.path.insert(0, ".")
from src.core.model import model_from_ckpt
KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
model = model_from_ckpt(sys.argv[1]); model.eval()
move_in = model.embeddings.move_encoder.fc[0].in_features - 48
P, T, V = [], [], []
for path in sys.argv[2:]:
    z0 = np.load(path); z = {k: z0[k] for k in KEYS + ["action_mask", "target", "lens"]}
    for k in ("my_move_num", "opp_move_num"):
        if z[k].shape[-1] < move_in:
            z[k] = np.concatenate([z[k], np.zeros(z[k].shape[:-1] + (move_in - z[k].shape[-1],), dtype=z[k].dtype)], -1)
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    with torch.no_grad():
        for b in range(min(len(z["lens"]), int(os.environ.get("MAXB", 10**9)))):
            s, e = off[b], off[b + 1]; n = e - s
            out = model.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(n, dtype=torch.long), torch.arange(n), 1, torch.tensor(z["action_mask"][s:e]))
            P.append(torch.softmax(out["policy_logits"], -1).numpy()); T.append(z["target"][s:e])
            am = z["action_mask"][s:e].astype(bool); V.append(am[:, :8].any(-1) & am[:, 8:14].any(-1))
P, T, V = np.concatenate(P), np.concatenate(T), np.concatenate(V)
P, T = P[V], T[V]
t_sw = (T.argmax(-1) >= 8) & (T.argmax(-1) < 14)
sm, mm = P[:, 8:14].sum(-1), P[:, :8].sum(-1)
a_arg = P.argmax(-1); s_sw_action = (a_arg >= 8) & (a_arg < 14)
s_sw_cat = sm > mm  # 범주 먼저: 교체 질량 합 > 기술 질량 합이면 교체
cat_act = np.where(s_sw_cat[:, None], np.where(np.arange(22) >= 8, P, -1), np.where(np.arange(22) < 8, P, -1)).argmax(-1)
print(f"자발 결정 {len(P)}건 | 교사 교체 비율 {t_sw.mean():.3f}")
print(f"[기존: 행동 argmax] 교체 비율 {s_sw_action.mean():.3f} | 교체/기술 범주 일치 {(s_sw_action == t_sw).mean():.3f} | 정확한 행동 일치 {(a_arg == T.argmax(-1)).mean():.3f}")
print(f"[범주 먼저: 질량합 비교] 교체 비율 {s_sw_cat.mean():.3f} | 교체/기술 범주 일치 {(s_sw_cat == t_sw).mean():.3f} | 정확한 행동 일치 {(cat_act == T.argmax(-1)).mean():.3f}")
m = t_sw & ~s_sw_action
print(f"교사는 교체인데 학생 argmax는 기술인 {m.sum()}건: 학생 교체질량합 {sm[m].mean():.3f}, 최상위 교체 1개 {P[m][:, 8:14].max(-1).mean():.3f}, 최상위 기술 1개 {P[m][:, :8].max(-1).mean():.3f}, 질량합이 기술합을 넘는 비율 {(sm[m] > mm[m]).mean():.2f}")
tm = T[:, 8:14]; top_share = tm.max(-1) / np.clip(tm.sum(-1), 1e-9, None)
print(f"교사 교체 상태에서 교사의 교체 후보 집중도(최상위/교체합) {top_share[t_sw].mean():.2f} | 학생 {(P[t_sw][:, 8:14].max(-1) / np.clip(sm[t_sw], 1e-9, None)).mean():.2f}")
