"""FP로 증류한 학생(v8_full)이 사람 행동과 얼마나 맞는가 (사람 데이터 파싱 결과 data/human/gen9ou_s0_r1600_*.npz, 추론만):
사람 행동의 NLL과 일치율, 행동 종류(기술/교체/테라) 비율 비교, 레이팅 구간별.
    PYTHONPATH=. python tools/human_agree.py [체크포인트]"""
import glob
import sys

import numpy as np
import torch

from src.core.model import model_from_ckpt
from src.training.replay_dataset import OBS_KEYS

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/supervised_v2_fp_v8_full.pt"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model_from_ckpt(ckpt).to(dev).eval()
CLS = np.array([0] * 4 + [2] * 4 + [1] * 14)         # 0 기술, 1 교체, 2 테라 기술 (22칸 → 종류)
rec = {k: [] for k in ("nll", "agree", "h_cls", "s_cls", "p_sw", "rating")}
with torch.no_grad():
    for f in sorted(glob.glob("data/human/gen9ou_s0_r1600_*.npz")):
        p = np.load(f)
        lens, rating, act, mask = p["lens"], p["rating"], p["action"], p["action_mask"]
        obs = [p[k] for k in OBS_KEYS]
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for a in range(0, len(lens), 16):
            ids = np.arange(a, min(a + 16, len(lens)))
            pos = np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in ids])
            seq_index = torch.from_numpy(np.repeat(np.arange(len(ids)), lens[ids])).to(dev)
            time_index = torch.from_numpy(np.concatenate([np.arange(lens[s]) for s in ids])).to(dev)
            m = torch.from_numpy(mask[pos]).to(dev)
            lp = net.forward_sequences([torch.from_numpy(o[pos]).to(dev) for o in obs], seq_index, time_index, len(ids), action_mask=m)["policy_logits"].float().cpu().numpy()
            lab = act[pos] >= 0
            la, lpl = act[pos][lab], lp[lab]
            rec["nll"].append(-lpl[np.arange(len(la)), la]); rec["agree"].append(lpl.argmax(1) == la)
            rec["h_cls"].append(CLS[la]); rec["s_cls"].append(CLS[lpl.argmax(1)])
            rec["p_sw"].append(np.exp(lpl[:, 8:14]).sum(1)); rec["rating"].append(np.repeat(rating[ids], lens[ids])[lab])
        print(f"{f} 완료", flush=True)
r = {k: np.concatenate(v) for k, v in rec.items()}
lines = [f"체크포인트 {ckpt}, 라벨 있는 사람 결정 {len(r['nll'])}"]
for name, sel in [("전체", np.ones(len(r["nll"]), bool)), ("1600-1699", (r["rating"] >= 1600) & (r["rating"] < 1700)),
                  ("1700-1799", (r["rating"] >= 1700) & (r["rating"] < 1800)), ("1800+", r["rating"] >= 1800)]:
    h = np.bincount(r["h_cls"][sel], minlength=3) / sel.sum()
    s = np.bincount(r["s_cls"][sel], minlength=3) / sel.sum()
    lines.append(f"{name:10s} n={sel.sum():6d} | 사람 행동 NLL {r['nll'][sel].mean():.3f}, 일치율 {r['agree'][sel].mean():.3f} | "
                 f"사람 비율(기술/교체/테라) {h[0]:.3f}/{h[1]:.3f}/{h[2]:.3f}, 학생 argmax 비율 {s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f}, "
                 f"학생 평균 교체 확률 {r['p_sw'][sel].mean():.3f}")
for c, nm in enumerate(("기술", "교체", "테라 기술")):
    sel = r["h_cls"] == c
    lines.append(f"사람이 {nm}를 고른 결정에서: 행동 일치율 {r['agree'][sel].mean():.3f}, 같은 종류로 고른 비율 {(r['s_cls'][sel] == c).mean():.3f}")
open("logs/human_agree.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
