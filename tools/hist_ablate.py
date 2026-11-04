"""모델이 히스토리를 실제로 얼마나 쓰는가 (재학습 없음, 추론만): 히스토리 어텐션을 최근 k턴으로 제한하거나 0으로 바꿔 홀드아웃 지표가 얼마나 떨어지는지.
k=1은 현재 턴 하나만 보는 히스토리 모듈(기억 없음), full은 원래, zero는 히스토리 출력을 0으로. 모든 지표가 같으면 그 모델은 히스토리를 쓰지 않는 것.
사용: python tools/hist_ablate.py <체크포인트 경로 또는 이름> (홀드아웃 d7_ho, d6_ho, dg5)"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "tools")
import legal_eval as L
import regret_eval as R
from src.core.model import MAX_TURNS, model_from_ckpt

SETS = ["data/dagger/d7_ho.npz", "data/dagger/d6_ho.npz", "data/dagger/dg5.npz"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = [L.load_set(p) for p in SETS]
qs = [np.load(p)["q_mean"] for p in SETS]; vs = [np.load(p)["q_vis"] for p in SETS]
q, vis = np.concatenate(qs), np.concatenate(vs)
T = np.concatenate([d["target"] for d in sets]).astype(np.float64); M = np.concatenate([d["action_mask"] for d in sets])
off = np.cumsum([0] + [len(d["lens"]) for d in sets])
battle = np.concatenate([np.repeat(np.arange(len(d["lens"])), d["lens"]) + o for d, o in zip(sets, off)])
pos = np.concatenate([np.concatenate([np.arange(n) for n in d["lens"]]) for d in sets])
rel = (vis >= 100) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)
rng = np.random.RandomState(0)

arg = sys.argv[1] if len(sys.argv) > 1 else "v8_full"
net = model_from_ckpt(arg if os.path.exists(arg) else f"checkpoints/supervised_v2_fp_{arg}.pt").to(dev).eval()
orig = net._history


def patch(k):
    if k == "full":
        net._history = orig
    elif k == "zero":
        net._history = lambda turn_seq: torch.zeros(turn_seq.size(0), turn_seq.size(1), net.history_out.out_features, device=turn_seq.device, dtype=turn_seq.dtype)
    else:
        def h(turn_seq):
            B, Tn = turn_seq.size(0), turn_seq.size(1)
            p = torch.arange(Tn, device=turn_seq.device).clamp(max=MAX_TURNS - 1)
            x = net.turn_proj(turn_seq) + net.turn_pos(p)
            i, j = torch.arange(Tn, device=x.device)[:, None], torch.arange(Tn, device=x.device)[None, :]
            mask = torch.where((j <= i) & (j > i - k), 0.0, float("-inf"))
            return net.history_out(net.history_seq(x, mask=mask))
        net._history = h


print(f"결정 {len(T)} (비교 가능 {int(valid.sum())}) | 초반 = 판 안 0~9번째 결정 {int((pos < 10).sum())}, 후반 = 30번째 이후 {int((pos >= 30).sum())}")
print(f"{'히스토리':10s} {'소프트CE':>8s} {'일치율':>6s} | {'argmax regret 전체':>24s} | {'초반':>7s} {'후반':>7s}")
base = None
for k in ("full", 16, 8, 4, 2, 1, "zero"):
    patch(k)
    P = np.concatenate([L.infer(net, d, dev) for d in sets])
    ce = float(-(T * np.log(np.clip(P, 1e-9, 1))).sum(1).mean()); agree = float((P.argmax(1) == T.argmax(1)).mean())
    reg_s, _, _, _, _, _ = R.regrets(P, q, rel, qstar, T)
    r = np.where(valid, reg_s, np.nan)
    m, lo, hi = R.boot_ci(r, battle, rng, 300)
    early, late = np.nanmean(r[pos < 10]), np.nanmean(r[pos >= 30])
    if base is None:
        base = (m, early, late)
    print(f"{('최근 ' + str(k) + '턴') if isinstance(k, int) else ('전체 (원래)' if k == 'full' else '0 (끔)'):10s} {ce:8.4f} {agree:6.3f} | {m:.4f} [{lo:.4f},{hi:.4f}] ({m - base[0]:+.4f}) | {early:.4f} {late:.4f}")
