"""regret을 종류(기술/교체) 수준 실수와 종류 안 선택 실수로 분해: 어디가 병목인지 (교체 경로 가설 검증). 사용: python tools/type_regret.py <모델 이름|경로> ... (학생 상태 홀드아웃 d7_ho,d6_ho,dg5)"""
import sys, os
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import regrets
from src.core.model import model_from_ckpt
sets_p = "data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz".split(",")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = [L.load_set(p) for p in sets_p]; zs = [np.load(p) for p in sets_p]
q = np.concatenate([z["q_mean"] for z in zs]); vis = np.concatenate([z["q_vis"] for z in zs])
d = {k: np.concatenate([s[k] for s in sets]) for k in sets[0] if k != "lens"}
T, M = d["target"].astype(np.float64), d["action_mask"]
rel = (vis >= 100) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)

kind = np.array([0]*8 + [1]*6 + [2]*8)
ar = np.arange(len(q)); qf = np.where(rel, q, 0.0); gap = np.where(rel, qstar[:, None] - qf, 0.0)
kbest = kind[np.where(rel, q, -np.inf).argmax(1)]            # 최선 행동의 종류
for name in sys.argv[1:]:
    net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
    P = np.concatenate([L.infer(net, s, dev) for s in sets])
    reg_s, *_ , a_s, a_t = regrets(P, q, rel, qstar, T)
    ok = valid & ~np.isnan(reg_s)
    r = np.where(ok, reg_s, 0.0); tot = r.sum()
    ks = kind[a_s]
    cat = {"종류 맞음, 안에서 틀림(기술끼리)": (ks == kbest) & (kbest == 0), "종류 맞음, 안에서 틀림(교체 후보끼리)": (ks == kbest) & (kbest == 1),
           "교체해야 하는데 기술(덜 교체)": (ks == 0) & (kbest == 1), "기술해야 하는데 교체(더 교체)": (ks == 1) & (kbest == 0)}
    print(f"\n== {name}: 결정 {int(ok.sum())} 평균 regret {r.sum()/ok.sum():.4f}")
    for n, m in cat.items():
        m = m & ok
        print(f"  {n:32s} 결정 {m.sum():6d} ({m.mean() if False else m.sum()/ok.sum():.3f}) regret 몫 {r[m].sum()/tot:.3f}  평균 {r[m].mean() if m.sum() else 0:.4f}")
    # 가상: 종류를 최선 종류로 고정하고 학생이 그 종류 안에서 고르면 (종류 오라클) / 학생 종류를 유지하고 그 안에서 최선을 고르면 (안 선택 오라클)
    pk = np.stack([(P * (kind == k)).sum(1) for k in range(2)], 1)
    Pt = np.where((kind[None] == kbest[:, None]) & rel, P, -1); a_to = Pt.argmax(1)
    okt = valid & rel[ar, a_to]; rt = np.where(okt, gap[ar, a_to], np.nan)
    gi = np.where((kind[None] == ks[:, None]) & rel, gap, np.inf).min(1)
    gi = np.where(np.isfinite(gi), gi, np.nan)
    print(f"  종류 오라클(종류를 최선으로 고정, 안은 학생 선택): regret {np.nanmean(np.where(okt, rt, np.nan)):.4f}  (원래 {r.sum()/ok.sum():.4f})")
    print(f"  안 오라클(학생이 고른 종류 유지, 안은 최선): regret {np.nanmean(np.where(ok, gi, np.nan)):.4f}")
    # 종류 먼저 디코딩: 종류 = argmax(P(기술), P(교체)) 후 그 종류 안에서 argmax (행동 전체 argmax와 비교)
    pt = np.stack([P[:, :8].sum(1), P[:, 8:14].sum(1)], 1); tk = pt.argmax(1)
    inside = np.where(tk[:, None] == 1, P[:, :14] * (kind[None, :14] == 1), P[:, :14] * (kind[None, :14] == 0)); a_k = inside.argmax(1)
    okk = valid & rel[ar, a_k]
    print(f"  종류 먼저 디코딩 regret {gap[ar, a_k][okk].mean():.4f} | 학생 교체 비율 전체 argmax {np.mean(kind[a_s][ok] == 1):.3f} 종류 먼저 {np.mean(kind[a_k][okk] == 1):.3f} 최선 행동이 교체인 비율 {np.mean(kbest[ok] == 1):.3f}")
    # 교체가 최선인 결정에서 교체 총확률 평균 / 기술이 최선인 결정에서 교체 총확률 평균
    print(f"  P(교체) 평균: 최선이 교체 {pt[ok & (kbest == 1), 1].mean():.3f}, 최선이 기술 {pt[ok & (kbest == 0), 1].mean():.3f}")
