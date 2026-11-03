"""국면 평가 포텐셜 Φ(src/core/battle_eval.py)가 보상 설계에 쓸 만한지 오프라인 검증 (학습 없음).
데이터: Foul Play(300ms) 대전에서 학생이 둔 결정마다 FP가 다시 매긴 행동별 가치(Q)가 붙은 npz (lab_stu.npz)와, 그 npz를 만든 FP 결정 기록.
1) 상태 Φ(s_t)가 그 판의 승패를 맞히는 정도(AUC): Φ(지금 코드) / Φ(npz에 저장돼 있던 옛 코드) / 학습된 가치망 V / FP 탐색 가치
2) k턴 뒤 차이 Φ(s_{t+k}) - Φ(s_t)가 FP가 매긴 '고른 행동의 이득'과 얼마나 같이 움직이는가 (adv_mean = Q(고른 행동) - sum(pi*Q), adv_star = Q(고른 행동) - Q*,
   믿을 수 있는 행동(방문 >= 100) 위), 같은 검사를 V 차이/FP 가치 차이에도 해서 비교. 지연(k)을 얼마나 줄지 정하는 근거
Φ는 npz에서 value_target = γ^(남은 결정 수)·승패 - Φ 로 복원하되, 지금 코드의 Φ는 기록을 다시 재생해서(build_fp_dataset.build) 새로 계산 (결과는 --cache에 저장).
사용: python tools/phi_check.py [--set data/winrate_v6_fp300/lab_stu.npz --decisions "data/winrate_v6_fp300/stu/lab_rec_*.jsonl"]"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

GAMMA, MIN_VIS, KS = 0.995, 100, (1, 2, 3, 5, 8, 12)


def phi_from_value_target(d):
    """npz의 value_target = γ^(남은 결정 수)·승패 - Φ → Φ"""
    out = np.empty(len(d["value_target"]), np.float64)
    s = 0
    for n, o in zip(d["lens"], d["outcome"]):
        out[s:s + n] = GAMMA ** np.arange(n - 1, -1, -1) * float(o) - d["value_target"][s:s + n]
        s += n
    return out


def rebuild_phi(decisions, d, cache):
    """FP 결정 기록을 다시 재생해 지금 코드의 Φ를 계산 (npz와 같은 순서여야 함: 길이와 FP 가치가 같은지 확인)"""
    if os.path.exists(cache):
        return np.load(cache)
    from src.training.build_fp_dataset import build, load_jsonl
    seqs, _, _ = build(load_jsonl(decisions), workers=8)
    lens = np.array([len(s["action_mask"]) for s in seqs])
    tv = np.concatenate([s["teacher_value"] for s in seqs])
    if len(lens) != len(d["lens"]) or (lens != d["lens"]).any() or not np.allclose(tv, d["teacher_value"], atol=1e-6):
        raise SystemExit("❌ 다시 만든 결정이 npz와 순서/내용이 다름: --decisions가 이 npz를 만든 기록이 맞는지 확인")
    phi = np.concatenate([GAMMA ** np.arange(len(s["action_mask"]) - 1, -1, -1) * s["outcome"] - s["value_target"] for s in seqs])
    np.save(cache, phi)
    return phi


def model_values(path, d):
    from src.core.model import model_from_ckpt
    from src.training import train_fp_distill as T
    from src.training.replay_dataset import OBS_KEYS
    from ablate_inputs import load_set
    dd, off, _ = load_set(args.set)
    model = model_from_ckpt(path).to(T.DEVICE).eval()
    vs = []
    with torch.no_grad():
        for b in T.make_batches(dd, off, list(range(len(dd["lens"]))), T.BATCH_BATTLES, False):
            b = {k: v.to(T.DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}
            out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
            vs.append(out["value"].view(-1).float().cpu().numpy())
    return np.concatenate(vs).astype(np.float64)


def shifted(x, lens, k):
    """각 결정의 k칸 뒤 값 (배틀 끝을 넘으면 그 판의 마지막 값)"""
    out, s = np.empty_like(x), 0
    for n in lens:
        idx = np.minimum(np.arange(n) + k, n - 1)
        out[s:s + n] = x[s + idx]
        s += n
    return out


def auc(score, win):
    r = rankdata(score)
    n1, n0 = int(win.sum()), int((~win).sum())
    return (r[win].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(fn, battle, rng, n=200):
    """판 단위 부트스트랩 95% 구간"""
    ub = np.unique(battle)
    idx = {u: np.where(battle == u)[0] for u in ub}
    vals = [fn(np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def main():
    d = {k: v for k, v in np.load(args.set).items()}
    lens, N, nb = d["lens"], len(d["target"]), len(d["lens"])
    battle = np.repeat(np.arange(nb), lens)
    win = np.repeat(d["outcome"] > 0, lens)
    pos = np.concatenate([np.arange(n) / max(1, n - 1) for n in lens])      # 판 안에서의 진행도 0~1
    phi_new = rebuild_phi(args.decisions, d, args.cache).astype(np.float64)
    phi_old = phi_from_value_target(d)
    v = model_values(args.model, d)
    tv = d["teacher_value"].astype(np.float64) * 2 - 1
    rng = np.random.default_rng(0)

    print(f"결정 {N} / 판 {nb} (학생 승 {int((d['outcome'] > 0).sum())})")
    print("\n##### 1) 상태가 승패를 맞히는 정도 (AUC, 판 단위 부트스트랩 95%)")
    cand = {"Φ 지금 코드": phi_new, "Φ 옛 코드(npz 저장분)": phi_old, f"가치망 V({os.path.basename(args.model)})": v, "FP 탐색 가치": tv}
    print(f"{'추정량':34s} {'전체':>20s} | 판 초반(<1/3) {'중반':>8s} {'후반(>2/3)':>10s}")
    for name, x in cand.items():
        a = auc(x, win)
        lo, hi = boot(lambda i: auc(x[i], win[i]), battle, rng)
        ph = [auc(x[m], win[m]) for m in (pos < 1 / 3, (pos >= 1 / 3) & (pos <= 2 / 3), pos > 2 / 3)]
        print(f"{name:34s} {a:.3f} [{lo:.3f},{hi:.3f}] | {ph[0]:.3f}         {ph[1]:.3f}    {ph[2]:.3f}")

    q, m = d["q_mean"].astype(np.float64), d["action_mask"]
    rel = (d["q_vis"] >= MIN_VIS) & ~np.isnan(q) & m
    a = d["action_taken"]
    ai = np.clip(a, 0, 21)
    ok = (a >= 0) & rel[np.arange(N), ai] & (rel.sum(1) >= 2)
    pi = np.where(rel, d["target"], 0.0)
    psum = pi.sum(1)
    ok &= psum > 0
    adv_mean = q[np.arange(N), ai] - (pi * np.where(rel, q, 0.0)).sum(1) / np.where(psum > 0, psum, 1)
    adv_star = q[np.arange(N), ai] - np.where(rel, q, -np.inf).max(1)
    harm, best = adv_star <= -0.05, adv_star >= -0.01
    print(f"\n##### 2) 지연된 점수 차이와 FP가 매긴 행동 이득의 상관 (비교 가능한 결정 {int(ok.sum())}개: 믿을 행동 2개 이상, 고른 행동의 Q가 믿을 만함)")
    print(f"{'추정량':34s} | {'r(adv_mean)':>11s} {'95% 구간':>17s} | {'r(adv_star)':>11s} | {'d(해로움-최선)':>13s}")

    def row(name, f):
        f, am, ast, bi = f[ok], adv_mean[ok], adv_star[ok], battle[ok]
        r = np.corrcoef(f, am)[0, 1]
        lo, hi = boot(lambda i: np.corrcoef(f[i], am[i])[0, 1], bi, rng)
        g, nbst = ast <= -0.05, ast >= -0.01
        dd = (f[g].mean() - f[nbst].mean()) / np.sqrt((f[g].var() + f[nbst].var()) / 2)
        print(f"{name:34s} | {r:+11.3f} [{lo:+.3f},{hi:+.3f}] | {np.corrcoef(f, ast)[0, 1]:+11.3f} | {dd:+13.3f}")

    for k in KS:
        row(f"Φ 지금 코드: Φ(t+{k}) - Φ(t)", shifted(phi_new, lens, k) - phi_new)
    for k in (1, 3, 8):
        row(f"Φ 옛 코드:   Φ(t+{k}) - Φ(t)", shifted(phi_old, lens, k) - phi_old)
    for k in (1, 3):
        row(f"가치망 V:    V(t+{k}) - V(t)", shifted(v, lens, k) - v)
        row(f"FP 탐색 가치: TV(t+{k}) - TV(t)", shifted(tv, lens, k) - tv)
    print(f"\n(참고: 해로운 선택(adv_star <= -0.05) {int((harm & ok).sum())}개, 거의 최선(>= -0.01) {int((best & ok).sum())}개, adv_mean 표준편차 {adv_mean[ok].std():.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="data/winrate_v6_fp300/lab_stu.npz")
    ap.add_argument("--decisions", default="data/winrate_v6_fp300/stu/lab_rec_*.jsonl")
    ap.add_argument("--model", default="checkpoints/supervised_v2_fp_v8_full.pt")
    ap.add_argument("--cache", default="data/winrate_v6_fp300/phi_new.npy")
    args = ap.parse_args()
    main()
