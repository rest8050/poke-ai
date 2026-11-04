"""FP-Q 비판자 오프라인 관문 (롤아웃 없음). v8_full을 얼리고 그 latent 위에 "FP 탐색이 매긴 행동별 가치 Q(s,a)"를 예측하는 헤드만 학습한 뒤,
학습에 안 쓴 FP 대전 기록(lab_stu.npz)에서 tools/phi_check.py와 같은 정의로 재서 기존 추정량(승패 이점 -0.04, Φ 차이 +0.03, FP 가치 변화 +0.05, 학습된 V 차이 0)과 비교한다.
  r(adv_mean): 추정 이점 f(s, 고른 행동)과 FP가 매긴 이득 adv_mean = Q(고른 행동) - sum(pi_FP * Q) 의 상관 (믿을 행동 = 방문 >= 100)
  f 후보: (1) 비판자 Q^(s,a) - sum(pi_학생 * Q^) (2) 학생 정책 자체에서 나온 logpi(a) - sum(pi * logpi) (비판자가 정책에 없는 정보를 더하는지 보는 기준) (3) 비판자 Q^ - 믿을 행동의 평균
학습: 학생이 방문한 상태의 FP 라벨(d7_tr, d6_tr, dg2_tr), 검증: d7_ho + d6_ho, 시험: lab_stu (+ d7_ho)
사용: python tools/critic_gate.py"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from ablate_inputs import load_set
from src.core.model import model_from_ckpt
from src.training import train_fp_distill as T
from src.training.replay_dataset import OBS_KEYS

MIN_VIS = T.Q_MIN_VIS
DEV = T.DEVICE


@torch.no_grad()
def extract(model, path):
    d, off, _ = load_set(path)
    Z, LP, V = [], [], []
    for b in T.make_batches(d, off, list(range(len(d["lens"]))), T.BATCH_BATTLES, False):
        b = {k: v.to(DEV) if torch.is_tensor(v) else v for k, v in b.items()}
        out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
        Z.append(out["latent"].float().cpu()); LP.append(F.log_softmax(out["policy_logits"].float(), -1).cpu()); V.append(out["value"].float().view(-1).cpu())
    return {"Z": torch.cat(Z), "LP": torch.cat(LP), "V": torch.cat(V), "mask": torch.from_numpy(d["action_mask"]), "q": torch.from_numpy(d["q_mean"]).float(),
            "vis": torch.from_numpy(d["q_vis"]).float(), "a": torch.from_numpy(d["action_taken"]), "pi": torch.from_numpy(d["target"]).float(), "lens": d["lens"], "outcome": torch.from_numpy(d["outcome"]).float()}


def cat(parts):
    return {k: (np.concatenate([p[k] for p in parts]) if k == "lens" else torch.cat([p[k] for p in parts])) for k in parts[0]}


class QHead(nn.Module):
    def __init__(self, din, hid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 22))

    def forward(self, x):
        return self.net(x)


def feats(S, variant):
    return S["Z"] if variant == "latent" else torch.cat([S["Z"], S["LP"].clamp(min=-12)], 1)


def rel_of(S):
    return (S["vis"] >= MIN_VIS) & ~torch.isnan(S["q"]) & S["mask"]


def loss_fn(pred, S, idx):
    rel = rel_of({k: S[k][idx] for k in ("vis", "q", "mask")}).to(DEV)
    q = torch.nan_to_num(S["q"][idx]).to(DEV)
    n = rel.sum(1, keepdim=True).clamp_min(1)
    adv_t = (q - (q * rel).sum(1, keepdim=True) / n) * rel
    adv_p = (pred - (pred * rel).sum(1, keepdim=True) / n) * rel
    tot = rel.sum().clamp_min(1)
    return (((pred - q) ** 2) * rel).sum() / tot + 3 * ((adv_p - adv_t) ** 2).sum() / tot


def train_head(tr, va, variant, epochs=14, bs=4096):
    X, Xv = feats(tr, variant), feats(va, variant)
    head = QHead(X.size(1)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * ((len(X) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    best, state = 1e9, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = loss_fn(head(X[idx].to(DEV)), tr, idx)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        head.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_fn(head(Xv[i:i + bs].to(DEV)), va, torch.arange(i, min(i + bs, len(Xv))))) for i in range(0, len(Xv), bs)]))
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in head.state_dict().items()}
        print(f"  [{variant}] 에폭 {ep + 1}/{epochs} 검증 손실 {vl:.5f}", flush=True)
    head.load_state_dict(state)
    return head.eval(), best


def boot(fn, battle, rng, n=300):
    ub = np.unique(battle)
    idx = {u: np.where(battle == u)[0] for u in ub}
    vals = [fn(np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def evaluate(name, S, preds, rng):
    """phi_check.py와 같은 정의의 상관 + 비판자 자체의 품질 (argmax 일치, regret, 오차)"""
    q, vis, m, a = S["q"].numpy().astype(np.float64), S["vis"].numpy(), S["mask"].numpy(), S["a"].numpy()
    N = len(q)
    lens = S["lens"]
    battle = np.repeat(np.arange(len(lens)), lens)
    rel = (vis >= MIN_VIS) & ~np.isnan(q) & m
    ai, ar = np.clip(a, 0, 21), np.arange(N)
    ok = (a >= 0) & rel[ar, ai] & (rel.sum(1) >= 2)
    pi = np.where(rel, S["pi"].numpy(), 0.0)
    psum = pi.sum(1)
    ok &= psum > 0
    qz = np.where(rel, q, 0.0)
    adv_mean = q[ar, ai] - (pi * qz).sum(1) / np.where(psum > 0, psum, 1)
    adv_star = q[ar, ai] - np.where(rel, q, -np.inf).max(1)
    harm, best = adv_star <= -0.05, adv_star >= -0.01
    ps = np.where(rel, np.exp(S["LP"].numpy().astype(np.float64)), 0.0)
    ps = ps / np.where(ps.sum(1) > 0, ps.sum(1), 1)[:, None]
    lp = np.where(rel, S["LP"].numpy().astype(np.float64), 0.0)
    cands = {"정책 자체: logπ(a) - Σπ·logπ": lp[ar, ai] - (ps * lp).sum(1)}
    for pname, Qh in preds.items():
        Qh = Qh.astype(np.float64)
        cands[f"{pname}: Q^(a) - Σπ학생·Q^"] = Qh[ar, ai] - (ps * np.where(rel, Qh, 0.0)).sum(1)
        cands[f"{pname}: Q^(a) - 믿을 행동 평균"] = Qh[ar, ai] - np.where(rel, Qh, 0.0).sum(1) / np.maximum(rel.sum(1), 1)
    print(f"\n##### {name}: 결정 {N}, 판 {len(lens)}, 비교 가능 {int(ok.sum())}개 (adv_mean 표준편차 {adv_mean[ok].std():.4f}, 해로운 {int((harm & ok).sum())}, 거의 최선 {int((best & ok).sum())})")
    print(f"{'추정량':44s} | {'r(adv_mean)':>11s} {'95% 구간':>17s} | {'r(adv_star)':>11s} | {'d(해로움-최선)':>13s}")
    res = {}
    for k, f in cands.items():
        fo, am, bi, ast = f[ok], adv_mean[ok], battle[ok], adv_star[ok]
        r = np.corrcoef(fo, am)[0, 1]
        lo, hi = boot(lambda i: np.corrcoef(fo[i], am[i])[0, 1], bi, rng)
        g, nb = ast <= -0.05, ast >= -0.01
        dd = (fo[g].mean() - fo[nb].mean()) / np.sqrt((fo[g].var() + fo[nb].var()) / 2)
        res[k] = (r, lo, hi)
        print(f"{k:44s} | {r:+11.3f} [{lo:+.3f},{hi:+.3f}] | {np.corrcoef(fo, ast)[0, 1]:+11.3f} | {dd:+13.3f}")
    # 비판자를 그대로 정책으로 쓰면: 믿을 행동 안에서 argmax가 FP와 같은가 / regret
    two = rel.sum(1) >= 2
    qstar = np.where(rel, q, -np.inf).max(1)
    pol_pick = np.where(rel, S["LP"].numpy(), -np.inf).argmax(1)
    print(f"{'행동 선택 품질 (믿을 행동 2개 이상 %d개)' % int(two.sum()):44s} | {'argmax=FP':>9s} {'regret':>8s} {'Q 평균오차':>10s}")
    print(f"{'학생 정책 argmax':44s} | {(pol_pick == np.where(rel, q, -np.inf).argmax(1))[two].mean():9.3f} {(qstar - q[ar, pol_pick])[two].mean():8.4f}")
    for pname, Qh in preds.items():
        pick = np.where(rel, Qh, -np.inf).argmax(1)
        print(f"{'비판자 ' + pname + ' argmax':44s} | {(pick == np.where(rel, q, -np.inf).argmax(1))[two].mean():9.3f} {(qstar - q[ar, pick])[two].mean():8.4f} {np.abs(Qh - q)[rel].mean():10.4f}")
    return res


def main(args):
    model = model_from_ckpt(args.model).to(DEV).eval()
    sets = {}
    for tag, paths in (("train", args.train.split(",")), ("val", args.val.split(",")), ("test", [args.test]), ("d7_ho", ["data/dagger/d7_ho.npz"])):
        sets[tag] = cat([extract(model, p) for p in paths])
        print(f"{tag}: 결정 {len(sets[tag]['Z'])}, 판 {len(sets[tag]['lens'])}", flush=True)
    rng = np.random.RandomState(0)
    preds = {"test": {}, "d7_ho": {}}
    for variant in ("latent", "latent+정책"):
        head, _ = train_head(sets["train"], sets["val"], "latent" if variant == "latent" else "x")
        for tag in preds:
            with torch.no_grad():
                preds[tag][f"비판자[{variant}]"] = head(feats(sets[tag], "latent" if variant == "latent" else "x").to(DEV)).cpu().numpy()
    evaluate("시험 lab_stu (학습에 안 쓴 FP 대전, 기준 표와 같은 세트)", sets["test"], preds["test"], rng)
    evaluate("참고 d7_ho (학생이 방문한 상태의 검증 세트)", sets["d7_ho"], preds["d7_ho"], rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/supervised_v2_fp_v8_full.pt")
    ap.add_argument("--train", default="data/dagger/d7_tr.npz,data/dagger/d6_tr.npz,data/dagger/dg2_tr.npz")
    ap.add_argument("--val", default="data/dagger/d6_ho.npz")
    ap.add_argument("--test", default="data/winrate_v6_fp300/lab_stu.npz")
    main(ap.parse_args())
