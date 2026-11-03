"""TD 비판자 오프라인 관문: FP 탐색 Q 라벨 없이 승패(±1)만으로 Q(s,a)를 TD로 학습하고(얼린 v8_full latent 위 헤드, 목표망 + 학생 정책 기댓값 부트스트랩),
critic_gate.py와 같은 평가(FP가 매긴 이득 adv_mean과의 상관, 정책 logπ 기준선)로 잰다. 헤드 K=1(시드별 개별)과 K개 평균 앙상블을 같은 데이터에서 비교.
참고 상한: critic_gate.py는 FP Q를 직접 라벨로 학습하고도 시험 r +0.028(latent) / +0.130(latent+정책) -> 승패만으로 배우는 TD 비판자가 이보다 나을 근거는 약함
학습: --train (학생이 방문한 상태, 판 단위 연속 결정), 검증: --val, 시험: --test + d7_ho
사용: python tools/td_critic_gate.py [--k 4] [--steps 4000]"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from critic_gate import DEV, MIN_VIS, QHead, boot, cat, evaluate, extract, feats
from src.core.model import model_from_ckpt


def prep(S, variant, gamma):
    """결정별 (입력, 고른 행동, 다음 결정 위치, 종료 여부, 정책 분포, MC 반환)"""
    lens = S["lens"]
    N = len(S["Z"])
    ends = np.cumsum(lens) - 1
    term = np.zeros(N, bool)
    term[ends] = True
    left = np.concatenate([np.arange(n - 1, -1, -1) for n in lens])
    out = np.repeat(S["outcome"].numpy(), lens)
    pol = torch.exp(S["LP"].float()) * S["mask"].float()
    pol = pol / pol.sum(1, keepdim=True).clamp_min(1e-9)
    a = S["a"]
    ok = (a >= 0) & S["mask"][torch.arange(N), a.clamp(min=0)]
    return dict(X=feats(S, variant).to(DEV), a=a.clamp(min=0).to(DEV), ok=torch.from_numpy(np.where(ok.numpy())[0]).to(DEV),
                nxt=torch.from_numpy(np.minimum(np.arange(N) + 1, N - 1)).to(DEV), term=torch.from_numpy(term).to(DEV), pol=pol.to(DEV),
                out=torch.from_numpy(out).float().to(DEV), mc=torch.from_numpy((gamma ** left) * out).float().to(DEV))


def train_head(tr, va, seed, gamma, steps, bs=4096, tau=0.01):
    torch.manual_seed(seed)
    head = QHead(tr["X"].size(1)).to(DEV)
    tgt = copy.deepcopy(head)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    best, state = 1e9, None
    for step in range(steps):
        i = tr["ok"][torch.randint(len(tr["ok"]), (bs,), device=DEV)]
        q = head(tr["X"][i]).gather(1, tr["a"][i][:, None]).squeeze(1)
        with torch.no_grad():
            v_next = (tr["pol"][tr["nxt"][i]] * tgt(tr["X"][tr["nxt"][i]])).sum(1)
            y = torch.where(tr["term"][i], tr["out"][i], gamma * v_next)
        loss = ((q - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pt, ph in zip(tgt.parameters(), head.parameters()):
                pt.lerp_(ph, tau)
        if (step + 1) % 250 == 0:
            with torch.no_grad():   # 선택 기준은 목표 없이 안정적인 MC 반환 오차 (검증 판)
                vl = float(((head(va["X"][va["ok"]]).gather(1, va["a"][va["ok"]][:, None]).squeeze(1) - va["mc"][va["ok"]]) ** 2).mean())
            if vl < best:
                best, state = vl, copy.deepcopy(head.state_dict())
            print(f"    seed {seed} 스텝 {step + 1}/{steps} TD 손실 {loss.item():.4f} 검증 MC 오차 {vl:.4f}", flush=True)
    head.load_state_dict(state)
    return head.eval()


def outcome_eval(name, S, Qs, gamma, rng):
    """FP의 Q를 정답으로 쓰지 않는 판정: 실제 결과만으로 비판자가 정책/가치헤드가 모르는 행동 정보를 갖는지 본다.
    G = γ^(남은 결정)·승패. (1) 예측 오차: 상수 / v8_full 가치헤드 V(s) / 비판자 Q(s,a고른) 의 G에 대한 MSE
    (2) 이점 A = Q(a) - Σπ학생 Q 와 실제 잔차 G - V(s) 의 상관 (95% 구간은 판 단위 부트스트랩). 기준: 정책 이점 logπ(a)-Σπ logπ, FP가 매긴 이득 adv_mean"""
    lens, N = S["lens"], len(S["Z"])
    left = np.concatenate([np.arange(n - 1, -1, -1) for n in lens])
    G = (gamma ** left) * np.repeat(S["outcome"].numpy(), lens)
    battle = np.repeat(np.arange(len(lens)), lens)
    V = S["V"].numpy().astype(np.float64)
    a, mask = S["a"].numpy(), S["mask"].numpy()
    ar, ai = np.arange(N), np.clip(a, 0, 21)
    ok = (a >= 0) & mask[ar, ai]
    ps = np.exp(S["LP"].numpy().astype(np.float64)) * mask
    ps = ps / np.maximum(ps.sum(1, keepdims=True), 1e-9)
    lp = np.where(mask, S["LP"].numpy().astype(np.float64), 0.0)
    q, vis = S["q"].numpy().astype(np.float64), S["vis"].numpy()
    rel = (vis >= MIN_VIS) & ~np.isnan(q) & mask
    pi = np.where(rel, S["pi"].numpy(), 0.0)
    okf = ok & rel[ar, ai] & (rel.sum(1) >= 2) & (pi.sum(1) > 0)
    adv_fp = q[ar, ai] - (pi * np.where(rel, q, 0.0)).sum(1) / np.where(pi.sum(1) > 0, pi.sum(1), 1)
    print(f"\n##### 결과 기반 판정 {name}: 결정 {int(ok.sum())}, 판 {len(lens)} (G 분산 {G[ok].var():.3f})")
    print(f"{'G 예측 오차 MSE':40s} | {'MSE':>7s}   (낮을수록 좋음)")
    print(f"{'상수(평균)':40s} | {((G - G[ok].mean()) ** 2)[ok].mean():7.4f}")
    print(f"{'v8_full 가치헤드 V(s)':40s} | {((G - V) ** 2)[ok].mean():7.4f}")
    for k, Q in Qs.items():
        print(f"{k + ' Q(s,a고른)':40s} | {((G - Q[ar, ai]) ** 2)[ok].mean():7.4f}")
    resid = G - V
    cands = {"정책 이점 logπ(a)-Σπ·logπ": lp[ar, ai] - (ps * lp).sum(1)}
    for k, Q in Qs.items():
        cands[f"{k} A=Q(a)-Σπ·Q"] = Q[ar, ai] - (ps * Q).sum(1)
    print(f"{'이점 추정량':40s} | {'r(잔차 G-V)':>12s} {'95% 구간':>17s}")
    for k, f in cands.items():
        fo, rs, bi = f[ok], resid[ok], battle[ok]
        lo, hi = boot(lambda i: np.corrcoef(fo[i], rs[i])[0, 1], bi, rng, 200)
        print(f"{k:40s} | {np.corrcoef(fo, rs)[0, 1]:+12.3f} [{lo:+.3f},{hi:+.3f}]")
    fo, rs, bi = adv_fp[okf], resid[okf], battle[okf]
    lo, hi = boot(lambda i: np.corrcoef(fo[i], rs[i])[0, 1], bi, rng, 200)
    print(f"{'(참고) FP가 매긴 이득 adv_mean':40s} | {np.corrcoef(fo, rs)[0, 1]:+12.3f} [{lo:+.3f},{hi:+.3f}]  (FP의 이점조차 결과 잔차와 얼마나 닿는지 = 결과 신호의 상한)")


def main(args):
    model = model_from_ckpt(args.model).to(DEV).eval()
    sets = {}
    for tag, paths in (("train", args.train.split(",")), ("val", args.val.split(",")), ("test", [args.test]), ("d7_ho", ["data/dagger/d7_ho.npz"])):
        sets[tag] = cat([extract(model, p) for p in paths])
        print(f"{tag}: 결정 {len(sets[tag]['Z'])}, 판 {len(sets[tag]['lens'])}, 승률 {float((sets[tag]['outcome'] > 0).float().mean()):.3f}", flush=True)
    rng = np.random.RandomState(0)
    preds = {"test": {}, "d7_ho": {}}
    qmean = {"test": {}, "d7_ho": {}}
    for variant, vname in (("latent", "latent"), ("x", "latent+정책")):
        tr, va = prep(sets["train"], variant, args.gamma), prep(sets["val"], variant, args.gamma)
        heads = [train_head(tr, va, s, args.gamma, args.steps) for s in range(args.k)]
        for tag in preds:
            X = feats(sets[tag], variant).to(DEV)
            with torch.no_grad():
                qs = [h(X).cpu().numpy() for h in heads]
            for s, q in enumerate(qs[:args.k]):
                preds[tag][f"TD[{vname}] 헤드{s}"] = q
            preds[tag][f"TD[{vname}] 평균 K={args.k}"] = np.mean(qs, 0)
            qmean[tag][f"TD[{vname}] K={args.k}"] = np.mean(qs, 0)
            qmean[tag][f"TD[{vname}] K=1(헤드0)"] = qs[0]
            preds[tag][f"TD[{vname}] 헤드 불일치 크기"] = np.std(qs, 0)   # 정보 표시용 (Q가 아님: 이점 상관은 의미 없음)
    for tag, name in (("test", "시험 lab_stu"), ("d7_ho", "참고 d7_ho")):
        unc = preds[tag].pop("TD[latent] 헤드 불일치 크기"), preds[tag].pop("TD[latent+정책] 헤드 불일치 크기")
        print(f"\n(헤드 불일치 평균: latent {unc[0].mean():.4f}, latent+정책 {unc[1].mean():.4f})")
        evaluate(name, sets[tag], preds[tag], rng)
        outcome_eval(name, sets[tag], qmean[tag], args.gamma, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/supervised_v2_fp_v8_full.pt")
    ap.add_argument("--train", default="data/dagger/d7_tr.npz,data/dagger/d6_tr.npz,data/dagger/dg2_tr.npz")
    ap.add_argument("--val", default="data/dagger/d6_ho.npz")
    ap.add_argument("--test", default="data/winrate_v6_fp300/lab_stu.npz")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=0.995)
    main(ap.parse_args())
