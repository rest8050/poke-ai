"""regret 평가: 학생이 고른 행동이 교사 탐색이 본 행동별 가치(Q)로 얼마나 손해인가.
일치율은 교사 상위 두 수가 비슷한 결정에서 어긋나도 감점이라, 어긋남이 진짜 손해인지 가치 손실로 다시 잰다 (점수는 0~1 승률 척도).
q_mean/q_vis가 든 npz만 쓸 수 있음 (build_fp_dataset 출력. 옛 파일은 원본 결정기록으로 재빌드해야 함).
정의: 방문 수 >= --min-vis 인 행동만 믿을 수 있는 Q. 결정마다 Q* = 믿을 수 있는 행동 중 최대 Q (믿을 행동이 2개 미만이면 제외).
  학생 argmax 행동 a의 regret = Q* - Q_a (a가 믿을 수 없는 행동이면 '지원 밖'으로 따로 셈), 평균 정책 regret = 믿을 수 있는 행동 위에서 정책 평균 (지원 밖 질량은 따로)
Q 추정에는 탐색 잡음이 있어 (방문 N이면 표준오차 ~0.5/sqrt(N)) 작은 regret 차이는 잡음 → 교사 자신의 argmax(탐색 분포 최상위) regret을 기준선으로 같이 출력.
사용: python tools/regret_eval.py --models v6_full,v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz   (모델 = checkpoints/supervised_v2_fp_<이름>.pt, 또는 체크포인트 경로)"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

TIE, HARM = 0.02, 0.05        # regret이 이 미만이면 사실상 동점, 이 이상이면 해로운 선택
KINDS = (("기술", slice(0, 4)), ("테라 기술", slice(4, 8)), ("교체", slice(8, 14)))


def boot_ci(x, battle, rng, n=1000):
    """판 단위 부트스트랩 평균과 95% 구간 (x의 NaN은 빼고 센다)"""
    ok = ~np.isnan(x)
    ub, inv = np.unique(battle[ok], return_inverse=True)
    s, c = np.bincount(inv, weights=x[ok]), np.bincount(inv)
    ids = rng.randint(0, len(ub), (n, len(ub)))
    m = s[ids].sum(1) / c[ids].sum(1)
    return s.sum() / c.sum(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def regrets(P, q, rel, qstar, T):
    """결정마다: 학생 argmax regret(지원 밖이면 NaN), 교사 argmax regret, 평균 정책 regret, 지원 밖 질량"""
    qf = np.where(rel, q, 0.0)
    gap = np.where(rel, qstar[:, None] - qf, 0.0)
    a_s, a_t, ar = P.argmax(1), T.argmax(1), np.arange(len(P))
    mass = (P * rel).sum(1)
    reg_s = np.where(rel[ar, a_s], gap[ar, a_s], np.nan)
    reg_t = np.where(rel[ar, a_t], gap[ar, a_t], np.nan)
    reg_pol = (P * gap).sum(1) / np.maximum(mass, 1e-9)
    return reg_s, reg_t, reg_pol, 1 - mass, a_s, a_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v6_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--min-vis", type=float, default=100, help="이 방문 수 이상인 행동의 Q만 믿음")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = args.sets.split(",")
    sets, qs, vs = [], [], []
    for p in paths:
        z = np.load(p)
        if "q_mean" not in z.files:
            raise SystemExit(f"❌ {p}에 q_mean이 없음: 원본 결정기록으로 build_fp_dataset.py를 다시 돌려야 함")
        sets.append(L.load_set(p)); qs.append(z["q_mean"]); vs.append(z["q_vis"])
    q, vis = np.concatenate(qs), np.concatenate(vs)
    T = np.concatenate([d["target"] for d in sets]).astype(np.float64)
    M = np.concatenate([d["action_mask"] for d in sets])
    off = np.cumsum([0] + [len(d["lens"]) for d in sets])
    battle = np.concatenate([np.repeat(np.arange(len(d["lens"])), d["lens"]) + o for d, o in zip(sets, off)])
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    spread = np.where(valid, qstar - np.where(rel, q, np.inf).min(1), np.nan)
    print(f"결정 {len(T)} 중 비교 가능(믿을 수 있는 행동 2개 이상) {int(valid.sum())} | 방문 >= {args.min_vis:g} | 이 결정들의 Q* - 최저 Q 평균 {np.nanmean(spread):.3f}")
    rng = np.random.RandomState(0)
    res = {}
    for name in args.models.split(","):
        net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
        P = np.concatenate([L.infer(net, d, dev) for d in sets])
        reg_s, reg_t, reg_pol, off_mass, a_s, a_t = regrets(P, q, rel, qstar, T)
        res[name] = np.where(valid, reg_s, np.nan)
        v = valid
        rs, rt = reg_s[v], reg_t[v]
        dis = (a_s != a_t)[v]
        in_sup = ~np.isnan(rs)
        print(f"\n===== {name} =====")
        m, lo, hi = boot_ci(np.where(valid, reg_s, np.nan), battle, rng)
        mt, lot, hit = boot_ci(np.where(valid, reg_t, np.nan), battle, rng)
        mp, lop, hip = boot_ci(np.where(valid, reg_pol, np.nan), battle, rng)
        print(f"학생 argmax regret 평균 {m:.4f} [{lo:.4f},{hi:.4f}] | 교사 argmax 기준선 {mt:.4f} [{lot:.4f},{hit:.4f}] | 평균 정책 regret {mp:.4f} [{lop:.4f},{hip:.4f}]")
        print(f"학생 argmax: 동점(<{TIE}) {np.mean(rs[in_sup] < TIE):.3f} | 해로움(>={HARM}) {np.mean(rs[in_sup] >= HARM):.3f} | 지원 밖 선택 {np.mean(~in_sup):.3f} | 교사 argmax 해로움 {np.nanmean(rt >= HARM):.3f}")
        print(f"교사 argmax와 불일치 {dis.mean():.3f} (일치율 {1 - dis.mean():.3f}) | 불일치 중 동점 {np.mean(rs[dis & in_sup] < TIE):.3f}, 해로움 {np.mean(rs[dis & in_sup] >= HARM):.3f}, 지원 밖 {np.mean(~in_sup[dis]):.3f}")
        print(f"{'교사 선택 종류':14s} {'결정':>6s} | {'불일치':>6s} {'평균 regret':>11s} {'해로움':>7s}")
        tk = np.where(a_t < 4, 0, np.where(a_t < 8, 1, 2))[v]
        for k, (kn, _) in enumerate(KINDS):
            s = (tk == k)
            if s.sum() < 50:
                continue
            si = s & in_sup
            print(f"{kn:14s} {int(s.sum()):6d} | {dis[s].mean():6.3f} {np.mean(rs[si]):11.4f} {np.mean(rs[si] >= HARM):7.3f}")
        conf = T.max(1)[v]
        for cn, sel in (("교사 확신 >= 0.8", conf >= 0.8), ("교사 확신 < 0.8", conf < 0.8)):
            si = sel & in_sup
            print(f"{cn:14s} {int(sel.sum()):6d} | {dis[sel].mean():6.3f} {np.mean(rs[si]):11.4f} {np.mean(rs[si] >= HARM):7.3f}")
    names = list(res)
    if len(names) == 2:
        a, b = res[names[0]], res[names[1]]
        m, lo, hi = boot_ci(b - a, battle, rng)
        print(f"\n쌍 비교 {names[1]} - {names[0]} regret 평균 차이 {m:+.4f} [{lo:+.4f},{hi:+.4f}]  (음수 = {names[1]}이 손해가 적음, 둘 다 지원 안인 결정만)")
        tl = M[:, 4:8].any(1)      # 테라가 합법인 결정만
        m, lo, hi = boot_ci(np.where(tl, b - a, np.nan), battle, rng)
        print(f"  테라 합법 결정({int((tl & valid).sum())}개)만: {m:+.4f} [{lo:+.4f},{hi:+.4f}] | 평균 regret {names[0]} {np.nanmean(np.where(tl, a, np.nan)):.4f}, {names[1]} {np.nanmean(np.where(tl, b, np.nan)):.4f}")


if __name__ == "__main__":
    main()
