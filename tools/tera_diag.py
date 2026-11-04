"""테라스탈 결정 진단: 테라가 합법인 결정을 "테라하면 방어가 얼마나 좋아지나 / 공격이 얼마나 좋아지나"로 나눠, 교사와 학생의 테라 비율과 regret을 비교한다.
방어 이득 = 상대 활성의 공개 기술이 내 활성에 주는 최대 데미지의 (테라 전 - 테라 후) (model_v8._tera_feat의 열 5 - 열 7, 매치업 규칙 모듈이 계산한 값),
공격 이득 = 내 기술 4칸 중 테라 후 데미지 증가 최대값 (열 3).
사용: python tools/tera_diag.py <모델|경로> [<모델|경로> ...]   (둘 이상이면 첫 모델 대비 부분집합 쌍 비교도 출력)
읽는 법: 교사 테라 비율이 방어 이득 구간에서 크게 오르는데 학생 비율이 평평하면 학생이 방어 테라를 못 읽는 것."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import boot_ci, regrets
from src.core.model import model_from_ckpt

SETS = "data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz".split(",")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load():
    sets = [L.load_set(p) for p in SETS]
    zs = [np.load(p) for p in SETS]
    d = {k: np.concatenate([s[k] for s in sets]) for k in sets[0] if k != "lens"}
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    lens = np.concatenate([s["lens"] for s in sets])
    return sets, d, q, vis, np.repeat(np.arange(len(lens)), lens)


def run_model(name, sets):
    """학생 확률 P [N,22]와 결정마다 tera_feat [N,4,9] (모델의 _tera_feat 반환값을 가로채 모음)"""
    net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
    store, orig, flag0 = [], net._encode_turn, net.need_tera

    def wrap(*a, **k):
        net.need_tera = True              # tera 수치를 안 쓰는 모델은 인코더가 그 값을 계산하지 않으므로 인코딩 동안만 켬 (헤드는 tera_feat/tera_gate 설정으로 읽음)
        try:
            enc = orig(*a, **k)
        finally:
            net.need_tera = flag0
        store.append(enc["tera_feat"].detach().float().cpu().numpy())
        return enc
    net._encode_turn = wrap
    P = np.concatenate([L.infer(net, s, dev) for s in sets])
    tf = np.concatenate(store)
    assert len(tf) == len(P), (len(tf), len(P))
    return P, tf


def classes(tf, legal_tera):
    """방어 이득/공격 이득으로 나눈 결정 종류 (테라 합법인 결정만)"""
    dd = tf[:, 0, 5] - tf[:, 0, 7]                  # 방어 이득: 상대 공개 기술 최대 데미지 감소
    dk = tf[:, 0, 6] - tf[:, 0, 8]                  # 방어 이득: 상대 공개 기술 최대 KO 확률 감소
    do = tf[:, :, 3].max(1)                         # 공격 이득: 테라 후 데미지 증가 최대
    D, O = dd >= 0.25, do >= 0.1
    return {"방어만(방어 +0.25 이상, 공격 +0.1 미만)": legal_tera & D & ~O, "공격만(공격 +0.1 이상, 방어 +0.25 미만)": legal_tera & O & ~D,
            "둘 다": legal_tera & D & O, "둘 다 아님": legal_tera & ~D & ~O}, dd, dk, do


def main():
    names = sys.argv[1:]
    sets, d, q, vis, battle = load()
    T, M = d["target"].astype(np.float64), d["action_mask"]
    rel = (vis >= 100) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    legal_tera = M[:, 4:8].any(1)
    rng = np.random.RandomState(0)
    res = {}
    for name in names:
        P, tf = run_model(name, sets)
        reg_s, *_ = regrets(P, q, rel, qstar, T)
        res[name] = (P, tf, np.where(valid, reg_s, np.nan))
    P0, tf0, r0 = res[names[0]]
    cls, dd, dk, do = classes(tf0, legal_tera)
    best_tera = np.where(rel, q, -np.inf).argmax(1) >= 4
    best_tera &= np.where(rel, q, -np.inf).argmax(1) < 8
    print(f"테라 합법 결정 {int(legal_tera.sum())} / 전체 {len(T)} (교사 테라 비율 {T[legal_tera, 4:8].sum(1).mean():.3f}, 학생 {names[0]} {P0[legal_tera, 4:8].sum(1).mean():.3f})")
    print("방어 이득 열 기준 분포(테라 합법): 중앙 %.3f p90 %.3f | KO 감소 p90 %.3f | 공격 이득 p90 %.3f" % tuple(
        np.percentile(x[legal_tera], p) for x, p in ((dd, 50), (dd, 90), (dk, 90), (do, 90))))
    for title, groups in (("결정 종류별", cls), ("방어 이득 구간별", {f"방어 이득 {lo:+.2f}~{hi:+.2f}": legal_tera & (dd >= lo) & (dd < hi) for lo, hi in
                                                             ((-9, -0.05), (-0.05, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 9))})):
        print(f"\n== {title}")
        print(f"{'':44s} {'결정':>6s} {'교사 테라':>9s} {'최선이 테라':>10s} " + " ".join(f"{n[:10]:>22s}" for n in names))
        for g, m in groups.items():
            if m.sum() == 0:
                continue
            row = f"{g:44s} {int(m.sum()):6d} {T[m, 4:8].sum(1).mean():9.3f} {best_tera[m & valid].mean() if (m & valid).sum() else float('nan'):10.3f} "
            for n in names:
                P, _, r = res[n]
                mm = m & ~np.isnan(r)
                row += f"  테라 {P[m, 4:8].sum(1).mean():.3f} regret {r[mm].mean():.4f}   "
            print(row)
    if len(names) > 1:
        print("\n== 첫 모델 대비 쌍 비교 (음수 = 그 모델이 손해가 적음, 둘 다 지원 안인 결정만)")
        allg = dict(cls); allg["테라 합법 전체"] = legal_tera; allg["테라가 최선인 결정"] = best_tera & legal_tera
        for n in names[1:]:
            _, _, r = res[n]
            for g, m in allg.items():
                x = np.where(m, r - r0, np.nan)
                if np.isfinite(x).sum() < 30:
                    continue
                mean, lo, hi = boot_ci(x, battle, rng)
                print(f"  {n:20s} {g:44s} n={int(np.isfinite(x).sum()):5d} 차이 {mean:+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
