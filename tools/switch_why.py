"""교사가 교체를 고르는 이유가 상대의 미공개 정보 때문인가, 아니면 위치 가치인가.
A) 상대 활성 포켓몬의 공개된 기술 수(0~4)별로: 교사/학생 교체 질량, 교체 판별 AUC, 교사 확신 교체를 학생이 맞힌 비율, 평균 regret, regret 몫.
B) 세계별 가치(tools/world_q.py가 만든 *_worlds.npz, Foul Play가 표본 낸 상대 세트 2개): 교사 확신 교체의 세계별 이득 = 교체 점수 - 최선 비교체 점수.
   두 세계 모두 교체가 최선이면 "상대 세트와 무관한 이득", 한 세계에서만이면 "세트 의존". 학생이 맞힌 교체와 놓친 교체를 나눠 비교.
   세계가 2개뿐이라 탐색 잡음과 세트 차이를 못 가르므로, 같은 통계를 교사 확신 일반 기술 결정에도 내서 잡음 기준선으로 씀.
사용: python tools/switch_why.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz   (같은 이름의 *_worlds.npz가 있어야 B를 냄)"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import regrets
from src.core.model import model_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--min-vis", type=float, default=100)
    args = ap.parse_args()
    paths = args.sets.split(",")
    sets = [L.load_set(p) for p in paths]
    zs = [np.load(p) for p in paths]
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    T, M = np.concatenate([s["target"] for s in sets]).astype(np.float64), np.concatenate([s["action_mask"] for s in sets])
    opp_mv, opp_n = np.concatenate([s["opp_move_num"] for s in sets]), np.concatenate([s["opp_team_num"] for s in sets])
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    vol = (M[:, :4].sum(1) > 0) & (M[:, 8:14].sum(1) > 0) & valid
    net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
    P = np.concatenate([L.infer(net, s, torch.device("cpu")) for s in sets])
    reg, *_ = regrets(P, q, rel, qstar, T)
    reg = np.where(valid, reg, np.nan)
    a_t, a_s = T.argmax(1), P.argmax(1)
    conf_sw = vol & (T.max(1) >= args.conf) & (a_t >= 8) & (a_t < 14)
    hit = a_s == a_t
    ar = np.arange(len(P))
    oa = opp_n[:, :, 0].argmax(1)
    nrev = (np.abs(opp_mv[ar, oa]).sum(-1) > 0).sum(1)                                  # 상대 활성의 공개된 기술 수
    pos = T[:, 8:14].sum(1) >= 0.5
    p_sw = P[:, 8:14].sum(1)
    tot = np.nansum(np.where(vol, reg, np.nan))
    print(f"모델 {args.models} | 자발 결정 {int(vol.sum())} | 교사 확신 교체 {int(conf_sw.sum())}")
    print("\n[A] 상대 활성의 공개 기술 수별 (자발 결정)")
    print(f"{'공개':>4s} {'결정':>6s} | {'교사 교체질량':>11s} {'학생 교체질량':>11s} {'AUC':>6s} | {'확신 교체':>8s} {'학생 일치':>8s} | {'평균 regret':>11s} {'regret 몫':>9s}")
    for k in range(5):
        s = vol & (nrev == k)
        if s.sum() < 50:
            continue
        c = conf_sw & (nrev == k)
        print(f"{k:4d} {int(s.sum()):6d} | {T[s][:, 8:14].sum(1).mean():11.3f} {p_sw[s].mean():11.3f} {L.auc(p_sw[s], pos[s]):6.3f} | {int(c.sum()):8d} {np.mean(hit[c]) if c.sum() else np.nan:8.3f} | "
              f"{np.nanmean(reg[s]):11.4f} {np.nansum(reg[s]) / tot:9.3f}")
    wpaths = [p.replace(".npz", "_worlds.npz") for p in paths]
    if not all(os.path.exists(p) for p in wpaths):
        print("\n[B] 건너뜀: *_worlds.npz 없음 (tools/world_q.py)")
        return
    wz = [np.load(p) for p in wpaths]
    wp, wq, wv = (np.concatenate([z[k] for z in wz]) for k in ("wp", "wq", "wv"))      # [N,2], [N,2,22], [N,2,22]
    ok = (wv >= 30) & ~np.isnan(wq) & M[:, None, :]                                    # 세계 안에서 믿을 만한 행동
    sw_mask = np.zeros(22, bool); sw_mask[8:14] = True

    def adv(a_sel, other_mask):
        """세계별 이득: a_sel(결정마다 한 행동) 점수 - other_mask(행동 집합)의 최선 점수. 두 세계 모두 값이 있는 결정만"""
        a = wq[ar, :, a_sel]                                                          # [N,2]
        a_ok = ok[ar, :, a_sel]
        o = np.where(ok & other_mask[:, None, :], wq, -np.inf).max(-1)                # [N,2]
        o_ok = np.isfinite(o)
        good = a_ok.all(1) & o_ok.all(1)
        return np.where(a_ok & o_ok, a - o, np.nan), good

    for title, sel, other in (("교사 확신 교체: 교체 이득 = 교체 - 최선 비교체", conf_sw, ~np.broadcast_to(sw_mask, M.shape)),
                              ("교사 확신 일반 기술 (잡음 기준선): 이득 = 그 기술 - 최선 다른 행동", vol & (T.max(1) >= args.conf) & (a_t < 4),
                               np.ones_like(M) & ~np.eye(22, dtype=bool)[a_t])):
        A, good = adv(a_t, other)
        print(f"\n[B] {title}")
        print(f"{'':16s} {'결정':>5s} | {'평균 이득':>8s} {'두 세계 모두 이득>0':>16s} {'한 세계만':>8s} {'두 세계 이득 차 평균':>14s} {'이득 크기(평균)':>13s}")
        for nm, s in (("전체", sel), ("학생 맞힘", sel & hit), ("학생 틀림", sel & ~hit), ("학생 큰 실수", sel & (reg >= 0.15))):
            s = s & good
            if s.sum() < 20:
                continue
            a = A[s]
            both, one = (a > 0).all(1).mean(), ((a > 0).sum(1) == 1).mean()
            print(f"{nm:16s} {int(s.sum()):5d} | {np.nanmean(a):8.3f} {both:16.3f} {one:8.3f} {np.mean(np.abs(a[:, 0] - a[:, 1])):14.3f} {np.mean(np.abs(a).mean(1)):13.3f}")


if __name__ == "__main__":
    main()
