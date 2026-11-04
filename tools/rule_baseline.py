"""한 수 앞 규칙 대 학생: 학생이 한 수 앞 계산(데미지, KO, 선공, 상성)으로 판단 가능한 결정에서 규칙을 그대로 따르는 기준선만큼 하는가.
1) 교사 argmax가 공격기인 결정에서 (규칙 = 최대 데미지 비율 공격기) 기술 일치율/regret을 학생과 비교하고 "규칙만 맞음/학생만 맞음/둘 다 틀림"으로 나눔
2) 규칙이 확실한 상황 개입 실험: (R1) 선공으로 확정 KO(확률 > 0.9) 가능한 공격기가 있으면 그 기술을 고름, (R2) 반감/무효 기술은 고르지 않음(다른 공격기가 중립 이상일 때)
   -> 개입한 결정만 학생 regret 대 개입 regret을 판 단위 부트스트랩으로 쌍 비교 (음수 = 개입이 낫다). 개입이 낫다면 학생이 한 수 앞 정보를 덜 쓰고 있다는 뜻
데미지/KO/선공은 학생과 같은 MatchupV7 출력(forward hook)을 읽음: 근사(레벨 100, 랜덤 편차 평균, 상대는 종족값 추정)이므로 규칙도 같은 한계를 가짐.
사용: python tools/rule_baseline.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import boot_ci, regrets
from src.core.model import model_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--min-vis", type=float, default=100)
    args = ap.parse_args()
    paths = args.sets.split(",")
    sets = [L.load_set(p) for p in paths]
    zs = [np.load(p) for p in paths]
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    cat = lambda k: np.concatenate([s[k] for s in sets])
    T, M = cat("target").astype(np.float64), cat("action_mask")
    mn, on = cat("my_team_num"), cat("opp_team_num")
    off = np.cumsum([0] + [len(s["lens"]) for s in sets])
    battle = np.concatenate([np.repeat(np.arange(len(s["lens"])), s["lens"]) + o for s, o in zip(sets, off)])
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
    cap = []
    h = net.matchup.register_forward_hook(lambda m, i, o: cap.append({k: o[k].detach().cpu().numpy() for k in ("off", "fm_my")}))
    P = np.concatenate([L.infer(net, s, torch.device("cpu")) for s in sets])
    h.remove()
    O, FM = np.concatenate([c["off"] for c in cap]), np.concatenate([c["fm_my"] for c in cap])
    N = len(P); ar = np.arange(N)
    ma, oa = mn[:, :, 0].argmax(1), on[:, :, 0].argmax(1)
    o4 = O.reshape(N, 6, 6, 4, 4)[ar, oa, ma]                                         # [N,4(기술칸),4]: (배율/4, 데미지 비율, KO 확률, 공격기 여부)
    first = FM[ar, ma]                                                                 # [N,4] 선공 기대값 (+1 먼저)
    eff, frac, ko, dmg = o4[..., 0] * 4, o4[..., 1] * 1.5, o4[..., 2], o4[..., 3] > 0
    legal = M[:, :4]
    usable = dmg & legal
    pick_reg = lambda a: np.where(valid & rel[ar, a], qstar - q[ar, a], np.nan)        # 한 행동의 regret (믿을 Q 없으면 NaN)
    a_t, a_s = T.argmax(1), P.argmax(1)
    t_slot = np.where(a_t < 8, a_t % 4, -1)
    s_slot = np.where(a_s < 8, a_s % 4, -1)
    rng = np.random.RandomState(0)
    bci = lambda x: "{:.4f} [{:.4f},{:.4f}]".format(*boot_ci(x, battle, rng, 300))
    reg_s = pick_reg(a_s)

    # 1) 교사 argmax가 공격기
    att_t = valid & (t_slot >= 0) & usable[ar, np.maximum(t_slot, 0)] & usable.any(1)
    r_slot = np.where(usable, frac, -1).argmax(1)                                      # 규칙: 최대 데미지 비율
    reg_r = pick_reg(r_slot)
    ok_r, ok_s = r_slot == t_slot, s_slot == t_slot
    print(f"모델 {args.models} | 교사 argmax가 공격기인 결정 {int(att_t.sum())} (비교 가능 {int(valid.sum())} 중)")
    print(f"  기술 일치(테라 여부 무시): 규칙(최대 데미지) {ok_r[att_t].mean():.3f} | 학생 {ok_s[att_t].mean():.3f}")
    ok2 = att_t & ~np.isnan(reg_r) & ~np.isnan(reg_s)
    print(f"  regret: 규칙 {bci(np.where(ok2, reg_r, np.nan))} | 학생 {bci(np.where(ok2, reg_s, np.nan))} | 쌍 차이(학생-규칙) {np.nanmean(reg_s[ok2] - reg_r[ok2]):+.4f}")
    for nm, m in (("규칙만 맞음", ok_r & ~ok_s), ("학생만 맞음", ~ok_r & ok_s), ("둘 다 맞음", ok_r & ok_s), ("둘 다 틀림", ~ok_r & ~ok_s)):
        s = att_t & m & ~np.isnan(reg_s) & ~np.isnan(reg_r)
        print(f"  {nm:12s} {int(s.sum()):6d}건 ({s.sum() / ok2.sum():.1%}) | 학생 regret {np.mean(reg_s[s]):.4f} | 규칙 regret {np.mean(reg_r[s]):.4f}")

    # 2) 개입 실험
    print("\n개입 실험 (개입이 적용되는 결정만; 쌍 차이 = 개입 regret - 학생 regret, 음수면 개입이 낫다)")
    def report(nm, apply, pick):
        s = valid & apply
        a_new = np.where(apply, pick, a_s)
        r_new = pick_reg(a_new)
        s = s & ~np.isnan(r_new) & ~np.isnan(reg_s)
        changed = s & (a_new != a_s) & ~(a_s < 8) | s & (a_new != a_s) & (a_s < 8) & (a_s % 4 != a_new % 4)
        d = np.where(s, r_new - reg_s, np.nan)
        print(f"  {nm}: 적용 {int(s.sum())}건 | 학생이 이미 같은 기술 {np.mean((a_s % 4 == a_new % 4)[s] & (a_s < 8)[s]):.3f} | 교사와 기술 일치 개입 {np.mean((t_slot == a_new)[s]):.3f} 학생 {np.mean((t_slot == a_s % 4 * (a_s < 8) - (a_s >= 8))[s]):.3f}")
        print(f"      학생 regret {bci(np.where(s, reg_s, np.nan))} | 개입 regret {bci(np.where(s, r_new, np.nan))} | 쌍 차이 {bci(d)}")
        diff = s & (a_s % 4 != a_new % 4) | s & (a_s >= 8)
        if diff.sum():
            print(f"      학생과 다른 선택을 하게 된 {int(diff.sum())}건: 학생 {np.mean(reg_s[diff]):.4f} -> 개입 {np.mean(r_new[diff]):.4f}")
    sure = usable & (ko > 0.9) & (first > 0)
    pick1 = np.where(sure, frac, -1).argmax(1)
    report("R1 선공 확정 KO 기술이 있으면 선택", sure.any(1), pick1)
    best_eff = np.where(usable, eff, 0).max(1)
    bad = (best_eff >= 1.0)
    cand = np.where(usable & (eff >= 1.0), frac, -1).argmax(1)
    s_bad = (a_s < 8) & usable[ar, a_s % 4] & (eff[ar, a_s % 4] <= 0.5) & bad
    report("R2 학생이 반감/무효 기술을 골랐고 중립 이상 공격기가 있으면 그 중 최대 데미지로 교체", s_bad, cand)
    t_bad = (t_slot >= 0) & usable[ar, np.maximum(t_slot, 0)] & (eff[ar, np.maximum(t_slot, 0)] <= 0.5) & bad
    print(f"  (참고) 교사 argmax가 반감/무효 기술이면서 중립 이상 공격기 존재: {t_bad[valid].mean():.4f} | 학생 {s_bad[valid].mean():.4f}")


if __name__ == "__main__":
    main()
