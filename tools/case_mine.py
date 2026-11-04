"""교체/테라 오답 사례 분석: 교사가 교체나 테라를 확신(질량 >= --conf)한 결정에서 학생이 놓친 경우(regret >= --big)와 맞힌 경우를,
v8 매치업 모듈이 계산하는 한 수 앞 규칙 상황(내 활성이 KO 위협인가 / 내가 선공인가 / 내가 KO 낼 수 있는가 / HP / 남은 포켓몬)으로 나눠 비교한다.
놓친 쪽에서만 "규칙으로 설명되는 상황" 비율이 높으면 학생이 규칙 계산 가능한 개념도 못 쓰는 것, 두 쪽이 같으면 한 수 앞 규칙 밖(예측, 보존, 다수 턴 계획)의 판단이다.
테라는 학생의 대안 종류(같은 기술 테라 없이 / 다른 기술 / 교체)도 센다. 사용: python tools/case_mine.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
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


def rule_feats(net, sets, dev):
    """결정마다 규칙 특징 dict: 내 활성이 맞는 최대 데미지 비율/KO 확률 (드러난 공격기만), 내 최선 공격이 상대 활성에 주는 데미지 비율/KO 확률, 내가 선공일 확률"""
    cap = []
    h = net.matchup.register_forward_hook(lambda m, i, o: cap.append({k: o[k].detach().cpu().numpy() for k in ("def", "off", "spd_my")}))
    P = np.concatenate([L.infer(net, s, dev) for s in sets])
    h.remove()
    D, O, S = (np.concatenate([c[k] for c in cap]) for k in ("def", "off", "spd_my"))
    my_n = np.concatenate([s["my_team_num"] for s in sets]); opp_n = np.concatenate([s["opp_team_num"] for s in sets])
    ar = np.arange(len(P))
    ma, oa = my_n[:, :, 0].argmax(1), opp_n[:, :, 0].argmax(1)
    slots = lambda act: act[:, None] * 4 + np.arange(4)[None]                       # [N,4] 활성 포켓몬의 기술칸 번호
    # D: 내 포켓몬 i가 상대 기술칸 j에 맞을 때 (배율/4, 데미지 비율, KO 확률, 공격기 여부)
    d_in = D[ar[:, None], ma[:, None], slots(oa)]                                     # [N,4,4]
    atk_in = d_in[..., 3] > 0
    thr_dmg = np.where(atk_in, d_in[..., 1], 0).max(1) * 1.5                          # 매치업이 1.5 상한 후 /1.5 한 값을 되돌림
    thr_ko = np.where(atk_in, d_in[..., 2], 0).max(1)
    o_out = O[ar[:, None], oa[:, None], slots(ma)]                                     # [N,4,4] 내 기술 j가 상대 활성에 주는 값
    atk_out = o_out[..., 3] > 0
    my_dmg = np.where(atk_out, o_out[..., 1], 0).max(1) * 1.5
    my_ko = np.where(atk_out, o_out[..., 2], 0).max(1)
    # 벤치 포함 팀 전체의 한 수 앞 상성: 포켓몬 i가 상대 활성에 주는 최대 데미지 - 상대 활성의 드러난 공격기가 i에게 주는 최대 데미지
    d_all = D.reshape(len(P), 6, 6, 4, 4)[ar, :, oa]                                  # [N,6,4,4]
    o_all = O.reshape(len(P), 6, 6, 4, 4)[ar, oa]                                      # [N,6(내 포켓몬),4,4]
    inc = np.where(d_all[..., 3] > 0, d_all[..., 1], 0).max(-1) * 1.5
    out = np.where(o_all[..., 3] > 0, o_all[..., 1], 0).max(-1) * 1.5
    adv = out - inc                                                                    # [N,6]
    faster = S[ar, ma]                                                                 # 내 활성 스피드 / 상대 활성 (log2 /2): 0 이상이면 같거나 빠름 (자기 자신 대비 아님)
    return P, dict(thr_ko=thr_ko, thr_dmg=thr_dmg, my_ko=my_ko, my_dmg=my_dmg, adv=adv, ma=ma, spd=faster, my_hp=my_n[ar, ma, 2], opp_hp=opp_n[ar, oa, 2],
                   alive=((my_n[..., 11] > 0) & (my_n[..., 1] < 0.5)).sum(1), opp_left=(opp_n[..., 1] < 0.5).sum(1))


def show(title, sel, F):
    n = int(sel.sum())
    if n == 0:
        print(f"  {title:28s} 0건"); return
    f = {k: v[sel] for k, v in F.items()}
    thr = f["thr_ko"] > 0.5
    slow = f["spd"] < 0
    kill = f["my_ko"] > 0.5
    print(f"  {title:28s} {n:5d} | 내 활성 KO 위협 {thr.mean():.3f} (후공 {np.mean(thr & slow):.3f}) | 내가 KO 가능 {kill.mean():.3f} | 위협 없음&KO 불가 {np.mean(~thr & ~kill):.3f} | "
          f"활성 HP {np.mean(f['my_hp']):.2f} | 위협 데미지 {np.mean(f['thr_dmg']):.2f} | 내 데미지 {np.mean(f['my_dmg']):.2f} | 내 남은 {np.mean(f['alive']):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--conf", type=float, default=0.8, help="교사 질량이 이 이상인 선택만 확신으로 봄")
    ap.add_argument("--big", type=float, default=0.15, help="학생 regret이 이 이상이면 큰 실수")
    ap.add_argument("--min-vis", type=float, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    sets = [L.load_set(p) for p in args.sets.split(",")]
    zs = [np.load(p) for p in args.sets.split(",")]
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    T, M = np.concatenate([s["target"] for s in sets]).astype(np.float64), np.concatenate([s["action_mask"] for s in sets])
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    vol = (M[:, :4].sum(1) > 0) & (M[:, 8:14].sum(1) > 0)
    a_t, conf = T.argmax(1), T.max(1)
    net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").to(dev).eval()
    P, F = rule_feats(net, sets, dev)
    reg_s, *_ = regrets(P, q, rel, qstar, T)
    a_s = P.argmax(1)
    big, hit = valid & (reg_s >= args.big), valid & (a_s == a_t)
    print(f"모델 {args.models} | 자발 결정 {int((vol & valid).sum())} | 확신 >= {args.conf}, 큰 실수 regret >= {args.big}")
    print("규칙 열: KO 위협 = 내 활성이 드러난 상대 공격기에 맞아 쓰러질 확률 > 0.5, 후공 = 내 스피드 < 상대 / 내가 KO 가능 = 내 최선 기술이 상대 활성을 쓰러뜨릴 확률 > 0.5")
    base = vol & valid
    print("\n[전체 자발 결정 기준선]"); show("자발 결정 전체", base, F)
    for name, kinds, lo, hi in (("교체", None, 8, 14), ("테라 기술", None, 4, 8)):
        c = base & (conf >= args.conf) & (a_t >= lo) & (a_t < hi)
        print(f"\n[교사 확신 {name}: {int(c.sum())}건, 학생 argmax 일치 {np.mean(hit[c]):.3f}, 큰 실수 {np.mean(big[c]):.3f}]")
        show("학생 맞힘", c & hit, F); show("학생 틀림(큰 실수 아님)", c & ~hit & ~big, F); show("학생 큰 실수", c & big, F)
        if lo == 4:
            m_ = c & ~hit
            same = a_s == a_t - 4
            print(f"  틀린 {int(m_.sum())}건의 학생 선택: 같은 기술 테라 없이 {np.mean(same[m_]):.3f} | 다른 기술 {np.mean(((a_s < 8) & ~same)[m_]):.3f} | 교체 {np.mean((a_s >= 8)[m_]):.3f}")
        else:
            m_ = c & ~hit
            print(f"  틀린 {int(m_.sum())}건의 학생 선택: 공격/변화기 {np.mean((a_s < 4)[m_]):.3f} | 테라 {np.mean(((a_s >= 4) & (a_s < 8))[m_]):.3f} | 다른 교체 {np.mean((a_s >= 8)[m_]):.3f}")
    # 한 수 앞 상성 우위만으로 교체를 얼마나 가려내는가: (벤치 중 살아 있고 교체 가능한 최선 상성 - 활성 상성) 점수 vs 학생 교체 확률의 AUC (정답 = 교사 교체 질량 >= 0.5)
    n = len(P); ar = np.arange(n)
    alive_b = (np.concatenate([s["my_team_num"] for s in sets])[..., 1] < 0.5) & M[:, 8:14]
    benchadv = np.where(alive_b, F["adv"], -9).max(1)
    gain = benchadv - F["adv"][ar, F["ma"]]
    pos = (T[:, 8:14].sum(1) >= 0.5)
    ok = base
    p_sw = P[:, 8:14].sum(1)
    print("\n[교체 판별 AUC (자발 결정, 정답 = 교사 교체 질량 >= 0.5)] 학생 교체 확률 %.3f | 규칙: 벤치 최선 상성 - 활성 상성 %.3f | 규칙: 활성 KO 위협 %.3f" % (
        L.auc(p_sw[ok], pos[ok]), L.auc(gain[ok], pos[ok]), L.auc(F["thr_dmg"][ok], pos[ok])))
    hard = ok & (F["thr_ko"] <= 0.5)
    print("  위협 없는 결정만: 학생 %.3f | 규칙(상성 차) %.3f   (정답 비율 %.3f)" % (L.auc(p_sw[hard], pos[hard]), L.auc(gain[hard], pos[hard]), pos[hard].mean()))
    bins = [("상성 차 < 0", gain < 0), ("0~0.3", (gain >= 0) & (gain < 0.3)), ("0.3~0.6", (gain >= 0.3) & (gain < 0.6)), (">= 0.6", gain >= 0.6)]
    print("  규칙 상성 차 구간별 교사 교체 질량 평균 / 학생 교체 확률 평균 (자발 결정):")
    for nm, sel in bins:
        sel = sel & ok
        print(f"    {nm:10s} {int(sel.sum()):6d}건 교사 {T[sel][:, 8:14].sum(1).mean():.3f} 학생 {p_sw[sel].mean():.3f}")
    # 반대 방향: 교사는 공격인데 학생이 교체/테라 (과교체)
    for name, lo, hi in (("교체", 8, 14), ("테라", 4, 8)):
        c = base & (conf >= args.conf) & (a_t < 4) & (a_s >= lo) & (a_s < hi)
        print(f"\n[교사 확신 일반 기술 -> 학생 {name}: {int(c.sum())}건, 평균 regret {np.nanmean(reg_s[c]):.4f}]"); show("해당 결정", c, F)


if __name__ == "__main__":
    main()
