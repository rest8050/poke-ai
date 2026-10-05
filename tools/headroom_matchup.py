"""매치업 특징(데미지/한 방/속도/1대1 승부) 헤드룸 점검: 같은 입력 텐서에서 특징을 계산해, 교사(Foul Play)의 선택을
'신경망 출력만'으로 맞힐 때보다 '신경망 출력 + 매치업 특징'으로 얼마나 더 잘 맞히는지 교차검증(GBM, 배틀 단위 분할)으로 비교.
사용: python tools/headroom_matchup.py [체크포인트=v12r3] [npz ...=dg2_ho dg3_ho]
- 내 쪽 num[11:17]*255 = 실제 능력치(아이템 배율 포함), 상대 쪽 = 종족값 → 레벨 100 기준 노력치 0/84/252+성격 0.9/1.0/1.1 세 시나리오로 추정
- 기술 벡터: [0]위력/200 [1]명중 [3]우선도/7 [4]분류(0.33물리/0.67특수/1.0변화) [6]타입id/20 [7]상성배율/4(상대 활성 대상) [2]연속타격
- 근사: 급소/특성/테라/벽/날씨 보정 제외(날씨는 위력에 이미 반영). 정확한 계산기가 아니라 '이런 정보가 추가 신호를 주는가'를 보는 용도"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.model import model_from_ckpt

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
TYPES = ["normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock",
         "ghost", "dragon", "steel", "dark", "fairy"]
chart = json.load(open("data/type_chart.json"))
EFF = np.ones((21, 21))
for i, a in enumerate(TYPES, 1):
    for j, d in enumerate(TYPES, 1):
        EFF[i, j] = chart[a][d]
R = np.linspace(0.85, 1.0, 16)


def bm(s):
    s = np.clip(s, -6, 6)
    return np.where(s >= 0, (2 + s) / 2, 2 / (2 - np.minimum(s, 0)))


def opp_variants(n):
    """상대 능력치 3시나리오(약/중/강) [3,N,6,6]: 종족값 -> 레벨 L 실능력치"""
    B = n[..., 11:17] * 255
    L = np.rint(n[..., 8] * 100)[..., None]
    out = []
    for ev, nat in ((0, 0.9), (84, 1.0), (252, 1.1)):
        s = np.floor((2 * B + 31 + ev / 4) * L / 100)
        st = np.floor((s + 5) * nat)
        st[..., 0] = s[..., 0] + L[..., 0] + 10
        out.append(st)
    return np.stack(out)


def attack_eval(att_st, att_stage, att_types, L, mv, def_st, def_stage, def_hp, att_burn, eff=None):
    """att_st/def_st [S,N,6], att_stage/def_stage [N,5](atk,def,spa,spd,spe), mv [N,4,46], def_hp [S,N] (현재 HP).
    반환: mid 시나리오 평균 피해/현재HP [N,4], 한 방 확률(시나리오x난수 평균) [N,4], 명중 [N,4], 공격기 여부 [N,4], 우선도 [N,4]"""
    bp, acc, cat = mv[..., 0] * 200, np.clip(mv[..., 1], 0, 1), mv[..., 4]
    phys, spec = cat < 0.5, (cat > 0.5) & (cat < 0.9)
    is_att = (bp > 0) & (phys | spec)
    mtype = np.rint(mv[..., 6] * 20).astype(int)
    hits = 1 + 4 * mv[..., 2]
    eff = mv[..., 7] * 4 if eff is None else eff
    A = np.where(phys[None], att_st[:, :, None, 1] * bm(att_stage[None, :, None, 0]), att_st[:, :, None, 3] * bm(att_stage[None, :, None, 2]))
    D = np.where(phys[None], def_st[:, :, None, 2] * bm(def_stage[None, :, None, 1]), def_st[:, :, None, 4] * bm(def_stage[None, :, None, 3]))
    stab = np.where((mtype == att_types[:, 0:1]) | (mtype == att_types[:, 1:2]), 1.5, 1.0)
    burn = np.where(phys & att_burn[:, None], 0.5, 1.0)
    base = ((2 * L[None, :, None] / 5 + 2) * bp[None] * A / np.maximum(D, 1)) / 50 + 2
    tot = base * (stab * eff * burn * hits * is_att)[None]
    hp = np.maximum(def_hp, 1)
    frac = tot[1] * R.mean() / hp[1][:, None]
    ko = ((tot[..., None] * R) >= hp[:, :, None, None]).mean((0, -1))
    return frac, ko * is_att, acc, is_att, mv[..., 3] * 7


def assumed_move(o_st3, o_stage, o_types, my_types_k, N):
    """상대가 아직 안 보인 기술칸 대비: 위력 90 자속 공격기 하나를 가정한 가짜 기술 벡터 [N,1,46] (분류는 더 높은 공격 능력치 쪽)"""
    mv = np.zeros((N, 1, 46))
    mv[..., 0], mv[..., 1], mv[..., 3] = 90 / 200, 1.0, 0.0
    phys = (o_st3[1][:, 1] >= o_st3[1][:, 3])
    mv[:, 0, 4] = np.where(phys, 0.33, 0.67)
    mv[:, 0, 6] = o_types[:, 0] / 20
    return mv


def features(d):
    N = d["my_team_num"].shape[0]
    mc, mn, mm = d["my_team_cat"], d["my_team_num"], d["my_move_num"]
    oc, on, om = d["opp_team_cat"], d["opp_team_num"], d["opp_move_num"]
    ar = np.arange(N)
    a, o = mn[:, :, 0].argmax(1), on[:, :, 0].argmax(1)
    Lm, Lo = np.rint(mn[:, :, 8] * 100), np.rint(on[:, :, 8] * 100)
    my_st = mn[..., 11:17] * 255
    op_var = opp_variants(on)                       # [3,N,6,6]
    stage = lambda n: np.rint(n[..., 3:8] * 6)       # [N,6,5]
    my_stage, op_stage = stage(mn), stage(on)
    my_hpf, op_hpf = mn[:, :, 2], on[:, :, 2]
    my_status, op_status = mc[:, :, 4], oc[:, :, 4]
    types = lambda c: c[..., 2:4]
    F = {}
    o_st = op_var[:, ar, o]                           # [3,N,6]
    o_hp_cur = o_st[..., 0] * op_hpf[ar, o][None]
    # --- 속도: 내 활성 vs 상대 활성(범위) ---
    def p_faster(my_spe, op_spe3, op_stg, op_par, tr):
        spe = op_spe3[..., 5] * bm(op_stg[:, 4])[None] * np.where(op_par, 0.5, 1.0)[None]
        lo, hi = spe[0], spe[2]
        p = np.clip((my_spe - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        return np.where(tr, 1 - p, p)
    tr = d["field_vec"][:, 11] > 0
    op_par = op_status[ar, o] == 3
    # --- 내 활성의 공격 ---
    def my_attack(k):
        st = np.broadcast_to(my_st[ar, k][None], (3, N, 6))
        return attack_eval(st, my_stage[ar, k], types(mc)[ar, k], Lm[ar, k], mm[ar, k], o_st, op_stage[ar, o], o_hp_cur,
                           my_status[ar, k] == 2)
    # --- 상대 활성의 위협(공개 기술 + 미공개칸 가정) vs 내 포켓몬 k ---
    def threat(k, active_target):
        d_st = np.broadcast_to(my_st[ar, k][None], (3, N, 6))
        d_hp = d_st[..., 0] * my_hpf[ar, k][None]
        mv = om[ar, o].copy()
        eff = None
        dt = types(mc)[ar, k]
        if not active_target:   # 벤치 포켓몬 상대: 텐서의 배율은 내 활성 기준이라 타입표로 다시 계산
            mt = np.rint(mv[..., 6] * 20).astype(int)
            eff = EFF[mt, dt[:, 0:1]] * EFF[mt, dt[:, 1:2]]
        frac, ko, acc, is_att, pri = attack_eval(op_var[:, ar, o], op_stage[ar, o], types(oc)[ar, o], Lo[ar, o], mv, d_st,
                                                my_stage[ar, k], d_hp, op_status[ar, o] == 2, eff)
        known = is_att.sum(1)
        revealed = (np.abs(om[ar, o]).sum(-1) > 0).sum(1)
        # 미공개칸 가정 공격기
        am = assumed_move(o_st, None, types(oc)[ar, o], dt, N)
        am_eff = None
        if not active_target:
            mt = np.rint(am[..., 6] * 20).astype(int)
            am_eff = EFF[mt, dt[:, 0:1]] * EFF[mt, dt[:, 1:2]]
        else:
            mt = np.rint(am[..., 6] * 20).astype(int)
            am_eff = EFF[mt, dt[:, 0:1]] * EFF[mt, dt[:, 1:2]]
        af, ako, _, aatt, _ = attack_eval(op_var[:, ar, o], op_stage[ar, o], types(oc)[ar, o], Lo[ar, o], am, d_st, my_stage[ar, k], d_hp,
                                          op_status[ar, o] == 2, am_eff)
        has_unknown = revealed < 4
        t_ko = np.maximum((ko * acc).max(1), np.where(has_unknown, ako[:, 0], 0))
        t_frac = np.maximum((frac * acc).max(1), np.where(has_unknown, af[:, 0], 0))
        return t_ko, t_frac, known / 4.0
    hits = lambda fr: np.clip(np.ceil(1 / np.maximum(fr, 1e-3)), 1, 6)
    # 활성 vs 활성
    mf, mk, macc, matt, mpri = my_attack(a)
    am_ko, am_dmg = (mk * macc).max(1), (mf * macc).max(1)
    at_ko, at_dmg, at_known = threat(a, True)
    my_spe = my_st[ar, a][:, 5] * bm(my_stage[ar, a][:, 4]) * np.where(my_status[ar, a] == 3, 0.5, 1.0)
    pf = p_faster(my_spe, o_st, op_stage[ar, o], op_par, tr)
    h_me, h_op = hits(am_dmg), hits(at_dmg)
    race = pf * (h_me <= h_op) + (1 - pf) * (h_me < h_op)
    prio_ko = (((mpri > 0) & (mk * macc > 0.5)).any(1)).astype(float)
    F.update(am_ko=am_ko, am_dmg=np.minimum(am_dmg, 3), am_hits=h_me, at_ko=at_ko, at_dmg=np.minimum(at_dmg, 3), at_hits=h_op,
             at_known=at_known, p_faster=pf, race=race, prio_ko=prio_ko, surv=1 - at_ko)
    # 벤치 후보 (살아 있고 활성이 아닌 내 포켓몬)
    best_race = np.zeros(N); best_surv = np.zeros(N); best_ko = np.zeros(N); nb = np.zeros(N)
    for k in range(6):
        kk = np.full(N, k)
        valid = (mn[:, k, 1] == 0) & (a != k) & (mn[:, k, 11] > 0)
        f_, k_, acc_, _, pri_ = my_attack(kk)
        kd, kk_ko = (f_ * acc_).max(1), (k_ * acc_).max(1)
        t_ko, t_fr, _ = threat(kk, False)
        spe_k = my_st[ar, kk][:, 5] * np.where(my_status[ar, kk] == 3, 0.5, 1.0)
        pf_k = p_faster(spe_k, o_st, op_stage[ar, o], op_par, tr)
        hm, ho = hits(kd), np.maximum(hits(t_fr) - 1, 1)     # 교체 직후 한 대 맞고 시작
        rc = (1 - t_ko) * (pf_k * (hm <= ho) + (1 - pf_k) * (hm < ho))
        best_race = np.where(valid, np.maximum(best_race, rc), best_race)
        best_surv = np.where(valid, np.maximum(best_surv, 1 - t_ko), best_surv)
        best_ko = np.where(valid, np.maximum(best_ko, kk_ko), best_ko)
        nb += valid
    F.update(b_race=best_race, b_surv=best_surv, b_ko=best_ko, b_n=nb)
    base = dict(my_hp=my_hpf[ar, a], op_hp=op_hpf[ar, o], my_alive=nb + 1, op_alive=(on[:, :, 1] == 0).sum(1),
                my_atk=my_stage[ar, a][:, 0], my_spa=my_stage[ar, a][:, 2], my_spe_stg=my_stage[ar, a][:, 4],
                op_atk=op_stage[ar, o][:, 0], op_spa=op_stage[ar, o][:, 2], op_spe_stg=op_stage[ar, o][:, 4],
                my_status=my_status[ar, a], op_status=op_status[ar, o], trick_room=tr.astype(float))
    return base, F, dict(move_frac=mf, move_ko=mk * macc)


def load(paths, model):
    arrs, P, bid, tix = {k: [] for k in KEYS + ["action_mask", "target"]}, [], [], []
    b0 = 0
    for path in paths:
        z0 = np.load(path)
        z = {k: z0[k] for k in KEYS + ["action_mask", "target", "lens"]}
        for k in ("my_move_num", "opp_move_num"):
            if z[k].shape[-1] < 46:
                z[k] = np.concatenate([z[k], np.zeros(z[k].shape[:-1] + (46 - z[k].shape[-1],), dtype=z[k].dtype)], -1)
        off = np.concatenate([[0], np.cumsum(z["lens"])])
        with torch.no_grad():
            for b in range(len(z["lens"])):
                s, e = off[b], off[b + 1]
                out = model.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(e - s, dtype=torch.long),
                                              torch.arange(e - s), 1, torch.tensor(z["action_mask"][s:e]))
                P.append(torch.softmax(out["policy_logits"], -1).numpy())
                bid += [b0 + b] * (e - s); tix += list(range(e - s))
        for k in arrs:
            arrs[k].append(z[k])
        b0 += len(z["lens"])
    return {k: np.concatenate(v) for k, v in arrs.items()}, np.concatenate(P), np.array(bid), np.array(tix)


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/supervised_v2_fp_v12r3.pt"
    paths = sys.argv[2:] or ["data/dagger/dg2_ho.npz", "data/dagger/dg3_ho.npz"]
    model = model_from_ckpt(ck).eval()
    d, P, bid, tix = load(paths, model)
    base, M, per_move = features(d)
    am = d["action_mask"].astype(bool)
    V = am[:, :8].any(1) & am[:, 8:14].any(1)
    T = d["target"]; ta, sa = T.argmax(1), P.argmax(1)
    S = dict(sw_mass=P[:, 8:14].sum(1), top_move=P[:, :8].max(1), ent=-(P * np.log(np.clip(P, 1e-9, 1))).sum(1), top_is_sw=((sa >= 8) & (sa < 14)).astype(float),
             turn=tix.astype(float))
    print(f"상태 {len(ta)}개, 자발 결정 {V.sum()}개, 배틀 {bid.max() + 1}판")
    from sklearn.ensemble import HistGradientBoostingClassifier as G
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    def cv(feats, y, mask, name):
        X = np.column_stack([f[mask] for f in feats]); yy = y[mask]; g = bid[mask]
        pred = np.zeros(len(yy))
        for tr_, te_ in GroupKFold(5).split(X, yy, g):
            m = G(max_iter=150, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0).fit(X[tr_], yy[tr_])
            pred[te_] = m.predict_proba(X[te_])[:, 1]
        print(f"  {name:22s} logloss {log_loss(yy, pred):.4f}  AUC {roc_auc_score(yy, pred):.4f}  정확도 {((pred > 0.5) == yy).mean():.4f}")
        return log_loss(yy, pred)
    s, b, m = list(S.values()), list(base.values()), list(M.values())
    y1 = ((ta >= 8) & (ta < 14)).astype(int)
    print(f"\n[과제 1] 교사가 교체를 고르는가 (자발 결정 {V.sum()}개, 교사 교체 비율 {y1[V].mean():.3f}, 상수 예측 logloss "
          f"{log_loss(y1[V], np.full(V.sum(), y1[V].mean())):.4f})")
    cv(b, y1, V, "기본 상태만(신경망 없이)"); cv(b + m, y1, V, "기본+매치업(신경망 없이)")
    l_s = cv(s, y1, V, "신경망 출력만"); l_sb = cv(s + b, y1, V, "+ 기본 상태"); l_sbm = cv(s + b + m, y1, V, "+ 매치업 특징")
    print(f"  -> 매치업 특징이 더한 개선: logloss {l_sb - l_sbm:+.4f} (기본 상태 대비)")
    mv_t = V & (ta < 8)
    y2 = (sa == ta).astype(int)
    print(f"\n[과제 2] 교사가 기술을 고른 상태에서 신경망이 맞히는가 (상태 {mv_t.sum()}개, 정답률 {y2[mv_t].mean():.3f})")
    l_s2 = cv(s, y2, mv_t, "신경망 출력만"); l_sb2 = cv(s + b, y2, mv_t, "+ 기본 상태"); l_sbm2 = cv(s + b + m, y2, mv_t, "+ 매치업 특징")
    print(f"  -> 매치업 특징이 더한 개선: logloss {l_sb2 - l_sbm2:+.4f}")
    # 기술 선택: 추정 피해가 교사/학생 선택과 얼마나 맞는가
    fr = per_move["move_frac"]
    both = V & (ta < 8) & (sa < 8) & (ta != sa)
    ts, ss = ta % 4, sa % 4
    t_f, s_f = fr[np.arange(len(ta)), ts], fr[np.arange(len(ta)), ss]
    best = fr.argmax(1)
    print(f"\n[기술 선택] 교사/학생이 모두 기술을 골랐지만 서로 다른 상태 {both.sum()}개 (기술칸 기준, 테라 여부 무시)")
    print(f"  교사 선택의 추정 피해가 학생 선택보다 큼: {(t_f[both] > s_f[both]).mean():.3f} / 작음: {(t_f[both] < s_f[both]).mean():.3f}")
    ok = V & (ta < 8)
    print(f"  교사 선택 = 추정 최대 피해 기술: {(ts[ok] == best[ok]).mean():.3f} | 학생 선택 = 추정 최대 피해 기술: {(ss[ok & (sa < 8)] == best[ok & (sa < 8)]).mean():.3f}")


if __name__ == "__main__":
    main()
