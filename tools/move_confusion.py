"""기능별 혼동표: 학생이 교사와 어떤 "종류"의 행동에서 어긋나는가 (공격/셋업/회복/설치/보호/교체/테라 ...).
행동을 내 활성 포켓몬의 기술 기능으로 분류(기술 id -> 부가효과 벡터 + 공격/변화 구분)하고, 자발 결정(기술과 교체가 둘 다 가능)에서
(1) 교사 정책 질량 대 학생 정책 질량의 기능별 평균 (argmax 노이즈에 덜 민감), (2) 교사 argmax 기능별 학생 불일치율/평균 regret/전체 regret 몫, (3) 기능 간 혼동(교사 -> 학생 argmax)을 낸다.
날카로운 결정(Q* - 최저Q >= 0.3)에서 따로 한 번 더. 사용: python tools/move_confusion.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import regrets
from src.core.model import model_from_ckpt
from src.core.tensor_encoder import MOVE_EFFECT_NAMES, _move_obj, move_effect_features

FUNCS = ["공격", "공격(교체기)", "셋업", "회복", "설치", "제거", "필드/벽", "보호", "상태이상/디버프", "교체기(변화)", "강제교체", "방해/대타", "기타 변화", "테라", "교체"]
F = {n: i for i, n in enumerate(FUNCS)}
EF = {n: i for i, n in enumerate(MOVE_EFFECT_NAMES)}


def move_func(name):
    mv = _move_obj(name)
    if mv is None:
        return F["기타 변화"]
    e = move_effect_features(name)
    pos = lambda *ks: any(e[EF[k]] > 0 for k in ks)
    if mv.category.name != "STATUS":
        return F["공격(교체기)"] if pos("self_switch") else F["공격"]
    for cond, f in ((pos("self_switch"), "교체기(변화)"), (pos("protect"), "보호"), (pos("heal"), "회복"), (pos("hazard_set"), "설치"), (pos("hazard_removal"), "제거"),
                    (pos("screen", "trick_room", "weather_terrain_tailwind"), "필드/벽"), (pos("self_offense", "self_defense", "self_speed"), "셋업"),
                    (pos("tgt_brn", "tgt_par", "tgt_psn", "tgt_slp", "tgt_frz", "tgt_confusion") or any(e[EF[k]] < 0 for k in ("tgt_offense", "tgt_defense", "tgt_speed")), "상태이상/디버프"),
                    (pos("force_switch"), "강제교체"), (pos("disrupt", "substitute"), "방해/대타")):
        if cond:
            return F[f]
    return F["기타 변화"]


def action_funcs(d, id2name):
    """[N,22] 각 행동의 기능 번호 (교체 = '교체', 테라 기술 = '테라', 나머지는 활성 포켓몬의 기술 기능)"""
    N = len(d["lens"]) and d["action_mask"].shape[0]
    act = d["my_team_num"][:, :, 0].argmax(1)
    ids = d["my_team_cat"][np.arange(N), act][:, 5:9]
    cache = {}
    mf = np.zeros((N, 4), int)
    for i in range(N):
        for k in range(4):
            m = int(ids[i, k])
            if m not in cache:
                cache[m] = move_func(id2name.get(m, ""))
            mf[i, k] = cache[m]
    out = np.full((N, 22), F["교체"])
    out[:, :4] = mf
    out[:, 4:8] = F["테라"]
    return out


def table(title, sel, T, P, AF, reg, tot):
    """sel 결정에서: 기능별 교사/학생 평균 질량, 교사 argmax 기능별 불일치율/regret"""
    n = int(sel.sum())
    onehot = np.eye(len(FUNCS))[AF]                                   # [N,22,F]
    mt = (T[sel][..., None] * onehot[sel]).sum(1).mean(0)
    ms = (P[sel][..., None] * onehot[sel]).sum(1).mean(0)
    ft, fs = AF[sel, T[sel].argmax(1)], AF[sel, P[sel].argmax(1)]
    r = reg[sel]
    print(f"\n--- {title} (결정 {n}) ---")
    print(f"{'기능':14s} {'교사 질량':>9s} {'학생 질량':>9s} {'학생-교사':>9s} | {'교사 argmax 수':>13s} {'기능 불일치':>10s} {'평균 regret':>11s} {'regret 몫':>9s}")
    for i, name in enumerate(FUNCS):
        s = ft == i
        cnt = int(s.sum())
        miss = np.mean(fs[s] != i) if cnt else np.nan
        rr = np.nanmean(r[s]) if cnt and (~np.isnan(r[s])).any() else np.nan
        share = np.nansum(r[s]) / tot if cnt else 0.0
        print(f"{name:14s} {mt[i]:9.3f} {ms[i]:9.3f} {ms[i] - mt[i]:+9.3f} | {cnt:13d} {miss:10.3f} {rr:11.4f} {share:9.3f}")
    conf = np.zeros((len(FUNCS), len(FUNCS)))
    np.add.at(conf, (ft, fs), 1)
    print("혼동 (교사 argmax 기능 -> 학생 argmax 기능, 행 기준 비율. 대각선 제외 상위 3개만)")
    for i, name in enumerate(FUNCS):
        if conf[i].sum() < 100:
            continue
        row = conf[i] / conf[i].sum()
        off = [(j, row[j]) for j in np.argsort(-row) if j != i and row[j] >= 0.03][:3]
        print(f"  {name:14s} 일치 {row[i]:.3f} | " + ", ".join(f"{FUNCS[j]} {v:.3f}" for j, v in off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--min-vis", type=float, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    id2name = {v: k for k, v in json.load(open("data/vocab.json", encoding="utf-8"))["move"].items()}
    sets = [L.load_set(p) for p in args.sets.split(",")]
    zs = [np.load(p) for p in args.sets.split(",")]
    cat = lambda k: np.concatenate([s[k] for s in sets])
    d = {k: cat(k) for k in ("my_team_cat", "my_team_num", "action_mask")}
    d["lens"] = np.concatenate([s["lens"] for s in sets])
    T, M = cat("target").astype(np.float64), d["action_mask"]
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    spread = np.where(valid, qstar - np.where(rel, q, np.inf).min(1), np.nan)
    vol = (M[:, :4].sum(1) > 0) & (M[:, 8:14].sum(1) > 0)
    AF = action_funcs(d, id2name)
    dev = torch.device(args.device)
    for name in args.models.split(","):
        net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
        P = np.concatenate([L.infer(net, s, dev) for s in sets])
        reg_s, *_ = regrets(P, q, rel, qstar, T)
        reg = np.where(valid, reg_s, np.nan)
        tot = np.nansum(np.where(vol, reg, np.nan))
        print(f"\n===== {name} | 자발 결정 {int((vol & valid).sum())} (규칙: 기술과 교체가 둘 다 가능) =====")
        table("자발 결정 전체", vol & valid, T, P, AF, reg, tot)
        table("자발 + 날카로운 결정 (Q*-최저Q >= 0.3)", vol & valid & (spread >= 0.3), T, P, AF, reg, tot)


if __name__ == "__main__":
    main()
