"""쌍 비교: 같은 결정에서 두 모델의 KL / 일치 / regret 차이(B - A)를 "규칙이 닿는 결정" 범주별로 판 단위 부트스트랩으로 잰다.
범주는 matchup_v7 규칙이 영향을 줄 수 있는 결정 (규칙 특성이 활성에 있음, 위협 교체 후보, 스피드 규칙이 걸리는 상황 등). 결정 간 분산이 사라져서 묶음별 표(ability_eval)보다 훨씬 민감함.
음수 = B가 더 낮음 (KL/regret은 B가 낫다, 일치는 B가 나쁘다). q_mean/q_vis가 든 npz만 가능 (regret 때문).
사용: python tools/paired_eval.py --a v8_full --b v9 --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz   (모델 = checkpoints/supervised_v2_fp_<이름>.pt)"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import boot_ci, regrets
from src.core.matchup_v7 import IMMUNE_ABILITIES, INTIM_RESULT, RAIN, SUN, TAILWIND_MY, TAILWIND_OPP, TRICK_ROOM, WEATHER_SPEED
from src.core.model import model_from_ckpt

IMMUNE = " ".join(IMMUNE_ABILITIES)
REACT = " ".join(INTIM_RESULT)
FAMILIES = {                       # 이름: 특성 목록 (활성 포켓몬 양쪽 중 하나라도)
    "정밀기술 (technician)": "technician",
    "두꺼운지방 (thickfat)": "thickfat",
    "흡수/부유 면역 계열": IMMUNE,
    "옹골찬 몸 (sturdy)": "sturdy",
    "장난꾸러기 (prankster)": "prankster",
    "위협 (intimidate)": "intimidate",
    "재생력 (regenerator)": "regenerator",
    "날씨 스피드 (엽록소/쓱쓱)": " ".join(WEATHER_SPEED),
    "위협 반응 특성 (분발/오기 등)": REACT,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--min-vis", type=float, default=100)
    args = ap.parse_args()
    V = json.load(open("data/vocab.json"))
    ab, st, it = V["ability"], V["status"], V["item"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets, qs, vs = [], [], []
    for p in args.sets.split(","):
        z = np.load(p)
        if "q_mean" not in z.files:
            raise SystemExit(f"❌ {p}에 q_mean이 없음")
        sets.append(L.load_set(p)); qs.append(z["q_mean"]); vs.append(z["q_vis"])
    cat = lambda k: np.concatenate([d[k] for d in sets])
    T, M = cat("target").astype(np.float64), cat("action_mask")
    q, vis = np.concatenate(qs), np.concatenate(vs)
    off = np.cumsum([0] + [len(d["lens"]) for d in sets])
    battle = np.concatenate([np.repeat(np.arange(len(d["lens"])), d["lens"]) + o for d, o in zip(sets, off)])
    n, ar = len(T), np.arange(len(T))
    ok = M.sum(1) >= 2
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = ok & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)

    mc, mn, oc, on = cat("my_team_cat"), cat("my_team_num"), cat("opp_team_cat"), cat("opp_team_num")
    fv = cat("field_vec")
    ma, oa = mn[..., 0].argmax(1), on[..., 0].argmax(1)
    my_ab, opp_ab = mc[..., 1][ar, ma], oc[..., 1][ar, oa]
    bench = (mn[..., 0] < 0.5) & (mn[..., 1] < 0.5) & (mn[..., 11] > 0)
    ids = lambda names: [ab[x] for x in names.split() if x in ab]
    groups = {}
    for name, names in FAMILIES.items():
        i = ids(names)
        groups[name] = np.isin(my_ab, i) | np.isin(opp_ab, i)
    groups["위협 교체 후보 (내 벤치에 위협)"] = (np.isin(mc[..., 1], ids("intimidate")) & bench).any(1)
    wi = ids(" ".join(WEATHER_SPEED))
    sunrain = (fv[:, SUN] > 0.5) | (fv[:, RAIN] > 0.5)
    groups["날씨 스피드 특성 + 쾌청/비"] = (np.isin(my_ab, wi) | np.isin(opp_ab, wi)) & sunrain
    groups["순풍 (어느 쪽이든)"] = (fv[:, TAILWIND_MY] > 0.5) | (fv[:, TAILWIND_OPP] > 0.5)
    groups["트릭룸"] = fv[:, TRICK_ROOM] > 0.5
    groups["마비 (활성)"] = (mc[..., 4][ar, ma] == st["par"]) | (oc[..., 4][ar, oa] == st["par"])
    groups["내 활성 구애스카프"] = mc[..., 0][ar, ma] == it["choicescarf"]
    anyrule = np.zeros(n, bool)
    for g in groups.values():
        anyrule |= g
    groups["위 중 하나라도"] = anyrule
    groups["해당 없음 (기준)"] = ~anyrule
    groups["전체"] = np.ones(n, bool)

    res = {}
    for key, name in (("a", args.a), ("b", args.b)):
        net = model_from_ckpt(f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
        P = np.concatenate([L.infer(net, d, dev) for d in sets])
        reg_s = regrets(P, q, rel, qstar, T)[0]
        res[key] = (L.kl(T, P), (P.argmax(1) == T.argmax(1)).astype(float), np.where(valid, reg_s, np.nan))
    d_kl = np.where(ok, res["b"][0] - res["a"][0], np.nan)
    d_ag = np.where(ok, res["b"][1] - res["a"][1], np.nan)
    d_rg = res["b"][2] - res["a"][2]
    rng = np.random.RandomState(0)
    print(f"A = {args.a}, B = {args.b} | 결정 {int(ok.sum())} 판 {len(np.unique(battle))} | 차이는 B - A (KL/regret 음수 = B가 좋음, 일치 양수 = B가 좋음)")
    print(f"{'범주':34s} {'결정':>6s} {'판':>5s} | {'KL 차이 [95%]':>26s} | {'일치 차이 [95%]':>26s} | {'regret 차이 [95%]':>26s}")
    for name, g in groups.items():
        m = g & ok
        if m.sum() < 50:
            continue
        cols = []
        for x in (d_kl, d_ag, d_rg):
            mean, lo, hi = boot_ci(np.where(m, x, np.nan), battle, rng)
            cols.append(f"{mean:+.4f} [{lo:+.4f},{hi:+.4f}]")
        print(f"{name:34s} {int(m.sum()):6d} {len(np.unique(battle[m])):5d} | {cols[0]:>26s} | {cols[1]:>26s} | {cols[2]:>26s}")
    print("범주가 겹치므로 (한 결정이 여러 범주에 속함) 범주별 구간은 독립이 아님. 범주가 많으니 구간이 0을 살짝 벗어나는 한두 개는 우연일 수 있음")


if __name__ == "__main__":
    main()
