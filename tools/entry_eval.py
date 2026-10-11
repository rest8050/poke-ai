"""진입 후 매치업 범주 평가 (v6 합격선): 특징 포켓몬으로의 교체 질량이 교사와 얼마나 같은가.
범주: A2 상대 공/특공 +2 이상 & 교체 후보에 천진 (특징 = 천진 후보) / A0 상대 랭크업 없음 & 천진 후보 있음 (대조: 오작동 점검) /
      C4 내 쪽 스텔스록 & 약점(2배+) 후보 / C5 스텔스록 & 저항(0.5배-) 후보 — 모두 자발 결정(기술·교체 둘 다 가능)만
지표: 결정마다 오차 = 학생의 특징 질량 - 교사의 특징 질량. 편향 = 오차 평균(0이면 교사와 같음), 평균 |오차|. 구간은 판(배틀) 단위 부트스트랩 95%
기준 모델 대비 평균 |오차|의 차이도 냄 (음수가 개선).
사용: python tools/entry_eval.py --models v5_leg,v6_ent --base v5_leg --sets sp4=data/v6/fp_data_sp4.npz,dg5=data/v6/dg5.npz  (모델 = checkpoints/supervised_v2_fp_<이름>.pt)"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.matchup import Matchup
from src.core.model import model_from_ckpt


def categories(path, d):
    """이름 -> (결정 마스크, 특징 슬롯 [N,6])"""
    z = np.load(path)
    V = json.load(open("data/vocab.json"))
    ab, fv, am = z["my_team_cat"][..., 1], z["field_vec"], d["action_mask"]
    N = len(am)
    ar = np.arange(N)
    vol = am[:, :4].any(1) & am[:, 8:14].any(1)
    bench = am[:, 8:14]
    on = d["opp_team_num"]
    ob = on[ar, on[..., 0].argmax(1), 3:8] * 6
    un = bench & (ab == V["ability"]["unaware"])
    chart, rock = Matchup().chart, V["type"]["rock"]
    t1, t2 = torch.tensor(z["my_team_cat"][..., 2]), torch.tensor(z["my_team_cat"][..., 3])
    eff = (chart[rock, t1] * chart[rock, t2]).numpy()
    sr = fv[:, 28] > 0
    weak, resist = bench & (eff >= 2), bench & (eff <= 0.5)
    boosted = (ob[:, 0] >= 2) | (ob[:, 2] >= 2)
    any_up = (ob[:, 0] >= 1) | (ob[:, 2] >= 1)
    return {"A2 상대 +2 이상 & 천진 후보": (vol & boosted & un.any(1), un),
            "A0 상대 랭크업 없음 & 천진 후보 (대조)": (vol & ~any_up & un.any(1), un),
            "C4 스텔스록 & 약점 후보": (vol & sr & weak.any(1), weak),
            "C5 스텔스록 & 저항 후보": (vol & sr & resist.any(1), resist)}


def ci(x, battle, rng, n_boot=2000):
    u, inv = np.unique(battle, return_inverse=True)
    s, n = np.bincount(inv, weights=x), np.bincount(inv)
    ds = [(s[p].sum() / n[p].sum()) for p in (rng.randint(0, len(u), len(u)) for _ in range(n_boot))]
    return x.mean(), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    models = args.models.split(",")
    base = args.base or models[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = {n: model_from_ckpt(f"checkpoints/supervised_v2_fp_{n}.pt").to(dev).eval() for n in models}
    rng = np.random.RandomState(0)
    for spec in args.sets.split(","):
        name, path = spec.split("=", 1)
        d = L.load_set(path)
        _, battle = L.categorize(d)
        T = d["target"].astype(np.float64)
        P = {n: L.infer(net, d, dev) for n, net in nets.items()}
        print(f"\n===== {name} ({path}): 배틀 {len(d['lens'])}, 결정 {len(battle)} =====")
        for cn, (m, slots) in categories(path, d).items():
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            t_mass = (T[:, 8:14] * slots).sum(1)
            err = {n: (P[n][:, 8:14] * slots).sum(1) - t_mass for n in models}
            print(f"\n{cn} (n={len(idx)}, 배틀 {len(np.unique(battle[idx]))}) 교사 특징 질량 {t_mass[idx].mean():.3f}")
            for n in models:
                b = ci(err[n][idx], battle[idx], rng)
                print(f"  {n:12s} 학생 특징 질량 {(P[n][:, 8:14] * slots).sum(1)[idx].mean():.3f} | 편향 {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] | 평균 |오차| {np.abs(err[n][idx]).mean():.3f}")
                if n != base:
                    dm = ci(np.abs(err[n][idx]) - np.abs(err[base][idx]), battle[idx], rng)
                    print(f"  {'':12s} {base} 대비 평균 |오차| 차이 {dm[0]:+.4f} [{dm[1]:+.4f}, {dm[2]:+.4f}]")


if __name__ == "__main__":
    main()
