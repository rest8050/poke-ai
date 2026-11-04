"""필드/도구/상태 범주별 평가: 곱셈 보정이 필요한 특수 상황(트릭룸, 순풍, 벽, 날씨, 지형, 구애·증폭 도구, 마비)에서 학생이 교사와 얼마나 다른가.
지표: 결정마다 KL(교사 || 학생)과 교사 argmax 일치. 범주 평균 - 나머지 평균(차이)을 판(배틀) 단위 부트스트랩 95%로 냄 (양수 = 범주에서 더 나쁨)
범주는 서로 겹칠 수 있고 게임 단계와 얽히므로(예: 트릭룸은 중후반) 차이는 "특수 상황의 난이도 + 학생의 약점"의 합으로 읽을 것
사용: python tools/field_eval.py --models v6_full --sets data/dagger/dg5.npz,data/dagger/d6_ho.npz   (모델 = checkpoints/supervised_v2_fp_<이름>.pt, 판단 가능한 결정만 = 합법 행동 2개 이상)"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

V = json.load(open("data/vocab.json"))
ITEMS = lambda *names: [V["item"][n] for n in names]
CHOICE = ITEMS("choiceband", "choicespecs", "choicescarf")
BOOST = ITEMS("lifeorb", "assaultvest", "eviolite")
PAR = V["status"]["par"]
# field_vec 위치: 날씨 0 쾌청 / 2 비 / 4 모래 / 6 눈, 지형 12 14 16 18, 트릭룸 10, 내 쪽 벽 20 22 24, 순풍 26, 상대 쪽 벽 32 34 36, 순풍 38


def categories(d):
    fv, n = d["field_vec"], len(d["field_vec"])
    ar = np.arange(n)
    ma, oa = d["my_team_num"][..., 0].argmax(1), d["opp_team_num"][..., 0].argmax(1)
    my_item, opp_item = d["my_team_cat"][ar, ma, 0], d["opp_team_cat"][ar, oa, 0]
    my_st, opp_st = d["my_team_cat"][ar, ma, 4], d["opp_team_cat"][ar, oa, 4]
    on = lambda *cols: (fv[:, list(cols)] > 0.5).any(1)
    cats = {
        "트릭룸": on(10),
        "순풍 (양쪽)": on(26, 38),
        "벽 (상대 쪽: 내 데미지 감소)": on(32, 34, 36),
        "벽 (내 쪽: 받는 데미지 감소)": on(20, 22, 24),
        "쾌청": on(0), "비": on(2), "모래": on(4), "눈": on(6),
        "지형": on(12, 14, 16, 18),
        "내 활성 구애 계열 (밴드/안경/스카프)": np.isin(my_item, CHOICE),
        "상대 활성 구애 계열 (공개된 것만)": np.isin(opp_item, CHOICE),
        "내 활성 증폭 도구 (구슬/조끼/진화의휘석)": np.isin(my_item, BOOST),
        "마비 (내 활성 또는 상대 활성)": (my_st == PAR) | (opp_st == PAR),
    }
    any_special = np.zeros(n, bool)
    for m in cats.values():
        any_special |= m
    cats["위 중 하나라도"] = any_special
    cats["아무것도 없음 (기준)"] = ~any_special
    return cats


def cat_vs_rest(x, mask, battle, rng, n_boot=2000):
    u, inv = np.unique(battle, return_inverse=True)
    nb = len(u)
    s_in = np.bincount(inv, weights=x * mask, minlength=nb)
    n_in = np.bincount(inv, weights=mask.astype(float), minlength=nb)
    s_all, n_all = np.bincount(inv, weights=x, minlength=nb), np.bincount(inv, minlength=nb).astype(float)
    s_out, n_out = s_all - s_in, n_all - n_in
    m_in, m_out = s_in.sum() / max(1, n_in.sum()), s_out.sum() / max(1, n_out.sum())
    ds = []
    for _ in range(n_boot):
        p = rng.randint(0, nb, nb)
        ni, no = n_in[p].sum(), n_out[p].sum()
        if ni > 0 and no > 0:
            ds.append(s_in[p].sum() / ni - s_out[p].sum() / no)
    return m_in, m_out, m_in - m_out, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v6_full")
    ap.add_argument("--sets", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(0)
    sets = [L.load_set(p) for p in args.sets.split(",")]
    cat_parts, battle, off = {}, [], 0
    for d in sets:
        for k, v in categories(d).items():
            cat_parts.setdefault(k, []).append(v)
        battle.append(np.repeat(np.arange(len(d["lens"])), d["lens"]) + off)
        off += len(d["lens"])
    cats = {k: np.concatenate(v) for k, v in cat_parts.items()}
    battle, T = np.concatenate(battle), np.concatenate([d["target"] for d in sets]).astype(np.float64)
    amask = np.concatenate([d["action_mask"] for d in sets])
    ok = amask.sum(1) >= 2                                                         # 고를 게 있는 결정만
    for name in args.models.split(","):
        net = model_from_ckpt(f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
        P = np.concatenate([L.infer(net, d, dev) for d in sets])
        pd = L.per_decision(P, T)
        print(f"\n===== {name}: 배틀 {len(np.unique(battle[ok]))}, 결정 {int(ok.sum())} | 평균 KL {pd['total'][ok].mean():.4f} | 일치 {pd['agree'][ok].mean():.3f} =====")
        print(f"{'범주':36s} {'결정':>6s} {'배틀':>5s} | {'KL 범주':>8s} {'KL 나머지':>9s} {'차이 [95%]':>26s} | {'일치 범주':>8s} {'나머지':>7s} {'차이 [95%]':>24s}")
        for cn, m in cats.items():
            m = m & ok
            if m.sum() < 50:
                print(f"{cn:36s} {int(m.sum()):6d}   (결정 50개 미만 생략)")
                continue
            sub = ok                                                                # 나머지 = 같은 판단 가능 결정 중 범주 밖
            xs = {k: pd[k][sub] for k in ("total", "agree")}
            mm, bb = m[sub], battle[sub]
            row = {k: cat_vs_rest(xs[k], mm, bb, rng) for k in xs}
            k, a = row["total"], row["agree"]
            print(f"{cn:36s} {int(m.sum()):6d} {len(np.unique(battle[m])):5d} | {k[0]:8.4f} {k[1]:9.4f} {k[2]:+8.4f} [{k[3]:+.4f}, {k[4]:+.4f}] | {a[0]:8.3f} {a[1]:7.3f} {a[2]:+8.3f} [{a[3]:+.3f}, {a[4]:+.3f}]")


if __name__ == "__main__":
    main()
