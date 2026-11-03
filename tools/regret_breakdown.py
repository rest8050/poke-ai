"""학생 regret이 어떤 결정에 몰려 있나: regret_eval과 같은 정의(방문 수 >= --min-vis인 Q만 믿음)로 결정을 여러 축으로 나눠 평균 regret, 해로운 선택 비율, 전체 regret 중 몫을 낸다.
축: 교사 선택 종류 / 판 진행(턴) / 교체 가능 벤치 수 / 강제 교체 / 결정의 중요도(Q* - 최저 Q) / 교사 확신 / 드러난 상대 기술 수(상대 정보량).
교사 자기 일치 상한(logs/teacher_ceiling.txt, 같은 상태를 두 번 라벨링한 regret 0.0095, 몰표 결정 포함. 구간별은 logs/teacher_ceiling_stakes.txt)과 비교해 구조로 줄일 수 있는 몫이 어디인지 가늠하는 용도.
사용: python tools/regret_breakdown.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import HARM, boot_ci, regrets
from src.core.model import model_from_ckpt


def axes(d, a_t, spread, conf):
    """(축 이름, [(구간 이름, 결정 마스크)]) 목록"""
    m = d["action_mask"]
    n_mv, n_sw = m[:, :4].sum(1), m[:, 8:14].sum(1)
    turn = np.concatenate([np.arange(n) for n in d["lens"]])
    seen = (np.abs(d["opp_move_num"]).sum(-1) > 0).sum((1, 2))          # 드러난 상대 기술 칸 수 (6마리 x 4칸 중)
    return [
        ("교사 선택", [("기술", a_t < 4), ("테라 기술", (a_t >= 4) & (a_t < 8)), ("교체", a_t >= 8)]),
        ("턴", [("1~5", turn < 5), ("6~15", (turn >= 5) & (turn < 15)), ("16~30", (turn >= 15) & (turn < 30)), ("31+", turn >= 30)]),
        ("교체 가능 벤치", [("0", n_sw == 0), ("1~2", (n_sw >= 1) & (n_sw <= 2)), ("3+", n_sw >= 3)]),
        ("강제 교체(기술 불가)", [("예", n_mv == 0), ("아니오", n_mv > 0)]),
        ("중요도 Q*-최저Q", [("<0.1", spread < 0.1), ("0.1~0.3", (spread >= 0.1) & (spread < 0.3)), (">=0.3", spread >= 0.3)]),
        ("교사 확신", [("<0.5", conf < 0.5), ("0.5~0.8", (conf >= 0.5) & (conf < 0.8)), (">=0.8", conf >= 0.8)]),
        ("드러난 상대 기술 칸", [("0~3", seen <= 3), ("4~7", (seen >= 4) & (seen <= 7)), ("8+", seen >= 8)]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--min-vis", type=float, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = [L.load_set(p) for p in args.sets.split(",")]
    zs = [np.load(p) for p in args.sets.split(",")]
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    d = {k: np.concatenate([s[k] for s in sets]) for k in sets[0] if k != "lens"}
    d["lens"] = np.concatenate([s["lens"] for s in sets])
    T, M = d["target"].astype(np.float64), d["action_mask"]
    off = np.cumsum([0] + [len(s["lens"]) for s in sets])
    battle = np.concatenate([np.repeat(np.arange(len(s["lens"])), s["lens"]) + o for s, o in zip(sets, off)])
    rel = (vis >= args.min_vis) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    spread = np.where(valid, qstar - np.where(rel, q, np.inf).min(1), np.nan)
    a_t, conf = T.argmax(1), T.max(1)
    groups = axes(d, a_t, spread, conf)
    rng = np.random.RandomState(0)
    for name in args.models.split(","):
        net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
        P = np.concatenate([L.infer(net, s, dev) for s in sets])
        reg_s, _, _, _, _, _ = regrets(P, q, rel, qstar, T)
        reg = np.where(valid, reg_s, np.nan)
        tot = np.nansum(reg)
        print(f"\n===== {name} | 비교 가능 결정 {int(valid.sum())} | 평균 regret {np.nanmean(reg):.4f} (교사 자기 일치 상한 0.0095) =====")
        print(f"{'축':22s} {'구간':10s} {'결정':>6s} {'평균 regret':>11s} {'95% 구간':>17s} {'해로움':>7s} {'regret 몫':>9s}")
        for ax, bins in groups:
            for bn, sel in bins:
                s = sel & valid
                if s.sum() < 50:
                    continue
                m, lo, hi = boot_ci(np.where(s, reg, np.nan), battle, rng, 300)
                print(f"{ax:22s} {bn:10s} {int(s.sum()):6d} {m:11.4f} [{lo:.4f},{hi:.4f}] {np.mean(reg[s] >= HARM):7.3f} {np.nansum(reg[s]) / tot:9.3f}")


if __name__ == "__main__":
    main()
