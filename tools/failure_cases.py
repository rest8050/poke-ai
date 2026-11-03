"""학생이 크게 틀린 결정(학생 argmax regret >= --big)을 무작위로 --n개 뽑아 읽을 수 있는 형태로 출력: 내/상대 활성(HP, 상태, 랭크, 도구, 기술), 벤치, 필드, 교사/학생의 상위 행동과 정책 질량, 탐색 Q.
사용: python tools/failure_cases.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz --n 30 --seed 0"""
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

V = json.load(open("data/vocab.json", encoding="utf-8"))
INV = {k: {i: n for n, i in V[k].items()} for k in ("species", "move", "item", "status", "type")}
SIDE = [("리플렉트", 0), ("빛의장막", 2), ("오로라베일", 4), ("순풍", 6), ("스텔스록", 8), ("압정", 9), ("독압정", 10), ("끈적끈적네트", 11)]
GLOBAL = [("쾌청", 0), ("비", 2), ("모래바람", 4), ("설경", 6), ("일렉트릭필드", 12), ("그래스필드", 14), ("사이코필드", 16), ("미스트필드", 18), ("트릭룸", 10), ("중력", 46)]
BOOST = ((3, "공"), (4, "방"), (5, "특공"), (6, "특방"), (7, "스피드"))


def side_str(fv, base):
    out = [n if o not in (9, 10) else f"{n}x{round(fv[base + o] * (3 if o == 9 else 2))}" for n, o in SIDE if fv[base + o] > 0]
    return ",".join(out) or "없음"


def mon(cat, num, i, with_moves=False):
    sp = INV["species"].get(int(cat[i, 10]), "?")
    st = INV["status"].get(int(cat[i, 4]), "")
    s = f"{sp} {num[i, 2] * 100:.0f}%" + (f" [{st}]" if st not in ("", "none", "<none>", "?") and int(cat[i, 4]) > 1 else "")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v8_full")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--big", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = args.sets.split(",")
    sets = [L.load_set(p) for p in paths]
    zs = [np.load(p) for p in paths]
    q, vis = np.concatenate([z["q_mean"] for z in zs]), np.concatenate([z["q_vis"] for z in zs])
    cat = lambda k: np.concatenate([s[k] for s in sets])
    T, M = cat("target").astype(np.float64), cat("action_mask")
    rel = (vis >= 100) & ~np.isnan(q) & M
    valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
    qstar = np.where(rel, q, -np.inf).max(1)
    net = model_from_ckpt(args.models if os.path.exists(args.models) else f"checkpoints/supervised_v2_fp_{args.models}.pt").eval()
    P = np.concatenate([L.infer(net, s, torch.device("cpu")) for s in sets])
    reg, *_ = regrets(P, q, rel, qstar, T)
    reg = np.where(valid, reg, np.nan)
    mc, mn, oc, on, fv = cat("my_team_cat"), cat("my_team_num"), cat("opp_team_cat"), cat("opp_team_num"), cat("field_vec")
    turn = np.concatenate([np.arange(n) for s in sets for n in s["lens"]])
    src = np.concatenate([np.full(len(s["target"]), i) for i, s in enumerate(sets)])
    a_t, a_s = T.argmax(1), P.argmax(1)
    big = np.where(reg >= args.big)[0]
    kind = lambda a: "교체" if a >= 8 else ("테라" if a >= 4 else "기술")
    print(f"모델 {args.models} | 큰 실수(regret >= {args.big}) {len(big)}건 / 비교 가능 {int(valid.sum())}건 ({len(big) / valid.sum():.1%}), 전체 regret의 {np.nansum(reg[big]) / np.nansum(reg):.0%}")
    for kt in ("기술", "테라", "교체"):
        m = [i for i in big if kind(a_t[i]) == kt]
        print(f"  교사 선택 {kt}: {len(m)}건 ({len(m) / len(big):.0%}), 평균 regret {np.mean(reg[m]):.3f}")
    print("  (아래 표본은 큰 실수 전체에서 무작위 추출: 위 비율이 곧 표본의 기대 구성)\n")
    rng = np.random.RandomState(args.seed)
    for n_, i in enumerate(sorted(rng.choice(big, min(args.n, len(big)), replace=False)), 1):
        ma, oa = mn[i, :, 0].argmax(), on[i, :, 0].argmax()
        names = [INV["move"].get(int(c), "?") for c in mc[i, ma, 5:9]]
        onames = [INV["move"].get(int(c), "?") for c in oc[i, oa, 5:9] if int(c) > 2]
        bst = " ".join(f"{n}{mn[i, ma, k] * 6:+.0f}" for k, n in BOOST if abs(mn[i, ma, k]) > 0.01)
        obst = " ".join(f"{n}{on[i, oa, k] * 6:+.0f}" for k, n in BOOST if abs(on[i, oa, k]) > 0.01)
        tera = f"테라 {INV['type'].get(int(mc[i, ma, 9]), '?')}" + (" (가능)" if mn[i, ma, 10] > 0 else " (사용됨/불가)")
        bench = ", ".join(mon(mc[i], mn[i], k) for k in range(6) if k != ma and mc[i, k, 10] > 2)
        obench = ", ".join(mon(oc[i], on[i], k) for k in range(6) if k != oa and oc[i, k, 10] > 2)
        glob = ",".join(n for n, o in GLOBAL if fv[i, o] > 0) or "없음"
        def act(a):
            if a >= 8:
                return "교체 -> " + mon(mc[i], mn[i], a - 8)
            return ("테라+" if a >= 4 else "") + names[a % 4]
        def top(p, k=3):
            return " / ".join(f"{act(a)} {p[i, a]:.2f}" + (f"(Q {q[i, a]:.2f})" if rel[i, a] else "") for a in np.argsort(-p[i])[:k] if p[i, a] > 0.03)
        Q = np.where(rel[i], q[i], np.nan)
        print(f"[{n_}] 세트{src[i]} 턴{turn[i] + 1} | 교사 {kind(a_t[i])} 대 학생 {kind(a_s[i])} | regret {reg[i]:.2f} (Q* {qstar[i]:.2f}, 최저 {np.nanmin(Q):.2f})")
        print(f"  내 활성: {mon(mc[i], mn[i], ma)} {bst} | 도구 {INV['item'].get(int(mc[i, ma, 0]), '?')} | {tera} | 기술 {names}")
        print(f"  상대 활성: {mon(oc[i], on[i], oa)} {obst} | 공개 기술 {onames}")
        print(f"  내 벤치: {bench} | 상대 벤치(공개): {obench}")
        print(f"  필드: {glob} | 내 쪽 {side_str(fv[i], 20)} | 상대 쪽 {side_str(fv[i], 32)}")
        print(f"  교사: {top(T)}")
        print(f"  학생: {top(P)}\n")


if __name__ == "__main__":
    main()
