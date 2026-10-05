"""장부 전체(개별 판 + 이식된 집계)로 Bradley–Terry 레이팅(Elo 척도)과 95% 구간 계산.
사용: python tools/bt_report.py [--ref v10boot] [--ledger PATH] [--boot 300]
구간은 적합된 승률로 쌍별 승수를 다시 뽑아(모수적 부트스트랩) 재적합한 분포. 대전이 이어지지 않은 모델 묶음은 따로 경고"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from tools.ledger import LEDGER, read

ELO = 400 / math.log(10)


def win_table(rows):
    wins = defaultdict(float)   # (이긴 모델, 진 모델) -> 승수
    for r in rows:
        if r["kind"] == "game":
            if r["winner"] == "a":
                wins[(r["a"], r["b"])] += 1
            elif r["winner"] == "b":
                wins[(r["b"], r["a"])] += 1
        elif r["kind"] == "agg":
            wins[(r["a"], r["b"])] += r["wins_a"]
            wins[(r["b"], r["a"])] += r["wins_b"]
    return wins


def fit_matrix(W, iters=300):
    N = W + W.T
    W = W + 0.5 * (N > 0)   # 약한 사전(쌍마다 양방향 0.5승): 전패 모델이 -무한대로 발산하지 않게
    N = W + W.T
    p = np.ones(len(W))
    for _ in range(iters):
        p = W.sum(1) / np.maximum((N / (p[:, None] + p[None, :])).sum(1), 1e-12)
        p /= math.exp(np.log(p).mean())
    return np.log(p) * ELO


def fit_bt(wins, ref=None, boot=0, seed=0):
    """wins -> {모델: (elo, lo, hi)}  (ref가 있으면 그 모델이 0, 없으면 평균 0). boot=0이면 구간 생략"""
    names = sorted({m for pair in wins for m in pair})
    idx = {m: i for i, m in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for (i, j), c in wins.items():
        W[idx[i], idx[j]] += c
    elo = fit_matrix(W)
    off = elo[idx[ref]] if ref in idx else 0.0
    out = {m: [elo[idx[m]] - off, None, None] for m in names}
    if boot:
        rng, N = np.random.default_rng(seed), W + W.T
        P = 1 / (1 + np.exp(-(elo[:, None] - elo[None, :]) / ELO))
        samples = []
        for _ in range(boot):
            Wb = np.zeros_like(W)
            iu = np.triu_indices(len(names), 1)
            Wb[iu] = rng.binomial(N[iu].astype(int), P[iu])
            Wb.T[iu] = N[iu] - Wb[iu]
            e = fit_matrix(Wb, 100)
            samples.append(e - (e[idx[ref]] if ref in idx else 0.0))
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        for m in names:
            out[m][1], out[m][2] = lo[idx[m]], hi[idx[m]]
    return {m: tuple(v) for m, v in out.items()}, names, W


def components(names, W):
    seen, groups = set(), []
    for s in range(len(names)):
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in range(len(names)) if W[u, v] + W[v, u] > 0]
        seen |= comp
        groups.append(sorted(names[i] for i in comp))
    return groups


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="v10boot"); ap.add_argument("--ledger", default=LEDGER); ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    wins = win_table(read(a.ledger))
    if not wins:
        sys.exit("장부가 비어 있음 (python tools/ledger.py backfill 로 기존 결과를 넣을 수 있음)")
    res, names, W = fit_bt(wins, a.ref, a.boot)
    games = {m: int(sum(c for pair, c in wins.items() if m in pair)) for m in names}
    print(f"{'모델':12s} {'Elo(vs ' + a.ref + ')':>16s} {'95% 구간':>18s} {'판 수':>8s}")
    for m, (e, lo, hi) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print(f"{m:12s} {e:+16.1f} {f'[{lo:+.0f}, {hi:+.0f}]':>18s} {games[m]:8d}")
    groups = components(names, W)
    if len(groups) > 1:
        print("\n경고: 서로 대전한 적 없는 모델 묶음이 있어 묶음 사이 비교는 무의미:", groups)
