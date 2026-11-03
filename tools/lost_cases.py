"""학생이 FP300에게 진 판에서 "실제로 둔 수가 크게 틀리고(regret >= --big) 바로 다음 결정에서 FP 승률이 --drop 이상 떨어진" 결정을 판마다 하나씩 모아, 진행도가 고르게 퍼지도록 --n개를 읽을 수 있는 형태로 출력.
failure_cases.py와 같은 형식 (홀드아웃이 아니라 실제 대전 기록: data/winrate_v6_fp300/lab_stu.npz, 학생 시점, action_taken = 학생이 둔 수).
사용: python tools/lost_cases.py --n 10 --seed 0"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from failure_cases import GLOBAL, INV, BOOST, mon, side_str   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/winrate_v6_fp300/lab_stu.npz")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--big", type=float, default=0.15)
ap.add_argument("--drop", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
d = np.load(a.data)
lens, out, V, A, q, vis, M, T = (d[k] for k in ("lens", "outcome", "teacher_value", "action_taken", "q_mean", "q_vis", "action_mask", "target"))
mc, mn, oc, on, fv = (d[k] for k in ("my_team_cat", "my_team_num", "opp_team_cat", "opp_team_num", "field_vec"))
off = np.concatenate([[0], np.cumsum(lens)])
rel = (vis >= 100) & ~np.isnan(q) & M
qstar = np.where(rel, q, -np.inf).max(1)
ar = np.arange(len(V))
reg = np.where(rel[ar, A.clip(0)] & (rel.sum(1) >= 2), qstar - q[ar, A.clip(0)], np.nan)
cands = []                                                   # (진행도, 행)
rng = np.random.RandomState(a.seed)
for g in np.where(out < 0)[0]:
    rows = [i for i in range(off[g], off[g + 1] - 1) if reg[i] >= a.big and V[i + 1] - V[i] <= -a.drop]
    if rows:
        i = int(rng.choice(rows)); cands.append(((i - off[g] + 0.5) / lens[g], i, g))
cands.sort()
pick = [cands[int(j)] for j in np.linspace(0, len(cands) - 1, a.n)]
kind = lambda x: "교체" if x >= 8 else ("테라 기술" if x >= 4 else "기술")
print(f"진 판 {int((out < 0).sum())}판 중 조건(실제 수 regret >= {a.big}, 다음 결정 V 하락 >= {a.drop}) 만족 {len(cands)}판, 진행도 순으로 고르게 {a.n}개\n")
for n_, (prog, i, g) in enumerate(pick, 1):
    ma, oa = mn[i, :, 0].argmax(), on[i, :, 0].argmax()
    names = [INV["move"].get(int(c), "?") for c in mc[i, ma, 5:9]]
    onames = [INV["move"].get(int(c), "?") for c in oc[i, oa, 5:9] if int(c) > 2]
    bst = " ".join(f"{n}{mn[i, ma, k] * 6:+.0f}" for k, n in BOOST if abs(mn[i, ma, k]) > 0.01)
    obst = " ".join(f"{n}{on[i, oa, k] * 6:+.0f}" for k, n in BOOST if abs(on[i, oa, k]) > 0.01)
    tera = f"테라 {INV['type'].get(int(mc[i, ma, 9]), '?')}" + (" (가능)" if mn[i, ma, 10] > 0 else " (사용됨/불가)")
    bench = ", ".join(mon(mc[i], mn[i], k) for k in range(6) if k != ma and mc[i, k, 10] > 2)
    obench = ", ".join(mon(oc[i], on[i], k) for k in range(6) if k != oa and oc[i, k, 10] > 2)
    glob = ",".join(n for n, o in GLOBAL if fv[i, o] > 0) or "없음"
    def act(x):
        return ("교체 -> " + mon(mc[i], mn[i], x - 8)) if x >= 8 else ("테라+" if x >= 4 else "") + names[x % 4]
    best = [x for x in np.argsort(-np.where(rel[i], q[i], -np.inf))[:3] if rel[i, x]]
    pol = " / ".join(f"{act(x)} {T[i, x]:.2f}" for x in np.argsort(-T[i])[:3] if T[i, x] > 0.03)
    print(f"[{n_}] 판{g} 진행도 {prog:.0%} (턴 {i - off[g] + 1}/{lens[g]}) | 학생 {kind(A[i])} | regret {reg[i]:.2f} | V {V[i]:.2f} -> 다음 결정 {V[i + 1]:.2f}")
    print(f"  내 활성: {mon(mc[i], mn[i], ma)} {bst} | 도구 {INV['item'].get(int(mc[i, ma, 0]), '?')} | {tera} | 기술 {names}")
    print(f"  상대 활성: {mon(oc[i], on[i], oa)} {obst} | 공개 기술 {onames}")
    print(f"  내 벤치: {bench} | 상대 벤치(공개): {obench}")
    print(f"  필드: {glob} | 내 쪽 {side_str(fv[i], 20)} | 상대 쪽 {side_str(fv[i], 32)}")
    sw = lambda x: f"{INV['species'].get(int(mc[i, x - 8, 10]), '?')} 기술 {[INV['move'].get(int(c), '?') for c in mc[i, x - 8, 5:9]]} 도구 {INV['item'].get(int(mc[i, x - 8, 0]), '?')}"
    for lab_, x in (("학생 선택", A[i]), ("FP Q 최선", best[0] if best else -1)):
        if x >= 8:
            print(f"  {lab_} 교체 대상: {sw(x)}")
    print(f"  학생이 둔 수: {act(A[i])} (Q {q[i, A[i]]:.2f}) | FP 선호: {pol} | Q 상위: " + " / ".join(f"{act(x)} {q[i, x]:.2f}" for x in best) + "\n")
