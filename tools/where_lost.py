"""학생이 FP300에게 지는 판에서 어디서 지는가: 판 진행도별 FP 가치 곡선(승/패), 결정 regret의 진행도별 몫, 패한 판의 가치 하락이 소수의 큰 실수에 몰렸는지 서서히 깎였는지.
데이터: data/winrate_v6_fp300/lab_stu.npz (학생 시점, teacher_value = FP가 본 학생 쪽 승률, action_taken = 학생이 둔 행동, q_mean/q_vis = 행동별 FP 가치).
사용: python tools/where_lost.py"""
import sys

import numpy as np

sys.path.insert(0, ".")
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/winrate_v6_fp300/lab_stu.npz")
lens, out, V, A, Q, vis, M = d["lens"], d["outcome"], d["teacher_value"], d["action_taken"], d["q_mean"], d["q_vis"], d["action_mask"]
off = np.concatenate([[0], np.cumsum(lens)])
rel = (vis >= 100) & ~np.isnan(Q) & M
qstar = np.where(rel, Q, -np.inf).max(1)
ar = np.arange(len(V))
qa = np.where(rel[ar, A.clip(0)], Q[ar, A.clip(0)], np.nan)
reg = np.where(np.isfinite(qstar) & (rel.sum(1) >= 2), qstar - qa, np.nan)
win = out > 0
print(f"{len(lens)}판, 학생 승률 {win.mean():.3f}, 판당 결정 {lens.mean():.1f}, 결정 regret 유효 {np.isfinite(reg).mean():.2f}")
ph = np.concatenate([(np.arange(l) + 0.5) / l for l in lens])                    # 판 진행도 0~1
bins = np.linspace(0, 1, 6)
print(f"\n{'진행도':10s} | {'평균 V 승':>8s} {'평균 V 패':>8s} | {'regret 승':>9s} {'regret 패':>9s} | {'패한 판 regret 몫':>14s}")
gw = np.repeat(win, lens)
tot_loss = np.nansum(reg[~gw])
for a, b in zip(bins[:-1], bins[1:]):
    m = (ph >= a) & (ph < b + (b == 1))
    print(f"{a:.1f}~{b:.1f}   | {V[m & gw].mean():8.3f} {V[m & ~gw].mean():8.3f} | {np.nanmean(reg[m & gw]):9.4f} {np.nanmean(reg[m & ~gw]):9.4f} | {np.nansum(reg[m & ~gw]) / tot_loss:14.3f}")
# 패한 판: 단계별 가치 변화 V_{t+1}-V_t 가 큰 하락 몇 번에 몰렸는가
top3, first, tot, lo_t = [], [], [], []
for g in np.where(~win)[0]:
    v = V[off[g]:off[g + 1]]
    if len(v) < 6:
        continue
    dv = np.diff(v)
    neg = np.sort(dv[dv < 0])
    if len(neg) < 3:
        continue
    top3.append(neg[:3].sum() / neg.sum()); tot.append(v[0] - v[-1])
    first.append((np.argmax(v < 0.35) + 1) / len(v) if (v < 0.35).any() else np.nan)   # 처음으로 승률 0.35 아래로 내려간 시점(진행도)
    lo_t.append((np.argmin(dv) + 1) / len(v))
print(f"\n패한 판 {len(top3)}판: 하락(V 감소 단계)의 합 중 가장 큰 3번의 몫 평균 {np.mean(top3):.3f} (균등하게 깎이면 3/{int(np.mean([(np.diff(V[off[g]:off[g+1]]) < 0).sum() for g in np.where(~win)[0] if lens[g] >= 6]))}≈0.1대)")
print(f"가장 큰 한 번의 하락이 일어난 진행도: 평균 {np.mean(lo_t):.2f}, 사분위 {np.percentile(lo_t, [25, 50, 75]).round(2)}")
f = np.array(first); f = f[~np.isnan(f)]
print(f"처음 V<0.35로 내려간 진행도: 평균 {f.mean():.2f}, 사분위 {np.percentile(f, [25, 50, 75]).round(2)} ({len(f)}판)")
print(f"시작 V 평균 승 {np.mean([V[off[g]] for g in np.where(win)[0]]):.3f} 패 {np.mean([V[off[g]] for g in np.where(~win)[0]]):.3f}")
# 남은 결정 수별 "학생 승률 부족분": V가 0.4~0.6(거의 대등)인 결정에서 판의 실제 결과와 V의 차. 남은 길이에 비례하면 결정마다 조금씩 깎이는 것, 초반에 몰려 있으면 장기 운영 문제
rem = np.concatenate([np.arange(l)[::-1] for l in lens])
print(f"\nV 0.4~0.6 결정만: 남은 결정 수별 평균 V, 실제 학생 승률, 부족분 (V - 승률), 95% 구간(판 단위 부트스트랩)")
rng = np.random.RandomState(0)
game = np.repeat(np.arange(len(lens)), lens)
for a, b in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 99)):
    m = (V >= 0.4) & (V <= 0.6) & (rem >= a) & (rem < b)
    gap = V[m] - gw[m]
    gs = np.unique(game[m])
    sums = {g: (gap[game[m] == g].sum(), (game[m] == g).sum()) for g in gs}
    bs = [sum(sums[g][0] for g in s) / sum(sums[g][1] for g in s) for s in (rng.choice(gs, len(gs)) for _ in range(300))]
    print(f"남은 {a:2d}~{b - 1 if b < 99 else '+':>2}  n={int(m.sum()):5d} ({len(gs):3d}판) | V {V[m].mean():.3f} 승률 {gw[m].mean():.3f} 부족분 {gap.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
