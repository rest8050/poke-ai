"""교사 잡음 상한: 같은 상태를 Foul Play가 두 번 라벨링했을 때 두 결과가 얼마나 다른가 (학생의 일치율/KL/regret이 어디까지 갈 수 있는지의 상한).
한 상태를 같은 설정(300ms)으로 다시 돌린 결정기록 두 벌(--a 첫 라벨, --b 재라벨)을 (파일 이름, 배틀 태그, 턴, 강제 교체 여부)로 맞춰서 비교한다.
지표는 학생 평가와 같은 정의: 탐색 분포(방문이 최고의 0.2배 이상인 행동만 정규화)의 argmax 일치율, KL(a||b), 상대 라벨의 행동별 가치(Q, 방문 >= --min-vis)로 잰 regret.
regret은 학생 평가(regret_eval)와 같이 "한쪽 라벨의 argmax를 다른 쪽 라벨의 Q로 평가"하므로 비교 가능 (두 방향 평균).
사용: python tools/teacher_ceiling.py --a "data/dagger/raw/d7/lab_rec_*_h*.jsonl" --b "data/dagger/raw/d7b/lab_rec_*_h*.jsonl"
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import boot_ci
from src.training.build_fp_dataset import KEEP_RATIO as KEEP, load_jsonl



def load(pattern):
    """(파일, 태그, 턴, 강제교체, 순번) -> 결정 기록"""
    seen, out = defaultdict(int), {}
    for r in load_jsonl(pattern):
        if r["type"] != "decision":
            continue
        base = re.sub(r"\.jsonl(\.gz)?$", "", os.path.basename(r["_src"]))
        k = (base, r["tag"], r["turn"], int(r["force_switch"]))
        out[k + (seen[k],)] = r
        seen[k] += 1
    return out


def dist(r):
    pol = r["policy"]
    best = max(w for _, w in pol)
    keep = {c: w for c, w in pol if w >= KEEP * best}
    s = sum(keep.values())
    return {c: w / s for c, w in keep.items()}


def qvals(r, min_vis):
    num, den, vis = defaultdict(float), defaultdict(float), defaultdict(float)
    for w in r.get("worlds") or []:
        p = float(w.get("p", 1.0))
        for name, v, s in w.get("o", []):
            num[name] += p * s; den[name] += p * v; vis[name] += v
    return {c: num[c] / den[c] for c in num if den[c] > 0 and vis[c] >= min_vis}


def kind(name):
    return "교체" if name.startswith("switch ") else ("테라 기술" if name.endswith("-tera") else "기술")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--min-vis", type=float, default=100)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) & set(B))
    print(f"첫 라벨 {len(A)}결정, 재라벨 {len(B)}결정, 맞춘 결정 {len(keys)} (첫 라벨에만 {len(set(A) - set(B))}, 재라벨에만 {len(set(B) - set(A))})")
    bid = {}
    rows = []
    for k in keys:
        da, db = dist(A[k]), dist(B[k])
        names = sorted(set(da) | set(db))
        pa, pb = np.array([da.get(c, 0.0) for c in names]), np.array([db.get(c, 0.0) for c in names])
        a1, a2 = names[int(pa.argmax())], names[int(pb.argmax())]
        qa, qb = qvals(A[k], args.min_vis), qvals(B[k], args.min_vis)
        def reg(pick, q):                     # pick(한쪽 argmax)를 다른 쪽 Q로 평가. 믿을 행동이 2개 미만이거나 pick이 지원 밖이면 NaN
            return max(q.values()) - q[pick] if len(q) >= 2 and pick in q else np.nan
        r_ab, r_ba = reg(a1, qb), reg(a2, qa)
        rows.append((bid.setdefault(k[:2], len(bid)), float(a1 == a2), float(L.kl(pa[None], pb[None])[0]), float(L.kl(pb[None], pa[None])[0]),
                     np.nanmean([r_ab, r_ba]) if not (np.isnan(r_ab) and np.isnan(r_ba)) else np.nan, r_ab, kind(a1), float(pa.max()), max(qa.values()) - min(qa.values()) if len(qa) >= 2 else np.nan, r_ba))
    battle = np.array([r[0] for r in rows]); agree = np.array([r[1] for r in rows])
    kl_ab = np.array([r[2] for r in rows]); kl_ba = np.array([r[3] for r in rows]); reg_m = np.array([r[4] for r in rows])
    kinds = np.array([r[6] for r in rows]); conf = np.array([r[7] for r in rows]); reg_ab = np.array([r[5] for r in rows])
    spread = np.array([r[8] for r in rows]); reg_ba = np.array([r[9] for r in rows])
    rng = np.random.RandomState(0)
    f = lambda x: "{:.4f} [{:.4f},{:.4f}]".format(*boot_ci(x, battle, rng))
    print(f"비교 대상 {len(rows)}결정 {len(np.unique(battle))}판")
    print(f"교사 자기 일치율 {f(agree)} | 불일치 {1 - agree.mean():.3f}")
    print(f"KL(첫 라벨 || 재라벨) {f(kl_ab)} | 반대 방향 {f(kl_ba)}")
    print(f"regret (한쪽 argmax를 다른 쪽 Q로) 평균 {f(reg_m)} | 해로움(>=0.05) {np.nanmean(np.where(np.isnan(reg_m), np.nan, reg_m >= 0.05)):.3f} | 동점(<0.02) {np.nanmean(np.where(np.isnan(reg_m), np.nan, reg_m < 0.02)):.3f}")
    d = agree < 1
    print(f"불일치 결정 중 regret<0.02(동점) {np.nanmean(np.where(np.isnan(reg_m[d]), np.nan, reg_m[d] < 0.02)):.3f}, regret>=0.05(해로움) {np.nanmean(np.where(np.isnan(reg_m[d]), np.nan, reg_m[d] >= 0.05)):.3f}")
    print(f"{'첫 라벨 argmax 종류':18s} {'결정':>6s} | {'일치':>6s} {'regret':>8s}")
    for kn in ("기술", "테라 기술", "교체"):
        m = kinds == kn
        if m.sum() >= 50:
            print(f"{kn:18s} {int(m.sum()):6d} | {agree[m].mean():6.3f} {np.nanmean(reg_ab[m]):8.4f}")
    for cn, m in (("첫 라벨 확신 >= 0.8", conf >= 0.8), ("첫 라벨 확신 < 0.8", conf < 0.8)):
        print(f"{cn:18s} {int(m.sum()):6d} | {agree[m].mean():6.3f} {np.nanmean(reg_ab[m]):8.4f}")
    # 중요도(첫 라벨 Q의 최대-최소) 구간별: 재라벨의 argmax를 첫 라벨 Q로 평가 = 학생 평가(첫 라벨 Q로 잼)와 같은 조건의 독립 라벨 기준선 -> tools/regret_breakdown.py의 같은 구간과 비교
    print(f"\n중요도 구간별 (재라벨 argmax를 첫 라벨 Q로 평가, 첫 라벨 Q 기준 구간)  {'결정':>6s} | {'일치':>6s} {'regret':>8s} {'해로움':>7s}")
    for cn, m in (("<0.1", spread < 0.1), ("0.1~0.3", (spread >= 0.1) & (spread < 0.3)), (">=0.3", spread >= 0.3)):
        ok = m & ~np.isnan(reg_ba)
        print(f"  {cn:10s} {int(ok.sum()):28d} | {agree[m].mean():6.3f} {f(np.where(ok, reg_ba, np.nan))} {np.mean(reg_ba[ok] >= 0.05):7.3f}")
    sharp = spread >= 0.3
    for kn in ("기술", "테라 기술", "교체"):
        ok = sharp & (kinds == kn) & ~np.isnan(reg_ba)
        if ok.sum() >= 30:
            print(f"  중요도>=0.3 & 첫 라벨 {kn:6s} {int(ok.sum()):10d} | 일치 {agree[sharp & (kinds == kn)].mean():.3f} regret {np.mean(reg_ba[ok]):.4f}")


if __name__ == "__main__":
    main()
