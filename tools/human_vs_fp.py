"""사람(고수) 결정 vs Foul Play: 사람이 실제로 둔 상태를 FP(300ms)가 다시 라벨링하고, 사람 행동이 FP의 분포/Q에서 어디에 놓이는지 잰다.
  convert: Metamon parquet에서 레이팅 구간별로 판을 뽑아 (판마다 한쪽 시점) tools/human_to_fp.py로 변환(세트 추정만) -> rec_hv_h<j>.jsonl (--chunks개로 나눔, 컨테이너 하나씩) + sel.json (rec별 구간/레이팅)
  compare: 라벨(lab_rec_hv_h<j>.jsonl)과 rec에서 읽은 사람 행동을 맞춰 지표를 낸다. 기준선 = 같은 변환 잡음을 가진 "FP가 직접 둔 결정"(data/fp_fidelity_est2의 변환 라벨 B에 대해, 첫 라벨 A의 FP 선택을 사람 자리에 놓음).
지표(결정마다, 학생이 고를 수 있는 선택지 = 드러난 기술 + 교체로 제한): FP argmax와 일치, 사람 행동의 FP 확률(−log), FP 탐색이 거른 행동(방문 최고의 0.2배 미만) 비율,
행동의 regret = 최선 Q − 그 행동 Q (방문 >= 100인 행동만 믿음). 변환 잡음(내 쪽 세트 추정)은 사람/기준선에 똑같이 실려 있으므로 둘의 차이만 읽는다.
사용: python tools/human_vs_fp.py convert --shard 0 --bands top:1800:9999:216,mid:1300:1500:108 --out data/human_vs_fp --chunks 9
      python tools/human_vs_fp.py compare --dir data/human_vs_fp"""
import argparse
import glob
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from teacher_ceiling import KEEP, qvals                                  # noqa: E402
from regret_eval import boot_ci                                          # noqa: E402

NORM = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def cmd_convert(args):
    import pyarrow.parquet as pq
    from human_to_fp import convert
    t = pq.read_table(f"data/metamon_raw/train-{args.shard:05d}-of-00046.parquet", columns=["rating", "log"]).to_pydict()
    rng = np.random.RandomState(args.seed)
    chosen, fails = [], defaultdict(int)
    for band, lo, hi, n in (b.split(":") for b in args.bands.split(",")):
        idx = [i for i, r in enumerate(t["rating"]) if str(r).isdigit() and int(lo) <= int(r) < int(hi)]
        rng.shuffle(idx)
        got = 0
        for i in idx:
            if got == int(n):
                break
            side = ("p1", "p2")[rng.randint(2)]
            log = t["log"][i]
            user = re.search(rf"\|player\|{side}\|([^|]*)\|", log)
            rec = convert(log, side, user.group(1) if user else side, f"battle-gen9ou-{9000000 + i}")
            if isinstance(rec, str):
                fails[rec.split(":")[0]] += 1
                continue
            rec["meta"].update(band=band, rating=int(t["rating"][i]))
            chosen.append(rec); got += 1
        print(f"{band}: 후보 {len(idx)}판 -> 변환 {got}판", flush=True)
    order = rng.permutation(len(chosen))                                # 구간이 컨테이너마다 고르게 섞이게
    os.makedirs(args.out, exist_ok=True)
    for j in range(args.chunks):
        with open(f"{args.out}/rec_hv_h{j}.jsonl", "w", encoding="utf-8") as f:
            for k in order[j::args.chunks]:
                f.write(json.dumps(chosen[k]) + "\n")
    print(f"변환 실패 {dict(fails)} | rec {len(chosen)}개 -> {args.chunks}파일")


def human_actions(rec):
    """rec 프레임에서 결정마다 사람이 실제로 한 행동 -> {(턴, 강제, 순번): 행동 이름}. fp/replay_label.py의 actual_action과 같은 규칙 + 테라(-terastallize)"""
    side = re.search(r'"id":"(p\d)"', next(f for f in rec["frames"] if "|request|" in f)).group(1)
    out, seen, turn, fr = {}, defaultdict(int), 0, rec["frames"]
    for i, f in enumerate(fr):
        m = re.search(r"\|turn\|(\d+)", f)
        if m:
            turn = int(m.group(1))
        if "|request|" not in f:
            continue
        req = json.loads(f.split("|request|")[1].split("\n")[0])
        kind = "tp" if req.get("teamPreview") else "forced" if req.get("forceSwitch") else "turn" if req.get("active") else None
        if kind is None or kind == "tp":
            continue
        key = (turn, kind == "forced"); n = seen[key]; seen[key] += 1
        tera, act = False, None
        for g in fr[i + 1:]:
            if "|request|" in g:
                break
            for line in g.split("\n"):
                p = line.split("|")
                if len(p) > 3 and p[2][:4] in (side + "a:", side + "b:"):
                    if p[1] == "-terastallize":
                        tera = True
                    elif p[1] == "move":
                        act = NORM(p[3]) + ("-tera" if tera else ""); break
                    elif p[1] == "switch":
                        act = "switch " + NORM(p[3].split(",")[0]); break
                    elif p[1] == "cant" and len(p) > 4 and p[4]:
                        act = NORM(p[4]) + ("-tera" if tera else ""); break
            if act or "|turn|" in g or "|win|" in g:
                break
        if act:
            out[key + (n,)] = act
    return out


def load_labels(path):
    seen, out = defaultdict(int), {}
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "decision":
                k = (r["tag"], r["turn"], bool(r["force_switch"]))
                out[k + (seen[k],)] = r; seen[k] += 1
    return out


def measure(r, act, revealed):
    """행동 act를 라벨 r에서 잼 -> (일치, -log p, 거름, regret, FP 확신, Q 폭, 선택지 수) 또는 None"""
    ok = lambda c: c.startswith("switch ") or c.removesuffix("-tera") in revealed
    pol = {c: w for c, w in r["policy"] if ok(c)}
    if len(pol) < 2 or act not in pol:
        return None
    z = sum(pol.values()); best = max(pol.values())
    q = {c: v for c, v in qvals(r, 100).items() if ok(c)}
    reg = max(q.values()) - q[act] if len(q) >= 2 and act in q else math.nan
    spread = max(q.values()) - min(q.values()) if len(q) >= 2 else math.nan
    top = max(pol, key=pol.get)
    return (float(top == act), -math.log(max(pol[act] / z, 1e-4)), float(pol[act] < KEEP * best), reg, best / z, spread, len(pol))


def kind(a):
    return "교체" if a.startswith("switch ") else "테라 기술" if a.endswith("-tera") else "기술"


def table(title, rows, battle, rng, mask):
    X = np.array([r[:7] for r in rows], float)
    f = lambda x, m: "{:.3f} [{:.3f},{:.3f}]".format(*boot_ci(x[m], battle[m], rng))
    m = mask
    if m.sum() < 30:
        return
    reg = X[:, 3]; ok = m & ~np.isnan(reg)
    print(f"{title:22s} {int(m.sum()):6d}결정 {len(np.unique(battle[m])):4d}판 | 일치 {f(X[:, 0], m)} | −log p {X[m, 1].mean():.2f} | FP가 거른 행동 {X[m, 2].mean():.3f} | "
          f"regret {f(np.where(ok, reg, np.nan), ok) if ok.sum() >= 30 else 'n/a'} 해로움(>=0.05) {np.mean(reg[ok] >= 0.05):.3f} 큰 실수(>=0.15) {np.mean(reg[ok] >= 0.15):.3f}")


def cmd_compare(args):
    sel = defaultdict(dict)
    rows, meta = [], []                                                  # 사람 행동 지표 행 / (판 번호, 구간, 종류, 강제)
    bid = {}
    for path in sorted(glob.glob(f"{args.dir}/rec_hv_h*.jsonl")):
        name = os.path.basename(path).removesuffix(".jsonl")
        lab = f"{args.dir}/lab_{name}.jsonl"
        if not os.path.exists(lab):
            continue
        L = load_labels(lab)
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            ha = human_actions(rec)
            rev = set(rec["meta"]["revealed"])
            for k, act in ha.items():
                if (rec["tag"],) + k not in L:
                    continue
                r = measure(L[(rec["tag"],) + k], act, rev)
                if r is None:
                    sel["skip"][rec["meta"]["band"]] = sel["skip"].get(rec["meta"]["band"], 0) + 1
                    continue
                rows.append(r); meta.append((bid.setdefault(rec["tag"], len(bid)), rec["meta"]["band"], kind(act), k[1]))
    battle = np.array([m[0] for m in meta]); band = np.array([m[1] for m in meta]); kinds = np.array([m[2] for m in meta]); forced = np.array([m[3] for m in meta])
    rng = np.random.RandomState(0)
    print(f"사람 결정 {len(rows)}개 (측정 불가로 뺀 결정 {dict(sel.get('skip', {}))})")
    for b in sorted(set(band)):
        table(f"사람 [{b}]", rows, battle, rng, band == b)
    # 기준선: FP가 직접 둔 결정 (est2 변환 라벨 B 위에서 A의 FP 선택). 같은 변환 잡음
    base, bb = [], []
    for lab in sorted(glob.glob(f"{args.baseline}/lab_rec_*_h*.jsonl")):
        n = os.path.basename(lab).removeprefix("lab_").removesuffix(".jsonl")
        A = load_labels(f"data/dagger/raw/d7/lab_{n}.jsonl.gz")
        B = load_labels(lab)
        rev = {json.loads(l)["tag"]: set(json.loads(l)["meta"]["revealed"]) for l in open(f"{args.baseline}/{n}.jsonl", encoding="utf-8")}
        for k, b in B.items():
            if k in A and k[0] in rev:
                r = measure(b, A[k]["choice"], rev[k[0]])
                if r is not None:
                    base.append(r); bb.append((n, k[0]))
    ids = {t: i for i, t in enumerate(sorted(set(bb)))}
    bbat = np.array([ids[t] for t in bb])
    print()
    table("기준선: FP 자기 선택", base, bbat, rng, np.ones(len(base), bool))
    print("\n[사람 행동 종류별]")
    for b in sorted(set(band)):
        for kn in ("기술", "테라 기술", "교체"):
            table(f"[{b}] {kn}", rows, battle, rng, (band == b) & (kinds == kn))
        table(f"[{b}] 강제 교체", rows, battle, rng, (band == b) & forced)
    print("\n[FP 확신도별 (사람 행동 기준)]")
    X = np.array([r[:7] for r in rows], float)
    for b in sorted(set(band)):
        for nm, m in (("FP 확신>=0.8", X[:, 4] >= 0.8), ("FP 확신 0.5~0.8", (X[:, 4] >= 0.5) & (X[:, 4] < 0.8)), ("FP 확신<0.5", X[:, 4] < 0.5)):
            table(f"[{b}] {nm}", rows, battle, rng, (band == b) & m)
    print("\n[Q 폭(중요도)별]")
    for b in sorted(set(band)):
        for nm, m in (("폭<0.1", X[:, 5] < 0.1), ("폭 0.1~0.3", (X[:, 5] >= 0.1) & (X[:, 5] < 0.3)), ("폭>=0.3", X[:, 5] >= 0.3)):
            table(f"[{b}] {nm}", rows, battle, rng, (band == b) & m)
    np.savez(f"{args.dir}/human_rows.npz", X=X, band=band, kinds=kinds, forced=forced, battle=battle)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--bands", required=True, help="이름:최소:최대(미만):판수 콤마 목록")
    c.add_argument("--out", required=True)
    c.add_argument("--chunks", type=int, default=9)
    c.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("compare")
    k.add_argument("--dir", required=True)
    k.add_argument("--baseline", default="data/fp_fidelity_est2")
    a = ap.parse_args()
    {"convert": cmd_convert, "compare": cmd_compare}[a.cmd](a)
