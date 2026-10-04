"""같은 상태를 여러 탐색 시간(ms)으로 라벨링한 결과 비교. 기준 = 가장 긴 ms.
사용: ms_compare.py <디렉터리> <ms 목록(오름차순, 마지막이 기준)> 예: ms_compare.py data/ms_test 300 1000 3000
전제: 디렉터리에 rec_sXX.jsonl(재생 입력)과 dec_ms<ms>_sXX.jsonl(라벨 결과)"""
import glob
import json
import math
import sys
from collections import defaultdict

d, ms_list = sys.argv[1], [int(x) for x in sys.argv[2:]]
ref_ms = ms_list[-1]


def game_kind():
    kind = {}
    for f in glob.glob(f"{d}/rec_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            g = json.loads(l)
            t = "\n".join(g["frames"])
            kind[g["tag"]] = "트릭룸" if ("|move|" in t and "Trick Room" in t) else "일반"
    return kind


def load(ms):
    out = {}
    for f in sorted(glob.glob(f"{d}/dec_ms{ms}_s*.jsonl")):
        seen = defaultdict(int)
        for l in open(f, encoding="utf-8"):
            if not l.startswith('{"type": "decision"'):
                continue
            r = json.loads(l)
            base = (r["tag"], r["turn"], r["force_switch"])
            k = base + (seen[base],)
            seen[base] += 1
            pol = {a: p for a, p in r["policy"]}
            z = sum(pol.values()) or 1
            pol = {a: p / z for a, p in pol.items()}
            visits = sum(w["n"] for w in r.get("worlds", []))
            out[k] = (pol, visits, len(r.get("worlds", [])))
    return out


def kl(p, q, eps=1e-4):
    ks = set(p) | set(q)
    return sum(p.get(a, 0) * math.log((p.get(a, 0) + eps) / (q.get(a, 0) + eps)) for a in ks if p.get(a, 0) > 0)


def ent(p):
    return -sum(v * math.log(v) for v in p.values() if v > 0)


kind = game_kind()
data = {ms: load(ms) for ms in ms_list}
ref = data[ref_ms]
print(f"기준 {ref_ms}ms 결정 {len(ref)}개")
for grp in ("전체", "트릭룸", "일반"):
    print(f"\n== {grp}")
    print(f"{'ms':>6} {'결정':>5} {'방문수(세트당)':>13} {'세트수':>5} {'엔트로피':>7} {'KL(기준||ms)':>12} {'1위일치':>7} {'TR확률평균':>9}")
    for ms in ms_list:
        keys = [k for k in ref if k in data[ms] and (grp == "전체" or kind.get(k[0]) == grp)]
        if not keys:
            continue
        n = len(keys)
        vis = sum(data[ms][k][1] / max(1, data[ms][k][2]) for k in keys) / n
        nw = sum(data[ms][k][2] for k in keys) / n
        e = sum(ent(data[ms][k][0]) for k in keys) / n
        kls = sum(kl(ref[k][0], data[ms][k][0]) for k in keys) / n
        top = sum(max(data[ms][k][0], key=data[ms][k][0].get) == max(ref[k][0], key=ref[k][0].get) for k in keys) / n
        trk = [k for k in keys if any("trickroom" in a for a in ref[k][0])]
        trp = sum(sum(p for a, p in data[ms][k][0].items() if "trickroom" in a) for k in trk) / max(1, len(trk))
        print(f"{ms:>6} {n:>5} {vis:>13.0f} {nw:>5.2f} {e:>7.3f} {kls:>12.4f} {top:>7.3f} {trp:>9.3f} (트릭룸 후보 결정 {len(trk)})")
