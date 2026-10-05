"""학생(vs Foul Play) 대전 기록 + 교사 재생 라벨을 합쳐 "어떤 판단이 부족해서 지는가"를 집계.
사용: loss_report.py [디렉터리=data/loss_analysis]   (rec_*.jsonl 학생 기록 + lab_*.jsonl 교사 라벨 필요)
교사 value = Foul Play 탐색이 본 '학생 쪽 승률'(학생 시점 상태 기준). 선택 이름은 소문자·영숫자로 정규화해 맞춤."""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "data/loss_analysis"
toid = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def norm_student(name):
    if name is None:
        return None
    if name.startswith("switch:"):
        return "switch " + toid(name[7:])
    base, tera = name.split("+")[0], "+tera" in name
    return toid(base) + ("-tera" if tera else "")


def norm_teacher(name):
    if name.startswith("switch "):
        return "switch " + toid(name[7:])
    tera = name.endswith("-tera")
    return toid(name[:-5] if tera else name) + ("-tera" if tera else "")


def kind(name):
    return "switch" if name.startswith("switch ") else "move"


games = {}
for f in sorted(glob.glob(f"{D}/rec_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            g = json.loads(line)
            games[g["tag"]] = {"won": g["won"], "turns": g["turns"]}
labels = defaultdict(dict)
for f in sorted(glob.glob(f"{D}/lab_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") == "decision":
            labels[d["tag"]][(int(d["turn"]), bool(d["force_switch"]))] = d

n_games = len(games); n_won = sum(g["won"] for g in games.values())
print(f"대전 {n_games}판: 학생 {n_won}승 ({n_won / max(1, n_games):.1%}), 라벨 있는 판 {sum(1 for t in games if t in labels)}")

rows = []  # 결정 하나 = 학생 선택 + 교사 분포 + 형세 변화
unmatched = 0
for tag, g in games.items():
    if tag not in labels:
        continue
    seq = []
    for t in g["turns"]:
        d = labels[tag].get((int(t["turn"]), bool(t["forced"])))
        if d is None:
            unmatched += 1
            continue
        P = {norm_teacher(n): p for n, p in d["policy"]}
        seq.append((t, d, P))
    for i, (t, d, P) in enumerate(seq):
        c = norm_student(t["chosen"])
        top = max(P, key=P.get)
        v = float(d["value"])
        v_next = float(seq[i + 1][1]["value"]) if i + 1 < len(seq) else (1.0 if g["won"] else 0.0)
        pc = P.get(c)
        if pc is None:  # 교사가 후보로도 안 본 행동(탐색 밖) → 아주 낮은 값으로
            pc = 0.0
        cat = ("forced_switch" if t["forced"] else
               "same" if c == top else
               "tera_only" if c.replace("-tera", "") == top.replace("-tera", "") else
               {("switch", "move"): "under_switch(교사 교체, 학생 기술)", ("move", "switch"): "over_switch(교사 기술, 학생 교체)",
                ("switch", "switch"): "switch_target(교체 대상 다름)", ("move", "move"): "move_choice(기술 선택 다름)"}[(kind(top), kind(c))])
        rows.append(dict(tag=tag, won=g["won"], turn=int(t["turn"]), forced=t["forced"], c=c, top=top, ptop=P[top], pc=pc, v=v,
                         dv=v_next - v, cat=cat, P=P, student_p=t["p"]))
print(f"분석 결정 {len(rows)}개 (라벨 못 맞춘 학생 결정 {unmatched}개 제외)\n")


def table(title, rs):
    print(f"== {title} (결정 {len(rs)}개)")
    by = defaultdict(list)
    for r in rs:
        by[r["cat"]].append(r)
    print(f"{'유형':36s} {'비율':>6s} {'교사가 학생선택에 준 확률':>22s} {'선택 후 평균 형세변화':>20s} {'<5% 선택 비율':>12s}")
    for cat, x in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"{cat:36s} {len(x) / len(rs):6.1%} {sum(r['pc'] for r in x) / len(x):22.3f} {sum(r['dv'] for r in x) / len(x):+20.3f} {sum(r['pc'] < 0.05 for r in x) / len(x):12.1%}")
    print()


table("전체", rows)
table("패배한 판만", [r for r in rows if not r["won"]])
table("승리한 판만", [r for r in rows if r["won"]])

# 낮은 확률 선택(교사가 <5%)의 형세 비용: 선택 직후 형세가 얼마나 깎이는지 (교사 일치 선택과 비교)
low = [r for r in rows if r["pc"] < 0.05 and not r["forced"]]
ok = [r for r in rows if r["pc"] >= 0.30 and not r["forced"]]
print(f"교사가 <5%로 본 선택 {len(low)}개: 평균 형세변화 {sum(r['dv'] for r in low) / max(1, len(low)):+.3f} | 교사가 30% 이상으로 본 선택 {len(ok)}개: {sum(r['dv'] for r in ok) / max(1, len(ok)):+.3f}\n")

# 패배한 판에서 형세가 가장 크게 꺾인 결정
print("== 패배한 판의 큰 악수 후보 (강제교체 제외, 교사가 20% 미만으로 본 선택만, 형세 하락 큰 순, 판당 최대 1개)")
worst = {}
for r in rows:
    if not r["won"] and not r["forced"] and r["pc"] < 0.2 and (r["tag"] not in worst or r["dv"] < worst[r["tag"]]["dv"]):
        worst[r["tag"]] = r
for r in sorted(worst.values(), key=lambda r: r["dv"])[:25]:
    alts = sorted(r["P"].items(), key=lambda kv: -kv[1])[:3]
    print(f"{r['tag'][-6:]} T{r['turn']:2d}{'(강제)' if r['forced'] else '     '} 승률 {r['v']:.2f}->{r['v'] + r['dv']:.2f} | 학생: {r['c']} (교사확률 {r['pc']:.2f}) | 교사: " +
          ", ".join(f"{n} {p:.2f}" for n, p in alts))

# 교체/기술 판단: 교사 교체 비중이 높은 상태에서 학생이 놓친 비율
sw = [r for r in rows if not r["forced"]]
t_sw = [r for r in sw if kind(r["top"]) == "switch"]
s_sw = [r for r in sw if kind(r["c"]) == "switch"]
print(f"\n자발 결정 {len(sw)}개: 교사 교체 {len(t_sw) / len(sw):.1%} / 학생 교체 {len(s_sw) / len(sw):.1%}")
print("승률대별 학생 교체 비율 vs 교사 교체 비율 (교사 value 기준 학생 쪽 승률)")
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    x = [r for r in sw if lo <= r["v"] < hi]
    if x:
        print(f"  승률 {lo:.1f}-{min(hi, 1):.1f}: 결정 {len(x):4d} | 교사 교체 {sum(kind(r['top']) == 'switch' for r in x) / len(x):5.1%} / 학생 교체 {sum(kind(r['c']) == 'switch' for r in x) / len(x):5.1%}")

# 턴 진행에 따른 승률 궤적: 언제 지는가
print("\n판 진행도별 평균 교사 승률 (패배한 판 / 승리한 판)")
for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 99)]:
    for w, nm in ((False, "패"), (True, "승")):
        x = [r["v"] for r in rows if r["won"] == w and lo <= r["turn"] <= hi]
        if x:
            print(f"  턴 {lo:2d}-{hi:2d} {nm}: {sum(x) / len(x):.2f} (n={len(x)})", end="")
    print()


# 기술 선택 불일치(교사 기술 vs 학생 기술)를 기술 종류로 분해: 어떤 종류의 판단을 놓치는가
MC = json.load(open("data/move_classes.json"))
cls = lambda n: MC.get(n.replace("-tera", ""), "?")
mm = [r for r in rows if not r["forced"] and r["cat"].startswith("move_choice")]
conf = Counter((cls(r["top"]), cls(r["c"])) for r in mm)
print(f"\n== 기술 선택 불일치 {len(mm)}개의 종류별 혼동 (교사 최선 종류 -> 학생 선택 종류)")
tot_top = Counter(cls(r["top"]) for r in mm)
for (a, b), n in conf.most_common(14):
    print(f"  {a:15s} -> {b:15s} {n:4d} ({n / len(mm):5.1%})")
print("교사가 고른 종류별로 학생이 같은 종류를 고른 비율(종류는 맞췄고 세부 기술만 다른 경우):")
for a, n in tot_top.most_common(6):
    same = sum(v for (x, y), v in conf.items() if x == a and y == a)
    print(f"  {a:15s} 불일치 {n:4d}개 중 같은 종류 {same / n:5.1%}")
# 전체 자발 결정에서 종류별 교사 vs 학생 빈도
vol = [r for r in rows if not r["forced"]]
def cls_all(n): return "switch" if n.startswith("switch ") else cls(n)
ct, cs = Counter(cls_all(r["top"]) for r in vol), Counter(cls_all(r["c"]) for r in vol)
print("\n== 자발 결정에서 행동 종류 빈도: 교사 최선 vs 학생 선택")
for k in sorted(set(ct) | set(cs), key=lambda k: -ct[k]):
    print(f"  {k:15s} 교사 {ct[k] / len(vol):6.1%} / 학생 {cs[k] / len(vol):6.1%}")
