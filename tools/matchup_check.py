"""매치업 모듈(MatchupV7)의 데미지를 Showdown 실제 계산(tools/sd_damage.js, 최대 난수/급소 없음)과 대조.
실제 상태(홀드아웃 npz)에서 (공격 포켓몬, 공개 공격기, 방어 포켓몬)을 무작위로 뽑아 같은 입력(능력치/랭크/HP/도구/특성/상태/테라/날씨/지형/벽)을 Showdown에 주고,
모듈의 최대 난수 데미지 비율(= 평균 난수 채널 / 0.925, 1.5 상한 미만만)과 비교한다. 상대 쪽 능력치는 모듈의 가정(종족값 -> 252노력치 무보정)을 Showdown에도 줌 = 계산식만 검증.
변형: 0 전체 / 1 날씨·지형·벽 제외 / 2 거기에 도구 제외 -> 모듈이 어느 변형과 맞는지로 "모듈이 안 쓰는 요인"을 분리. 다단히트는 한 타 데미지끼리(모듈 값 / 타수) 비교.
사용: python tools/matchup_check.py --set data/dagger/d7_ho.npz --n 400 --k 8"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.matchup_v7 import MatchupV7

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="data/dagger/d7_ho.npz")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dump", type=int, default=0, help="N번째로 어긋난 테라 쌍의 입력 전체를 출력")
a = ap.parse_args()
V = json.load(open("data/vocab.json", encoding="utf-8"))
INV = {k: {i: n for n, i in V[k].items()} for k in ("species", "move", "item", "ability", "status", "type")}
d = np.load(a.set)
rng = np.random.RandomState(a.seed)
rows = np.sort(rng.choice(len(d["my_team_num"]), a.n, replace=False))
T = {k: torch.tensor(d[k][rows]) for k in ("my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec")}
mod = MatchupV7().eval()
with torch.no_grad():
    out = mod(T["my_team_cat"], T["my_team_num"], T["my_move_num"][..., :46], T["opp_team_cat"], T["opp_team_num"], T["opp_move_num"][..., :46], T["field_vec"],
              mod._onehot(T["opp_team_cat"]), torch.zeros(len(rows), 6))
Dm, Om = out["def"].numpy(), out["off"].numpy()                                  # [B,6(방어),24(공격 칸),4]
UNK = {k: V[k].get("<unk>", -1) for k in ("species", "ability", "item", "move")}
TYPE = lambda i: INV["type"].get(int(i), "").capitalize() or None
STATUS = {"brn", "par", "slp", "psn", "tox", "frz"}


def mon(cat, num, mine):
    sp = INV["species"].get(int(cat[10]), "")
    ab, it = int(cat[1]), int(cat[0])
    st = num[11:17] * 255.0
    stats = list(st) if mine else [2 * st[0] + 204] + [2 * s + 99 for s in st[1:]]
    s_name = INV["status"].get(int(cat[4]), "")
    return {"species": sp, "ability": INV["ability"].get(ab, "") if ab != UNK["ability"] else None, "item": INV["item"].get(it, "") if it != UNK["item"] else "",
            "status": s_name if s_name in STATUS else "", "boosts": [float(x) * 6 for x in num[3:8]], "stats": [float(x) for x in stats], "hp": float(num[2]),
            "tera": TYPE(cat[9]) if num[9] > 0.5 else None, "types": [TYPE(cat[2]), TYPE(cat[3])]}


def field(fv, def_mine):
    wx = next((n for n, o in (("sunnyday", 0), ("raindance", 2), ("sandstorm", 4), ("snowscape", 6)) if fv[o] > 0), None)
    tr = next((n for n, o in (("electricterrain", 12), ("grassyterrain", 14), ("psychicterrain", 16), ("mistyterrain", 18)) if fv[o] > 0), None)
    base = 20 if def_mine else 32
    return wx, tr, [int(fv[base + o] > 0) for o in (0, 2, 4)]


scen, meta = [], {}
for bi, r in enumerate(rows):
    cat = {s: d[f"{s}_team_cat"][r] for s in ("my", "opp")}
    num = {s: d[f"{s}_team_num"][r] for s in ("my", "opp")}
    mv = {s: d[f"{s}_move_num"][r][..., :46] for s in ("my", "opp")}
    pairs = []
    for att_s, def_s in (("my", "opp"), ("opp", "my")):
        for ai in range(6):
            if not (num[att_s][ai, 11] > 0 and num[att_s][ai, 1] < 0.5):
                continue
            for j in range(4):
                mid = int(cat[att_s][ai, 5 + j])
                if not np.abs(mv[att_s][ai, j]).sum() > 0 or mid <= 2 or mv[att_s][ai, j, 0] <= 0 or mv[att_s][ai, j, 4] >= 0.85:
                    continue
                for di in range(6):
                    if num[def_s][di, 11] > 0 and num[def_s][di, 1] < 0.5:
                        pairs.append((att_s, ai, j, def_s, di, mid))
    for p in (pairs[i] for i in rng.permutation(len(pairs))[:a.k]):
        att_s, ai, j, def_s, di, mid = p
        A, Dd = mon(cat[att_s][ai], num[att_s][ai], att_s == "my"), mon(cat[def_s][di], num[def_s][di], def_s == "my")
        if A["ability"] is None or Dd["ability"] is None or not A["species"] or not Dd["species"]:
            continue
        mval = Om[bi, di, ai * 4 + j] if att_s == "my" else Dm[bi, di, ai * 4 + j]        # 모듈: [방어 포켓몬, 공격 칸]
        hits = 1.0 + 4.0 * mv[att_s][ai, j, 10]
        wx, tr, sc_ = field(d["field_vec"][r], def_s == "my")
        k = len(meta)
        for var in (0, 1, 2):
            scen.append({"id": k * 3 + var, "att": A, "def": Dd, "move": INV["move"].get(mid, ""), "weather": wx, "terrain": tr, "screens": sc_, "variant": var})
        meta[k] = {"mod": float(mval[1]) * 1.5 / 0.925 / hits, "capped": bool(mval[1] >= 0.999), "att": A, "def": Dd, "move": INV["move"].get(mid, ""),
                   "wx": wx, "tr": tr, "sc": sc_, "side": att_s, "tera": bool(A["tera"] or Dd["tera"])}
proc = subprocess.run(["node", "tools/sd_damage.js"], input="\n".join(json.dumps(s) for s in scen), capture_output=True, text=True, encoding="utf-8")
res = {}
for line in proc.stdout.splitlines():
    o = json.loads(line)
    res[o["id"]] = o
errs = Counter(v["err"][:60] for v in res.values() if "err" in v)
print(f"상태 {len(rows)}개, 시나리오 {len(meta)}쌍 (x3 변형), Showdown 오류 {sum(errs.values())}건 {errs.most_common(3)}")
R = []
for i, m in meta.items():
    r = [res.get(3 * i + v) for v in (0, 1, 2)]
    if any(x is None or "err" in x for x in r):
        continue
    m["sd"] = [x["dmg"] / x["maxhp"] for x in r]
    m["dyn"] = r[0]["dyn"]
    R.append(m)
print(f"비교 가능 {len(R)}쌍")
arr = np.array
mod = arr([m["mod"] for m in R])
sd = arr([m["sd"] for m in R])
cap = arr([m["capped"] for m in R])
dyn = arr([m["dyn"] for m in R])
plain = (dyn == "") | (dyn == "multihit")
sd0, m0 = sd[:, 0] == 0, mod == 0
print("\n[면역/무효 일치] (변형 0 기준)")
print(f"  Showdown 0 {int(sd0.sum())}쌍 중 모듈도 0: {np.mean(m0[sd0]):.3f} | 모듈 0인데 Showdown > 0: {int((m0 & ~sd0).sum())}쌍 | Showdown 0인데 모듈 > 0: {int((~m0 & sd0).sum())}쌍")
ok = ~sd0 & ~m0 & ~cap
print(f"\n[최대 난수 데미지 비율: 모듈 / Showdown] (둘 다 > 0, 1.5 상한 미만 {int(ok.sum())}쌍)")
print(f"{'대상':34s} {'쌍':>5s} | {'중앙 비':>7s} {'<=3%':>6s} {'<=10%':>6s} {'평균 오차':>9s} | 변형 0/1/2 중 가장 가까운 비율")


def row(name, m):
    m = m & ok
    if m.sum() < 15:
        return
    rat = [mod[m] / sd[m, v] for v in (0, 1, 2)]
    rel = [np.abs(x - 1) for x in rat]
    best = np.argmin(np.stack(rel, 1), 1)
    print(f"{name:34s} {int(m.sum()):5d} | {np.median(rat[0]):7.3f} {np.mean(rel[0] <= 0.03):6.3f} {np.mean(rel[0] <= 0.10):6.3f} {np.mean(rat[0] - 1):+9.3f} | " + "/".join(f"{np.mean(best == v):.2f}" for v in range(3)))


row("전체", np.ones(len(R), bool))
row("일반 기술(위력 고정, 다단 제외)", dyn == "")
row("다단히트(한 타)", dyn == "multihit")
row("위력 변동/특수 규칙 기술", ~plain)
fx = arr([m["wx"] is not None or m["tr"] is not None or any(m["sc"]) for m in R])
row("날씨/지형/벽 있음 (일반 기술)", fx & (dyn == ""))
row("날씨/지형/벽 없음 (일반 기술)", ~fx & (dyn == ""))
for nm, who, ids in (("공격측 구애 도구", "att", ("choiceband", "choicespecs")), ("공격측 생명의구슬", "att", ("lifeorb",)), ("방어측 돌격조끼/진화의휘석", "def", ("assaultvest", "eviolite"))):
    row(nm + " (일반 기술)", arr([m[who]["item"] in ids for m in R]) & (dyn == ""))
row("테라 있음 (일반 기술)", arr([m["tera"] for m in R]) & (dyn == ""))
row("랭크 변화 있음 (일반 기술)", arr([any(abs(b) > 0.5 for b in m["att"]["boosts"] + m["def"]["boosts"]) for m in R]) & (dyn == ""))
row("공격측 화상 (일반 기술)", arr([m["att"]["status"] == "brn" for m in R]) & (dyn == ""))
row("공격측이 상대(능력치 추정)", arr([m["side"] == "opp" for m in R]) & (dyn == ""))
row("공격측이 나(능력치 실제)", arr([m["side"] == "my" for m in R]) & (dyn == ""))
rel2 = np.where(ok & (dyn == ""), np.abs(mod / np.maximum(sd[:, 0], 1e-6) - 1), 0)
print("\n[변형 0 기준 가장 크게 어긋난 일반 기술 12건]")
for i in np.argsort(-rel2)[:12]:
    m = R[i]
    print(f"  {m['att']['species']}({m['att']['ability']},{m['att']['item'] or '-'}) {m['move']} -> {m['def']['species']}({m['def']['ability']},{m['def']['item'] or '-'}) 타입 {m['att']['types']}->{m['def']['types']} | 모듈 {mod[i]:.3f} SD(0/1/2) {sd[i, 0]:.3f}/{sd[i, 1]:.3f}/{sd[i, 2]:.3f} | 날씨 {m['wx']} 지형 {m['tr']} 벽 {m['sc']} 테라 {m['att']['tera'] or m['def']['tera']}")
bad = defaultdict(list)
for i, m in enumerate(R):
    if not plain[i] and ok[i]:
        bad[m["move"]].append(mod[i] / sd[i, 0])
print("\n[위력 변동/특수 규칙 기술: 쌍 수 x |log 중앙 비| 큰 순]")
for mv_, v in sorted(bad.items(), key=lambda kv: -len(kv[1]) * abs(np.log(np.median(kv[1]))))[:30]:
    print(f"  {mv_:16s} {len(v):3d}쌍 | 중앙 비 {np.median(v):.2f} ({'/'.join(sorted({x['dyn'] for x in R if x['move'] == mv_}))})")
clean = ok & (dyn == "") & ~arr([m["tera"] or m["att"]["item"] in ("choiceband", "choicespecs", "lifeorb") or m["def"]["item"] in ("assaultvest", "eviolite") for m in R]) & ~fx
badm = clean & (np.abs(mod / np.maximum(sd[:, 0], 1e-6) - 1) > 0.10)
print(f"\n[아이템·테라·필드 없는 일반 기술 {int(clean.sum())}쌍 중 10% 넘게 어긋남 {int(badm.sum())}쌍 -> 원인 후보별 (어긋난 수/해당 수, 어긋난 것의 중앙 비)]")
for who in ("def", "att"):
    g = defaultdict(lambda: [0, 0, []])
    for i, m in enumerate(R):
        if clean[i]:
            x = g[m[who]["ability"]]
            x[0] += int(badm[i]); x[1] += 1
            if badm[i]:
                x[2].append(mod[i] / sd[i, 0])
    print(f" {who} 특성:", " | ".join(f"{k} {v[0]}/{v[1]} ({np.median(v[2]):.2f})" for k, v in sorted(g.items(), key=lambda kv: -kv[1][0])[:8] if v[0]))
g = defaultdict(list)
for i, m in enumerate(R):
    if badm[i] and m["def"]["ability"] not in ("multiscale", "iceface", "thickfat", "waterabsorb", "levitate", "voltabsorb", "flashfire", "sapsipper", "stormdrain", "lightningrod", "motordrive", "dryskin", "eartheater"):
        g[(m["def"]["species"], m["att"]["species"])].append(m["move"])
print(" 방어 특성이 원인 후보가 아닌 어긋남 (방어 종 <- 공격 종: 기술):", "; ".join(f"{k[0]}<-{k[1]}:{','.join(sorted(set(v)))}" for k, v in list(g.items())[:12]))
tm = ok & (dyn == "") & arr([m["tera"] for m in R])
print("\n[테라 있는 일반 기술 중 어긋난 쌍: 공격측 테라 / 방어측 테라 / 둘 다]")
for nm, f in (("공격측 테라", lambda m: bool(m["att"]["tera"]) and not m["def"]["tera"]), ("방어측 테라", lambda m: bool(m["def"]["tera"]) and not m["att"]["tera"]), ("둘 다", lambda m: bool(m["att"]["tera"] and m["def"]["tera"]))):
    sel = tm & arr([f(m) for m in R])
    r = np.abs(mod[sel] / sd[sel, 0] - 1)
    print(f"  {nm}: {int(sel.sum())}쌍, 3% 이내 {np.mean(r <= 0.03):.3f}, 중앙 비 {np.median(mod[sel] / sd[sel, 0]):.3f}")
bt = np.where(tm, np.abs(mod / np.maximum(sd[:, 0], 1e-6) - 1), 0)
for i in np.argsort(-bt)[:8]:
    m = R[i]
    print(f"  {m['att']['species']}({m['att']['ability']}) {m['move']} 타입 {m['att']['types']} 테라 {m['att']['tera']} -> {m['def']['species']}({m['def']['ability']}) 타입 {m['def']['types']} 테라 {m['def']['tera']} | 모듈 {mod[i]:.3f} SD {sd[i, 0]:.3f}")
if a.dump:
    i = int(np.argsort(-bt)[a.dump - 1])
    print(json.dumps({k: R[i][k] for k in ("att", "def", "move", "wx", "tr", "sc", "side")}, ensure_ascii=False), f"모듈 {mod[i]:.4f} SD {sd[i]}")
print("\n[면역 불일치 (SD 0인데 모듈 > 0 / 모듈 0인데 SD > 0) 기술 -> 방어 특성 개수]")
for nm, sel in (("SD 0 / 모듈 >0", sd0 & ~m0), ("모듈 0 / SD >0", m0 & ~sd0)):
    print(f"  {nm}:", Counter(f"{R[i]['move']}->{R[i]['def']['species']}({R[i]['def']['ability']}{',' + R[i]['def']['item'] if R[i]['def']['item'] else ''})" for i in np.where(sel)[0]).most_common(14))
