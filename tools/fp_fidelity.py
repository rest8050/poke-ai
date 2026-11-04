"""사람 로그 → Foul Play 변환기(tools/human_to_fp.py)의 충실도 측정.
정답이 있는 판(학생 시점 녹화 rec: 실제 |request|와 FP 첫 라벨이 있음)의 내 쪽 정보를 일부러 관전자 형태로 깎고(퍼센트 HP, request 제거),
변환기로 되살린 뒤 비교한다.
  convert: 깎기 → 변환 → rec 저장 + V0 리포트 (request 위치/종류가 실제와 같은가, 추정 세트/HP/PP가 실제 request와 얼마나 다른가). FP 불필요
  compare: 변환 rec를 FP로 다시 라벨링한 결과(B)를 첫 라벨(A), 같은 상태의 재라벨(A2 = 교사 잡음)과 비교 (V1)
사용:
  python tools/fp_fidelity.py convert --src data/dagger/raw/d7 --recs rec_fp_h0,rec_fp_h1,rec_fp_h2 --out data/fp_fidelity
  python tools/fp_fidelity.py compare --a data/dagger/raw/d7 --a2 data/dagger/raw/d7b --b data/fp_fidelity --recs rec_fp_h0,rec_fp_h1,rec_fp_h2
비교는 학생이 고를 수 있는 선택지(드러난 기술 + 교체)로 제한해 다시 정규화한 분포 위에서 한다: A는 안 드러난 진짜 기술을 빼고, B는 PAD 기술을 뺌.
지표(teacher_ceiling.py와 같은 정의): top-1 일치, KL(A||X), X의 argmax를 A의 Q로 잰 regret. X = A2(교사 잡음 기준선) 또는 B(변환). B가 A2만큼 A와 가까우면 변환은 충분히 충실."""
import argparse
import glob
import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from human_to_fp import _norm, convert

DROP = ("init", "title", "j", "l", "t:", "inactive", "inactiveoff", "c", "c:", "n", "raw", "html", "uhtml", "request")


def load_recs(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def requests(rec):
    """프레임에서 |request| JSON들 (wait 요청은 제외) → [(종류, JSON)]"""
    out = []
    for fr in rec["frames"]:
        if "|request|" in fr:
            r = json.loads(fr.split("|request|")[1].split("\n")[0])
            kind = "tp" if r.get("teamPreview") else "forced" if r.get("forceSwitch") else "turn" if r.get("active") else None
            if kind:
                out.append((kind, r))
    return out


def degrade(rec):
    """학생 시점 녹화 → (관전자 형태 로그, 내 쪽). 내 쪽 HP는 PS의 퍼센트 규칙(올림, 깎였으면 최대 99)으로 되돌림"""
    first = requests(rec)[0][1]["side"]
    side, maxhp = first["id"], {p["ident"]: int(p["condition"].split("/")[1].split(" ")[0]) for p in first["pokemon"]}

    def pct(ident, cond):
        key = ident[:2] + ident[3:]
        if "fnt" in cond or key not in maxhp:
            return cond
        hp, _, rest = cond.partition(" ")
        cur = int(hp.split("/")[0])
        p = math.ceil(cur * 100 / maxhp[key])
        p = 99 if p == 100 and cur < maxhp[key] else p
        return f"{p}/100" + (f" {rest}" if rest else "")

    out = []
    for fr in rec["frames"]:
        if "|request|" in fr:
            continue
        for l in fr.split("\n")[1:]:
            s = l.split("|")
            if not l.startswith("|") or len(s) < 2 or s[1] in DROP:
                continue
            if len(s) > 2 and s[2].startswith(side) and len(s[2]) > 2 and s[2][2] in "abc":
                if s[1] in ("switch", "drag") and len(s) > 4:
                    s[4] = pct(s[2], s[4])
                elif s[1] in ("-damage", "-heal") and len(s) > 3:
                    s[3] = pct(s[2], s[3])
            out.append("|".join(s))
    return "\n".join(out), side


POOL_GROUP = {"train_a2": "사람형", "train_a": "사람형", "metamon": "사람형", "metamon_holdout": "사람형", "randomset_train3": "무작위 세트", "rare_train3": "희귀"}


def pool_groups():
    """팀 서명(종 + 기술 집합) -> 풀 묶음 (사람형 / 무작위 세트 / 희귀). 여러 풀에 겹치면 사람형 우선. 학생 판의 팀이 어디서 왔는지 가름"""
    from poke_env.teambuilder import Teambuilder
    out = {}
    for path in glob.glob("data/team_pool_*.json"):
        g = POOL_GROUP.get(os.path.basename(path)[10:-5])
        if g is None:
            continue
        for tm in json.load(open(path, encoding="utf-8-sig"))["teams"]:
            out_k = frozenset((_norm(m.species or m.nickname), frozenset(_norm(x) for x in m.moves)) for m in Teambuilder.parse_showdown_team(tm["export"]))
            if out.get(out_k) != "사람형":
                out[out_k] = g
    return out


def game_groups(src, names):
    """(rec 이름, 배틀 태그) -> 풀 묶음 (원래 rec의 export로 판정)"""
    from poke_env.teambuilder import Teambuilder
    pg, out = pool_groups(), {}
    for n in names:
        for rec in load_recs(os.path.join(src, n + ".jsonl.gz")):
            k = frozenset((_norm(m.species or m.nickname), frozenset(_norm(x) for x in m.moves)) for m in Teambuilder.parse_showdown_team(rec["export"]))
            out[(n, rec["tag"])] = pg.get(k, "기타")
    return out


def species_of(details):
    return _norm(details.split(",")[0])


ORACLE_PARTS = ("moves", "item", "ability", "tera", "stats")


def oracle_from(rec, parts):
    """실제 팀 프리뷰 request에서 고른 세트 필드(parts)만 정답으로 뽑음 → human_to_fp.convert(oracle=)"""
    out = {}
    for p in requests(rec)[0][1]["side"]["pokemon"]:
        o = {"moves": list(p["moves"]), "item": _norm(p["item"]), "ability": _norm(p["baseAbility"]), "tera": _norm(p["teraType"]),
             "stats": dict(p["stats"]), "maxhp": int(p["condition"].split("/")[1].split(" ")[0])}
        out[species_of(p["details"])] = {k: v for k, v in o.items() if k in parts or (k == "maxhp" and "stats" in parts)}
    return out


def v0_report(pairs):
    """pairs = [(실제 rec, 변환 rec)]: request 순서/종류 일치와 필드별 오차"""
    n = Counter()
    err = defaultdict(list)
    for real, conv in pairs:
        a, b = requests(real), requests(conv)
        n["games"] += 1
        ka, kb = [k for k, _ in a], [k for k, _ in b]
        if ka != kb:
            n["skeleton_mismatch_games"] += 1
            n["real_requests_in_mismatch"] += len(ka)
            n["conv_requests_in_mismatch"] += len(kb)
            continue
        n["requests"] += len(a)
        tp_real, tp_conv = a[0][1]["side"]["pokemon"], b[0][1]["side"]["pokemon"]
        by = {species_of(p["details"]): p for p in tp_conv}
        for p in tp_real:                                         # 팀 세트 추정 정확도 (팀 프리뷰 요청 = 게임 시작 시점의 정답)
            q = by.get(species_of(p["details"]))
            if q is None:
                n["species_missing"] += 1
                continue
            e = conv["meta"]["est"].get(species_of(p["details"]), {})
            n["mons"] += 1
            for f in ("item", "ability"):
                n[f + "_ok"] += _norm(p[f]) == _norm(q[f])
            n["tera_ok"] += _norm(p["teraType"]) == _norm(q["teraType"])
            hp_real = int(p["condition"].split("/")[1].split(" ")[0])
            err["hp_rel"].append(abs(q_hp(q) - hp_real) / hp_real)
            for st in ("atk", "def", "spa", "spd", "spe"):
                err["stat_" + st].append(abs(q["stats"][st] - p["stats"][st]) / p["stats"][st])
            n["spe_exact"] += q["stats"]["spe"] == p["stats"]["spe"]
            n["moves_total"] += len(p["moves"])
            n["moves_revealed"] += e.get("nrev", 0)
            n["moves_revealed_correct"] += sum(m in p["moves"] for m in q["moves"][:e.get("nrev", 0)])
        for (kind, ra), (_, rb) in zip(a, b):
            if kind == "tp":
                continue
            mb = {species_of(p["details"]): p for p in rb["side"]["pokemon"]}
            for p in ra["side"]["pokemon"]:
                q = mb.get(species_of(p["details"]))
                if q is None:
                    continue
                n["cond_n"] += 1
                hr, mr = (int(p["condition"].split("/")[0]) if "fnt" not in p["condition"] else 0), int(p["condition"].split("/")[1].split(" ")[0]) if "fnt" not in p["condition"] else 1
                hq = int(q["condition"].split("/")[0]) if "fnt" not in q["condition"] else 0
                if "fnt" in p["condition"] or "fnt" in q["condition"]:
                    n["faint_ok"] += ("fnt" in p["condition"]) == ("fnt" in q["condition"])
                else:
                    err["hp_frac_err"].append(abs(hq / q_hp(q) - hr / mr))
                    n["status_ok"] += (p["condition"].split(" ")[1:] == q["condition"].split(" ")[1:])
                    n["status_n"] += 1
                n["item_now_ok"] += _norm(p["item"]) == _norm(q["item"])
            if kind == "turn":
                ma, mb_ = ra["active"][0], rb["active"][0]
                n["turn_req"] += 1
                n["tera_flag_ok"] += ma.get("canTerastallize") == mb_.get("canTerastallize")
                ia = {m["id"]: m for m in ma["moves"]}
                for m in mb_["moves"]:
                    if m["id"] in ia and "pp" in m and "pp" in ia[m["id"]]:
                        n["pp_n"] += 1
                        n["pp_ok"] += m["pp"] == ia[m["id"]]["pp"]
                        n["dis_n"] += 1
                        n["dis_ok"] += bool(m["disabled"]) == bool(ia[m["id"]]["disabled"])
                n["single_move_real"] += len(ma["moves"]) == 1
                n["single_move_conv"] += len(mb_["moves"]) == 1
                n["trapped_real"] += bool(ma.get("trapped"))
    return n, err


def q_hp(q):
    return int(q["condition"].split("/")[1].split(" ")[0])


def print_v0(n, err):
    r = lambda a, b: f"{a / b:.3f} ({a}/{b})" if b else "n/a"
    q = lambda x: f"중앙 {np.median(x):.4f} 평균 {np.mean(x):.4f} p90 {np.percentile(x, 90):.4f}" if len(x) else "n/a"
    print(f"판 {n['games']}, request 순서 불일치 판 {n['skeleton_mismatch_games']} (실제 {n['real_requests_in_mismatch']} / 변환 {n['conv_requests_in_mismatch']}개), 비교한 request {n['requests']}")
    print("--- 세트 추정 (팀 프리뷰 시점, 포켓몬 %d마리)" % n["mons"])
    print(f"도구 일치 {r(n['item_ok'], n['mons'])} | 특성 {r(n['ability_ok'], n['mons'])} | 테라 {r(n['tera_ok'], n['mons'])}")
    print(f"기술: 드러난 칸 {r(n['moves_revealed'], n['moves_total'])}, 드러난 기술이 실제 세트에 있음 {r(n['moves_revealed_correct'], n['moves_revealed'])}")
    print(f"최대 HP 상대오차 {q(err['hp_rel'])}")
    for st in ("atk", "def", "spa", "spd", "spe"):
        print(f"능력치 {st} 상대오차 {q(err['stat_' + st])}")
    print(f"스피드 정확히 일치 {r(n['spe_exact'], n['mons'])}")
    print("--- 결정 시점 상태 (매 request, 포켓몬 6마리 모두)")
    print(f"HP 비율 오차 {q(err['hp_frac_err'])} | 기절 일치 {r(n['faint_ok'], n['cond_n'] - n['status_n'])} | 상태이상 일치 {r(n['status_ok'], n['status_n'])} | 현재 도구 일치 {r(n['item_now_ok'], n['cond_n'])}")
    print(f"턴 request {n['turn_req']}: 테라 가능 표시 일치 {r(n['tera_flag_ok'], n['turn_req'])} | PP 일치 {r(n['pp_ok'], n['pp_n'])} | disabled 일치 {r(n['dis_ok'], n['dis_n'])}")
    print(f"기술이 하나뿐인 요청(고정/충전): 실제 {n['single_move_real']}, 변환 {n['single_move_conv']} | 실제 갇힘(trapped) {n['trapped_real']}")


def restricted(r, revealed):
    """정책/가치에서 학생이 못 고르는 기술(안 드러난 기술, PAD)을 뺀 (policy 리스트, 뺀 질량 비율)"""
    ok = lambda c: c.startswith("switch ") or c.removesuffix("-tera") in revealed
    pol = [(c, w) for c, w in r["policy"] if ok(c)]
    tot = sum(w for _, w in r["policy"])
    return pol, 1 - sum(w for _, w in pol) / tot if tot else 0.0, ok


def load_worlds(d, names, js, load):
    """월드 j의 라벨 파일(lab_<rec>_w<j>.jsonl)들 → [키(파일 이름에서 _w<j> 뺌) -> 결정 기록]"""
    out = []
    for j in js:
        raw = load(",".join(f"{d}/lab_{n}_w{j}.jsonl" for n in names))
        out.append({(re.sub(r"_w\d+$", "", k[0]),) + k[1:]: v for k, v in raw.items()})
    return out


def mix_worlds(worlds, revealed):
    """월드별 정책을 학생이 고를 수 있는 선택지로 제한해 정규화한 뒤 평균 → 키 -> {"policy": ...}. 그 결정이 없는 월드는 빼고 평균"""
    out = {}
    for k in set().union(*[set(w) for w in worlds]):
        rev = revealed.get((k[0].removeprefix("lab_"), k[1]))
        if rev is None:
            continue
        acc, n = defaultdict(float), 0
        for w in worlds:
            if k not in w:
                continue
            pol, _, _ = restricted(w[k], rev)
            s = sum(x for _, x in pol)
            if not pol or s <= 0:
                continue
            for c, x in pol:
                acc[c] += x / s
            n += 1
        if n:
            out[k] = {"policy": [(c, x / n) for c, x in acc.items()]}
    return out


SMOOTH = 0.02


def cmd_compare(args):
    import legal_eval as L
    from regret_eval import boot_ci
    from teacher_ceiling import KEEP, kind, load, qvals
    names = args.recs.split(",")
    srcs = lambda d, pre="lab_": ",".join(f"{d}/{pre}{n}.jsonl" for n in names)
    js = [int(x) for x in args.worlds.split(",")] if args.worlds else None
    A, A2 = load(srcs(args.a)), load(srcs(args.a2))
    revealed = {}
    for n in names:
        path = os.path.join(args.b, f"{n}_w{js[0]}.jsonl" if js else f"{n}.jsonl")
        if os.path.exists(path):
            for rec in load_recs(path):
                revealed[(n, rec["tag"])] = set(rec["meta"]["revealed"])
    B = mix_worlds(load_worlds(args.b, names, js, load), revealed) if js else load(srcs(args.b))
    keys = sorted(set(A) & set(A2) & set(B))
    print(f"첫 라벨 {len(A)}, 재라벨 {len(A2)}, 변환 라벨 {len(B)}결정 → 셋 다 있는 결정 {len(keys)} (변환에만 {len(set(B) - set(A))}, 첫 라벨에만 {len(set(A) - set(B))})" + (f" | 월드 {js} 평균" if js else ""))

    def dist(pol):
        best = max(w for _, w in pol)
        keep = {c: w for c, w in pol if w >= KEEP * best}
        s = sum(keep.values())
        return {c: w / s for c, w in keep.items()}

    groups = game_groups(args.a, names)
    bid, rows, grp = {}, [], []
    for k in keys:
        g = groups.get((k[0].removeprefix("lab_"), k[1]), "기타")
        rev = revealed.get((k[0].removeprefix("lab_"), k[1]))
        if rev is None or (args.only_group and g != args.only_group):
            continue
        pols = {}
        for tag, src in (("a", A), ("2", A2), ("b", B)):
            pol, mass, ok = restricted(src[k], rev)
            if len(pol) < 2:
                break
            pols[tag] = (dist(pol), mass, ok)
        else:
            da = pols["a"][0]
            names_ = sorted(set(da) | set(pols["2"][0]) | set(pols["b"][0]))
            pa = np.array([da.get(c, 0.0) for c in names_])
            qa = {c: v for c, v in qvals(A[k], 100).items() if pols["a"][2](c)}
            qmax = max(qa.values()) if len(qa) >= 2 else None
            row = [bid.setdefault(k[:2], len(bid)), pa.max(), kind(names_[int(pa.argmax())]), k[3], pols["a"][1]]
            for tag in ("2", "b"):
                px = np.array([pols[tag][0].get(c, 0.0) for c in names_])
                pick = names_[int(px.argmax())]
                reg = qmax - qa[pick] if qmax is not None and pick in qa else np.nan
                ps = (1 - SMOOTH) * px + SMOOTH / len(names_)                      # 평활 KL: 지원이 다른 분포끼리 무한대가 되지 않게
                kls = float((pa[pa > 0] * (np.log(pa[pa > 0]) - np.log(ps[pa > 0]))).sum())
                xs = np.array([pols[tag][0].get(c, 0.0) for c in qa]) if qmax is not None else np.zeros(0)
                mreg = float((xs / xs.sum() * np.array([qmax - qa[c] for c in qa])).sum()) if qmax is not None and xs.sum() > 0 else np.nan   # 평균 정책 regret
                row += [float(pa.argmax() == px.argmax()), float(L.kl(pa[None], px[None])[0]), reg, kls, mreg]
            rows.append(row)
            grp.append(g)
    grp = np.array(grp)
    battle = np.array([r[0] for r in rows]); conf = np.array([r[1] for r in rows]); kinds = np.array([r[2] for r in rows])
    forced = np.array([r[3] for r in rows]); mass = np.array([r[4] for r in rows]); X = np.array([r[5:] for r in rows], dtype=float)
    rng = np.random.RandomState(0)
    f = lambda x: "{:.4f} [{:.4f},{:.4f}]".format(*boot_ci(x, battle, rng))
    print(f"비교 가능(선택지 2개 이상) {len(rows)}결정 {len(np.unique(battle))}판 | 첫 라벨이 학생이 못 고르는 기술에 둔 질량 평균 {mass.mean():.3f}" + (f" | {args.only_group}만" if args.only_group else ""))
    print(f"{'지표':28s} {'A2(교사 잡음)':>26s} {'B(변환)':>26s}   B - A2 [95% 구간]")
    for name, i in (("top-1 일치(높을수록 좋음)", 0), ("KL(A||X) (낮을수록)", 1), ("argmax regret(A의 Q)", 2), ("평활 KL (λ=0.02)", 3), ("평균 정책 regret(A의 Q)", 4)):
        a2, b = X[:, i], X[:, 5 + i]
        print(f"{name:28s} {f(a2):>26s} {f(b):>26s}   {f(b - a2)}")
    dis2, disb = 1 - X[:, 0], 1 - X[:, 5]
    print(f"B / A2 비: 불일치율 {disb.mean() / dis2.mean():.2f}배 | 평활 KL {X[:, 8].mean() / X[:, 3].mean():.2f}배 | 평균 정책 regret {np.nanmean(X[:, 9]) / np.nanmean(X[:, 4]):.2f}배 (1에 가까울수록 변환이 재실행 잡음 수준)")
    print(f"{'구분':14s} {'결정':>6s} | {'일치 A2':>8s} {'일치 B':>8s} | {'평활KL A2':>9s} {'평활KL B':>9s} | {'평균regret A2':>13s} {'평균regret B':>12s}")
    kinds_ = [(f"A argmax {kn}", kinds == kn) for kn in ("기술", "테라 기술", "교체")] + [("강제 교체 결정", forced == 1), ("일반 턴 결정", forced == 0),
                                                                                ("확신 >= 0.8", conf >= 0.8), ("확신 < 0.8", conf < 0.8)]
    for gn, m in kinds_:
        if m.sum() >= 30:
            print(f"{gn:14s} {int(m.sum()):6d} | {X[m, 0].mean():8.3f} {X[m, 5].mean():8.3f} | {X[m, 3].mean():9.4f} {X[m, 8].mean():9.4f} | {np.nanmean(X[m, 4]):13.4f} {np.nanmean(X[m, 9]):12.4f}")
    print(f"{'팀 풀 묶음':12s} {'판':>4s} {'결정':>6s} | {'일치 A2':>8s} {'일치 B':>8s} | {'regret A2':>10s} {'regret B':>9s} | {'불일치 B/A2':>11s}")
    for gname in ("사람형", "무작위 세트", "희귀"):
        m = grp == gname
        if m.sum() >= 30:
            print(f"{gname:12s} {len(np.unique(battle[m])):4d} {int(m.sum()):6d} | {X[m, 0].mean():8.3f} {X[m, 5].mean():8.3f} | {np.nanmean(X[m, 2]):10.4f} {np.nanmean(X[m, 7]):9.4f} | {(1 - X[m, 5].mean()) / (1 - X[m, 0].mean()):11.2f}")


def cmd_convert(args):
    os.makedirs(args.out, exist_ok=True)
    names = args.recs.split(",")
    gmap = game_groups(args.src, names) if args.group else None
    pairs, fails = [], Counter()
    for name in names:
        recs = [r for r in load_recs(os.path.join(args.src, name + ".jsonl.gz")) if gmap is None or gmap.get((name, r["tag"])) == args.group]
        outs = defaultdict(list)                       # 파일 접미사("" 또는 _w<j>) -> rec들
        for rec in recs:
            log, side = degrade(rec)
            if args.worlds or args.map:
                from set_posterior import make_world
                fns = make_world(rec["tag"], args.worlds or 1, "map" if args.map else "sample")
                convs = [(f"_w{j}" if args.worlds else "", convert(log, side, rec["user"], rec["tag"], world=fn)) for j, fn in enumerate(fns)]
            else:
                parts = ORACLE_PARTS if args.oracle == "all" else tuple(x for x in args.oracle.split(",") if x)
                convs = [("", convert(log, side, rec["user"], rec["tag"], oracle_from(rec, parts) if parts else None))]
            for suf, conv in convs:
                if isinstance(conv, str):
                    fails[conv.split(":")[0]] += 1
                    continue
                outs[suf].append(conv)
                if suf in ("", "_w0"):
                    pairs.append((rec, conv))
        for suf, out in outs.items():
            with open(os.path.join(args.out, name + suf + ".jsonl"), "w", encoding="utf-8") as f:
                for c in out:
                    f.write(json.dumps(c) + "\n")
        print(f"{name}: 실제 {len(recs)}판 → 변환 {len(outs.get('', outs.get('_w0', [])))}판" + (f" x 월드 {len(outs)}" if args.worlds else ""), flush=True)
    print(f"변환 실패 사유 {dict(fails)}")
    print_v0(*v0_report(pairs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", required=True)
    c.add_argument("--recs", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--oracle", default="", help="정답으로 줄 세트 필드 (콤마: moves,item,ability,tera,stats 또는 all). 비우면 추정만")
    c.add_argument("--map", action="store_true", help="완결 세트 표의 사후 최빈 세트 하나로 변환 (tools/set_posterior.py)")
    c.add_argument("--worlds", type=int, default=0, help="사후에서 뽑은 세트 월드 K개 (파일 이름 <rec>_w<j>.jsonl)")
    c.add_argument("--group", default="", help="이 팀 풀 묶음의 판만 변환 (사람형 / 무작위 세트 / 희귀)")
    k = sub.add_parser("compare")
    k.add_argument("--a", required=True)
    k.add_argument("--a2", required=True)
    k.add_argument("--b", required=True)
    k.add_argument("--recs", required=True)
    k.add_argument("--worlds", default="", help="평균할 월드 번호들 (콤마, 예 0,1,2,3). 비우면 b의 단일 변환 라벨")
    k.add_argument("--only-group", default="", help="이 팀 풀 묶음의 판만 비교")
    args = ap.parse_args()
    {"convert": cmd_convert, "compare": cmd_compare}[args.cmd](args)
