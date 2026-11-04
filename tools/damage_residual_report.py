"""행동 사건(tools/extract_events.py 출력)으로 모델의 손계산(Matchup과 같은 근사, src/core/damage_calc.py)이 실제 피해와 얼마나 어긋나는지
특성/도구/기술/날씨/필드/벽/랭크별로 집계 → 메커니즘 표에 무엇부터 넣을지, 학습된 보정(결과 예측기)이 얼마나 필요한지 판단.

지표 r = ln(실제 피해 / 손계산 평균 피해). 그룹의 평균 r(편향)을 배율 exp(평균 r)로 보여줌 (1.50 = 실제가 손계산의 1.5배)
  제외: 급소/빗나감/실패/방어/대타/반사된 기술/가변 위력 기술/자기 대상, 실제 피해 3% 미만(HP 1% 반올림 잡음), 기띠/옹골참 발동
  정상 범위: 난수(0.85~1.0)와 HP 반올림 때문에 개별 |r| 0.1 안팎은 잡음. 표본이 많은 그룹의 편향이 1에서 멀수록 손계산이 놓친 메커니즘
  KO 사건은 피해가 남은 HP에서 잘리므로 r 대신 "최대 난수 손계산이 KO를 예측했나"로 따로 셈. 무효는 손계산/실제 불일치를 따로 셈
능력치: 기본은 팀 데이터의 실제 능력치(노력치/성격, 세트를 모르는 쪽만 Matchup의 표준 가정). --stats standard면 모두 표준 가정 (노력치 추정 오차 크기 확인용)
등장 사건: 장판 피해(스텔스록 1/8 x 바위 상성, 압정 1/8·1/6·1/4, 두꺼운부츠/매직가드/비접지 예외)와 실제 차이, 위협 발동 시 상대 랭크 변화 분포
데이터: 종족값/기술은 poke-env(GenData 9세대), 상성표는 data/type_chart.json (Matchup과 같은 표)
사용: python tools/damage_residual_report.py data/events/ms100.jsonl.gz [더 많은 파일] [--min-n 30] [--top 20] [--tsv out.tsv] [--stats standard]"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from src.core.damage_calc import calc_stats, damage, item_stat_mult, standard_stats, supported, type_eff
from src.training.battle_events import find_set

MIN_OBS = 0.03
SCREENS = {"Physical": ("reflect", "auroraveil"), "Special": ("lightscreen", "auroraveil")}
SPIKES = {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}


def species_entry(species, sp, cache={}):
    """poke-env pokedex에서 종 찾기 (폼 차이는 접두사로)"""
    key = (id(species), sp)
    if key not in cache:
        cache[key] = species.get(sp) or next((v for k, v in species.items() if k.startswith(sp) or (sp.startswith(k) and len(k) > 3)), None)
    return cache[key]


def mon_view(state, sset, entry, stats_mode):
    """사건의 상태 표기 + 세트 + 종 데이터 → damage_calc.damage가 받는 dict (능력치는 도구 배율까지 적용)"""
    base = {k: entry["baseStats"][k] for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    if sset and stats_mode == "true":
        stats = calc_stats(base, sset["evs"], sset["ivs"], sset["level"], sset["nature"])
    else:
        stats = standard_stats(base)
    hp = stats["hp"]
    return {"stats": item_stat_mult(stats, state.get("item"), state.get("ab")), "maxhp": hp,
            "types": [str(t).lower() for t in entry.get("types", [])], "tera": state.get("tera"), "boosts": state.get("b") or {},
            "status": state.get("st"), "ability": state.get("ab"), "item": state.get("item")}


def evaluate(ev, header, species, moves, chart, stats_mode="true"):
    """기술 사건 하나 → {"kind": 제외 사유 | "immune_ok" | "immune_miss" | "immune_extra" | "ko" | "dmg", ...}"""
    out, u, t = ev["out"], ev.get("u"), ev.get("t")
    if ev.get("via"):
        return {"kind": "via"}
    if not u or not t or ev.get("ts") == ev.get("us"):
        return {"kind": "self_or_none"}
    mv = moves.get(ev["move"])
    if mv is None or mv.get("category") == "Status":
        return {"kind": "status"}
    mv = dict(mv, id=ev["move"])
    if not supported(mv):
        return {"kind": "variable"}
    if out.get("miss") or out.get("fail") or out.get("protect") or out.get("sub") or t.get("sub"):
        return {"kind": "no_contact"}
    if out.get("crit"):
        return {"kind": "crit"}
    ue, te = species_entry(species, u["sp"]), species_entry(species, t["sp"])
    if ue is None or te is None:
        return {"kind": "unknown_species"}
    sets = header.get("sets", {})
    a = mon_view(u, find_set(sets.get(ev["us"]), u["sp"]), ue, stats_mode)
    d = mon_view(t, find_set(sets.get(ev["ts"]), t["sp"]), te, stats_mode)
    calc = damage(a, d, mv, chart, (ev.get("field") or {}).get("weather"))
    hits = out.get("hits") or 1
    pred, pred_max = calc["mean"] * hits / d["maxhp"], calc["max"] * hits / d["maxhp"]
    obs = out.get("dmg", 0.0)
    if calc["eff"] == 0:
        return {"kind": "immune_ok" if out.get("immune") or obs == 0 else "immune_miss"}
    if out.get("immune"):
        return {"kind": "immune_extra"}
    ends = {x[1] for x in out.get("items_end", [])} | {x if isinstance(x, str) else x[1] for x in out.get("activates", [])}
    if "focussash" in ends or "sturdy" in ends:
        return {"kind": "sash"}
    if out.get("ko"):
        return {"kind": "ko", "pred_ko": pred_max >= t["hp"] - 1e-6, "pred": pred, "hp": t["hp"]}
    if obs < MIN_OBS or pred <= 0:
        return {"kind": "small"}
    return {"kind": "dmg", "r": math.log(obs / pred), "pred": pred, "obs": obs}


def groups(ev, mv):
    """이 기술 사건이 들어가는 집계 그룹들 [(구분, 값)]"""
    u, t, field = ev["u"], ev["t"], ev.get("field") or {}
    cat = mv.get("category")
    a_stat, d_stat = ("atk", "def") if cat == "Physical" else ("spa", "spd")
    screen = next((s for s in SCREENS.get(cat, ()) if (ev.get("sc") or {}).get(ev["ts"], {}).get(s)), "-")
    g = [("공격 특성", u.get("ab") or "?"), ("공격 도구", u.get("item") or "-"), ("방어 특성", t.get("ab") or "?"),
         ("방어 도구", t.get("item") or "-"), ("기술", ev["move"]), ("날씨", field.get("weather") or "-"), ("필드", field.get("terrain") or "-"),
         ("방어측 벽", screen), ("공격측 테라", "on" if u.get("tera") else "off"), ("방어측 테라", "on" if t.get("tera") else "off"),
         ("공격 랭크", str((u.get("b") or {}).get(a_stat, 0))), ("방어 랭크", str((t.get("b") or {}).get(d_stat, 0)))]
    if u.get("ab") == "supremeoverlord":
        g.append(("총대장 아군 기절 수", str((ev.get("fnt") or {}).get(ev["us"], 0))))
    return g


def hazard_expected(ev, species, chart):
    """등장 사건의 예상 장판 피해 (스텔스록 1/8 x 바위 상성, 압정 단수별, 두꺼운부츠/매직가드/비접지 예외)"""
    mon, sc = ev["mon"], (ev.get("sc") or {}).get(ev["side"], {})
    if not mon or mon.get("item") == "heavydutyboots" or mon.get("ab") == "magicguard":
        return 0.0
    entry = species_entry(species, mon["sp"])
    types = [mon["tera"]] if mon.get("tera") else [str(x).lower() for x in (entry or {}).get("types", [])]
    exp = 0.125 * type_eff(chart, "rock", types) if sc.get("stealthrock") else 0.0
    grounded = "flying" not in types and mon.get("ab") != "levitate" and mon.get("item") != "airballoon"
    if grounded and sc.get("spikes"):
        exp += SPIKES[min(3, sc["spikes"])]
    return min(exp, mon.get("hp", 1.0))


def load_data(chart_path="data/type_chart.json"):
    from poke_env.data import GenData
    gd = GenData.from_gen(9)
    chart = json.load(open(chart_path, encoding="utf-8"))
    return gd.pokedex, gd.moves, {a.lower(): {d.lower(): m for d, m in row.items()} for a, row in chart.items()}


def read_events(paths):
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def fmt_row(name, rs):
    n = len(rs)
    bias = sum(rs) / n
    return f"| {name} | {n} | ×{math.exp(bias):.2f} | {sum(abs(r) for r in rs) / n:.3f} | {sum(abs(r) > 0.15 for r in rs) / n:.0%} |"


def main(args):
    species, moves, chart = load_data(args.chart)
    kinds, by_group, ko, ko_miss = Counter(), defaultdict(list), Counter(), defaultdict(Counter)
    immune_miss, all_r = defaultdict(Counter), []
    haz_n, haz_err, haz_bad = 0, 0.0, defaultdict(Counter)
    intim = defaultdict(Counter)
    header = {}
    for ev in read_events(args.events):
        if ev["type"] == "battle":
            header = ev
            continue
        if ev["type"] == "switch":
            sc = (ev.get("sc") or {}).get(ev["side"], {})
            if ev.get("mon") and (sc.get("stealthrock") or sc.get("spikes")):
                exp, obs = hazard_expected(ev, species, chart), ev["out"].get("hazard", 0.0)
                haz_n += 1
                haz_err += abs(obs - exp)
                if abs(obs - exp) > 0.03:
                    haz_bad["도구"][ev["mon"].get("item") or "-"] += 1
                    haz_bad["특성"][ev["mon"].get("ab") or "?"] += 1
            if [ev["side"], "intimidate"] in ev["out"].get("abilities", []) and ev.get("opp"):
                ob = ev["out"].get("opp_boosts", {})
                intim[ev["opp"].get("ab") or "?"][",".join(f"{k}{v:+d}" for k, v in sorted(ob.items())) or "변화 없음"] += 1
            continue
        res = evaluate(ev, header, species, moves, chart, args.stats)
        kinds[res["kind"]] += 1
        if res["kind"] == "dmg":
            all_r.append(res["r"])
            for g in groups(ev, moves[ev["move"]]):
                by_group[g].append(res["r"])
        elif res["kind"] == "ko":
            ko[res["pred_ko"]] += 1
            if not res["pred_ko"]:
                for g in groups(ev, moves[ev["move"]]):
                    ko_miss[g[0]][g[1]] += 1
        elif res["kind"] in ("immune_miss", "immune_extra"):
            immune_miss[res["kind"]][f"{ev['t'].get('ab') or '?'} / {ev['t'].get('item') or '-'} / {ev['move']}"] += 1

    print(f"## 손계산 오차 진단 (능력치 {args.stats})\n")
    print("사건 분류:", dict(kinds))
    if all_r:
        n = len(all_r)
        print(f"\n비교한 피해 사건 {n}개 | 전체 편향 ×{math.exp(sum(all_r) / n):.3f} | 평균 |r| {sum(abs(r) for r in all_r) / n:.3f} | "
              f"|r|≤0.10 {sum(abs(r) <= 0.10 for r in all_r) / n:.0%} | |r|≤0.15 {sum(abs(r) <= 0.15 for r in all_r) / n:.0%}")
    if ko:
        print(f"KO 사건 {sum(ko.values())}개 중 손계산(최대 난수)이 KO를 예측한 비율 {ko[True] / sum(ko.values()):.0%}")

    print(f"\n### 그룹별 편향 (표본 {args.min_n}개 이상, 영향 = 표본 수 × |평균 r| 순)\n")
    dims = []
    for g in by_group:
        if g[0] not in dims:
            dims.append(g[0])
    for dim in dims:
        rows = [(v, rs) for (d, v), rs in by_group.items() if d == dim and len(rs) >= args.min_n]
        rows.sort(key=lambda x: -len(x[1]) * abs(sum(x[1]) / len(x[1])))
        if not rows:
            continue
        print(f"\n#### {dim}\n\n| 값 | n | 실제/손계산 | 평균 |r| | |r|>0.15 |\n|---|---|---|---|---|")
        for v, rs in rows[:args.top]:
            print(fmt_row(v, rs))

    if ko_miss:
        print("\n### 손계산이 못 맞힌 KO (실제 KO인데 최대 난수로도 KO 아님) — 많은 순")
        for dim in ("기술", "공격 특성", "공격 도구", "방어 특성", "방어측 벽"):
            top = ko_miss[dim].most_common(8)
            if top:
                print(f"- {dim}: " + ", ".join(f"{k} {c}" for k, c in top))
    for k, title in (("immune_miss", "손계산은 무효인데 실제로 맞음"), ("immune_extra", "손계산은 맞는데 실제로 무효")):
        if immune_miss[k]:
            print(f"\n### {title} (방어 특성 / 방어 도구 / 기술)\n- " + "\n- ".join(f"{x} {c}" for x, c in immune_miss[k].most_common(12)))
    if haz_n:
        print(f"\n### 장판 피해: 장판이 깔린 쪽 등장 {haz_n}번, 예상과 실제 차이 평균 {haz_err / haz_n:.3f}")
        for dim in ("도구", "특성"):
            print(f"- 차이 3%p 넘는 등장의 {dim}: " + (", ".join(f"{k} {c}" for k, c in haz_bad[dim].most_common(8)) or "없음"))
    if intim:
        print("\n### 위협 발동 시 상대 랭크 변화 (상대 특성별)")
        for ab, c in sorted(intim.items(), key=lambda x: -sum(x[1].values()))[:12]:
            print(f"- {ab}: " + ", ".join(f"{k} {n}" for k, n in c.most_common(4)))

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as f:
            f.write("구분\t값\tn\t편향배율\t평균abs_r\n")
            for (dim, v), rs in sorted(by_group.items()):
                f.write(f"{dim}\t{v}\t{len(rs)}\t{math.exp(sum(rs) / len(rs)):.4f}\t{sum(abs(r) for r in rs) / len(rs):.4f}\n")
        print(f"\n전체 그룹 표: {args.tsv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+", help="tools/extract_events.py 출력 (.jsonl.gz)")
    ap.add_argument("--stats", choices=["true", "standard"], default="true", help="true=팀 데이터의 실제 능력치, standard=Matchup의 표준 가정")
    ap.add_argument("--chart", default="data/type_chart.json")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--tsv", default=None)
    main(ap.parse_args())
