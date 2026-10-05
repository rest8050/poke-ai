"""설치기(스텔스락/압정/독압정/끈적끈적네트)·벽(리플렉터/빛의장막/오로라베일) 포켓몬을 얼마나 일찍 쓰는지: 학생 vs Foul Play (같은 판 안에서 비교).
사용: hazard_timing.py [디렉터리=data/loss_analysis]   (rec_*.jsonl 학생 시점 프레임 + fpside_*.jsonl Foul Play 쪽 로그 필요)
팀 구성: 학생=export 텍스트, Foul Play=start 레코드의 team. 사건: 학생 프레임의 |switch|/|move|/|turn| 줄 (양쪽 공개 행동이 다 들어 있음)"""
import ast
import glob
import json
import re
import statistics
import sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "data/loss_analysis"
toid = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
GROUPS = {"설치기(락/압정/독압정/네트)": {"stealthrock", "spikes", "toxicspikes", "stickyweb"},
          "벽(리플렉터/빛의장막/오로라베일)": {"reflect", "lightscreen", "auroraveil"}}


def team_from_export(exp):
    team = {}
    for blk in exp.strip().split("\n\n"):
        lines = blk.strip().split("\n")
        first = re.sub(r"\s*\((M|F)\)\s*$", "", lines[0].split(" @")[0].strip())
        m = re.search(r"\(([^()]+)\)$", first)
        team[toid(m.group(1) if m else first)] = {toid(l[2:]) for l in lines if l.startswith("- ")}
    return team


fp_start = {}
for f in glob.glob(f"{D}/fpside_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") == "start":
            fp_start[d["tag"]] = d["team"]

stats = {g: {"학생": defaultdict(list), "FoulPlay": defaultdict(list)} for g in GROUPS}
for f in glob.glob(f"{D}/rec_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        g = json.loads(line)
        if g["tag"] not in fp_start:
            continue
        frames = "\n".join(g["frames"]).split("\n")
        side_of = {}
        for ln in frames:
            p = ln.split("|")
            if len(p) > 3 and p[1] == "player":
                side_of[p[3]] = p[2]
        stu, fpl = side_of.get(g["user"]), [s for n, s in side_of.items() if n != g["user"]]
        if not stu or not fpl:
            continue
        teams = {stu: team_from_export(g["export"]), fpl[0]: {}}
        raw = fp_start[g["tag"]]
        raw = raw if isinstance(raw, list) else ast.literal_eval(raw)
        for mon in raw:
            teams[fpl[0]][toid(mon.get("species"))] = {toid(m) for m in mon.get("moves", [])}
        name_of = {stu: "학생", fpl[0]: "FoulPlay"}
        lead, first_turn, turn = {}, {}, 0
        for ln in frames:
            p = ln.split("|")
            if len(p) >= 3 and p[1] == "turn":
                turn = int(p[2])
                continue
            if len(p) < 4:
                continue
            if p[1] == "switch" and p[2][:2] in name_of and p[2][:2] not in lead:
                lead[p[2][:2]] = toid(p[3].split(",")[0])
            elif p[1] == "move" and p[2][:2] in name_of:
                mv = toid(p[3])
                for grp, moves in GROUPS.items():
                    if mv in moves:
                        first_turn.setdefault((p[2][:2], grp), turn)
        for side, who in name_of.items():
            for grp, moves in GROUPS.items():
                setters = {sp for sp, mv in teams[side].items() if mv & moves}
                if not setters:
                    continue
                s = stats[grp][who]
                s["games"].append(1)
                s["lead_is_setter"].append(int(lead.get(side) in setters))
                t = first_turn.get((side, grp))
                s["ever_set"].append(int(t is not None))
                s["early3"].append(int(t is not None and t <= 3))
                if t is not None:
                    s["turns"].append(t)

for grp in GROUPS:
    print(f"== {grp}: 팀에 해당 기술을 가진 포켓몬이 있는 경우")
    print(f"{'':10s} {'판':>4s} {'선두가 그 포켓몬':>14s} {'한 번이라도 사용':>14s} {'턴3 이내 사용':>12s} {'첫 사용 턴(중앙값)':>16s}")
    for who in ("학생", "FoulPlay"):
        s = stats[grp][who]
        n = len(s["games"])
        if n:
            print(f"{who:10s} {n:4d} {sum(s['lead_is_setter']) / n:14.1%} {sum(s['ever_set']) / n:14.1%} {sum(s['early3']) / n:12.1%} "
                  f"{statistics.median(s['turns']) if s['turns'] else float('nan'):16.1f}")
    print()
