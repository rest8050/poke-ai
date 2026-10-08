"""Foul Play 대전 기록 → 행동 사건 JSONL.gz (src/training/battle_events.py). 결과 예측기 학습과 손계산 오차 진단(tools/damage_residual_report.py)의 입력.
자가대전이면 같은 배틀의 양쪽 start 레코드로 두 팀의 세트(노력치/성격/도구/특성)를 모두 앎. 외부 상대(poke-env 수집기)는 그쪽 세트를 모름(known=false).
메모리: 1차로 start/end 레코드(팀/승자)만 모으고, 2차로 프로토콜을 만나는 대로 배틀 하나씩 처리해 씀 (프로토콜을 쌓아두지 않음)
사용: python tools/extract_events.py --decisions "data/fp_selfplay/ms100/dec_*.jsonl" --out data/events/ms100.jsonl.gz [--protocols ...] [--max-battles N]
출력 줄: {"type": "battle", tag, users, sets, known, winner} 다음에 그 배틀의 사건들 {"type": "move"|"switch", "tag", ...}"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from src.training.battle_events import parse_battle


def load_jsonl(pattern):
    """콤마로 여러 glob 패턴 (build_fp_dataset.load_jsonl과 같은 규칙, numpy/poke-env 없이)"""
    for path in sorted({p for pat in pattern.split(",") for p in glob.glob(pat.strip())}):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main(args):
    teams, winners = defaultdict(dict), {}
    for r in load_jsonl(args.decisions):
        if r.get("type") == "start":
            teams[r["tag"]][r["user"]] = r["team"]
        elif r.get("type") == "end":
            winners[r["tag"]] = r.get("winner")

    def protocols():   # 외부 상대가 남긴 프로토콜이 있으면 우선 (build_fp_dataset과 같은 우선순위), 없으면 Foul Play 자신의 기록
        if args.protocols:
            for p in load_jsonl(args.protocols):
                yield p["tag"], p["protocol"]
        for r in load_jsonl(args.decisions):
            if r.get("type") == "protocol":
                yield r["tag"], r["lines"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stat, done = Counter(), set()
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for tag, lines in protocols():
            if tag in done:
                continue
            done.add(tag)
            header, events = parse_battle(lines, teams.get(tag, {}))
            f.write(json.dumps({"type": "battle", "tag": tag, "winner": winners.get(tag), **header}, ensure_ascii=False) + "\n")
            for ev in events:
                f.write(json.dumps({"tag": tag, **ev}, ensure_ascii=False) + "\n")
            stat["배틀"] += 1
            stat["양쪽 세트 앎"] += all(header["known"].values()) and len(header["known"]) == 2
            stat["기술 사건"] += sum(e["type"] == "move" for e in events)
            stat["등장 사건"] += sum(e["type"] == "switch" for e in events)
            if args.max_battles and stat["배틀"] >= args.max_battles:
                break
    print(f"✅ {args.out}: {dict(stat)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help="Foul Play 결정 JSONL glob (콤마로 여러 개)")
    ap.add_argument("--protocols", default=None, help="상대편(poke-env)이 남긴 프로토콜 JSONL glob. 자가대전이면 생략")
    ap.add_argument("--out", default="data/events/events.jsonl.gz")
    ap.add_argument("--max-battles", type=int, default=0, help="이 수만큼만 처리 (0=전부). 빠른 진단용")
    main(ap.parse_args())
