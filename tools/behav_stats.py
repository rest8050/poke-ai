"""record_games.py 출력(프레임)에서 트릭룸/웨더볼 사용 행태 집계. 사용: behav_stats.py <접두사>... (예: data/behav/all9_tr)
트릭룸: 사용 횟수, 이미 트릭룸일 때 또 씀(취소), 자기가 켠 걸 2턴 안에 자기가 끔(자가취소)
웨더볼: 사용 횟수, 날씨 없이 씀"""
import json
import re
import sys


def scan(path):
    s = dict(games=0, tr_use=0, tr_cancel=0, tr_self_cancel=0, wb_use=0, wb_noweather=0)
    for suffix in ("_a.jsonl", "_b.jsonl"):
        for line in open(path + suffix, encoding="utf-8"):
            g = json.loads(line)
            s["games"] += 1
            turn, tr_on, tr_setter, tr_turn, weather = 0, False, None, -9, "none"
            for frame in g["frames"]:
                for l in frame.split("\n"):
                    p = l.split("|")
                    if len(p) < 2:
                        continue
                    t = p[1]
                    if t == "turn":
                        turn = int(p[2])
                    elif t == "-weather":
                        weather = p[2].lower()
                    elif t == "-fieldstart" and "Trick Room" in p[2]:
                        tr_on, tr_turn = True, turn
                        m = re.search(r"\[of\] (p\d)", l)
                        tr_setter = m.group(1) if m else None
                    elif t == "-fieldend" and "Trick Room" in p[2]:
                        tr_on = False
                    elif t == "move" and len(p) > 3:
                        side = p[2][:2]
                        if p[3] == "Trick Room":
                            s["tr_use"] += 1
                            if tr_on:
                                s["tr_cancel"] += 1
                                if tr_setter == side and turn - tr_turn <= 2:
                                    s["tr_self_cancel"] += 1
                        elif p[3] == "Weather Ball":
                            s["wb_use"] += 1
                            if weather in ("none", ""):
                                s["wb_noweather"] += 1
    # 양쪽 시점 파일이 같은 배틀을 각자 기록하므로 판 수와 횟수 모두 2배로 잡힘(두 모델 비교엔 무관)
    return s


for pre in sys.argv[1:]:
    s = scan(pre)
    print(pre, s)
