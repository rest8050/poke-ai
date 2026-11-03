"""사람 데이터 BC 실험 결과 모음: logs/human_<태그>.txt의 첫 줄들을 표로.  python tools/human_table.py"""
import glob
import re

rows = []
for f in sorted(glob.glob("logs/human_*.txt")):
    t = open(f, encoding="utf-8").read().splitlines()
    if len(t) < 2 or not t[0].startswith("["):
        continue
    tag = re.match(r"\[(\w+)\]", t[0]).group(1)
    m = re.search(r"테스트 전체: 사람 행동 NLL ([\d.]+) \| 일치율 ([\d.]+) \| 상대 종류 ([\d.]+) \| 상대 기술 ([\d.]+)", t[1])
    ab = {k: re.search(rf"테스트 {k}.*ΔNLL ([+-][\d.]+)", "\n".join(t)) for k in ("마지막 기술 입력을 0으로", "히스토리 출력 0", "히스토리 최근 1턴")}
    rows.append((tag, m.group(1), m.group(2), m.group(4), *[ab[k].group(1) if ab[k] else "-" for k in ab]))
print(f"{'태그':8s} {'NLL':>7s} {'일치율':>6s} {'상대기술':>8s} | {'기술입력0':>9s} {'히스토리0':>9s} {'최근1턴':>8s}")
for r in rows:
    print(f"{r[0]:8s} {r[1]:>7s} {r[2]:>6s} {r[3]:>8s} | {r[4]:>9s} {r[5]:>9s} {r[6]:>8s}")
