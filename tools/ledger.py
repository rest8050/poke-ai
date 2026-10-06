"""대전 결과 장부(append-only JSONL). 한 줄 = 한 판(kind=game) | 집계(kind=agg, 예전 h2h 요약 이식용) | 판정(kind=decision).
사용: python tools/ledger.py backfill   # 지금까지의 알려진 h2h 결과를 장부에 이식 (이미 넣은 출처는 건너뜀)
승자 표기: game.winner = "a"|"b"|"draw"|"void"  (void = 시간 초과 등으로 강제 종료돼 집계에서 제외)"""
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

LEDGER = "results/ledger.jsonl"
POOL_OF = {"A": "holdout", "B": "rare", "C": "randomset"}   # h2h_run.sh가 쓰던 풀 이름


def model_id(path):
    name = os.path.basename(path).replace("supervised_v2_fp_", "").replace(".pt", "")
    with open(path, "rb") as f:
        return name, hashlib.sha1(f.read()).hexdigest()[:8]


def _lock(fd):
    """프로세스 간 배타 잠금 (Windows는 O_APPEND가 프로세스 사이에서 원자적이지 않아 여러 작업자가 쓰면 줄이 서로 덮어써짐)"""
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.002)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock(fd):
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, 0)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def append(rec, path=LEDGER):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    lock = os.open(path + ".lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
    try:
        _lock(lock)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        _unlock(lock)
    finally:
        os.close(lock)


def read(path=LEDGER):
    """깨진 줄(예전 동시 쓰기 사고로 생긴 조각)은 건너뜀"""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for l in f:
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except ValueError:
                    pass
    return rows


def pair_counts(rows, a, b):
    """a 대 b의 (a승, b승, 무, 무효). 장부에 b 대 a로 기록된 것도 방향을 뒤집어 합침"""
    w = l = d = v = 0
    for r in rows:
        if r.get("kind") == "game":
            x, y, win = r["a"], r["b"], r["winner"]
            if (x, y) == (b, a):
                win = {"a": "b", "b": "a"}.get(win, win)
            elif (x, y) != (a, b):
                continue
            w += win == "a"; l += win == "b"; d += win == "draw"; v += win == "void"
        elif r.get("kind") == "agg":
            if (r["a"], r["b"]) == (a, b):
                w += r["wins_a"]; l += r["wins_b"]
            elif (r["a"], r["b"]) == (b, a):
                w += r["wins_b"]; l += r["wins_a"]
            else:
                continue
            d += r.get("draws", 0)
    return w, l, d, v


def balanced(game_rows):
    """풀별 판 수를 같게 맞춘 승/패: 풀마다 시간순 앞 m판(m = 가장 적은 풀의 판 수, 무승부·무효 제외). 풀을 균등 혼합한 승률을 재기 위함"""
    by = defaultdict(list)
    for r in sorted(game_rows, key=lambda r: r["t"]):
        if r["winner"] in ("a", "b"):
            by[r["pool"]].append(r["winner"])
    if not by:
        return 0, 0, {}
    m = min(len(v) for v in by.values())
    w = sum(x == "a" for v in by.values() for x in v[:m])
    return w, m * len(by) - w, {p: (sum(x == "a" for x in v[:m]), m) for p, v in by.items()}


# 지금까지 돌린 대전 중 결과 파일이 남아 있는 것 (파일에 풀별 줄이 있음). 파일이 덮어써진 것은 값을 직접 적음
KNOWN_FILES = [
    ("logs/h2h_summary_v12r3_run1.txt", "v1_full_r3", "v1_full_boot"), ("logs/h2h_summary_v12r3_run2.txt", "v1_full_r3", "v1_full_boot"),
    ("logs/h2h_summary_v11head.txt", "v11head", "v1_full_boot"),
    ("logs/h2h_summary_v13_run1.txt", "v13deep", "v1_full_r3"), ("logs/h2h_summary_v13_run2.txt", "v13deep", "v1_full_r3"),
    ("logs/h2h_summary_v14pair_all.txt", "v14pair", "v1_full_r3"), ("logs/h2h_summary_v14res_all.txt", "v14res", "v1_full_r3"),
]
KNOWN_COUNTS = [  # (출처, a, b, {풀: (a승, b승)})  — 대화 중 기록된 값 (v1_full_r3 대 v12r1은 875판+1000판 중 A풀 일부가 중단돼 있음)
    ("v12r3_vs_v12r1", "v1_full_r3", "v12r1", {"A": (473, 402), "B": (501, 499), "C": (492, 508)}),
]


def backfill(path=LEDGER):
    done = {r.get("src") for r in read(path) if r.get("kind") == "agg"}
    added = 0
    for f, a, b in KNOWN_FILES:
        if f in done or not os.path.exists(f):
            continue
        for line in open(f, encoding="utf-8", errors="replace"):
            m = re.match(r"H2H (\w): A (\d+) / B (\d+) / \S+ (\d+)", line)
            if m:
                append({"kind": "agg", "src": f, "a": a, "b": b, "pool": POOL_OF.get(m[1], m[1]),
                        "wins_a": int(m[2]), "wins_b": int(m[3]), "draws": int(m[4])}, path)
                added += 1
    for src, a, b, pools in KNOWN_COUNTS:
        if src in done:
            continue
        for p, (wa, wb) in pools.items():
            append({"kind": "agg", "src": src, "a": a, "b": b, "pool": POOL_OF[p], "wins_a": wa, "wins_b": wb, "draws": 0}, path)
            added += 1
    print(f"집계 {added}줄 추가")


if __name__ == "__main__":
    if sys.argv[1:] == ["backfill"]:
        backfill()
    else:
        print(__doc__)
