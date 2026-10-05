"""장부 동시 쓰기 self-check: PYTHONPATH=. python tests/test_ledger_concurrency.py
6개 프로세스가 동시에 각 300줄을 쓰면 1800줄이 모두 온전해야 함 (Windows O_APPEND 경합으로 줄이 덮어써지던 사고의 회귀 방지)"""
import json
import multiprocessing as mp
import os
import tempfile

from tools.ledger import append, read


def writer(path, k):
    for i in range(300):
        append({"kind": "game", "w": k, "i": i, "pad": "x" * 150}, path)


if __name__ == "__main__":
    path = os.path.join(tempfile.mkdtemp(), "l.jsonl")
    ps = [mp.Process(target=writer, args=(path, k)) for k in range(6)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    raw = [r for r in open(path, "rb").read().split(b"\n") if r.strip()]
    ok = sum(1 for r in raw if json.loads(r))
    got = {(r["w"], r["i"]) for r in read(path)}
    assert len(raw) == 1800 and ok == 1800 and len(got) == 1800, (len(raw), ok, len(got))
    print("동시 쓰기 1800줄 모두 정상")
