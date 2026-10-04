"""설치 확인용(표준 라이브러리만 사용): 모드가 남긴 실제 배틀 로그의 메시지를 브리지에 choose 호출 단위로 다시 보내 응답과 지연시간을 출력.
사용: python3 replay_check.py <모드 로그 파일> [브리지 주소=http://127.0.0.1:8765]"""
import json
import re
import sys
import time
import urllib.request

path = sys.argv[1]
url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8765") + "/choose"
calls, cur, chunk = [], None, None
for ln in open(path, encoding="utf-8").read().split("\n"):
    if ln.startswith("==== choose()"):
        cur, chunk = {"chunks": [], "head": ln}, None
        calls.append(cur)
    elif re.match(r"msg\[\d+\] ", ln):
        chunk = [re.sub(r"^msg\[\d+\] ", "", ln)]
        cur["chunks"].append(chunk)
    elif ln.startswith(("-> ", "moveset:", "request(", "  후보", "!!")):
        chunk = None
    elif chunk is not None:
        chunk.append(ln)
side = re.search(r"actor=(p[12])", calls[0]["head"]).group(1)
ok = 0
for i, c in enumerate(calls):
    body = json.dumps({"battle_id": "replaycheck", "side": side, "chunks": ["\n".join(x) for x in c["chunks"]]}).encode()
    t0 = time.time()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=10).read())
    print(f"#{i} {r['kind']} {r.get('id') or r.get('ident')}  서버 {r.get('ms')}ms / 왕복 {(time.time() - t0) * 1000:.0f}ms")
    ok += r["kind"] in ("move", "switch")
print(f"유효한 행동 {ok}/{len(calls)}")
