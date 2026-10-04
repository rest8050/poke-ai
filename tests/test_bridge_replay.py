"""브리지 서버 오프라인 검증: 모드가 남긴 실제 로그(coble-test/pokeai-log*/…log)의 메시지 덩어리를 choose() 호출 단위로 서버에 다시 넣어
(1) 정규화/필터(상대 정보 제거, UUID→종 이름)가 예외 없이 되고 (2) 유효한 행동이 나오는지 확인.
사용: PYTHONPATH=. python tests/test_bridge_replay.py [로그 파일]"""
import glob
import os
import re
import sys

from fastapi.testclient import TestClient

from src.core import bridge_server as bs

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(r"C:\Users\lsh\Desktop\coble-test\pokeai-log*\*.log"), key=os.path.getsize)[-1]
text = open(path, encoding="utf-8").read().split("\n")

calls, cur = [], None            # 호출마다 (선택 결과 줄, 이 호출 전에 새로 쌓인 덩어리들)
chunks = None
for ln in text:
    if ln.startswith("==== choose()"):
        cur = {"chunks": [], "head": ln}
        calls.append(cur)
        chunks = None
    elif re.match(r"msg\[\d+\] ", ln):
        chunks = [re.sub(r"^msg\[\d+\] ", "", ln)]
        cur["chunks"].append(chunks)
    elif ln.startswith("-> ") or ln.startswith("moveset:") or ln.startswith("request(") or ln.startswith("  후보") or ln.startswith("!!"):
        chunks = None
    elif chunks is not None:
        chunks.append(ln)

client = TestClient(bs.app)
side = re.search(r"actor=(p[12])", calls[0]["head"]).group(1)
print(f"로그 {os.path.basename(path)}: choose 호출 {len(calls)}회, 우리 쪽 {side}")
ok = 0
for i, c in enumerate(calls):
    body = {"battle_id": "replaytest", "side": side, "chunks": ["\n".join(x) for x in c["chunks"]]}
    r = client.post("/choose", json=body).json()
    print(f"#{i} chunks={len(body['chunks'])} -> {r}")
    ok += r["kind"] in ("move", "switch")
assert ok >= 1, "유효한 행동이 하나도 없음"
s = bs.sessions["replaytest"]
print("우리 팀:", list(s.battle.team.keys()), "| 상대 팀:", list(s.battle.opponent_team.keys()))
print("정규화 표(uuid→종):", len(s.species_of_uuid), "개 | ident→uuid:", s.uuid_of_ident)
# 상대 정보가 새지 않았는지: 상대 쪽 요청에만 있던 기술(Taunt/Leer/Scratch 등)이 상대 포켓몬 기술로 들어가 있으면 안 됨 (공개 프로토콜에서 쓴 것만)
opp = list(s.battle.opponent_team.values())[0]
print("상대가 아는 기술(공개된 것만이어야 함):", list(opp.moves.keys()))
print("OK")
