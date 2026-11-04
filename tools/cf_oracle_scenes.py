"""오라클 한 수 앞보기 (docs/chain_oracle.md): cf jsonl의 칸마다 '위치 + 그 칸의 한 턴(rep 0)' 장면을 만들고 샤드로 나눠 저장.
기절로 p1 강제 교체가 걸린 칸은 교체 대상마다 장면을 더 만들어 교체 뒤 결정까지 진행 (id 끝 숫자 = 교체 대상 시뮬레이터 팀 번호, 일반 칸은 -).
사용: cf_oracle_scenes.py <cf.jsonl> <out_prefix> <샤드 수>   -> <out_prefix><k>_in.json"""
import json, sys

src, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
mk = lambda id, p, script: dict(id=id, p1=dict(name="scn1", team=p["p1"]), p2=dict(name="scn2", team=p["p2"]), seed=p["seed"], script=script)
sc = []
for line in open(src, encoding="utf-8"):
    p = json.loads(line)
    for c in p["cells"]:
        r = c["reps"][0]
        st = [dict(s) for s in c["steps0"]]
        st[0]["reseed"] = p["rseeds"][0]
        base = p["prefix"] + st
        if r["req"] == "move":
            sc.append(mk(f'{p["id"]}|{c["a"]}|{c["o"]}|-', p, base))
        elif r["req"] == "switch":
            for i, m in enumerate(r["p1"]):
                if m["hp"] > 0 and not m["act"]:
                    sc.append(mk(f'{p["id"]}|{c["a"]}|{c["o"]}|{i}', p, base + [{"p1": f"switch {i + 1}"}]))
for k in range(K):
    json.dump(sc[k::K], open(f"{out}{k}_in.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(sc), "장면 ->", K, "샤드")
