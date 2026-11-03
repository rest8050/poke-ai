"""cf_gen.js 출력 -> scenario_gen.js 입력 (M5): 위치 장면(접두까지) 또는 동치 검증 장면(접두 + 한 턴, 난수 고정).
사용: cf_scenes.py <cf.jsonl> <out_in.json> pos | equiv <N> [seed]"""
import json, random, sys

src, out, mode = sys.argv[1:4]
P = [json.loads(l) for l in open(src, encoding="utf-8")]
mk = lambda id, p, script, seed: dict(id=id, p1=dict(name="scn1", team=p["p1"]), p2=dict(name="scn2", team=p["p2"]), seed=seed, script=script)
if mode == "pos":
    sc = [mk(p["id"], p, p["prefix"], p["seed"]) for p in P]
else:
    n, rng = int(sys.argv[4]), random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    cells = [(p, c) for p in P for c in p["cells"]]
    sc = []
    for p, c in rng.sample(cells, min(n, len(cells))):
        st = [dict(s) for s in c["steps0"]]
        st[0]["reseed"] = p["rseeds"][0]
        sc.append(mk(f'{p["id"]}|{c["a"]}|{c["o"]}', p, p["prefix"] + st, p["seed"]))
json.dump(sc, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(len(sc), "장면 ->", out)
