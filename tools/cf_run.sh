#!/bin/bash
# M5 장면 파이프라인: data/cf/<이름>_in.json(cf_scenes.py가 만듦) -> 시뮬레이터 진행(rec) -> FP 재생(탐색 ms는 라벨이 필요 없으니 짧게, CF_MS) -> npz + ids
# 사용: bash tools/cf_run.sh <이름>
cd /c/Users/lsh/Desktop/orcl_server/poke-ai
PY=/c/Users/lsh/anaconda3/envs/kosa/python.exe; ROOT="C:/Users/lsh/Desktop/orcl_server"; N=$1; D=data/cf; MS=${CF_MS:-20}
node tools/scenario_gen.js $D/${N}_in.json $D/${N}_rec.jsonl || exit 1
rm -f $D/${N}_lab.jsonl
MSYS_NO_PATHCONV=1 docker run --rm --name cf_lab_$N -v "$ROOT/foul-play:/foul-play" -v "$ROOT/poke-ai/$D:/rec" -e FP_DECISION_LOG=/rec/${N}_lab.jsonl -e REPLAY_LAST_MS=${CF_LAST_MS:-0} \
  --entrypoint python3 fp-runner -m fp.replay_label /rec/${N}_rec.jsonl $MS > $D/${N}_lab.log 2>&1
grep -a "재생 완료" $D/${N}_lab.log
PYTHONPATH=. $PY src/training/build_fp_dataset.py --decisions "$D/${N}_lab.jsonl" --out $D/${N}.npz > $D/${N}_build.log 2>&1 || { tail -3 $D/${N}_build.log; exit 1; }
$PY - <<PYEOF
import json
tags={json.loads(l)["tag"]:json.loads(l)["id"] for l in open("$D/${N}_rec.jsonl",encoding="utf-8")}
seen=[tags[r["tag"]] for r in map(json.loads,open("$D/${N}_lab.jsonl",encoding="utf-8")) if r["type"]=="start"]
json.dump(seen,open("$D/${N}.ids.json","w"))
print(len(seen),"시퀀스")
PYEOF
