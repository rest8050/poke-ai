#!/bin/bash
# 장면 파이프라인: 정의(.py가 <이름>_in.json을 씀) -> 시뮬레이터로 위치까지 진행(rec) -> FP 300ms 재생 라벨 -> npz -> 학생 평가
# 사용: bash tools/scenario_run.sh <이름> [ckpt ...]   (data/scenarios/<이름>_defs.py 필요; FP 라벨링은 시간 기반이라 다른 CPU 작업이 없을 때만)
cd /c/Users/lsh/Desktop/orcl_server/poke-ai
PY=/c/Users/lsh/anaconda3/envs/kosa/python.exe; ROOT="C:/Users/lsh/Desktop/orcl_server"; N=$1; shift; D=data/scenarios
$PY $D/${N}_defs.py || exit 1
node tools/scenario_gen.js $D/${N}_in.json $D/${N}_rec.jsonl || exit 1
rm -f $D/${N}_lab.jsonl
MSYS_NO_PATHCONV=1 docker run --rm --name scn_lab -v "$ROOT/foul-play:/foul-play" -v "$ROOT/poke-ai/$D:/rec" -e FP_DECISION_LOG=/rec/${N}_lab.jsonl \
  --entrypoint python3 fp-runner -m fp.replay_label /rec/${N}_rec.jsonl 300 > $D/${N}_lab.log 2>&1
grep -a "재생 완료" $D/${N}_lab.log
PYTHONPATH=. $PY src/training/build_fp_dataset.py --decisions "$D/${N}_lab.jsonl" --out $D/${N}.npz > $D/${N}_build.log 2>&1 || { tail -3 $D/${N}_build.log; exit 1; }
$PY - <<PYEOF
import json
ids=[json.loads(l)["id"] for l in open("$D/${N}_rec.jsonl",encoding="utf-8")]      # 라벨 실패한 장면은 npz에 없을 수 있어 순서 맞춤은 태그로
tags={json.loads(l)["tag"]:json.loads(l)["id"] for l in open("$D/${N}_rec.jsonl",encoding="utf-8")}
seen=[]
for l in open("$D/${N}_lab.jsonl",encoding="utf-8"):
    r=json.loads(l)
    if r["type"]=="start": seen.append(tags[r["tag"]])
json.dump(seen,open("$D/${N}.ids.json","w"))
PYEOF
[ $# -gt 0 ] && $PY tools/scenario_eval.py $D/${N}.npz "$@"
