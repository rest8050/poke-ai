#!/bin/bash
# 체크포인트를 Foul Play(컨테이너)와 붙여 승률 측정. 사용: bash tools/fp_eval.sh <쌍 수> <쌍당 판 수> <탐색 ms> <태그> <ckpt>
# 환경변수: POOL(우리 팀 풀 JSON, 기본 holdout) TEAMDIR(Foul Play 팀 폴더, 같은 풀이어야 함, 기본 gen9/holdout)
# 출력: 마지막 줄 "TOTAL ... 우리 W승 / N판"
pairs=$1; n=$2; ms=$3; tag=$4; ckpt=$5
ROOT="C:/Users/lsh/Desktop/orcl_server"; PY=/c/Users/lsh/anaconda3/envs/kosa/python.exe
cd "$ROOT/poke-ai"; mkdir -p logs/fp_eval
pair() {
  i=$1; name=fp${tag}$i
  $PY tools/fp_match.py bc $name $n $ckpt > logs/fp_eval/m_${tag}_$i.log 2>&1 &
  mp=$!
  sleep 30  # 상대 로그인 대기
  MSYS_NO_PATHCONV=1 docker run --rm --name fpe_${tag}_$i -v "$ROOT/foul-play:/foul-play" fp-runner \
    --websocket-uri ws://host.docker.internal:8000/showdown/websocket --ps-username $name --bot-mode challenge_user \
    --user-to-challenge bc${tag}$i --pokemon-format gen9ou --team-name ${TEAMDIR:-gen9/holdout} --search-time-ms $ms --run-count $n \
    --log-level INFO > logs/fp_eval/f_${tag}_$i.log 2>&1 &
  dp=$!
  wait $mp
  docker rm -f fpe_${tag}_$i > /dev/null 2>&1
  wait $dp
}
for i in $(seq 0 $((pairs-1))); do pair $i & sleep 1; done
wait
tw=0; tn=0
for i in $(seq 0 $((pairs-1))); do
  r=$(grep -a RESULT logs/fp_eval/m_${tag}_$i.log)
  w=$(echo "$r" | sed -E 's/.*우리 ([0-9]+)승 \/ ([0-9]+)판.*/\1/'); f=$(echo "$r" | sed -E 's/.*우리 ([0-9]+)승 \/ ([0-9]+)판.*/\2/')
  [[ "$w" =~ ^[0-9]+$ ]] && { tw=$((tw+w)); tn=$((tn+f)); }
done
echo "TOTAL Foul Play(${ms}ms) ${ckpt}: 우리 ${tw}승 / ${tn}판"
