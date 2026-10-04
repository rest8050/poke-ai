#!/bin/bash
# 새 구조(belief 없음, 사건 특징, 특징 표 = BoW+훅/플래그, ID 드롭아웃)로 BC를 처음부터 학습: 샤드 8개씩 4덩어리를 이어서 (각 덩어리 1에폭, 이전 결과에서 이어 학습)
# 끝나면 서버를 켜서 RCT 상대로 평가(A holdout 1000판, B/C 각 500판)하고 서버를 끔.
# 사용: bash tools/train_bc_chain.sh [시작 덩어리=1] [끝 덩어리=4]   (예: 2 2 = 덩어리 2만 이어서). 평가는 끝 덩어리가 4일 때만.
# 체크포인트 checkpoints/supervised_v2.pt, 결과 logs/train_v2_results.txt
PY=/c/Users/lsh/anaconda3/envs/kosa/python.exe
cd "C:/Users/lsh/Desktop/orcl_server/poke-ai"
START=${1:-1}; END=${2:-4}
OUT=logs/train_v2_results.txt; CK=checkpoints/supervised_v2.pt; [ $START = 1 ] && : > "$OUT"
log(){ echo "$(date +%H:%M:%S) $1" | tee -a "$OUT"; }
pat(){ p=""; for i in $(seq $1 $2); do p="$p,data/metamon/raw/train-$(printf %05d $i)-*.parquet"; done; echo "${p#,}"; }
i=0
for range in "0 7" "8 15" "16 23" "24 31"; do
  i=$((i+1)); set -- $range
  [ $i -lt $START ] || [ $i -gt $END ] && continue
  if [ $i = 1 ]; then INIT=none; LR=2e-4; else INIT=$CK; LR=1e-4; fi
  log "=== 덩어리 $i (샤드 $1~$2) 시작: init=$INIT lr=$LR ==="
  $PY -u src/training/train_bc.py --parquet "$(pat $1 $2)" --max-battles 200000 --epochs 1 --batch-battles 32 \
     --init $INIT --lr $LR --out $CK --save-last --id-dropout 0.2 > logs/train_v2_c$i.log 2>&1
  log "덩어리 $i 끝: $(grep -a -E '학습 정확도' logs/train_v2_c$i.log | tail -1)"
done
[ $END != 4 ] && { log "덩어리 $START~$END 끝 (평가 생략)"; exit 0; }
up(){ netstat -ano | grep LISTENING | grep -q ":8000"; }
started=0
if ! up; then (cd pokemon-showdown && nohup node pokemon-showdown start --no-security > ../logs/train_v2_showdown.log 2>&1 &); started=1; for _ in $(seq 30); do up && break; sleep 2; done; fi
for pool in metamon_holdout:1000 rare:500 randomset:500; do
  name=${pool%%:*}; n=${pool##*:}
  for try in 1 2 3; do
    $PY src/evaluation/analyze_model.py --ckpt $CK --mode argmax --battles $n --opponent rct --battle-timeout 900 --workers 6 --pool data/team_pool_$name.json > logs/train_v2_eval_$name.txt 2>&1
    grep -a -q PermissionError logs/train_v2_eval_$name.txt || break
  done
  log "$($PY tools/summ_eval.py supervised_v2 $name)"
done
[ $started = 1 ] && for pid in $(netstat -ano | grep LISTENING | grep ":8000" | awk '{print $5}' | sort -u); do taskkill //PID $pid //F > /dev/null 2>&1; done
log "ALL DONE (서버 종료)"
