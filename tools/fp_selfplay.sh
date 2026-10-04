#!/bin/bash
# FoulPlay 자가대전(FP vs FP) 데이터 수집: 양쪽 다 Foul Play라 상대편에 별도 수집기가 필요 없음 —
# 각 컨테이너가 자기 시점 공개 로그를 스스로 남기고(decision_log "protocol" 레코드), build_fp_dataset.py가 이걸 그대로 씀.
# 상대가 약하지 않은 강한 데이터를 얻으려는 용도. 한 쌍 = 컨테이너 2개(도전 A, 수락 B), 판마다 랜덤으로 다시 도전.
#
# 사용: bash tools/fp_selfplay.sh <쌍 수> <쌍당 판 수> <탐색 ms> <태그(영숫자, 실행마다 새 값)> [팀 풀 이름=train_a]
# 환경변수: OUTDIR (기본 data/fp_selfplay/ms<ms>), FORMAT (기본 gen9ou; Uber/AG 데이터는 gen9anythinggoes)
# 출력: dec_<tag>_<쌍번호>a.jsonl(도전 쪽) / ..._b.jsonl(수락 쪽) — 각자 start/decision/end/protocol 다 있음
pairs=$1; n=$2; ms=$3; tag=$4; pool=${5:-train_a}
OUTDIR=${OUTDIR:-data/fp_selfplay/ms$(printf %03d "$ms")}
ROOT="C:/Users/lsh/Desktop/orcl_server"
mkdir -p "$ROOT/poke-ai/$OUTDIR"
cd "$ROOT/poke-ai"

run_pair() {
  i=$1; ii=$(printf %02d "$i"); nameA=fpA${tag}$ii; nameB=fpB${tag}$ii
  logB="$OUTDIR/f_${tag}_${ii}b.log"
  # B(수락 쪽)를 먼저 띄우고 로그인할 때까지 기다린 뒤 A(도전 쪽)를 띄움 — 도전이 로그인 전에 와서 유실되는 것 방지
  MSYS_NO_PATHCONV=1 docker run --rm --name fps_${tag}_${ii}b -v "$ROOT/foul-play:/foul-play" -v "$ROOT/poke-ai/$OUTDIR:/out" \
    -e FP_DECISION_LOG=/out/dec_${tag}_${ii}b.jsonl fp-runner \
    --websocket-uri ws://host.docker.internal:8000/showdown/websocket --ps-username $nameB --bot-mode accept_challenge \
    --pokemon-format ${FORMAT:-gen9ou} --team-name gen9/$pool --search-time-ms $ms --run-count $n \
    --log-level INFO > "$logB" 2>&1 &
  bp=$!
  for _ in $(seq 60); do
    grep -aq "Successfully logged in" "$logB" 2>/dev/null && break
    kill -0 $bp 2>/dev/null || return
    sleep 2
  done
  MSYS_NO_PATHCONV=1 docker run --rm --name fps_${tag}_${ii}a -v "$ROOT/foul-play:/foul-play" -v "$ROOT/poke-ai/$OUTDIR:/out" \
    -e FP_DECISION_LOG=/out/dec_${tag}_${ii}a.jsonl fp-runner \
    --websocket-uri ws://host.docker.internal:8000/showdown/websocket --ps-username $nameA --bot-mode challenge_user \
    --user-to-challenge $nameB --pokemon-format ${FORMAT:-gen9ou} --team-name gen9/$pool --search-time-ms $ms --run-count $n \
    --log-level INFO > "$OUTDIR/f_${tag}_${ii}a.log" 2>&1 &
  ap=$!
  wait $ap
  docker rm -f fps_${tag}_${ii}a fps_${tag}_${ii}b > /dev/null 2>&1
  wait $bp
}
for i in $(seq 0 $((pairs-1))); do run_pair $i & sleep 1; done
wait
echo "완료 → $OUTDIR/dec_${tag}_*.jsonl"
grep -aho '"winner": "[^"]*"' "$OUTDIR"/dec_${tag}_*.jsonl 2>/dev/null | sort | uniq -c
