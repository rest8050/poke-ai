#!/bin/bash
# 서버 모델 교체 (재시작 없음): 체크포인트(+구조 설정 .cfg.json)를 올리고 포인터(checkpoints/CURRENT)만 바꾼다.
# 브리지는 새 배틀이 시작될 때 포인터를 보고 새 모델로 바꿔 끼움(진행 중 배틀은 이전 모델로 마무리). 읽기에 실패하면 이전 모델을 유지.
#   bash deploy/push_model.sh <이름>            예: v3_full  (checkpoints/supervised_v2_fp_<이름>.pt)
#   bash deploy/push_model.sh <이름> --code     코드(src/core 모델 파일, type_chart.json)도 같이 올림 — 새 구조(예: v3 첫 배포)일 때 필요.
#                                               코드 파일을 바꿨으면 서비스 재시작이 한 번 필요: ssh oci 'sudo systemctl restart pokeai-bridge' (접속자 없을 때)
#   bash deploy/push_model.sh --rollback        직전 포인터로 되돌림
#   bash deploy/push_model.sh --status          서버가 지금 쓰는 모델 확인
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${POKEAI_HOST:-oci}; REMOTE=/home/opc/pokeai-bridge
ssh_() { ssh -q "$HOST" "$@"; }
health() { sleep 7; ssh_ "curl -s http://127.0.0.1:8765/health"; echo; }

case "${1:-}" in
  --status)   health; exit 0 ;;
  --rollback) ssh_ "cd $REMOTE/checkpoints && test -f CURRENT.prev && cp CURRENT CURRENT.tmp && cp CURRENT.prev CURRENT && mv CURRENT.tmp CURRENT.prev && cat CURRENT"; health; exit 0 ;;
  ""|-*)      sed -n 2,12p "$0"; exit 1 ;;
esac

NAME=$1; FILE=supervised_v2_fp_$NAME.pt; SRC=checkpoints/$FILE
[ -f "$SRC" ] || { echo "체크포인트 없음: $SRC"; exit 1; }
if [ "${2:-}" = "--code" ]; then
  tar cf - src/__init__.py src/core/{__init__,bridge_server,events,model,model_v2,model_v3,matchup,rct_player,search,set_prior,tensor_encoder}.py data/type_chart.json | ssh_ "cd $REMOTE && tar xf -"
  echo "코드 올림 (bridge_server/model 등을 바꿨다면 재시작 필요)"
fi
# 올리는 도중 반쯤 쓰인 파일을 읽지 않도록 임시 이름으로 올린 뒤 이름 변경, 구조 설정 파일이 먼저 가도록 순서 유지
for f in "$FILE.cfg.json" "$FILE"; do
  if [ -f "checkpoints/$f" ]; then scp -q "checkpoints/$f" "$HOST:$REMOTE/checkpoints/$f.tmp" && ssh_ "mv $REMOTE/checkpoints/$f.tmp $REMOTE/checkpoints/$f"; fi
done
ssh_ "cd $REMOTE/checkpoints && { test -f CURRENT && cp CURRENT CURRENT.prev || true; } && echo '$FILE' > CURRENT.tmp && mv CURRENT.tmp CURRENT && cat CURRENT"
echo "포인터 변경 완료. 새 배틀부터 적용됩니다 (최대 5초 지연). 서버 상태:"
health
