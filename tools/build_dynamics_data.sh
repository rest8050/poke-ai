#!/bin/bash
# 동역학/상대 행동 예측 학습용 데이터: 자기 대전 원본 로그를 그룹(접두사)별로 build_fp_dataset.py로 변환해 data/dyn/에 저장.
# 기존 학습 데이터와 같은 빌더(결정마다 상태/FP 정책/Q) + turn, kind(0 턴 결정/1 강제 교체), opp_action(그 턴 상대 선택: 0 기술 1 교체 2 테라+기술 -1 행동 못 함 -2 강제 교체라 해당 없음), opp_move.
# 다음 상태 = 같은 배틀(lens)의 다음 행. 워커 8개를 쓰므로 GPU 학습/FP 라벨링과 겹치지 않게 실행. 사용: bash tools/build_dynamics_data.sh
cd /c/Users/lsh/Desktop/orcl_server/poke-ai
PY=/c/Users/lsh/anaconda3/envs/kosa/python.exe
mkdir -p data/dyn
for dir in data/fp_selfplay/ms100 data/fp_selfplay/ms300 data/fp_selfplay/uber300; do
  for g in $(ls $dir | grep '^dec_' | sed -E 's/_[0-9]+[ab]\.jsonl\.gz$//; s/\.jsonl\.gz$//' | sort -u); do
    out=data/dyn/$(basename $dir)_${g#dec_}.npz
    [ -f $out ] && continue
    echo "[$(date +%H:%M:%S)] $g -> $out"
    $PY src/training/build_fp_dataset.py --decisions "$dir/${g}_*" --out $out 2>&1 | grep -E "✅|❌|추적기"
  done
done
echo DYNDONE
