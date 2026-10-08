#!/bin/bash
# 배포 묶음 만들기: deploy/dist/pokeai-bridge/ 와 pokeai-bridge.tar.gz
set -e
cd "$(dirname "$0")/.."
OUT=deploy/dist/pokeai-bridge
mkdir -p $OUT/src/core $OUT/data $OUT/checkpoints
for f in __init__ bridge_server events model model_v3 model_v3_legacy model_v4 model_v5 belief matchup rct_player search set_prior tensor_encoder; do cp src/core/$f.py $OUT/src/core/$f.py; done
cp src/__init__.py $OUT/src/__init__.py
cp data/vocab.json data/entity_features.npz data/type_chart.json $OUT/data/
# 최초 설치용 모델: 이후 교체는 deploy/push_model.sh (포인터 방식). 구조 설정 파일(.cfg.json)이 있으면 같이 복사하고 포인터를 만든다
INIT=${1:-v3_full}
cp checkpoints/supervised_v2_fp_$INIT.pt $OUT/checkpoints/
[ -f checkpoints/supervised_v2_fp_$INIT.pt.cfg.json ] && cp checkpoints/supervised_v2_fp_$INIT.pt.cfg.json $OUT/checkpoints/ || true
echo "supervised_v2_fp_$INIT.pt" > $OUT/checkpoints/CURRENT
cp deploy/requirements.txt deploy/pokeai-bridge.service deploy/install.sh deploy/uninstall.sh $OUT/
tar czf deploy/dist/pokeai-bridge.tar.gz -C deploy/dist pokeai-bridge
echo "묶음 완료: $(du -h deploy/dist/pokeai-bridge.tar.gz | cut -f1)"
