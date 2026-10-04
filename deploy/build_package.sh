#!/bin/bash
# 배포 묶음 만들기: deploy/dist/pokeai-bridge/ 와 pokeai-bridge.tar.gz
set -e
cd "$(dirname "$0")/.."
OUT=deploy/dist/pokeai-bridge
mkdir -p $OUT/src/core $OUT/data $OUT/checkpoints
for f in __init__ bridge_server events model rct_player search set_prior tensor_encoder; do cp src/core/$f.py $OUT/src/core/$f.py; done
cp src/__init__.py $OUT/src/__init__.py
cp data/vocab.json data/entity_features.npz $OUT/data/
cp checkpoints/supervised_v2_fp_v10boot.pt $OUT/checkpoints/
cp deploy/requirements.txt deploy/pokeai-bridge.service deploy/install.sh deploy/uninstall.sh $OUT/
tar czf deploy/dist/pokeai-bridge.tar.gz -C deploy/dist pokeai-bridge
echo "묶음 완료: $(du -h deploy/dist/pokeai-bridge.tar.gz | cut -f1)"
