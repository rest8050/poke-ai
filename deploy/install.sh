#!/bin/bash
# OCI(Oracle Linux 9, aarch64)에 PokeAI 브리지 설치. 시스템 파이썬은 건드리지 않고, 사용자 영역에 새 파이썬(uv가 받아옴)과 가상환경을 만든다.
# 마지막의 systemd 등록만 sudo가 필요. 게임 서버(minecraft.service)는 건드리지 않음.
# 사용: (pokeai-bridge 폴더를 ~/pokeai-bridge 에 풀어 둔 뒤)  bash ~/pokeai-bridge/install.sh
set -euo pipefail
BASE="$HOME/pokeai-bridge"
cd "$BASE"

# 1) uv를 PyPI에서 (원격 스크립트를 실행하지 않고 pip로 설치) → uv가 독립 실행형 파이썬 3.12를 내려받음 (poke-env는 파이썬 3.10 이상 필요)
python3 -m venv .bootstrap
.bootstrap/bin/pip install -q uv
.bootstrap/bin/uv venv --python 3.12 venv

# 2) 의존성 (torch는 aarch64 CPU 빌드)
.bootstrap/bin/uv pip install --python venv/bin/python -r requirements.txt
venv/bin/python - <<'PY'
import torch, poke_env, fastapi
print("설치 확인: torch", torch.__version__, "| threads", torch.get_num_threads())
PY

# 3) systemd 서비스 등록 (sudo)
sudo cp pokeai-bridge.service /etc/systemd/system/pokeai-bridge.service
sudo systemctl daemon-reload
sudo systemctl enable --now pokeai-bridge
sleep 10
curl -s http://127.0.0.1:8765/health && echo
echo "완료. 상태: systemctl status pokeai-bridge / 로그: journalctl -u pokeai-bridge -f"
