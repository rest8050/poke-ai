#!/bin/bash
# 롤백: 브리지 서비스와 파일 제거. 게임 서버 쪽은 별도로 mods에서 pokeai-*.jar를 지우고 재시작하면 원래 상태로 돌아감.
sudo systemctl disable --now pokeai-bridge 2>/dev/null || true
sudo rm -f /etc/systemd/system/pokeai-bridge.service
sudo systemctl daemon-reload
rm -r "$HOME/pokeai-bridge"
echo "브리지 제거 완료"
