import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir != os.path.dirname(_current_dir):
    if os.path.exists(os.path.join(_current_dir, "data", "vocab.json")):
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        break
    _current_dir = os.path.dirname(_current_dir)
import asyncio
import torch
import warnings
from poke_env.player import Player
from poke_env import LocalhostServerConfiguration
from src.training.train_ppo import PPOCollectingPlayer
from src.core.model import DeepPokemonBattleTransformerNet

from poke_env import AccountConfiguration

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def main():
    print("🚀 [유저 대전 모드] 86% 승률 최신 가중치 로딩 중...")
    
    # 모델 로드
    model = DeepPokemonBattleTransformerNet()
    model.load_state_dict(torch.load("checkpoints/deep_checkpoint_ppo.pt", map_location=device))
    model.to(device)
    model.eval() # 빡겜 모드(평가 모드) 고정

    # 대전용 AI 플레이어 생성
    ai_player = PPOCollectingPlayer(
        model=model,
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("PPO_Boss_Bot", "password123"),
    )
    ai_player.is_eval = True # 탐색 완전히 끄기

    print("\n==================================================")
    print(f"✅ 봇 서버 가동 완료!")
    print(f"1. 웹브라우저에서 http://localhost:8000 접속")
    print(f"2. 아무 이름으로 Choose name (로그인)")
    print(f"3. 우측 하단 [Find a user] 클릭")
    print(f"4. 'PPO_Boss_Bot' 입력 후 [Open]")
    print(f"5. [Challenge] 클릭 후 포맷을 [Gen 9] -> [Random Battle]로 설정 후 대결 신청!")
    print("==================================================\n")
    
    # 챌린지 무한 수락 루프
    while True:
        try:
            print("⌛ 챌린지 대기 중...")
            await ai_player.accept_challenges(opponent=None, n_challenges=1)
            print("⚔️ 배틀 종료! 메모리를 초기화하고 다음 도전을 기다립니다.")
            ai_player.clear_memory() # RNN 잔여 기억 제거
        except Exception as e:
            print(f"에러 발생: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
