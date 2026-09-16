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
from poke_env import LocalhostServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration
from src.core.player import SmartPokemonPlayer
from src.core.model import DeepPokemonBattleTransformerNet

async def main():
    print("🤖 AI 모델 불러오는 중...")
    model = DeepPokemonBattleTransformerNet()
    
    try:
        model.load_state_dict(torch.load("checkpoints/deep_checkpoint_ppo.pt", map_location="cpu", weights_only=True))
        print("✅ 학습된 PPO 모델 가중치 로드 완료!")
    except FileNotFoundError:
        print("⚠️ 'checkpoints/deep_checkpoint_ppo.pt'를 찾을 수 없어 갓 태어난 AI로 진행합니다.")

    model.eval()

    # AI 봇 계정 설정
    bot_account = AccountConfiguration("MyAlphaBot", "")

    print("🔌 로컬 쇼다운 서버에 봇 접속 중...")
    ai_player = SmartPokemonPlayer(
        model=model,
        account_configuration=bot_account,
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )

    print("================================================================")
    print("🔥 봇 구동 완료! 이제 유저님과 배틀할 준비가 되었습니다.")
    print("1. 웹 브라우저를 열고 http://localhost:8000 에 접속하세요.")
    print("2. 우측 상단 'Choose name'을 눌러 아무 닉네임이나 설정하세요.")
    print("3. 좌측 하단 'Find a user' 버튼을 클릭 후 'MyAlphaBot' 을 검색하세요.")
    print("4. Challenge 버튼을 누르고 포맷을 [Gen 9 Random Battle]로 선택한 뒤 배틀을 거세요!")
    print("================================================================")
    
    # 유저가 배틀을 신청할 때까지 무한정 대기하며 수락합니다. (1판 끝나면 종료됨)
    await ai_player.accept_challenges(None, 1)

if __name__ == "__main__":
    asyncio.run(main())
