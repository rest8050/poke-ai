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
from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration
from src.training.train_ppo import PPOCollectingPlayer
from src.core.model import DeepPokemonBattleTransformerNet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def main():
    print("🚀 [테스트 모드] 86% 승률 가중치 로딩 중...")
    model = DeepPokemonBattleTransformerNet()
    model.load_state_dict(torch.load("checkpoints/deep_checkpoint_ppo.pt", map_location=device))
    model.to(device)
    model.eval() # 평가 모드 (드롭아웃/무작위성 제거)

    # 평가용 플레이어 생성
    ai_player = PPOCollectingPlayer(
        model=model,
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )
    ai_player.is_eval = True # 탐색 완전히 끄기

    opp_player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )

    n_battles = 50
    print(f"🔥 Random 봇을 상대로 {n_battles}판 실력 검증 매치를 시작합니다!\n(잠시만 기다려주세요...)")
    
    await ai_player.battle_against(opp_player, n_battles=n_battles)
    
    eval_wins = sum(1 for b in ai_player.battles.values() if b.won)
    print(f"==================================================")
    print(f"📊 최종 테스트 승률: {eval_wins}승 / {n_battles}판 ({eval_wins/n_battles*100:.1f}%)")
    print(f"==================================================")

if __name__ == "__main__":
    asyncio.run(main())
