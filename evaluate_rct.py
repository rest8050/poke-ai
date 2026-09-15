import asyncio
import torch
import warnings
from poke_env.player import RandomPlayer
from rct_player import RCTBattleAIPlayer
from poke_env import LocalhostServerConfiguration
from train_ppo_rct import PPOCollectingPlayer
from model import DeepPokemonBattleTransformerNet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def main():
    print("🚀 [테스트 모드] 86% 승률 가중치 로딩 중...")
    model = DeepPokemonBattleTransformerNet()
    model.load_state_dict(torch.load("deep_checkpoint_ppo.pt", map_location=device))
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

    opp_player = RCTBattleAIPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )

    n_battles = 50
    print(f"🔥 RCT 커뮤니티 봇을 상대로 {n_battles}판 실력 검증 매치를 시작합니다!\n(잠시만 기다려주세요...)")
    
    await ai_player.battle_against(opp_player, n_battles=n_battles)
    
    eval_wins = sum(1 for b in ai_player.battles.values() if b.won)
    print(f"==================================================")
    print(f"📊 최종 테스트 승률: {eval_wins}승 / {n_battles}판 ({eval_wins/n_battles*100:.1f}%)")
    print(f"==================================================")

if __name__ == "__main__":
    asyncio.run(main())
