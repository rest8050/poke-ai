"""
쇼다운 로컬 서버에서 사용자가 직접 AI를 상대로 배틀해 볼 수 있도록 봇을 띄우는 서빙 스크립트.

사용 예:
  python src/evaluation/serve_bot.py --bot_name MyRLBot --team "HO Physical"
"""
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir != os.path.dirname(_current_dir):
    if os.path.exists(os.path.join(_current_dir, "data", "vocab.json")):
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        break
    _current_dir = os.path.dirname(_current_dir)

import argparse
import asyncio
import json
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from src.core.model import DeepPokemonBattleTransformerNet, load_compatible
from src.core.player import SmartPokemonPlayer
from src.training.train_ppo_rct import load_team_pool, RandomPoolTeambuilder

async def main(args):
    print(f"로컬 서버에 '{args.bot_name}' 계정으로 로그인합니다...")
    
    # 1. 모델 로드 (가장 최신 체크포인트)
    print(f"모델 체크포인트 불러오는 중: {args.ckpt}")
    model = DeepPokemonBattleTransformerNet()
    expanded, skipped = load_compatible(model, torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    print(f"✅ 모델 로드 완료 (확장: {expanded}, 버림: {skipped})")

    # 2. 사용할 팀 로드
    pool_data = json.load(open("data/team_pool_train_a.json", encoding="utf-8-sig"))["teams"]
    
    if args.team == "random":
        print("팀 설정: 매 판 랜덤한 팀을 사용합니다.")
        team = RandomPoolTeambuilder(load_team_pool())
    else:
        team_export = next((t["export"] for t in pool_data if t["name"] == args.team), None)
        if team_export is None:
            print(f"⚠️ '{args.team}' 팀을 찾을 수 없습니다. 기본 팀(풀의 첫 번째)을 사용합니다.")
            team_export = pool_data[0]["export"]
            args.team = pool_data[0]["name"]
        else:
            print(f"팀 설정: '{args.team}' 고정")
        team = team_export

    # 3. 플레이어 (봇) 인스턴스 생성
    player = SmartPokemonPlayer(
        model=model,
        search_lambda=args.search,
        battle_format="gen9ou",
        team=team,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(args.bot_name, None),
    )

    print("\n" + "="*50)
    print(f"🤖 봇 '{args.bot_name}'이(가) 준비되었습니다!")
    print(f"   서버: localhost:8000 (LocalhostServerConfiguration)")
    print(f"   이제 브라우저에서 서버에 접속한 후 '{args.bot_name}'에게 배틀을 신청하세요!")
    print("="*50 + "\n")

    # 4. 도전 수락 대기 (무한 루프)
    # poke-env는 n_challenges 횟수만큼 도전을 받습니다. 1만 판으로 설정하여 사실상 무한 대기.
    await player.accept_challenges(None, 10000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot_name", default="AntiGravityBot", help="쇼다운에서 봇이 사용할 닉네임")
    ap.add_argument("--ckpt", default="checkpoints/supervised_v2.pt", help="사용할 모델 체크포인트 경로")
    ap.add_argument("--team", default="random", help="봇이 사용할 팀 이름 (ex: 'HO Physical', 'Stall'), 'random' 지정 시 매 판 랜덤")
    ap.add_argument("--search", type=float, default=None, help="1턴 탐색 람다 값 (기본값: 탐색 안함)")
    
    asyncio.run(main(ap.parse_args()))
