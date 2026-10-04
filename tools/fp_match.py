"""우리 봇(BC 모델 또는 RCT)이 컨테이너의 Foul Play에게 도전해 승률을 측정. 사용: fp_match.py <bc|rct> <fp_username> <n> [ckpt] [search_lambda]"""
import asyncio
import json
import sys
import time

import os; ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from src.core.model import model_from_ckpt
from src.core.rct_player import RCTBattleAIPlayer
from src.evaluation.analyze_model import LoggingPlayer
from src.training.train_ppo_rct import RandomPoolTeambuilder
import torch


async def main(kind, fp_user, n, ckpt, search_lambda=None):
    pool = [t["export"] for t in json.load(open(os.environ.get("POOL") or os.path.join(ROOT, "data", "team_pool_metamon_holdout.json"), encoding="utf-8-sig"))["teams"]]
    common = dict(battle_format="gen9ou", server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
    acct = AccountConfiguration(f"{kind}{fp_user[2:]}", None)  # 고정 이름: Foul Play가 이 이름을 도전
    if kind == "bc":
        model = model_from_ckpt(ckpt)
        model.eval()
        me = LoggingPlayer(mode="argmax", model=model, search_lambda=search_lambda, account_configuration=acct, **common)
    else:
        me = RCTBattleAIPlayer(account_configuration=acct, **common)
    me._team = RandomPoolTeambuilder(pool)
    t0 = time.time()
    try:
        await asyncio.wait_for(me.accept_challenges(fp_user, n), timeout=float(n) * 120)
    except asyncio.TimeoutError:
        print("시간 초과", flush=True)
    fin = sum(b.finished for b in me.battles.values())
    with open(os.path.join(ROOT, "logs", "fp_eval", f"detail_{fp_user}.jsonl"), "w", encoding="utf-8") as f:  # 판별 결과 요약: 승패, 턴 수, 남은 포켓몬 수(끝난 시점)
        for b in me.battles.values():
            if b.finished:
                mine, opp = list(b.team.values()), list(b.opponent_team.values())
                f.write(json.dumps({"won": bool(b.won), "turns": b.turn, "my_alive": sum(not m.fainted for m in mine),
                                    "opp_alive": 6 - sum(m.fainted for m in opp),
                                    "my_hp": round(sum(m.current_hp_fraction for m in mine if not m.fainted), 2),
                                    "opp_hp": round(sum(m.current_hp_fraction for m in opp if not m.fainted), 2)}) + chr(10))
    print(f"RESULT {kind} vs {fp_user}: 우리 {me.n_won_battles}승 / {fin}판 ({time.time() - t0:.0f}초)", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                      sys.argv[4] if len(sys.argv) > 4 else "checkpoints/supervised_v2.pt",
                      float(sys.argv[5]) if len(sys.argv) > 5 else None))
