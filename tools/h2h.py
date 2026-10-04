"""체크포인트 두 개를 서로 붙여 승률 비교 (Showdown 서버 :8000 필요). 사용: h2h.py <ckptA> <ckptB> <판 수> <태그(영숫자 6자 이하)> [팀 풀 JSON]
양쪽 팀은 같은 풀에서 독립적으로 뽑음 → 기대 승률 50%. 마지막 줄 "RESULT A승 B승 무"를 여러 프로세스에서 합산해 쓸 것"""
import asyncio
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from src.core.model import model_from_ckpt
from src.evaluation.analyze_model import LoggingPlayer
from src.training.train_ppo_rct import RandomPoolTeambuilder


def make(ckpt, name, pool):
    model = model_from_ckpt(ckpt)
    model.eval()
    p = LoggingPlayer(mode="argmax", model=model, search_lambda=None, account_configuration=AccountConfiguration(name, None),
                      battle_format="gen9ou", server_configuration=LocalhostServerConfiguration, max_concurrent_battles=8)
    p._team = RandomPoolTeambuilder(pool)
    return p


async def main(a, b, n, tag, pool_path):
    pool = [t["export"] for t in json.load(open(pool_path, encoding="utf-8-sig"))["teams"]]
    pa, pb = make(a, f"h{tag}a", pool), make(b, f"h{tag}b", pool)
    await pa.battle_against(pb, n_battles=n)
    wa = pa.n_won_battles
    wb = pb.n_won_battles
    print(f"RESULT {wa} {wb} {pa.n_finished_battles - wa - wb}", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4],
                     sys.argv[5] if len(sys.argv) > 5 else os.path.join(ROOT, "data", "team_pool_metamon_holdout.json")))
