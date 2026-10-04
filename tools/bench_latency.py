"""결정 한 번당 지연시간 비교: 우리 모델(인코딩 + 신경망 + 행동 선택) vs RCT 봇(파이썬 이식본). Showdown 서버(:8000) 필요.
사용: bench_latency.py <ckpt> [판 수=40] [torch 스레드 수=1]
주의: RCT 쪽은 mod의 자바 AI가 아니라 저장소의 파이썬 이식본이라 절대값은 참고용 (자바는 더 빠를 것)."""
import asyncio
import json
import os
import statistics
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from src.core.model import model_from_ckpt
from src.core.rct_player import RCTBattleAIPlayer
from src.evaluation.analyze_model import LoggingPlayer
from src.training.train_ppo_rct import RandomPoolTeambuilder


def timed(player, store):
    orig = player.choose_move

    def wrapped(battle):
        t = time.perf_counter()
        r = orig(battle)
        store.append((time.perf_counter() - t) * 1000)
        return r

    player.choose_move = wrapped


def summary(name, xs):
    xs = sorted(xs)
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    print(f"{name}: 결정 {len(xs)}회 | 평균 {statistics.mean(xs):.2f}ms | 중앙값 {p(0.5):.2f}ms | p95 {p(0.95):.2f}ms | 최대 {xs[-1]:.1f}ms", flush=True)


async def main(ckpt, n, threads):
    torch.set_num_threads(threads)
    pool = [t["export"] for t in json.load(open(os.path.join(ROOT, "data", "team_pool_metamon_holdout.json"), encoding="utf-8-sig"))["teams"]]
    model = model_from_ckpt(ckpt)
    model.eval()
    common = dict(battle_format="gen9ou", server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
    me = LoggingPlayer(mode="argmax", model=model, search_lambda=None, account_configuration=AccountConfiguration("latme", None), **common)
    opp = RCTBattleAIPlayer(account_configuration=AccountConfiguration("latrct", None), **common)
    me._team, opp._team = RandomPoolTeambuilder(pool), RandomPoolTeambuilder(pool)
    mine, theirs = [], []
    timed(me, mine)
    timed(opp, theirs)
    await me.battle_against(opp, n_battles=n)
    print(f"스레드 {threads} | 판 수 {n} | 우리 {me.n_won_battles}승")
    summary("우리 모델 (인코딩+추론+선택)", mine)
    summary("RCT 봇 (파이썬 이식본)", theirs)
    print(f"배율: 우리 모델 평균 / RCT 평균 = {statistics.mean(mine) / statistics.mean(theirs):.1f}배")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40, int(sys.argv[3]) if len(sys.argv) > 3 else 1))
