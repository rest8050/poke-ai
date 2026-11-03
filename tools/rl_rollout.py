"""RL 롤아웃 작업자: 현재 정책(확률 샘플링)이 자가대전 또는 과거 체크포인트와 두며, 정책 쪽 시점의 결정 전부(인코더 입력/마스크/행동)와 승패를 npz로 저장.
tools/arena.py 작업자와 같은 구조 (Showdown :8000 필요). 학습 쪽은 src/training/train_rl.py가 이걸 띄움.
사용: rl_rollout.py <현재 ckpt> <상대 ckpt|self> <판 수> <팀 풀 JSON> <출력 npz> <태그(영숫자 4자)> [--port 8000 --concurrent 10 --chunk 60 --max-battle-sec 900]
출력 npz: OBS_KEYS 7개 + action_mask/legal_mask + action_taken(-1 = 실행 못 해 무작위로 대체된 결정) + phi(결정 시점의 국면 포텐셜) + outcome(시퀀스별 +1/-1) + lens
상대가 self면 양쪽 시점 모두 저장 (한 판 = 시퀀스 2개), 과거 ckpt면 현재 정책 쪽만. 무승부와 시간 초과 판은 버림. 판을 끝낼 때마다 파일을 덮어써서 중간에 죽어도 지금까지의 판은 남음"""
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch   # poke_env보다 먼저 (Windows DLL 로딩 순서)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.concurrency import POKE_LOOP

from src.core.battle_eval import potential
from src.core.model import model_from_ckpt
from src.core.player import SmartPokemonPlayer
from src.core.team_pool import RandomPoolTeambuilder
from src.training.replay_dataset import OBS_KEYS
from tools.arena import watchdog


class RLPlayer(SmartPokemonPlayer):
    def __init__(self, record, voided, **kwargs):
        super().__init__(**kwargs)
        self.record, self.voided = record, voided
        self.steps = defaultdict(list)    # 배틀 태그 -> [(obs 7개, action_mask, legal_mask, 행동, 국면 포텐셜 Φ)]
        self.episodes = []                # (steps, +1/-1)

    def choose_move(self, battle):
        tag = battle.battle_tag
        t = self.encoder.encode_battle(battle)
        _, probs, _, self.history[tag] = self.model.get_action(*t[:7], self.history.get(tag), t[7], t[8])
        idx = int(torch.multinomial(probs[0], 1))
        order = self.order_for(battle, idx)
        if self.record:
            self.steps[tag].append(([x[0].cpu().numpy() for x in t[:7]], t[7][0].cpu().numpy(), t[8][0].cpu().numpy(), idx if order else -1, potential(battle, drop=("tera",))))
        return order or self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        tag = battle.battle_tag
        steps = self.steps.pop(tag, None)
        self.history.pop(tag, None)
        if steps and battle.won is not None and tag not in self.voided:
            self.episodes.append((steps, 1.0 if battle.won else -1.0))


def dump(episodes, path):
    steps = [s for ep, _ in episodes for s in ep]
    out = {}
    for i, k in enumerate(OBS_KEYS):
        a = np.stack([s[0][i] for s in steps])
        out[k] = a.astype(np.int16) if a.dtype == np.int64 else a.astype(np.float16)   # 어휘 번호 int16, 실수 반정밀도 (train_fp_distill.load와 같은 보관 방식)
    out["action_mask"] = np.stack([s[1] for s in steps])
    out["legal_mask"] = np.stack([s[2] for s in steps])
    out["action_taken"] = np.array([s[3] for s in steps], dtype=np.int64)
    out["phi"] = np.array([s[4] for s in steps], dtype=np.float32)
    out["outcome"] = np.array([o for _, o in episodes], dtype=np.float32)
    out["lens"] = np.array([len(ep) for ep, _ in episodes], dtype=np.int64)
    tmp = os.path.join(os.path.dirname(path), "tmp_" + os.path.basename(path))   # shard_*.npz glob에 안 걸리게 이름을 다르게
    np.savez(tmp, **out)
    os.replace(tmp, path)


async def main(a):
    exports = [t["export"] for t in json.load(open(a.pool, encoding="utf-8-sig"))["teams"]]
    cur = model_from_ckpt(a.cur).eval()
    selfplay = a.opp == "self"
    opp = cur if selfplay else model_from_ckpt(a.opp).eval()
    server = ServerConfiguration(f"ws://localhost:{a.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
    voided, episodes, done, chunk = set(), [], 0, 0
    while done < a.games:     # 일정 판 수마다 플레이어를 새로 만들어 메모리를 제한 (arena.py와 같음)
        chunk += 1
        n = min(a.chunk, a.games - done)

        def make(model, record, suffix):
            p = RLPlayer(record, voided, model=model, battle_format="gen9ou", server_configuration=server,
                         account_configuration=AccountConfiguration(f"r{a.tag}{chunk % 100}{suffix}", None), max_concurrent_battles=a.concurrent)
            p._team = RandomPoolTeambuilder(exports)
            return p
        pa, pb = make(cur, True, "a"), make(opp, selfplay, "b")
        watch = asyncio.run_coroutine_threadsafe(watchdog(pa, a.max_battle_sec, voided), POKE_LOOP)
        await pa.battle_against(pb, n_battles=n)
        watch.cancel()
        episodes += pa.episodes + pb.episodes
        done += n
        if episodes:
            dump(episodes, a.out)
        print(f"{done}/{a.games}판, 저장된 시퀀스 {len(episodes)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cur"); ap.add_argument("opp"); ap.add_argument("games", type=int); ap.add_argument("pool"); ap.add_argument("out"); ap.add_argument("tag")
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--concurrent", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=60); ap.add_argument("--max-battle-sec", type=int, default=900)
    asyncio.run(main(ap.parse_args()))
