"""Metamon 사람 대전 리플레이(parquet) → 레이팅 상위 판만 골라 양쪽 시점 시퀀스 npz 조각으로 저장.
    PYTHONPATH=. python tools/parse_human.py --shard 0 --min-rating 1600
조각 = <--out-dir, 기본 data/human>/gen9ou_s<샤드>_r<최소레이팅>_<번호>.npz (이미 있으면 건너뜀 = 이어 돌리기).
action = 사람 행동 (0-3 기술, 4-7 테라 기술, 8-13 교체), 선택이 로그에 안 보인 결정 지점은 -1 (상태만 있음).
opp_action = 그 턴 상대의 선택 (0 기술, 1 교체, 2 테라+기술, 모르면/기절 전이면 -1; 기절 후 교체 지점은 항상 -1), turn = 배틀 턴 번호,
opp_move/own_move = 그 턴 상대/내가 쓴 기술의 어휘 번호 (data/vocab.json의 move, 기술이 아니거나 모르면 0),
game = 판 번호 (같은 판의 p1/p2 시점이 같은 값), side = 0(p1)/1(p2).
rating = 판의 레이팅 열 (두 플레이어 중 낮은 쪽에 가까움; 로그 안 개별 레이팅은 대부분 비어 있어 안 씀), win = 그 시점 플레이어의 승리."""
import argparse
import os
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from src.training.replay_dataset import OBS_KEYS, replay_to_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--min-rating", type=int, default=1600)
ap.add_argument("--games-per-file", type=int, default=1000)
ap.add_argument("--out-dir", default="data/human")
args = ap.parse_args()

src = f"data/metamon_raw/train-{args.shard:05d}-of-00046.parquet"
t = pq.read_table(src, columns=["rating", "log"]).to_pydict()
rows = [(int(r), lg) for r, lg in zip(t["rating"], t["log"]) if str(r).isdigit() and int(r) >= args.min_rating]
os.makedirs(args.out_dir, exist_ok=True)
print(f"{src}: 레이팅 {args.min_rating} 이상 {len(rows)}판 → 조각당 {args.games_per_file}판", flush=True)

drops, failed, t0 = Counter(), 0, time.time()
for fi, start in enumerate(range(0, len(rows), args.games_per_file)):
    path = f"{args.out_dir}/gen9ou_s{args.shard}_r{args.min_rating}_{fi:03d}.npz"
    if os.path.exists(path):
        continue
    seqs, rating, win, game, side_id = [], [], [], [], []
    for gi, (r, lg) in enumerate(rows[start:start + args.games_per_file], start):
        for si, side in enumerate(("p1", "p2")):
            out = replay_to_sequence(lg, side)
            if out is None:
                failed += 1
                continue
            drops.update(out.pop("drop"))
            seqs.append(out); rating.append(r); win.append(out["outcome"] > 0); game.append(gi); side_id.append(si)
    data = {k: np.concatenate([s[k] for s in seqs]) for k in OBS_KEYS + ["action_mask", "action", "value_target", "turn", "opp_action", "opp_move", "own_move"]}
    data.update(lens=np.array([len(s["action"]) for s in seqs]), outcome=np.array([s["outcome"] for s in seqs], np.float32),
                rating=np.array(rating), win=np.array(win), game=np.array(game), side=np.array(side_id))
    np.savez(path + ".tmp.npz", **data)
    os.replace(path + ".tmp.npz", path)
    print(f"[{time.time() - t0:.0f}s] {path} 시퀀스 {len(seqs)}, 결정 {len(data['action'])}, 라벨 없음 {(data['action'] < 0).mean():.1%}", flush=True)
print(f"끝: 실패 {failed} | 라벨 없음 이유 {dict(drops)}")
