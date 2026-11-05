"""학생 + Foul Play 탐색(foul-play/fp/online_service.py 컨테이너)의 융합 플레이어가 Foul Play 컨테이너의 도전을 받아 대전.
모드:  student  탐색 안 씀 (기준선, 서비스 필요 없음)
       fp       서비스 탐색에서 Foul Play가 고른 수를 그대로 둠 = 탐색 시간만 다른 Foul Play (융합의 통제군)
       fuse     학생 확률^beta x 탐색 방문 분포 의 argmax (beta=0이면 탐색 분포의 argmax)
승패가 곧 이 모드의 강도: Foul Play 상대 승률이 fp 모드보다 오르면 학생이 탐색에 보탠 것, student보다 오르면 탐색이 학생에 보탠 것.
사용: hybrid_vs_fp.py <fp_user> <판 수> <ckpt> <서비스 포트> <모드> <beta> [팀 풀 JSON]   (Showdown :8000 필요, 서비스는 fp.online_service)"""
import asyncio
import json
import os
import socket
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from src.core.actions import decode_action
from src.core.model import model_from_ckpt
from src.core.team_pool import RandomPoolTeambuilder
from src.training.build_fp_dataset import _choice_index, _norm
from tools.record_games import RecordingPlayer, team_signature

DECIDE_TIMEOUT_S = 30


class HybridPlayer(RecordingPlayer):
    def __init__(self, exports, mode, beta, port, **kwargs):
        super().__init__(exports=exports, mode="argmax", **kwargs)
        self.fuse_mode, self.beta = mode, beta
        self.sock = socket.create_connection(("127.0.0.1", port)) if mode != "student" else None
        self.rf = self.sock.makefile("rb") if self.sock else None
        if self.sock:
            self.sock.settimeout(DECIDE_TIMEOUT_S)
        self.pending = defaultdict(list)        # 서비스에 아직 못 보낸 프레임 (팀을 알기 전)
        self.live = set()                       # 서비스에 init한 배틀
        self.stat = defaultdict(int)

    def _send(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0][1:]
        frame = "\n".join("|".join(m) for m in split_messages)
        if self.sock:
            if tag in self.live:
                self._send({"op": "feed", "frames": [frame]})
            else:
                self.pending[tag].append(frame)
        await super()._handle_battle_message(split_messages)

    def _ensure_service(self, battle, final=True):
        """팀을 알게 되면(팀 프리뷰 때 기술이 아직 안 채워졌으면 첫 턴에) 서비스를 시작하고 쌓인 프레임을 보냄"""
        tag = battle.battle_tag
        if not self.sock or tag in self.live:
            return
        export = self.by_sig.get(team_signature([(m.species, list(m.moves)) for m in battle.team.values()]))
        if export is None:
            if final:           # 풀에 없는 팀(폼 변화 등) → 이 판은 학생만
                self.stat["팀 매칭 실패 판"] += 1
                self.pending.pop(tag, None)
            return
        self._send({"op": "init", "tag": tag, "user": self.username, "export": export})
        self._send({"op": "feed", "frames": self.pending.pop(tag, [])})
        self.live.add(tag)

    def teampreview(self, battle):
        self._ensure_service(battle, final=False)
        return super().teampreview(battle)

    def _decide(self, battle, forced):
        try:
            self._send({"op": "decide", "turn": battle.turn, "forced": forced})
            rec = json.loads(self.rf.readline())
            return rec if rec.get("policy") else None
        except (OSError, ValueError):
            self.stat["서비스 응답 실패"] += 1
            return None

    def choose_move(self, battle):
        tag = battle.battle_tag
        self._ensure_service(battle)
        tensors = self.encoder.encode_battle(battle)
        mask = tensors[7][0].numpy().astype(bool)
        idx, probs, value, self.history[tag] = self.model.get_action(*tensors[:7], self.history.get(tag), tensors[7], tensors[8], prev_move=self.encoder.prev_move(battle))
        p = probs[0].double().numpy()
        forced = not bool(mask[:8].any())
        rec = self._decide(battle, forced) if tag in self.live else None
        self.stat["결정"] += 1
        student_idx = idx
        if rec is not None:
            self.stat["탐색 결과 받음"] += 1
            active = battle.active_pokemon
            moves = [_norm(m.id) for m in active.moves.values()] if active else []
            team = [_norm(m.species) for m in battle.team.values()]
            if self.fuse_mode == "fp":
                ci = _choice_index(rec["choice"], moves, team)
                idx = ci if ci is not None and mask[ci] else idx
            else:
                f = np.zeros(22)
                for name, w in rec["policy"]:
                    ci = _choice_index(name, moves, team)
                    if ci is not None and mask[ci]:
                        f[ci] += w
                if f.sum() > 0:
                    score = np.log(np.maximum(f / f.sum(), 1e-4)) + self.beta * np.log(np.maximum(p, 1e-4))
                    idx = int(np.where(mask, score, -np.inf).argmax())
        if idx != student_idx:
            self.stat["학생 argmax와 다른 선택"] += 1
        if tag in self.live:
            d = decode_action(battle, idx)
            chosen = {"kind": None, "name": None} if d is None else (
                {"kind": "move", "name": _norm(d[1].id)} if d[0] == "move" else {"kind": "switch", "name": _norm(d[1].species)})
            self._send({"op": "chosen", "turn": battle.turn, "forced": forced, **chosen})
        return self.order_for(battle, idx) or self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        print(f"GAME {'승' if battle.won else '패'} 누계 {self.n_won_battles}/{self.n_finished_battles}", flush=True)


async def main(fp_user, n, ckpt, port, mode, beta, pool_path):
    exports = [t["export"] for t in json.load(open(pool_path, encoding="utf-8-sig"))["teams"]]
    model = model_from_ckpt(ckpt)
    model.eval()
    me = HybridPlayer(exports=exports, mode=mode, beta=beta, port=port, model=model,
                      account_configuration=AccountConfiguration(f"bc{fp_user[2:]}", None), battle_format="gen9ou",
                      server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
    me._team = RandomPoolTeambuilder(exports)
    try:
        await asyncio.wait_for(me.accept_challenges(fp_user, n), timeout=float(n) * 180)
    except asyncio.TimeoutError:
        print("시간 초과", flush=True)
    print(f"RESULT {fp_user} {mode} beta={beta}: 우리 {me.n_won_battles}승 / {me.n_finished_battles}판 | {dict(me.stat)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], float(sys.argv[6]),
                     sys.argv[7] if len(sys.argv) > 7 else os.path.join(ROOT, "data", "team_pool_metamon_holdout.json")))
