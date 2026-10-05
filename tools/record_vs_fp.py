"""학생 모델이 Foul Play 컨테이너의 도전을 받아 대전하며, 학생 시점 원본 프레임 + 턴별 선택을 저장 (패배 원인 분석용).
저장된 프레임은 foul-play/fp/replay_label.py로 재생해 학생이 방문한 상태의 교사 분포를 얻음.
사용: record_vs_fp.py <fp_user> <판 수> <ckpt> <출력 jsonl> [팀 풀 JSON]   (Showdown :8000 필요, Foul Play 쪽 팀 폴더는 같은 풀이어야 함)"""
import asyncio
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from src.core.model import model_from_ckpt
from src.core.team_pool import RandomPoolTeambuilder
from tools.record_games import RecordingPlayer, team_signature


class VsFpPlayer(RecordingPlayer):
    def dump(self, path):
        n = skipped = 0
        with open(path, "w", encoding="utf-8") as f:
            for tag, b in self.battles.items():
                if not b.finished or tag not in self.frames:
                    continue
                sig = team_signature([(m.species, list(m.moves)) for m in b.team.values()])
                export = self.by_sig.get(sig)
                if export is None:
                    skipped += 1
                    continue
                turns = [{"turn": t["turn"], "forced": t["forced_switch"], "chosen": t["chosen"]["name"], "p": t["chosen"]["p"],
                          "value": t["value"], "top": [[e["name"], e["p"]] for e in t["legal"][:3]]} for t in self.turn_logs.get(tag, [])]
                f.write(json.dumps({"tag": tag, "user": self.username, "export": export, "frames": self.frames[tag],
                                    "values": [t["value"] for t in turns], "turns": turns, "won": bool(b.won)}) + "\n")
                n += 1
        print(f"{path}: {n}판 저장, {skipped}판 건너뜀(팀 매칭 실패)", flush=True)


async def main(fp_user, n, ckpt, out, pool_path):
    exports = [t["export"] for t in json.load(open(pool_path, encoding="utf-8-sig"))["teams"]]
    model = model_from_ckpt(ckpt)
    model.eval()
    me = VsFpPlayer(exports=exports, mode="argmax", model=model, search_lambda=None,
                    account_configuration=AccountConfiguration(f"bc{fp_user[2:]}", None), battle_format="gen9ou",
                    server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
    me._team = RandomPoolTeambuilder(exports)
    try:
        await asyncio.wait_for(me.accept_challenges(fp_user, n), timeout=float(n) * 120)
    except asyncio.TimeoutError:
        print("시간 초과", flush=True)
    me.dump(out)
    fin = sum(b.finished for b in me.battles.values())
    print(f"RESULT {fp_user}: 우리 {me.n_won_battles}승 / {fin}판", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4],
                     sys.argv[5] if len(sys.argv) > 5 else os.path.join(ROOT, "data", "team_pool_metamon_holdout.json")))
