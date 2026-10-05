"""DAgger용: 학생 모델끼리 자가대전(또는 서로 다른 두 체크포인트)하며 각자 시점의 Showdown 원본 메시지를 저장.
저장된 메시지는 foul-play/fp/replay_label.py가 Foul Play에 그대로 재생해서 그 상태들의 탐색 분포(교사 정답)를 뽑음.
사용: record_games.py <ckptA> <ckptB> <판 수> <태그(영숫자 6자 이하)> <팀 풀 JSON> <출력 접두사> [샘플링 on/off=1]
출력: <접두사>_a.jsonl / <접두사>_b.jsonl (한 줄 = 한 판, 각자 시점) — 같은 배틀 태그를 양쪽이 쓰므로 파일을 분리해야 build_fp_dataset가 안 섞임
각 줄의 "values"는 그 판 턴별 학생 자신의 value(승산 추정) 목록 — filter_dagger_hard.py로 불리했던 판만 추려낼 때 씀
Showdown 서버(:8000) 필요"""
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.teambuilder import Teambuilder

from src.core.model import model_from_ckpt
from src.evaluation.analyze_model import LoggingPlayer
from src.core.team_pool import RandomPoolTeambuilder


def _id(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def team_signature(mons):
    """[(종, 기술 집합)] 6마리 → 순서 무관 서명. 요청 JSON(우리 팀)과 풀의 export 텍스트를 같은 서명으로 맞춤"""
    return frozenset((_id(sp), frozenset(_id(m) for m in mv)) for sp, mv in mons)


class RecordingPlayer(LoggingPlayer):
    def __init__(self, exports, **kwargs):
        super().__init__(**kwargs)
        self.frames = defaultdict(list)
        self.by_sig = {}
        for e in exports:
            mons = [(p.species or p.nickname, p.moves) for p in Teambuilder.parse_showdown_team(e)]
            self.by_sig.setdefault(team_signature(mons), e)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0][1:]
        self.frames[tag].append("\n".join("|".join(m) for m in split_messages))  # 원본 프레임 그대로 복원 (split의 역연산)
        await super()._handle_battle_message(split_messages)

    def dump(self, path):
        n = skipped = 0
        with open(path, "w", encoding="utf-8") as f:
            for tag, b in self.battles.items():
                if not b.finished or tag not in self.frames:
                    continue
                sig = team_signature([(m.species, list(m.moves)) for m in b.team.values()])
                export = self.by_sig.get(sig)
                if export is None:  # 폼이 바뀌었거나 풀에 없는 팀 → 이 판은 건너뜀 (재생에서 팀 정보를 못 채움)
                    skipped += 1
                    continue
                values = [t["value"] for t in self.turn_logs.get(tag, [])]  # 턴별 학생 자신의 value (불리한 구간 필터링용)
                f.write(json.dumps({"tag": tag, "user": self.username, "export": export, "frames": self.frames[tag], "values": values}) + "\n")
                n += 1
        print(f"{path}: {n}판 저장, {skipped}판 건너뜀(팀 매칭 실패)", flush=True)


def make(ckpt, name, exports, mode):
    model = model_from_ckpt(ckpt)
    model.eval()
    p = RecordingPlayer(exports=exports, mode=mode, model=model, search_lambda=None,
                        account_configuration=AccountConfiguration(name, None), battle_format="gen9ou",
                        server_configuration=LocalhostServerConfiguration, max_concurrent_battles=8)
    p._team = RandomPoolTeambuilder(exports)
    return p


async def main(a, b, n, tag, pool_path, out, sample):
    exports = [t["export"] for t in json.load(open(pool_path, encoding="utf-8-sig"))["teams"]]
    mode = "sample" if sample else "argmax"
    pa, pb = make(a, f"r{tag}a", exports, mode), make(b, f"r{tag}b", exports, mode)
    await pa.battle_against(pb, n_battles=n)
    pa.dump(f"{out}_a.jsonl")
    pb.dump(f"{out}_b.jsonl")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6],
                     (sys.argv[7] if len(sys.argv) > 7 else "1") == "1"))
