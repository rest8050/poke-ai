"""대전 평가용 팀 풀 로더 (poke_env Teambuilder)"""
import json
import random

from poke_env.teambuilder import Teambuilder

TEAM_POOL_PATH = "data/team_pool_metamon.json"  # Metamon 학습 풀 (평가 전용 _holdout.json은 학습에 쓰지 말 것)


def load_team_pool(path: str = TEAM_POOL_PATH):
    """team_pool.json에서 Showdown export 문자열 목록 로드"""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        teams = [t["export"] for t in data["teams"]]
        print(f"✅ 팀 풀 로드 완료: {len(teams)}개 팀")
        return teams
    except Exception as e:
        print(f"⚠️ 팀 풀 로드 실패 ({e}). gen9randombattle로 폴백")
        return None


class RandomPoolTeambuilder(Teambuilder):
    def __init__(self, exports):
        self.packed = [self.join_team(self.parse_showdown_team(t)) for t in exports]

    def yield_team(self):
        return random.choice(self.packed)
