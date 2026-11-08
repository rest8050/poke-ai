"""대전 평가용 팀 풀 로더 (poke_env Teambuilder)"""
import random

from poke_env.teambuilder import Teambuilder


class RandomPoolTeambuilder(Teambuilder):
    def __init__(self, exports):
        self.packed = [self.join_team(self.parse_showdown_team(t)) for t in exports]

    def yield_team(self):
        return random.choice(self.packed)
