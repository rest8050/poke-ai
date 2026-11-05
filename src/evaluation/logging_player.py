"""평가/기록용 플레이어: 정책을 argmax 또는 확률 샘플링으로 행동으로 바꾸고, 턴마다 합법 행동별 확률과 가치를 turn_logs에 남김
(tools/arena.py, record_games.py, record_vs_fp.py, team_screen.py가 씀)"""
from collections import defaultdict

import torch

from src.core.actions import decode_action
from src.core.player import SmartPokemonPlayer


def action_name(d):
    if d is None:
        return None
    if d[0] == "switch":
        return f"switch:{d[1].species}"
    _, move, z, tera = d
    return f"{move.id}{'+tera' if tera else ''}{'+z' if z else ''}"


class LoggingPlayer(SmartPokemonPlayer):
    def __init__(self, mode="sample", **kwargs):
        """mode: argmax(가장 높은 확률) | sample(학습과 같은 확률 샘플링)"""
        super().__init__(**kwargs)
        self.mode = mode
        self.turn_logs = defaultdict(list)

    def choose_move(self, battle):
        tag = battle.battle_tag
        tensors = self.encoder.encode_battle(battle)
        mask = tensors[7][0]
        idx, probs, value, self.history[tag] = self.model.get_action(*tensors[:7], self.history.get(tag), tensors[7], tensors[8], prev_move=self.encoder.prev_move(battle))
        probs = probs[0]
        if self.mode == "sample":
            idx = int(torch.multinomial(probs, 1).item())
        legal = [{"idx": i, "name": action_name(decode_action(battle, i)), "p": round(probs[i].item(), 4)}
                 for i in torch.nonzero(mask).flatten().tolist()]
        self.turn_logs[tag].append({
            "turn": battle.turn, "forced_switch": not bool(mask[:8].any()),
            "legal": sorted(legal, key=lambda e: -e["p"]),
            "chosen": {"idx": idx, "name": action_name(decode_action(battle, idx)), "p": next((e["p"] for e in legal if e["idx"] == idx), None)},
            "value": round(value, 4),
        })
        return self.order_for(battle, idx) or self.choose_random_move(battle)
