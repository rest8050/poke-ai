import torch
from model import DeepPokemonBattleTransformerNet, PokemonBattleNet
from tensor_encoder import BattleTensorEncoder, VocabManager

try:
    from poke_env.player import Player
    from poke_env.battle import AbstractBattle
    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False
    Player = object
    AbstractBattle = object


# 호환성을 위한 FeatureExtractor 포워딩
class FeatureExtractor:
    @staticmethod
    def extract_from_battle(battle, device="cpu", vocab_path="vocab.json"):
        encoder = BattleTensorEncoder(vocab_path=vocab_path, device=device)
        return encoder.encode_battle(battle)


if POKE_ENV_AVAILABLE:
    class SmartPokemonPlayer(Player):
        """
        분리된 BattleTensorEncoder 모듈을 사용하는 스마트 poke-env 플레이어
        """
        def __init__(self, model: DeepPokemonBattleTransformerNet = None, vocab_path: str = "vocab.json", **kwargs):
            super().__init__(**kwargs)
            self.model = model or DeepPokemonBattleTransformerNet(vocab_path=vocab_path)
            self.encoder = BattleTensorEncoder(vocab_path=vocab_path)
            # 배틀별 GRU 히스토리 (battle_tag -> Tensor)
            # ponytail: 끝난 배틀도 남음, 장시간 돌리면 battle.finished 기준으로 정리
            self.history = {}

        def choose_move(self, battle: AbstractBattle):
            # 1. 분리된 전용 인코더 모듈로 텐서 및 마스크 변환
            tensors = self.encoder.encode_battle(battle)
            my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask = tensors

            # 2. 신경망 추론
            tag = battle.battle_tag
            action_idx, p_probs, opp_p_probs, val, self.history[tag] = self.model.get_action(
                my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, self.history.get(tag), action_mask
            )

            # 3. 행동 매핑 및 명령 생성
            active_pkmn = battle.active_pokemon
            my_team_list = list(battle.team.values()) if hasattr(battle, "team") else []

            # 3.1 기술 사용 (0~3 일반 기술, 4~7 같은 기술을 Z기술로)
            if action_idx < 8 and active_pkmn:
                move_idx = action_idx % 4
                active_moves = list(active_pkmn.moves.values())
                avail_ids = {m.id for m in battle.available_moves}
                # 몸부림/반동 턴 등 가진 기술이 선택 불가하면 무한 재요청 방지
                if move_idx < len(active_moves) and active_moves[move_idx].id in avail_ids:
                    target_move = active_moves[move_idx]
                    z_ids = {m.id for m in active_pkmn.available_z_moves} if battle.can_z_move else set()
                    z_move = action_idx >= 4 and target_move.id in z_ids
                    # 기믹은 한 턴에 하나만: Z(모델 선택) > 메가(즉시) > 다이맥스(즉시)
                    mega = not z_move and battle.can_mega_evolve
                    dynamax = not z_move and not mega and battle.can_dynamax

                    return self.create_order(target_move, z_move=z_move, mega=mega, dynamax=dynamax)

            # 3.2 포켓몬 교체 (8~13번 액션 -> 0~5번 슬롯)
            elif 8 <= action_idx <= 13:
                slot_idx = action_idx - 8
                if slot_idx < len(my_team_list):
                    target_pkmn = my_team_list[slot_idx]
                    if target_pkmn in battle.available_switches:
                        return self.create_order(target_pkmn)

            # 폴백: 안전한 무작위 선택
            return self.choose_random_move(battle)
