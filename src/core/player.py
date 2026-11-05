from src.core.model import DeepPokemonBattleTransformerNet
from src.core.tensor_encoder import BattleTensorEncoder

try:
    from poke_env.player import Player
    from poke_env.battle import AbstractBattle
    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False
    Player = object
    AbstractBattle = object


if POKE_ENV_AVAILABLE:
    from src.core.effectiveness import _id, eff_move, types
    from src.core.actions import decode_action

    def lead_score(battle, me, opps):
        """선두 점수 = 상대 팀 평균 (내 최고 공격 배율 - 상대 자속 최고 배율)"""
        if not opps:
            return 0.0
        my_types = types(me)
        attacks = [m for m in me.moves.values() if m.base_power]
        total = 0.0
        for o in opps:
            off = max([eff_move(battle, _id(m.type), o) * (1.5 if _id(m.type) in my_types else 1.0) for m in attacks]
                      or [1.0])
            dfn = max([eff_move(battle, t, me) * 1.5 for t in types(o)] or [1.0])
            total += off - dfn
        return total / len(opps)

    class SmartPokemonPlayer(Player):
        """
        분리된 BattleTensorEncoder 모듈을 사용하는 스마트 poke-env 플레이어
        """
        def __init__(self, model: DeepPokemonBattleTransformerNet = None, vocab_path: str = "data/vocab.json", **kwargs):
            super().__init__(**kwargs)
            self.model = model or DeepPokemonBattleTransformerNet(vocab_path=vocab_path)
            self.encoder = BattleTensorEncoder(vocab_path=vocab_path)
            # 배틀별 GRU 히스토리 (battle_tag -> Tensor)
            # ponytail: 끝난 배틀도 남음, 장시간 돌리면 battle.finished 기준으로 정리
            self.history = {}

        def teampreview(self, battle: AbstractBattle):
            """상대 6마리 상대로 상성이 가장 좋은 포켓몬을 선두로, 나머지는 원래 순서"""
            team = list(battle.team.values())
            opps = list(battle.teampreview_opponent_team or [])
            lead = max(range(len(team)), key=lambda i: lead_score(battle, team[i], opps))
            for mon in team:
                mon._selected_in_teampreview = True
            return "/team " + "".join(str(i + 1) for i in [lead] + [i for i in range(len(team)) if i != lead])

        def order_for(self, battle: AbstractBattle, action_idx: int):
            """22칸 행동 -> 주문. 실행 불가면 None"""
            d = decode_action(battle, action_idx)
            if d is None:
                return None
            if d[0] == "switch":
                return self.create_order(d[1])
            _, move, z_move, tera = d
            # 기믹은 한 턴에 하나만: Z/테라(모델 선택) > 메가(즉시) > 다이맥스(즉시)
            mega = not z_move and not tera and battle.can_mega_evolve
            dynamax = not z_move and not tera and not mega and battle.can_dynamax
            return self.create_order(move, z_move=z_move, mega=mega, dynamax=dynamax, terastallize=tera)

        def choose_move(self, battle: AbstractBattle):
            # 1. 분리된 전용 인코더 모듈로 텐서 및 마스크 변환
            tensors = self.encoder.encode_battle(battle)
            my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask, legal_mask = tensors

            # 2. 신경망 추론
            tag = battle.battle_tag
            action_idx, p_probs, val, self.history[tag] = self.model.get_action(
                my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, self.history.get(tag), action_mask, legal_mask,
                prev_move=self.encoder.prev_move(battle)
            )

            # 3. 행동 매핑 (폴백: 안전한 무작위 선택)
            return self.order_for(battle, action_idx) or self.choose_random_move(battle)
