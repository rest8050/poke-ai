"""v6 = v5 + 규칙 기반 매치업 보정 + 행동 제약 효과 입력. 교사가 교체를 고르는 이유 중 v5가 못 보던 두 가지를 채움.
1) 매치업 보정 (Matchup(entry=True)): 천진 포켓몬은 상대 공격/방어 랭크를 무시하고 계산(상대 +2 공격이 천진 후보에게 더 아프게 나오던 오류 수정),
   교체 후보의 데미지/KO는 장판(스텔스록·압정) 피해를 뺀 HP 기준, 후보별 진입 피해 비율을 후보 특징과 포켓몬 토큰에 추가
2) 행동 제약 효과: 양쪽 활성 포켓몬의 도발/앵콜/사슬묶기 + 교체 봉쇄 플래그와 남은 턴 비율(field_vec 뒤쪽 14열).
   필드 토큰과 턴 벡터(히스토리)로 들어가고, 합법성 요약과 함께 융합층/종류 헤드/교체 점수에도 직접 입력 (v5가 "기술이 막혔다"는 결과만 본 데 비해 원인과 남은 기간을 봄)
v3~v5 체크포인트는 field_vec의 새 열을 잘라내고 옛 매치업 계산 그대로 씀(model_v3._fit_field), v6는 옛 데이터의 짧은 field_vec을 0으로 채움
arch: {"model": "v6", "my_pos": false}"""
import torch

from src.core.matchup import Matchup
from src.core.model_v4 import CAND4, MON_EXTRA4
from src.core.model_v5 import LEG_G, EntityPokemonNetV5
from src.core.tensor_encoder import EFFECT_DIM, EFFECT_START, FIELD_DIM


class EntityPokemonNetV6(EntityPokemonNetV5):
    MON_X, CAND_N, G_DIM = MON_EXTRA4 + 1, CAND4 + 1, LEG_G + EFFECT_DIM            # + 후보별 진입 피해 비율 / + 양쪽 활성의 효과 열

    def __init__(self, vocab_path: str = "data/vocab.json", **kw):
        super().__init__(vocab_path=vocab_path, field_dim=FIELD_DIM, **kw)
        self.cfg["model"] = "v6"
        self.matchup = Matchup(vocab_path, entry=True)

    def _matchup_feats4(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat, field_vec=None):
        D, O, H, mon_extra, mv_extra, cand, ar, act_m = super()._matchup_feats4(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat, field_vec)
        with torch.no_grad():
            ent = self.matchup.entry_frac(my_cat, my_num, field_vec)                                # [B,6] 내 포켓몬이 교체해 들어올 때 장판에 깎이는 비율
        mon_extra = torch.cat([mon_extra, torch.cat([ent, torch.zeros_like(ent)], 1)[..., None]], -1)
        return D, O, H, mon_extra, mv_extra, torch.cat([cand, ent[..., None]], -1), ar, act_m

    def _global_feats(self, leg, off_act, n_rev, field_vec):
        return torch.cat([self._legal_summary(leg, off_act, n_rev), field_vec[:, EFFECT_START:]], -1)
