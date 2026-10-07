"""v5 = v4 + 상대 의도 예측 + 의도로 가중한 선택지 피처.

상대 의도(이번 턴 상대가 뭘 할지): 교체/테라 확률, 상대 활성의 기술 분포(공개 4칸 + 신념 후보 12칸 + 기타), 교체한다면 어느 포켓몬으로.
선택지 피처: 남을 때/각 후보로 교체할 때 이번 턴 기대로 받을 피해와 KO 확률, 상대가 교체하면 내가 줄 기대 피해 — 상대 의도의 기대값이고
남을 때와의 차이를 같이 줌. 이 피처는 기술/교체 점수와 종류 헤드(type_head)로 들어가며, 확률은 detach해서 의도 헤드는 보조 손실로만 학습.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model_v3 import _mx
from src.core.model_v4 import CAND4, K_GUESS, EntityPokemonNetV4

N_SLOT = 4 + K_GUESS    # 상대 활성의 기술 칸: 공개 4 + 후보 12 (+ '기타' 1칸은 따로)
F_OPT = 4               # 선택지 피처 수: 받을 기대 피해, 받을 KO 확률, 상대 교체 시 내 기대 피해, 상대 교체 시 내 KO 확률
CAND5 = CAND4 + 2 * F_OPT
TYPE_EXTRA = F_OPT + 3 + 2   # 남을 때 피처 + 후보 집계 3 + 상대 교체/테라 확률


class EntityPokemonNetV5(EntityPokemonNetV4):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.cfg["model"] = "v5"
        d, hw, ld = self.d, self.cfg["head_width"], self.cfg["latent_dim"]
        self.intent_kind = nn.Sequential(nn.Linear(ld + d, 128), nn.GELU(), nn.Linear(128, 2))                # 상대 교체, 상대 테라 (로짓)
        self.intent_move = nn.Sequential(nn.Linear(ld + d + 48 + 2, hw), nn.GELU(), nn.Linear(hw, 1))         # 기술 칸 점수: [잠재, 상대 활성, 기술 임베딩, 공개 여부, 세트 확률]
        self.intent_other = nn.Parameter(torch.zeros(1))                                                      # 위 칸에 없는 기술
        self.intent_tgt = nn.Sequential(nn.Linear(ld + d, 128), nn.GELU(), nn.Linear(128, 1))                 # 교체 대상 점수: [잠재, 상대 포켓몬 토큰]
        self.switch_score = nn.Sequential(nn.Linear(ld + d + CAND5, hw), nn.GELU(), nn.Linear(hw, 1))
        self.type_head = nn.Sequential(nn.Linear(ld + CAND4 + 2 + TYPE_EXTRA, 64), nn.GELU(), nn.Linear(64, 2))      # v4의 [잠재, 활성 요약, 최대 피해, 최소 위협] + 의도 피처

    def _intent(self, enc, latent):
        """상대 의도: kind [B,2] 로짓, move_lp [B,N_SLOT+1] (로그확률, 마지막 = 기타), tgt_lp [B,6], slot_ids/slot_ok [B,N_SLOT] (기술 칸의 기술 ID/유효 여부), tgt_ok [B,6]"""
        B, ar = latent.size(0), torch.arange(latent.size(0), device=latent.device)
        act_o, opp_act_mon, E = enc["act_o"], enc["opp_active_mon"], self.embeddings
        unk = self.belief.move.out_features - 2
        rev_ids = enc["opp_cat"][ar, act_o, 5:9].long()                                    # 공개된 기술 (없으면 0/unk)
        rev_ok = (rev_ids >= 1) & (rev_ids < unk)
        w = enc["guess"]["w"][ar, act_o]                                                   # 후보 기술의 세트 포함 확률 [B,K]
        cand_ids = enc["guess"]["idx"][ar, act_o]
        ids = torch.cat([rev_ids, cand_ids], 1)
        ok = torch.cat([rev_ok, w > 0], 1)
        flags = torch.cat([rev_ok.float(), torch.zeros_like(w)], 1)[..., None]
        wf = torch.cat([rev_ok.float(), w], 1)[..., None]
        x = torch.cat([latent[:, None].expand(-1, N_SLOT, -1), opp_act_mon[:, None].expand(-1, N_SLOT, -1), E._emb("move", ids).detach(), flags, wf], -1)
        score = self.intent_move(x).squeeze(-1).masked_fill(~ok, -1e4)
        move_lp = F.log_softmax(torch.cat([score, self.intent_other.expand(B, 1)], 1), -1)
        tgt = self.intent_tgt(torch.cat([latent[:, None].expand(-1, 6, -1), enc["opp_mon"]], -1)).squeeze(-1).masked_fill(~enc["opp_ok"], -1e4)
        return {"kind": self.intent_kind(torch.cat([latent, opp_act_mon], -1)), "move_lp": move_lp, "tgt_lp": F.log_softmax(tgt, -1), "slot_ids": ids,
                "slot_ok": ok, "tgt_ok": enc["opp_ok"]}

    def _option_features(self, enc, intent):
        """의도 기대값 선택지 피처 [B,6,F_OPT] (내 포켓몬 j마다) + 상대 교체/테라 확률 [B,2]. 기울기 없음"""
        with torch.no_grad():
            B = intent["kind"].size(0)
            ar = torch.arange(B, device=intent["kind"].device)
            p_kind = torch.sigmoid(intent["kind"].float())                                # [B,2] 교체, 테라
            can = enc["opp_ok"]                                                           # 상대가 교체할 수 있는 대상
            p_kind = torch.cat([p_kind[:, :1] * can.any(1, keepdim=True), p_kind[:, 1:]], 1)    # 갈 곳이 없으면 교체 확률 0
            p_sw = p_kind[:, :1]
            pi = intent["move_lp"].exp()[:, :N_SLOT]                                       # 상대가 공격할 때 칸별 확률 (기타 제외 → 합 <= 1)
            p_tgt = intent["tgt_lp"].exp() * can                                           # [B,6] 교체한다면 대상
            act_o = enc["act_o"]
            Dr = enc["D"].view(B, 6, 6, 4, 4)[ar, :, act_o]                                # [B,6(j),4,4] 상대 활성의 공개 기술이 내 j에게
            Dc = enc["Dk"][ar, :, act_o]                                                   # [B,6(j),K,2] 후보 기술 (데미지, KO)
            dmg = torch.cat([Dr[..., 1], Dc[..., 0]], -1)                                  # [B,6,N_SLOT]
            ko = torch.cat([Dr[..., 2], Dc[..., 1]], -1)
            e_dmg, e_ko = (1 - p_sw) * (pi[:, None] * dmg).sum(-1), (1 - p_sw) * (pi[:, None] * ko).sum(-1)       # [B,6]
            mx = _mx(enc["O"].view(B, 6, 6, 4, 4))                                         # [B,6(상대 o),6(내 j),3] 내 j의 기술이 상대 o에게 (배율, 데미지, KO)
            o_dmg, o_ko = p_sw * (p_tgt[..., None] * mx[..., 1]).sum(1), p_sw * (p_tgt[..., None] * mx[..., 2]).sum(1)     # 상대가 교체해 들어온 포켓몬에게
            return torch.stack([e_dmg, e_ko, o_dmg, o_ko], -1), torch.cat([p_sw, p_kind[:, 1:]], -1)

    def _option_inputs(self, enc, latent, ok):
        mv_in, _, type_in = super()._option_inputs(enc, latent, ok)
        B, ar = latent.size(0), torch.arange(latent.size(0), device=latent.device)
        intent = self._intent(enc, latent)
        enc["intent"] = intent
        feat, g = self._option_features(enc, intent)                                       # [B,6,F_OPT], [B,2]
        stay = feat[ar, enc["act_m"]]                                                      # [B,F_OPT]
        cand = torch.cat([enc["cand"], feat, feat - stay[:, None]], -1)                    # [B,6,CAND5]
        sw_in = torch.cat([latent[:, None].expand(-1, 6, -1), enc["my_mon"], cand], -1)
        agg = torch.stack([torch.where(ok, feat[..., 0], torch.full_like(feat[..., 0], 9.0)).amin(1),       # 합법 후보 중 받을 피해 최소
                           torch.where(ok, feat[..., 1], torch.full_like(feat[..., 1], 9.0)).amin(1),       # 받을 KO 확률 최소
                           torch.where(ok, feat[..., 2], torch.full_like(feat[..., 2], -1.0)).amax(1)], -1)  # 상대 교체 시 내 피해 최대
        agg = torch.where(ok.any(1, keepdim=True), agg, torch.zeros_like(agg))
        return mv_in, sw_in, torch.cat([type_in, stay, agg, g], -1)

    def _heads(self, enc, history, action_mask=None, opp_action_mask=None):
        out = super()._heads(enc, history, action_mask, opp_action_mask)
        out["intent"] = enc["intent"]
        return out
