"""v9 = v8 몸통 + 사슬 헤드 (docs/chain_design.md): 상대 응수 p(o|s) -> 결과 r(s,a,o) -> 가치 q̂(s,a,o), 정책 경로 Q(s,a)=Σ_o p(o|s) q̂(s,a,o).
- chain_variant "aux": 기존 정책 헤드 유지, 사슬은 보조 손실로만. "forced": 정책 = softmax(Q/τ) (사슬을 통과해야만 정책이 나옴)
- 출력 추가: opp_logits [B,K], opp_tera [B], q_hat [B,14,K], q_pi [B,22] (사슬 Q, 합법 마스크 전), q_logits [B,22] (log_softmax(Q/τ) 마스크 후), chain_h [B,14,K,R]
  (손실에서 r 헤드를 실제 둔 쌍에만 적용하려고 은닉 상태를 그대로 내보냄 — `outcome(h)`), policy_logits_base (forced일 때 원래 정책)"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model_v8 import EntityPokemonNetV8, CAND_N, G_DIM
from src.training.chain_targets import N_CLS, N_ACT


class EntityPokemonNetV9(EntityPokemonNetV8):
    def __init__(self, chain_variant: str = "aux", chain_h: int = 64, chain_r: int = 128, **kw):
        super().__init__(**kw)
        assert chain_variant in ("aux", "forced")
        self.cfg.update(model="v9", chain_variant=chain_variant, chain_h=chain_h, chain_r=chain_r)
        self.variant, H, R = chain_variant, chain_h, chain_r
        L, d = self.latent_dim, self.d
        self.act_mv = nn.Linear(L + 2 * d, H)                                   # 기술 a 특징 (mv_in과 같은 입력)
        self.act_sw = nn.Linear(L + d + CAND_N + G_DIM + 1, H)                  # 교체 k 특징 (sw_in과 같은 입력)
        self.tera_emb = nn.Parameter(torch.zeros(H))                            # 테라 동반 기술 표시
        self.opp_head = nn.Sequential(nn.Linear(L, 192), nn.GELU(), nn.Linear(192, N_CLS))
        self.opp_tera_head = nn.Linear(L, 1)
        self.o_emb = nn.Embedding(N_CLS, H)
        self.w_lat, self.w_act, self.w_opp = nn.Linear(L, R), nn.Linear(H, R, bias=False), nn.Linear(H, R, bias=False)
        self.chain2 = nn.Linear(R, R)
        self.q_out = nn.Linear(R, 1)
        # 결과 Δ: HP 변화 12, 새 기절 12(로짓), 다음 활성 슬롯 내 6/상대 6(로짓), 다음 활성 부스트 10
        self.r_out = nn.Linear(R, 12 + 12 + 6 + 6 + 10)
        self.r_my = nn.Linear(R, 2)                                             # 행동 상대형 내 쪽 결과: 이번 턴 필드에서 피해를 받는 내 포켓몬(비교체 = 현재 활성, 교체 = 들어오는 포켓몬)의 HP 변화, 기절 로짓. 내 쪽은 슬롯 번호 정보가 없어 12슬롯 벡터로는 배정이 안 됨
        self.eval_policy = os.environ.get("CHAIN_POLICY", "")                   # 평가용 스위치(학습엔 영향 없음): "q" = Q 정책만, "fuse" = 기존 정책과 Q 정책의 로그확률 평균
        self.log_tau = nn.Parameter(torch.tensor(-3.0))                         # Q 차이 ~0.05 -> 로짓 1 근처에서 시작 (τ≈0.05)

    def outcome(self, h):
        """h [N,R] -> Δ 예측 딕셔너리"""
        o = self.r_out(h)
        m = self.r_my(h)
        return {"dhp": o[:, :12], "faint": o[:, 12:24], "act_my": o[:, 24:30], "act_opp": o[:, 30:36], "boost": o[:, 36:46], "my_hp": m[:, 0], "my_faint": m[:, 1]}

    def _heads(self, enc, history, action_mask=None):
        out = super()._heads(enc, history, action_mask)
        lat = out["latent"]
        B = lat.size(0)
        lg = enc["leg_g"].to(lat.dtype)
        mv_in = torch.cat([lat[:, None].expand(-1, 4, -1), enc["active_mv"], enc["active_mon"][:, None].expand(-1, 4, -1)], -1)
        sw_in = torch.cat([lat[:, None].expand(-1, 6, -1), enc["my_mon"], enc["cand"], lg[:, None].expand(-1, 6, -1), enc["leg_sw"][..., None].to(lg.dtype)], -1)
        a_mv = self.act_mv(mv_in)
        A = torch.cat([a_mv, a_mv + self.tera_emb, self.act_sw(sw_in)], 1)                       # [B,14,H]
        h = F.gelu(self.w_lat(lat)[:, None, None, :] + self.w_act(A)[:, :, None, :] + self.w_opp(self.o_emb.weight)[None, None])    # [B,14,K,R]
        h = F.gelu(self.chain2(h))
        q_hat = torch.sigmoid(self.q_out(h).squeeze(-1).float())                                 # [B,14,K]
        opp_logits = self.opp_head(lat).float()
        p = opp_logits.softmax(-1)
        q = (q_hat * p[:, None, :]).sum(-1)                                                      # Q(s,a) [B,14]
        q_pi = torch.cat([q, q.new_zeros(B, self.ACTION_DIM - N_ACT)], 1)
        q_logits = q_pi / self.log_tau.exp()
        if action_mask is not None:
            q_logits = q_logits.masked_fill(~action_mask, -1e9)
        q_logits = F.log_softmax(q_logits, -1)
        out.update(opp_logits=opp_logits, opp_tera=self.opp_tera_head(lat).squeeze(-1).float(), q_hat=q_hat, q_pi=q_pi, q_logits=q_logits, chain_h=h)
        if self.variant == "forced" or self.eval_policy == "q":
            out["policy_logits_base"] = out["policy_logits"]
            out["policy_logits"] = q_logits
        elif self.eval_policy == "fuse":
            out["policy_logits_base"] = out["policy_logits"]
            out["policy_logits"] = 0.5 * (F.log_softmax(out["policy_logits"], -1) + q_logits)
        return out
