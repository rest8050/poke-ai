"""엔티티 토큰 트랜스포머의 공통 부분 (v3 설계). 토큰 구성과 신념 활용은 v4~v6(model_v4.py~model_v6.py)가 덮어쓰고, 여기에는 공통 모듈/히스토리/헤드/롤아웃만 남음 (단독 모델 아님, arch는 v4 이상).
1) 규모: d_model 256 / 6층 / 8헤드. 한 턴 = 포켓몬 12 + 기술 48 + 필드 3 + CLS 1 = 64개 토큰, 압축 없이 전부 어텐션
2) 매치업 특징(src/core/matchup.py): 상대 기술이 내 포켓몬에 주는 배율·데미지·KO, 내 기술이 상대 포켓몬에 주는 효과, 스피드 우열
   - 포켓몬/기술 토큰 입력에 요약치를 더하고, 어텐션 로짓에 편향 bias[h,i,j] = f_h(배율, 데미지, KO, 공격기여부)를 더함 (층마다 따로 학습)
   - 교체 후보 점수와 교체/기술 판단(type_head)에 후보별·활성 매치업 요약을 직접 입력
3) 히스토리: 턴 요약 = [CLS, 내 활성 토큰, 상대 활성 토큰, 필드/직전 사건 벡터]을 인과 트랜스포머로 요약
4) 미공개 기술 칸도 토큰으로 유지 (빈 슬롯만 마스킹) + 포켓몬 토큰에 공개된 기술 수/4
5) 상대 세트 신념(belief, src/core/belief.py): 자기 완결형 모듈, 출력은 detach라서 정책/가치 손실은 신념에 닿지 않음 (숨김정보 손실로만 학습)
인터페이스: forward/forward_sequences/get_action, cfg + save_ckpt/model_from_ckpt. 옛 구조(v3_full 등)는 model_v3_legacy.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.belief import SetBelief, build_move_table
from src.core.matchup import Matchup
from src.core.model import MAX_TURNS, DeepPokemonBattleTransformerNet, PokemonEmbeddingLayer
from src.core.tensor_encoder import MOVE_NUM_DIM

# 3차원 어텐션 마스크(헤드별 편향)는 추론용 빠른 커널에서 학습 경로와 값이 달라짐(CLS 기준 2e-3) → 학습과 같은 일반 경로만 쓰도록 끔
torch.backends.mha.set_fastpath_enabled(False)

N_TOK = 64            # 포켓몬 12 + 기술 48 + 필드 3 + CLS 1
MON_EXTRA, MV_EXTRA, CAND_FEAT = 7, 3, 9
BASE_IN = 204         # 포켓몬 토큰 기본 입력 차원 (종족값 투영 64 + 도구/특성/타입/상태/테라 임베딩 + 배틀 수치 + 이벤트 증거)


def _mx(x):           # [..., 4(기술), 4(채널)] 중 앞 3채널(배율, 데미지, KO)의 기술별 최댓값 → [..., 3]
    return x[..., :3].amax(-2)


class EntityPokemonNetV3(nn.Module):
    ACTION_DIM = 22

    def __init__(self, vocab_path: str = "data/vocab.json", d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 192, field_dim: int = 58, entity_features: bool = True,
                 species_dim: int = 32, belief_hidden: int = 256, belief_species_dropout: float = 0.3):
        super().__init__()
        self.cfg = dict(model="v3", version=3, d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width,
                        species_dim=species_dim, belief_hidden=belief_hidden, belief_species_dropout=belief_species_dropout)
        d = self.d = d_model
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        self.matchup = Matchup(vocab_path)
        self.belief = SetBelief(vocab_path, species_dim, belief_hidden, belief_species_dropout)
        self.register_buffer("move_table", build_move_table(vocab_path), persistent=False)
        self.unk_move = nn.Parameter(torch.zeros(d))                        # 미공개 기술 칸 표시
        self.mon_proj = nn.Sequential(nn.Linear(BASE_IN + MON_EXTRA + 1, d), nn.LayerNorm(d))
        self.move_proj = nn.Sequential(nn.Linear(142 + MV_EXTRA, d), nn.LayerNorm(d))
        self.field_proj = nn.Sequential(nn.Linear(field_dim, d), nn.GELU(), nn.Linear(d, 3 * d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.role_emb, self.side_emb, self.slot_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(6, d)
        self.mslot_emb, self.act_emb, self.fld_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(3, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout=dropout, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        # 층별 어텐션 편향: 입력 4채널(배율, 데미지, KO, 공격기) -> 헤드별 값. 작게 시작
        self.bias_def = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_layers)])
        self.bias_off = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_layers)])
        for lin in list(self.bias_def) + list(self.bias_off):
            nn.init.normal_(lin.weight, std=0.5)
            nn.init.zeros_(lin.bias)
        self.turn_proj = nn.Linear(3 * d + field_dim, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        hl = nn.TransformerEncoderLayer(history_dim, 4, history_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.history_seq = nn.TransformerEncoder(hl, hist_layers, enable_nested_tensor=False)
        self.history_out = nn.Linear(history_dim, history_dim)
        self.fusion = nn.Sequential(nn.Linear(d + history_dim, latent_dim * 2), nn.LayerNorm(latent_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.move_score = nn.Sequential(nn.Linear(latent_dim + 2 * d, head_width), nn.GELU(), nn.Linear(head_width, 2))
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + d + CAND_FEAT, head_width), nn.GELU(), nn.Linear(head_width, 1))
        self.type_head = nn.Sequential(nn.Linear(latent_dim + CAND_FEAT + 2, 64), nn.GELU(), nn.Linear(64, 2))
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.opp_action_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, self.ACTION_DIM))
        self.value_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def _bias(self, D, O, layer):
        """어텐션 편향 [B,H,64,64]: 내 포켓몬<->상대 기술 토큰(36..59), 상대 포켓몬<->내 기술 토큰(12..35)"""
        B, H = D.size(0), self.n_heads
        fd = self.bias_def[layer](D).permute(0, 3, 1, 2)                       # [B,H,6,24]
        fo = self.bias_off[layer](O).permute(0, 3, 1, 2)
        b = D.new_zeros(B, H, N_TOK, N_TOK)
        b[:, :, 0:6, 36:60] = fd
        b[:, :, 36:60, 0:6] = fd.transpose(2, 3)
        b[:, :, 6:12, 12:36] = fo
        b[:, :, 12:36, 6:12] = fo.transpose(2, 3)
        return b

    def _fit_field(self, fv):
        """field_vec 폭을 이 모델이 학습한 폭에 맞춤: 더 넓으면(인코더가 뒤에 열을 추가) 뒤를 버리고, 더 좁으면(옛 데이터) 0으로 채움"""
        n = self.field_proj[0].in_features
        return fv[..., :n] if fv.size(-1) >= n else F.pad(fv, (0, n - fv.size(-1)))

    def _history(self, turn_seq):
        B, T = turn_seq.size(0), turn_seq.size(1)
        pos = torch.arange(T, device=turn_seq.device).clamp(max=MAX_TURNS - 1)
        x = self.turn_proj(turn_seq) + self.turn_pos(pos)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=turn_seq.device), diagonal=1)
        return self.history_out(self.history_seq(x, mask=causal))

    @staticmethod
    def _turn_vec(enc):
        return torch.cat([enc["cls"], enc["active_mon"], enc["opp_active_mon"], enc["field"]], -1)

    def _heads(self, enc, history, action_mask=None, opp_action_mask=None):
        B = history.size(0)
        latent = self.fusion(torch.cat([enc["cls"], history], -1))
        cand = enc["cand"]                                                       # [B,6,9]
        ok = action_mask[:, 8:14] if action_mask is not None else torch.ones(B, 6, dtype=torch.bool, device=latent.device)
        act_feat = cand[torch.arange(B, device=latent.device), enc["act_m"]]    # 내 활성의 같은 요약
        best_off = torch.where(ok, cand[..., 4], torch.full_like(cand[..., 4], -1.0)).amax(1, keepdim=True).clamp_min(0)      # 교체 가능 후보가 줄 수 있는 최대 데미지
        min_thr = torch.where(ok, cand[..., 7], torch.full_like(cand[..., 7], 9.0)).amin(1, keepdim=True)
        min_thr = torch.where(ok.any(1, keepdim=True), min_thr, torch.zeros_like(min_thr))
        mv_in = torch.cat([latent[:, None].expand(-1, 4, -1), enc["active_mv"], enc["active_mon"][:, None].expand(-1, 4, -1)], -1)
        sw_in = torch.cat([latent[:, None].expand(-1, 6, -1), enc["my_mon"], cand], -1)
        mv, sw = self.move_score(mv_in), self.switch_score(sw_in).squeeze(-1)
        detail = torch.cat([mv[..., 0], mv[..., 1], sw, latent.new_full((B, 8), -1e9)], 1)
        type_logits = self.type_head(torch.cat([latent, act_feat, best_off, min_thr], -1))
        if action_mask is not None:
            detail = detail.masked_fill(~action_mask, -1e9)
            no_move = ~(action_mask & ~self.is_switch).any(dim=1)
            no_switch = ~(action_mask & self.is_switch).any(dim=1)
            type_logits = type_logits.masked_fill(torch.stack([no_move, no_switch], dim=1), -1e9)
        lp_type = F.log_softmax(type_logits, -1)
        lp_move = F.log_softmax(detail.masked_fill(self.is_switch, -1e9), -1)
        lp_switch = F.log_softmax(detail.masked_fill(~self.is_switch, -1e9), -1)
        policy_logits = torch.where(self.is_switch, lp_type[:, 1:2] + lp_switch, lp_type[:, 0:1] + lp_move)
        opp = self.opp_action_head(latent)
        if opp_action_mask is not None:
            opp = opp.masked_fill(~opp_action_mask, -1e9)
        bl = enc["belief_logits"]
        return {"policy_logits": policy_logits, "opp_action_logits": opp, "value": self.value_head(latent),
                "hid_move": bl["move"], "hid_item": bl["item"], "hid_ability": bl["ability"]}      # hid_*: 숨김정보 손실 입력

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None, opp_action_mask=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec)
        cur = self._turn_vec(enc).unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask, opp_action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None):
        enc = self._encode_turn(*obs)
        tv = self._turn_vec(enc)
        T = int(time_index.max().item()) + 1
        turn_seq = tv.new_zeros(n_seq, T, tv.size(-1))
        turn_seq[seq_index, time_index] = tv
        return self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask, None)

    get_action = DeepPokemonBattleTransformerNet.get_action
    get_action_rl = DeepPokemonBattleTransformerNet.get_action_rl
