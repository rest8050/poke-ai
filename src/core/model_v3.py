"""엔티티 토큰 트랜스포머 v3 (model_v2의 확장). 옵션 없이 확정된 설정이 기본 동작:
1) 규모: d_model 256 / 6층 / 8헤드. 한 턴 = 포켓몬 12 + 기술 48 + 필드 3 + CLS 1 = 64개 토큰, 압축 없이 전부 어텐션
2) 매치업 특징(src/core/matchup.py): 상대 기술이 내 포켓몬에 주는 배율·데미지·KO, 내 기술이 상대 포켓몬에 주는 효과, 스피드 우열
   - 포켓몬/기술 토큰 입력에 요약치를 더하고, 어텐션 로짓에 편향 bias[h,i,j] = f_h(배율, 데미지, KO, 공격기여부)를 더함
     (내 포켓몬 i <-> 상대 기술 토큰 j, 상대 포켓몬 o <-> 내 기술 토큰 j, 층마다 따로 학습)
   - 교체 후보 점수와 교체/기술 판단(type_head)에 후보별·활성 매치업 요약을 직접 입력
3) 히스토리: 턴 요약 = [CLS, 내 활성 토큰, 상대 활성 토큰, 필드/직전 사건 벡터]을 인과 트랜스포머로 요약
4) 미공개 기술 칸도 토큰으로 유지 (빈 슬롯만 마스킹) + 포켓몬 토큰에 공개된 기술 수/4. 안 드러난 칸을 통째로 빼면 "기술을 1개만 가진 포켓몬"과 구분이 안 됨
5) 상대 세트 신념(belief, src/core/belief.py): 자기 완결형 모듈(자기 임베딩, 트렁크와 공유 없음). 종 ID는 여기에만 들어가고(트렁크에는 없음) 팀 데이터 사전학습 +
   배틀에서는 숨김정보 손실(src/training/hidden_labels.py)로만 학습.
   확률 상위 기술을 미공개 칸에 "추측 기술"로 배정해, 그 기술의 위력/타입/분류로 매치업 위협(기대 데미지/KO)을 계산 → 어텐션 편향, 토큰 요약, 교체 헤드에 반영.
   신념 출력은 detach라서 정책/가치 손실은 신념에 닿지 않고(종 조합으로 승패를 외울 수 없음), 신념은 트렁크 입력을 안 쓰므로 숨김정보 손실도 트렁크에 닿지 않음
인터페이스는 model_v2와 같음 (forward/forward_sequences/get_action, cfg + save_ckpt/model_from_ckpt). 옛 구조(v3_full 등)는 model_v3_legacy.py
arch: {"model": "v3", "version": 3}"""
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
        self.guess_proj = nn.Linear(1 + MOVE_NUM_DIM + 48, d)               # 추측 기술(확률, 수치 특징, 기술 임베딩) -> 미공개 칸 토큰
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

    # ---- 매치업 요약 ----
    def _matchup_feats(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess):
        B = my_cat.size(0)
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv)
            D, O = M["def"], M["off"]                                          # [B,6,24,4]
            Dp = self.matchup._pair(opp_cat, opp_num, guess["pseudo"], False, my_cat, my_num, True)
        D_real = D
        D = D + Dp * guess["w"].reshape(B, 1, 24, 1)                           # 미공개 칸의 추측 기술이 내 포켓몬에게 주는 위협을 확률로 가중해 더함
        act_m, act_o = my_num[:, :, 0].argmax(1), opp_num[:, :, 0].argmax(1)
        DS, OS = D.view(B, 6, 6, 4, 4), O.view(B, 6, 6, 4, 4)                  # D[b,i(내),o,m,ch] / O[b,o(상대),i(내 공격),m,ch]
        def_opp_act = DS[ar, :, act_o]                                         # [B,6(i),4,4] 상대 활성 기술이 내 포켓몬 i에게 (추측 기술 포함)
        off_vs_opp_act = OS[ar, act_o]                                         # [B,6(i),4,4] 내 포켓몬 i의 기술이 상대 활성에게
        thr_on_my_act = DS[ar, act_m]                                          # [B,6(o),4,4] 상대 포켓몬 o의 기술이 내 활성에게
        my_act_on_opp = OS[ar, :, act_m]                                       # [B,6(o),4,4] 내 활성 기술이 상대 포켓몬 o에게
        mon_extra = torch.cat([
            torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None]], -1),                # 내 쪽 6
            torch.cat([_mx(thr_on_my_act), _mx(my_act_on_opp), M["spd_opp"][..., None]], -1)], 1)          # 상대 쪽 6 → [B,12,7]
        mv_extra = torch.cat([off_vs_opp_act[..., :3], thr_on_my_act[..., :3]], 1)                          # [B,12,4,3]
        # 벤치 위협은 실제로 공개된 기술만 (필드에 없는 상대의 추측 기술은 섞지 않음 — 그쪽 정보는 토큰/어텐션 편향 경로로 감)
        bench_thr = D_real.view(B, 6, 6, 4, 4)[..., :3].amax((2, 3))[..., 1:]                               # 후보 i가 상대 전체의 공개 기술에 맞는 최대 데미지/KO [B,6,2]
        cand = torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], bench_thr], -1)   # [B,6,9]
        return D, O, mon_extra, mv_extra, cand, ar, act_m

    def _make_guess(self, logits, opp_cat, opp_mv, opp_num):
        """신념의 기술 확률 -> 미공개 칸마다 추측 기술 배정 (확률 높은 순). 반환 idx/w [B,6,4], pseudo [B,6,4,46] (모두 기울기 없음)"""
        with torch.no_grad():
            lg = logits["move"].detach().float()
            V = lg.size(-1)
            unk = self.belief.move.out_features - 2
            cur = opp_cat[..., 5:9].long()
            known = torch.zeros_like(lg, dtype=torch.bool)
            known.scatter_(2, torch.where((cur >= 1) & (cur < unk), cur, torch.zeros_like(cur)).clamp(0, V - 1), True)
            bad = torch.zeros(V, dtype=torch.bool, device=lg.device)
            bad[0], bad[unk:] = True, True
            prob = torch.softmax(lg.masked_fill(known | bad, -1e4), -1)
            present = opp_num[..., 11] > 0
            unrev = (opp_mv.abs().sum(-1) == 0) & present[..., None]                # [B,6,4]
            n_unrev = unrev.sum(-1, keepdim=True).float()
            topv, topi = prob.topk(4, -1)
            marg = (topv * n_unrev).clamp(max=1.0)                                   # 미공개 칸이 n개면 각 기술이 세트에 있을 확률 ~ n x 확률
            rank = (unrev.long().cumsum(-1) - 1).clamp(0, 3)
            idx, w = topi.gather(-1, rank), marg.gather(-1, rank) * unrev
            pseudo = self.move_table[idx] * unrev[..., None]
        return {"idx": idx, "w": w, "pseudo": pseudo}

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

    # ---- 토큰 구성 + 트렁크 ----
    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec):
        B, dev, d = my_team_cat.size(0), my_team_cat.device, self.d
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        parts = [E.species_proj(num[:, 11:18]), E._emb("item", cat[:, 0]), E._emb("ability", cat[:, 1]), E.type_embed(cat[:, 2]),
                 E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:]]     # BASE_IN차원
        raw = E._emb("move", cat[:, 5:9])
        revealed = (mvn.abs().sum(-1) > 0)
        n_rev = revealed.float().sum(-1, keepdim=True) / 4.0
        # 신념: 상대 팀 텐서를 자기 입력으로 읽어(트렁크와 무관) 숨겨진 세트를 예측, 확률 높은 기술을 미공개 칸에 추측 기술로 배정
        belief_logits = self.belief(opp_team_cat, opp_team_num)
        guess = self._make_guess(belief_logits, opp_team_cat, opp_move_num, opp_team_num)
        D, O, mon_extra, mv_extra, cand, ar, act_m = self._matchup_feats(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, guess)
        mon_in = torch.cat(parts + [mon_extra.reshape(B * 12, MON_EXTRA), n_rev], -1)
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn, mv_extra.reshape(B * 12, 4, MV_EXTRA)], -1)
        side = torch.arange(12, device=dev) // 6
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] + self.act_emb(active)
        mon = self.mon_proj(mon_in).view(B, 12, d) + base + self.role_emb.weight[0]
        mv = self.move_proj(mv_in).view(B, 12, 4, d) + base[:, :, None] + self.mslot_emb.weight[None, None] + self.role_emb.weight[1]
        fld = self.field_proj(field_vec).view(B, 3, d) + self.fld_emb.weight[None] + self.role_emb.weight[2]
        cls = self.cls.expand(B, 1, d) + self.role_emb.weight[3]
        x = torch.cat([mon, mv.reshape(B, 48, d), fld, cls], 1)
        mon_pad = (num[:, 11] == 0).view(B, 12)
        unrevealed = ~revealed.view(B, 12, 4) & ~mon_pad[:, :, None]
        g = torch.cat([guess["w"][..., None], guess["pseudo"], E._emb("move", guess["idx"]).detach()], -1)       # 추측 기술 특징 (상대 쪽만)
        unk = self.unk_move.expand(B, 12, 4, d) + self.guess_proj(torch.cat([torch.zeros_like(g), g], 1))
        x = torch.cat([x[:, :12], x[:, 12:60] + unrevealed.reshape(B, 48, 1) * unk.reshape(B, 48, d), x[:, 60:]], 1)
        pad = torch.cat([mon_pad, mon_pad[:, :, None].expand(B, 12, 4).reshape(B, 48), mon_pad.new_zeros(B, 4)], 1)     # 빈 슬롯의 칸만 마스킹
        pad_f = torch.zeros(B, 1, 1, N_TOK, device=dev, dtype=x.dtype).masked_fill(pad[:, None, None, :], float("-inf"))
        for li, layer in enumerate(self.layers):
            mask = (self._bias(D, O, li).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        opp_act = opp_team_num[:, :, 0].argmax(1)
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m}

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
