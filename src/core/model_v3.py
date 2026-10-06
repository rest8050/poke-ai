"""엔티티 토큰 트랜스포머 v3 (model_v2의 확장). 달라진 점:
1) 규모: 기본 d_model 256 / 6층 / 8헤드
2) 매치업 특징(src/core/matchup.py): 상대 기술이 내 포켓몬에 주는 배율·데미지·KO, 내 기술이 상대 포켓몬에 주는 효과, 스피드 우열
   - 포켓몬/기술 토큰 입력에 요약치를 추가
   - 어텐션 로짓에 편향 bias[h, i, j] = f_h(배율, 데미지, KO, 공격기여부)를 더함: 내 포켓몬 i <-> 상대 기술 토큰 j, 상대 포켓몬 o <-> 내 기술 토큰 j (층마다 따로 학습)
   - 교체 후보 점수와 교체/기술 판단(type_head)에 후보별·활성 매치업 요약을 직접 입력
3) 히스토리: 턴 요약 = [CLS, 내 활성 토큰, 상대 활성 토큰, 필드/직전 사건 벡터] (v2는 CLS만)
4) unrevealed_tokens: 아직 안 드러난 상대 기술 칸도 토큰으로 남기고(미공개 표시 + 숨김 ID 임베딩), 포켓몬 토큰에 공개된 기술 수/4를 더함.
   (v3full까지는 안 드러난 칸을 어텐션에서 통째로 빼서 "기술을 1개만 가진 포켓몬"과 구분이 안 됐음 — 그 체크포인트는 model_from_ckpt가 False로 복원)
5) opp_species: 상대 포켓몬에만 종 ID 임베딩을 넣음. 내 쪽은 종 ID를 안 줌(종을 외우지 말고 능력치/기술로 일반화), 상대 쪽은 종별로 흔한 세트(숨겨진 기술/도구/특성)를
   기억에 의존해 추정하도록: 포켓몬 토큰 입력 + 미공개 기술 칸 토큰에 종 임베딩을 더함. 학습 중 species_dropout 확률로 종을 <unk>로 가려서 처음 보는 종도 능력치만으로 동작하게 함
6) hidden_head: 상대 포켓몬 토큰에서 숨겨진 기술/도구/특성을 예측하는 보조 헤드 (출력 hid_move/hid_item/hid_ability, 학습은 src/training/hidden_labels.py)
7) policy: "hier"(잠재 벡터가 기술/교체를 정함, 기존) | "lse"(기술/교체 로짓 = 각 점수의 logsumexp + 학습 보정, 보정 0이면 평탄 softmax와 같음) | "flat"(14칸 전체 softmax)
인터페이스는 model_v2와 같음 (forward/forward_sequences/get_action, cfg + save_ckpt/model_from_ckpt). arch에 {"model": "v3"}"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.matchup import Matchup
from src.core.model import MAX_TURNS, DeepPokemonBattleTransformerNet, PokemonEmbeddingLayer

# 3차원 어텐션 마스크(헤드별 편향)는 추론용 빠른 커널에서 학습 경로와 값이 달라짐(CLS 기준 2e-3) → 학습과 같은 일반 경로만 쓰도록 끔
torch.backends.mha.set_fastpath_enabled(False)

N_TOK = 64            # 포켓몬 12 + 기술 48 + 필드 3 + CLS 1
MON_EXTRA, MV_EXTRA, CAND_FEAT = 7, 3, 9


def _mx(x):           # [..., 4(기술), 4(채널)] 중 앞 3채널(배율, 데미지, KO)의 기술별 최댓값 → [..., 3]
    return x[..., :3].amax(-2)


class EntityPokemonNetV3(nn.Module):
    ACTION_DIM = 22

    def __init__(self, vocab_path: str = "data/vocab.json", d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 192, field_dim: int = 58, entity_features: bool = True,
                 unrevealed_tokens: bool = True, policy: str = "hier", opp_species: bool = True, species_dropout: float = 0.1, species_dim: int = 32, hidden_head: bool = False):
        super().__init__()
        self.cfg = dict(model="v3", d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width,
                        unrevealed_tokens=unrevealed_tokens, policy=policy, opp_species=opp_species, species_dropout=species_dropout, species_dim=species_dim, hidden_head=hidden_head)
        assert policy in ("hier", "lse", "flat"), policy
        self.hidden_head = hidden_head
        self.unrevealed_tokens, self.policy, self.opp_species, self.species_dropout = unrevealed_tokens, policy, opp_species, species_dropout
        d = self.d = d_model
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        self.matchup = Matchup(vocab_path)
        self.mon_proj = nn.Sequential(nn.Linear(204 + MON_EXTRA + int(unrevealed_tokens) + (species_dim if opp_species else 0), d), nn.LayerNorm(d))
        if unrevealed_tokens:
            self.unk_move = nn.Parameter(torch.zeros(d))                    # 미공개 기술 칸 표시
        if hidden_head:
            import json
            vv = json.load(open(vocab_path, encoding="utf-8"))
            self.hid_move = nn.Linear(d_model, vv["move"]["<unk>"] + 2)       # id 0..unk+1
            self.hid_item = nn.Linear(d_model, vv["item"]["<unk>"] + 2)
            self.hid_ability = nn.Linear(d_model, vv["ability"]["<unk>"] + 2)
        if opp_species:
            import json
            sp = json.load(open(vocab_path, encoding="utf-8"))["species"]
            self.unk_species = sp["<unk>"]
            self.species_emb = nn.Embedding(max(sp.values()) + 3, species_dim)
            nn.init.normal_(self.species_emb.weight, std=0.1)
            if unrevealed_tokens:
                self.sp_to_unk = nn.Linear(species_dim, d, bias=False)       # 종이 정하는 "숨겨진 기술에 대한 기대"
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
        # 히스토리: 턴 요약 = [CLS, 내 활성, 상대 활성, 필드 벡터]
        self.turn_proj = nn.Linear(3 * d + field_dim, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        hl = nn.TransformerEncoderLayer(history_dim, 4, history_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.history_seq = nn.TransformerEncoder(hl, hist_layers, enable_nested_tensor=False)
        self.history_out = nn.Linear(history_dim, history_dim)
        self.fusion = nn.Sequential(nn.Linear(d + history_dim, latent_dim * 2), nn.LayerNorm(latent_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.move_score = nn.Sequential(nn.Linear(latent_dim + 2 * d, head_width), nn.GELU(), nn.Linear(head_width, 2))
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + d + CAND_FEAT, head_width), nn.GELU(), nn.Linear(head_width, 1))
        if policy != "flat":
            self.type_head = nn.Sequential(nn.Linear(latent_dim + CAND_FEAT + 2, 64), nn.GELU(), nn.Linear(64, 2))
            if policy == "lse":                                              # 보정 0에서 시작 = 평탄 softmax
                nn.init.zeros_(self.type_head[2].weight)
                nn.init.zeros_(self.type_head[2].bias)
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.opp_action_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, self.ACTION_DIM))
        self.value_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

    # ---- 매치업 요약 ----
    def _matchup_feats(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv):
        B = my_cat.size(0)
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv)
        D, O = M["def"], M["off"]                                              # [B,6,24,4]
        act_m, act_o = my_num[:, :, 0].argmax(1), opp_num[:, :, 0].argmax(1)
        DS, OS = D.view(B, 6, 6, 4, 4), O.view(B, 6, 6, 4, 4)                  # D[b,i(내),o,m,ch] / O[b,o(상대),i(내 공격),m,ch]
        def_opp_act = DS[ar, :, act_o]                                         # [B,6(i),4,4] 상대 활성 기술이 내 포켓몬 i에게
        off_vs_opp_act = OS[ar, act_o]                                         # [B,6(i),4,4] 내 포켓몬 i의 기술이 상대 활성에게
        thr_on_my_act = DS[ar, act_m]                                          # [B,6(o),4,4] 상대 포켓몬 o의 기술이 내 활성에게
        my_act_on_opp = OS[ar, :, act_m]                                       # [B,6(o),4,4] 내 활성 기술이 상대 포켓몬 o에게
        mon_extra = torch.cat([
            torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None]], -1),                # 내 쪽 6
            torch.cat([_mx(thr_on_my_act), _mx(my_act_on_opp), M["spd_opp"][..., None]], -1)], 1)          # 상대 쪽 6 → [B,12,7]
        mv_extra = torch.cat([off_vs_opp_act[..., :3], thr_on_my_act[..., :3]], 1)                          # [B,12,4,3]
        bench_thr = DS[..., :3].amax((2, 3))[..., 1:]                                                       # 후보 i가 상대 전체의 기술에 맞는 최대 데미지/KO [B,6,2]
        cand = torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], bench_thr], -1)   # [B,6,9]
        return D, O, mon_extra, mv_extra, cand, ar, act_m

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
        D, O, mon_extra, mv_extra, cand, ar, act_m = self._matchup_feats(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num)
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        sp_feat = None
        if self.opp_species:                                                 # 상대 쪽만 종 ID (내 쪽은 0 = 일반화)
            sid = opp_team_cat[..., 10].long().clamp(0, self.species_emb.num_embeddings - 1)
            if self.training and self.species_dropout > 0:
                sid = torch.where(torch.rand(sid.shape, device=dev) < self.species_dropout, torch.full_like(sid, self.unk_species), sid)
            sp_e = self.species_emb(sid)
            sp_feat = torch.cat([torch.zeros_like(sp_e), sp_e], 1)           # [B,12,species_dim]
        mon_in = torch.cat([E.species_proj(num[:, 11:18]), E._emb("item", cat[:, 0]), E._emb("ability", cat[:, 1]), E.type_embed(cat[:, 2]),
                            E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:],
                            mon_extra.reshape(B * 12, MON_EXTRA)] + ([(mvn.abs().sum(-1) > 0).float().sum(-1, keepdim=True) / 4.0] if self.unrevealed_tokens else [])
                           + ([sp_feat.reshape(B * 12, -1)] if sp_feat is not None else []), -1)
        raw = E._emb("move", cat[:, 5:9])
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
        unrevealed = (mvn.abs().sum(-1) == 0).view(B, 12, 4) & ~mon_pad[:, :, None]
        if self.unrevealed_tokens:                                           # 안 드러난 칸도 토큰으로 남김 (빈 슬롯의 칸만 마스킹)
            mv_pad = mon_pad[:, :, None].expand(B, 12, 4)
            unk = self.unk_move.expand(B, 12, d)
            if sp_feat is not None:
                unk = unk + self.sp_to_unk(sp_feat)                          # 종별 기대가 미공개 기술 칸에 실림
            unk = unk[:, :, None, :].expand(B, 12, 4, d).reshape(B, 48, d)
            x = torch.cat([x[:, :12], x[:, 12:60] + unrevealed.reshape(B, 48, 1) * unk, x[:, 60:]], 1)
        else:
            mv_pad = (mvn.abs().sum(-1) == 0).view(B, 12, 4) | mon_pad[:, :, None]
        pad = torch.cat([mon_pad, mv_pad.reshape(B, 48), mon_pad.new_zeros(B, 4)], 1)
        pad_f = torch.zeros(B, 1, 1, N_TOK, device=dev, dtype=x.dtype).masked_fill(pad[:, None, None, :], float("-inf"))
        for li, layer in enumerate(self.layers):
            mask = (self._bias(D, O, li).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        opp_act = opp_team_num[:, :, 0].argmax(1)
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "opp_mon": out[:, 6:12], "cand": cand, "field": field_vec, "act_m": act_m}

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
        if action_mask is not None:
            detail = detail.masked_fill(~action_mask, -1e9)
        if self.policy == "flat":
            policy_logits = F.log_softmax(detail, -1)
        else:
            type_logits = self.type_head(torch.cat([latent, act_feat, best_off, min_thr], -1))
            if self.policy == "lse":
                type_logits = type_logits + torch.stack([torch.logsumexp(detail.masked_fill(self.is_switch, -1e9), -1),
                                                         torch.logsumexp(detail.masked_fill(~self.is_switch, -1e9), -1)], -1)
            if action_mask is not None:
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
        res = {"policy_logits": policy_logits, "opp_action_logits": opp, "value": self.value_head(latent)}
        if self.hidden_head:
            res.update(hid_move=self.hid_move(enc["opp_mon"]), hid_item=self.hid_item(enc["opp_mon"]), hid_ability=self.hid_ability(enc["opp_mon"]))
        return res

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
