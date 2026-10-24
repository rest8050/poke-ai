"""v4 구조 호환 전용 (서버에 탑재된 v4_full_tv 로드용). v4_full_tv가 새 구조 모델로 교체되어 퇴역하면 이 파일, tests/test_legacy_v4.py, build_model의 v4 분기,
Matchup의 entry 옵션(entry=False 경로)을 같이 삭제 (현행 구조는 model_v6.py 하나로 독립)
- 한 턴 = 포켓몬 12 + 기술 48 + 필드 3 + CLS 1 = 64 토큰, d_model 256 / 6층 / 8헤드, 매치업/숨은 기술 위협 기반 어텐션 편향, 히스토리 인과 트랜스포머 2층
- 숨은 기술은 마리당 확률 상위 12개 후보의 혼합으로 취급(P_KO, 그럴듯한 최대 데미지), 신념이 예측한 상대 능력치 배율을 매치업에 곱함 (신념 출력 전부 detach)
- field_vec은 58열까지만 읽음 (새 열은 잘라냄), 합법 마스크는 입력으로 안 받음 (로짓 마스킹에만 씀)
arch: {"model": "v4", "my_pos": false}"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.belief import STAT_SCALE, SetBelief, build_move_table
from src.core.matchup import Matchup
from src.core.model import MAX_TURNS, DeepPokemonBattleTransformerNet, PokemonEmbeddingLayer
from src.core.tensor_encoder import MOVE_NUM_DIM

# 3차원 어텐션 마스크(헤드별 편향)는 추론용 빠른 커널에서 학습 경로와 값이 달라짐(CLS 기준 2e-3) → 학습과 같은 일반 경로만 쓰도록 끔
torch.backends.mha.set_fastpath_enabled(False)

N_TOK = 64            # 포켓몬 12 + 기술 48 + 필드 3 + CLS 1
MV_EXTRA = 3
BASE_IN = 204         # 포켓몬 토큰 기본 입력 차원 (종족값 투영 64 + 도구/특성/타입/상태/테라 임베딩 + 배틀 수치 + 이벤트 증거)
FIELD_DIM_V4 = 58
K_GUESS = 12          # 마리당 후보 기술 수 (숨은 기술 재현율: 상위 4 0.73 / 상위 8 0.94 / 상위 12 0.98)
PLAUSIBLE = 0.15      # 이 이상의 세트 포함 확률이면 "그럴듯한" 후보
MON_EXTRA4 = 7 + 2 + 6   # 매치업 요약 7 + 숨은 기술 위협(P_KO, 그럴듯한 최대 데미지) + 상대 능력치 예측 6
CAND4 = 9 + 2            # 후보 요약 9 + 후보 i에게 상대 활성의 숨은 기술 위협 2
MULT_RANGE = (0.6, 1.25)


def _mx(x):           # [..., 4(기술), 4(채널)] 중 앞 3채널(배율, 데미지, KO)의 기술별 최댓값 → [..., 3]
    return x[..., :3].amax(-2)


class EntityPokemonNetV4Legacy(nn.Module):
    ACTION_DIM = 22

    def __init__(self, vocab_path: str = "data/vocab.json", my_pos: bool = True, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 192, entity_features: bool = True,
                 species_dim: int = 32, belief_hidden: int = 256, belief_species_dropout: float = 0.3):
        super().__init__()
        self.cfg = dict(model="v4", d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width,
                        species_dim=species_dim, belief_hidden=belief_hidden, belief_species_dropout=belief_species_dropout, my_pos=my_pos)
        d = self.d = d_model
        field_dim = FIELD_DIM_V4
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        pos = torch.ones(12) if my_pos else torch.tensor([0.0] * 6 + [1.0] * 6)       # 내 쪽 슬롯/기술 칸 순서는 임의(팀 export 순서)라서 끌 수 있음. 상대 쪽 순서는 공개된 순서라 정보가 있어 유지
        self.register_buffer("pos_gate", pos, persistent=False)
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        self.matchup = Matchup(vocab_path)
        self.belief = SetBelief(vocab_path, species_dim, belief_hidden, belief_species_dropout)
        self.register_buffer("move_table", build_move_table(vocab_path), persistent=False)
        self.unk_move = nn.Parameter(torch.zeros(d))                        # 미공개 기술 칸 표시
        self.hid_proj = nn.Linear(1 + MOVE_NUM_DIM + 48, d)                 # 후보 기술 혼합(확률 합, 수치, 임베딩) -> 상대 포켓몬 토큰
        self.mon_proj = nn.Sequential(nn.Linear(BASE_IN + MON_EXTRA4 + 1, d), nn.LayerNorm(d))
        self.move_proj = nn.Sequential(nn.Linear(142 + MV_EXTRA, d), nn.LayerNorm(d))
        self.field_proj = nn.Sequential(nn.Linear(field_dim, d), nn.GELU(), nn.Linear(d, 3 * d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.role_emb, self.side_emb, self.slot_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(6, d)
        self.mslot_emb, self.act_emb, self.fld_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(3, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout=dropout, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        # 층별 어텐션 편향: 입력 4채널(배율, 데미지, KO, 공격기) -> 헤드별 값 / 숨은 기술 위협 2채널 -> 헤드별 값. 작게 시작
        self.bias_def = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_layers)])
        self.bias_off = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_layers)])
        self.bias_hid = nn.ModuleList([nn.Linear(2, n_heads) for _ in range(n_layers)])
        for lin in list(self.bias_def) + list(self.bias_off) + list(self.bias_hid):
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
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + d + CAND4, head_width), nn.GELU(), nn.Linear(head_width, 1))
        self.type_head = nn.Sequential(nn.Linear(latent_dim + CAND4 + 2, 64), nn.GELU(), nn.Linear(64, 2))
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.value_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

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

    def _bias(self, D, O, H, layer):
        """어텐션 편향 [B,H,64,64]: 내 포켓몬<->상대 기술 토큰(36..59), 상대 포켓몬<->내 기술 토큰(12..35), 내 포켓몬<->상대 포켓몬(숨은 기술 위협)"""
        B, NH = D.size(0), self.n_heads
        fd = self.bias_def[layer](D).permute(0, 3, 1, 2)                       # [B,H,6,24]
        fo = self.bias_off[layer](O).permute(0, 3, 1, 2)
        b = D.new_zeros(B, NH, N_TOK, N_TOK)
        b[:, :, 0:6, 36:60] = fd
        b[:, :, 36:60, 0:6] = fd.transpose(2, 3)
        b[:, :, 6:12, 12:36] = fo
        b[:, :, 12:36, 6:12] = fo.transpose(2, 3)
        fh = self.bias_hid[layer](H).permute(0, 3, 1, 2)                       # [B,heads,6(내),6(상대)]
        b[:, :, 0:6, 6:12] = fh
        b[:, :, 6:12, 0:6] = fh.transpose(2, 3)
        return b

    def _top_guess(self, logits, opp_cat, opp_mv, opp_num):
        """신념의 기술 확률 -> 마리당 상위 K 후보. idx/w [B,6,K] (w = 세트 포함 확률 근사, 미공개 칸이 없으면 0), pseudo [B,6,K,46] (기울기 없음)"""
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
            n_unrev = ((opp_mv.abs().sum(-1) == 0) & present[..., None]).sum(-1, keepdim=True).float()      # [B,6,1]
            topv, idx = prob.topk(K_GUESS, -1)
            w = (topv * n_unrev).clamp(max=1.0)                                      # 미공개 칸이 n개면 각 기술이 세트에 있을 확률 ~ n x 확률
            pseudo = self.move_table[idx] * (w > 0)[..., None]
        return {"idx": idx, "w": w, "pseudo": pseudo}

    def _matchup_feats4(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat, field_vec=None):
        B, K = my_cat.size(0), K_GUESS
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            ent = self.matchup.entry_frac(my_cat, my_num, field_vec)               # 장판 진입 피해 (v6~, 아니면 None)
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, mult, entry=ent)
            D, O = M["def"], M["off"]                                              # 공개된 기술만 [B,6,24,4]
            Dp = self.matchup._pair(opp_cat, opp_num, guess["pseudo"], False, my_cat, my_num, True, att_mult=mult, entry=ent)   # [B,6(내),6K,4]
            w = guess["w"][:, None]                                                # [B,1,6(상대 o),K]
            dmg, ko = Dp[..., 1].view(B, 6, 6, K), Dp[..., 2].view(B, 6, 6, K)     # 후보별 데미지 비율/확정 KO [B,6(내 i),6(상대 o),K]
            H = torch.stack([1 - (1 - w * ko).prod(-1), (dmg * (w >= PLAUSIBLE)).amax(-1)], -1)   # [B,6,6,2] P_KO, 그럴듯한 최대 데미지
        act_m, act_o = my_num[:, :, 0].argmax(1), opp_num[:, :, 0].argmax(1)
        DS, OS = D.view(B, 6, 6, 4, 4), O.view(B, 6, 6, 4, 4)
        def_opp_act = DS[ar, :, act_o]                                             # [B,6(i),4,4] 상대 활성의 공개 기술이 내 포켓몬 i에게
        off_vs_opp_act = OS[ar, act_o]                                             # [B,6(i),4,4] 내 포켓몬 i의 기술이 상대 활성에게
        thr_on_my_act = DS[ar, act_m]                                              # [B,6(o),4,4] 상대 포켓몬 o의 공개 기술이 내 활성에게
        my_act_on_opp = OS[ar, :, act_m]                                           # [B,6(o),4,4] 내 활성 기술이 상대 포켓몬 o에게
        h_my, h_opp = H[ar, :, act_o], H[ar, act_m]                                # 상대 활성의 숨은 기술 -> 내 i / 상대 o의 숨은 기술 -> 내 활성 [B,6,6]
        mon_extra = torch.cat([
            torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], h_my, torch.zeros_like(stat)], -1),
            torch.cat([_mx(thr_on_my_act), _mx(my_act_on_opp), M["spd_opp"][..., None], h_opp, stat], -1)], 1)          # [B,12,15]
        mv_extra = torch.cat([off_vs_opp_act[..., :3], thr_on_my_act[..., :3]], 1)                                      # [B,12,4,3]
        bench_thr = DS[..., :3].amax((2, 3))[..., 1:]                                                                   # 후보 i가 상대 전체의 공개 기술에 맞는 최대 데미지/KO
        cand = torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], bench_thr, h_my], -1)         # [B,6,11]
        return D, O, H, mon_extra, mv_extra, cand, ar, act_m

    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec):
        field_vec = self._fit_field(field_vec)
        B, dev, d = my_team_cat.size(0), my_team_cat.device, self.d
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        parts = [E.species_proj(num[:, 11:18]), E._emb("item", cat[:, 0]), E._emb("ability", cat[:, 1]), E.type_embed(cat[:, 2]),
                 E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:]]
        raw = E._emb("move", cat[:, 5:9])
        revealed = (mvn.abs().sum(-1) > 0)
        n_rev = revealed.float().sum(-1, keepdim=True) / 4.0
        belief_logits = self.belief(opp_team_cat, opp_team_num)
        stat = belief_logits["stat"].detach().float()                              # [B,6,6] 상대 능력치 배율 예측 (STAT_SCALE 단위 로그)
        mult = torch.exp(STAT_SCALE * stat).clamp(*MULT_RANGE)
        guess = self._top_guess(belief_logits, opp_team_cat, opp_move_num, opp_team_num)
        D, O, H, mon_extra, mv_extra, cand, ar, act_m = self._matchup_feats4(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num,
                                                                             opp_move_num, guess, mult, stat, field_vec)
        mon_in = torch.cat(parts + [mon_extra.reshape(B * 12, MON_EXTRA4), n_rev], -1)
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn, mv_extra.reshape(B * 12, 4, 3)], -1)
        side = torch.arange(12, device=dev) // 6
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] * self.pos_gate[None, :, None] + self.act_emb(active)
        g = torch.cat([guess["w"].sum(-1, keepdim=True), (guess["w"][..., None] * guess["pseudo"]).sum(2),
                       (guess["w"][..., None] * E._emb("move", guess["idx"]).detach()).sum(2)], -1)      # 후보 기술 혼합 요약 [B,6,1+46+48]
        mon = self.mon_proj(mon_in).view(B, 12, d) + base + self.role_emb.weight[0]
        mon = mon + torch.cat([torch.zeros_like(mon[:, :6]), self.hid_proj(g)], 1)
        mv = self.move_proj(mv_in).view(B, 12, 4, d) + base[:, :, None] + self.mslot_emb.weight[None, None] * self.pos_gate[None, :, None, None] + self.role_emb.weight[1]
        fld = self.field_proj(field_vec).view(B, 3, d) + self.fld_emb.weight[None] + self.role_emb.weight[2]
        cls = self.cls.expand(B, 1, d) + self.role_emb.weight[3]
        x = torch.cat([mon, mv.reshape(B, 48, d), fld, cls], 1)
        mon_pad = (num[:, 11] == 0).view(B, 12)
        unrevealed = ~revealed.view(B, 12, 4) & ~mon_pad[:, :, None]
        x = torch.cat([x[:, :12], x[:, 12:60] + unrevealed.reshape(B, 48, 1) * self.unk_move, x[:, 60:]], 1)
        pad = torch.cat([mon_pad, mon_pad[:, :, None].expand(B, 12, 4).reshape(B, 48), mon_pad.new_zeros(B, 4)], 1)
        pad_f = torch.zeros(B, 1, 1, N_TOK, device=dev, dtype=x.dtype).masked_fill(pad[:, None, None, :], float("-inf"))
        for li, layer in enumerate(self.layers):
            mask = (self._bias(D, O, H, li).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        opp_act = opp_team_num[:, :, 0].argmax(1)
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m}

    def _heads(self, enc, history, action_mask=None):
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
        bl = enc["belief_logits"]
        return {"policy_logits": policy_logits, "value": self.value_head(latent),
                "hid_move": bl["move"], "hid_item": bl["item"], "hid_ability": bl["ability"]}      # hid_*: 숨김정보 손실 입력

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec)
        cur = self._turn_vec(enc).unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None):
        enc = self._encode_turn(*obs)
        tv = self._turn_vec(enc)
        T = int(time_index.max().item()) + 1
        turn_seq = tv.new_zeros(n_seq, T, tv.size(-1))
        turn_seq[seq_index, time_index] = tv
        return self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask)

    get_action = DeepPokemonBattleTransformerNet.get_action
