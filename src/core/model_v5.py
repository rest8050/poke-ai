"""v5 = v4 + 합법성 입력. 어떤 행동이 실제로 가능한지(구애·돌격조끼 고정, 도발·앵콜·사슬묶기, PP 소진, 교체 봉쇄)는 교사가 교체를 고르는 큰 이유인데,
v4는 이 정보를 입력으로 못 받음 (마스크는 로짓을 지우는 데만 쓰임). v5는 서버 요청 기반 순수 합법 마스크(legal_mask: 휴리스틱 가지치기 없음)를 특징으로 받음.
- 토큰: 내 활성 포켓몬의 기술 토큰마다 "사용 가능" 플래그, 내 포켓몬 토큰마다 "교체 가능" 플래그 (어텐션이 직접 봄)
- 요약(LEG_G=25): 합법 비트 14 + 합법 기술 수·공격기 수·교체 수·테라 가능·"기술 한 개만 가능"·"고정 의심" 6 + 사용 가능한 기술만으로 다시 계산한 공격 요약 3
  (배율·데미지·KO) + 고정으로 잃은 화력 2 (전체 최선 - 가능한 최선). 융합층·종류 헤드·교체 점수에 입력
- legal_mask를 안 주면 action_mask를 씀 (학습 데이터는 둘이 같음). 배포에서는 encode_battle의 legal_mask(합법성만)를 특징으로, action_mask(가지치기 포함)를 로짓 마스킹으로 따로 넘김
  -> 가지치기로 꺼진 비트가 '고정'으로 오해되지 않음
arch: {"model": "v5", "my_pos": false}"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.belief import STAT_SCALE
from src.core.model import DeepPokemonBattleTransformerNet
from src.core.model_v3 import BASE_IN, MV_EXTRA, N_TOK
from src.core.model_v4 import CAND4, MON_EXTRA4, MULT_RANGE, EntityPokemonNetV4

LEG_G = 14 + 6 + 3 + 2


class EntityPokemonNetV5(EntityPokemonNetV4):
    uses_legal = True
    MON_X, CAND_N, G_DIM = MON_EXTRA4, CAND4, LEG_G            # 하위 모델이 포켓몬 토큰 추가 수치 / 교체 후보 특징 / 전역 요약의 폭을 늘림

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cfg["model"] = "v5"
        d, hw, ld, hd = self.d, self.cfg["head_width"], self.cfg["latent_dim"], self.cfg["history_dim"]
        self.mon_proj = nn.Sequential(nn.Linear(BASE_IN + self.MON_X + 1 + 1, d), nn.LayerNorm(d))              # + 교체 가능 플래그
        self.move_proj = nn.Sequential(nn.Linear(142 + MV_EXTRA + 1, d), nn.LayerNorm(d))                      # + 사용 가능 플래그
        self.fusion = nn.Sequential(nn.Linear(d + hd + self.G_DIM, ld * 2), nn.LayerNorm(ld * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(ld * 2, ld), nn.LayerNorm(ld), nn.GELU())
        self.switch_score = nn.Sequential(nn.Linear(ld + d + self.CAND_N + self.G_DIM + 1, hw), nn.GELU(), nn.Linear(hw, 1))
        self.type_head = nn.Sequential(nn.Linear(ld + self.CAND_N + 2 + self.G_DIM, 64), nn.GELU(), nn.Linear(64, 2))

    @staticmethod
    def _legal_summary(leg, off_act, n_rev):
        """leg [B,22] (0/1), off_act [B,4,4] 활성 포켓몬의 기술 4칸 x (배율, 데미지, KO, 공격기) vs 상대 활성, n_rev [B] 공개된 활성 기술 수 -> [B, LEG_G]"""
        mv = leg[:, :4]
        n_normal, n_attack, n_sw = mv.sum(1), (mv * off_act[..., 3]).sum(1), leg[:, 8:14].sum(1)
        one_move = ((n_normal == 1) & (n_rev >= 2)).float()
        lock = (n_rev - n_normal).clamp_min(0) / 4.0                                     # 기술이 몇 개 막혀 있나 (고정·도발·PP 소진 등)
        full = off_act[..., :3].amax(1)
        legal_off = torch.where(mv[..., None] > 0, off_act[..., :3], torch.full_like(off_act[..., :3], -1.0)).amax(1).clamp_min(0)
        lost = torch.stack([(full[:, 1] - legal_off[:, 1]).clamp_min(0), (full[:, 2] - legal_off[:, 2]).clamp_min(0)], -1)
        return torch.cat([leg[:, :14], torch.stack([n_normal / 4.0, n_attack / 4.0, n_sw / 5.0, leg[:, 4:8].amax(1), one_move, lock], -1), legal_off, lost], -1)

    def _global_feats(self, leg, off_act, n_rev, field_vec):
        return self._legal_summary(leg, off_act, n_rev)

    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec, legal=None):
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
        stat = belief_logits["stat"].detach().float()
        mult = torch.exp(STAT_SCALE * stat).clamp(*MULT_RANGE)
        guess = self._top_guess(belief_logits, opp_team_cat, opp_move_num, opp_team_num)
        D, O, H, mon_extra, mv_extra, cand, ar, act_m = self._matchup_feats4(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num,
                                                                             opp_move_num, guess, mult, stat, field_vec)
        opp_act = opp_team_num[:, :, 0].argmax(1)
        # ---- 합법성: 토큰 플래그 + 요약 ----
        leg = torch.ones(B, 22, device=dev) if legal is None else legal.float()
        is_act = torch.zeros(B, 6, dtype=torch.bool, device=dev)
        is_act[ar, act_m] = True
        mon_flag = torch.ones(B, 12, device=dev)
        mon_flag[:, :6] = torch.where(is_act, torch.ones_like(leg[:, 8:14]), leg[:, 8:14])          # 활성은 해당 없음(1), 나머지는 교체 가능 여부
        mv_flag = torch.ones(B, 12, 4, device=dev)
        mv_flag[ar, act_m] = leg[:, :4]                                                              # 내 활성 포켓몬의 기술 4칸만 합법성이 의미 있음
        off_act = O.view(B, 6, 6, 4, 4)[ar, opp_act][ar, act_m]                                      # [B,4,4] 내 활성의 기술 4칸이 상대 활성에게
        n_rev_act = (my_move_num[ar, act_m].abs().sum(-1) > 0).sum(-1).float()
        leg_g = self._global_feats(leg, off_act, n_rev_act, field_vec)
        mon_in = torch.cat(parts + [mon_extra.reshape(B * 12, self.MON_X), n_rev, mon_flag.reshape(B * 12, 1)], -1)
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn, mv_extra.reshape(B * 12, 4, 3), mv_flag.reshape(B * 12, 4, 1)], -1)
        side = torch.arange(12, device=dev) // 6
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] * self.pos_gate[None, :, None] + self.act_emb(active)
        g = torch.cat([guess["w"].sum(-1, keepdim=True), (guess["w"][..., None] * guess["pseudo"]).sum(2),
                       (guess["w"][..., None] * E._emb("move", guess["idx"]).detach()).sum(2)], -1)
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
            mask = (self._bias4(D, O, H, li).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m, "leg_g": leg_g, "leg_sw": leg[:, 8:14]}

    def _heads(self, enc, history, action_mask=None, opp_action_mask=None):
        B = history.size(0)
        lg = enc["leg_g"].to(history.dtype)
        latent = self.fusion(torch.cat([enc["cls"], history, lg], -1))
        cand = enc["cand"]
        ok = action_mask[:, 8:14] if action_mask is not None else torch.ones(B, 6, dtype=torch.bool, device=latent.device)
        act_feat = cand[torch.arange(B, device=latent.device), enc["act_m"]]
        best_off = torch.where(ok, cand[..., 4], torch.full_like(cand[..., 4], -1.0)).amax(1, keepdim=True).clamp_min(0)
        min_thr = torch.where(ok, cand[..., 7], torch.full_like(cand[..., 7], 9.0)).amin(1, keepdim=True)
        min_thr = torch.where(ok.any(1, keepdim=True), min_thr, torch.zeros_like(min_thr))
        mv_in = torch.cat([latent[:, None].expand(-1, 4, -1), enc["active_mv"], enc["active_mon"][:, None].expand(-1, 4, -1)], -1)
        sw_in = torch.cat([latent[:, None].expand(-1, 6, -1), enc["my_mon"], cand, lg[:, None].expand(-1, 6, -1), enc["leg_sw"][..., None].to(lg.dtype)], -1)
        mv, sw = self.move_score(mv_in), self.switch_score(sw_in).squeeze(-1)
        detail = torch.cat([mv[..., 0], mv[..., 1], sw, latent.new_full((B, 8), -1e9)], 1)
        type_logits = self.type_head(torch.cat([latent, act_feat, best_off, min_thr, lg], -1))
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
                "hid_move": bl["move"], "hid_item": bl["item"], "hid_ability": bl["ability"]}

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None, opp_action_mask=None, legal_mask=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                                legal=legal_mask if legal_mask is not None else action_mask)
        cur = self._turn_vec(enc).unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask, opp_action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None, legal_mask=None):
        enc = self._encode_turn(*obs, legal=legal_mask if legal_mask is not None else action_mask)
        tv = self._turn_vec(enc)
        T = int(time_index.max().item()) + 1
        turn_seq = tv.new_zeros(n_seq, T, tv.size(-1))
        turn_seq[seq_index, time_index] = tv
        return self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask, None)

    get_action = DeepPokemonBattleTransformerNet.get_action
    get_action_rl = DeepPokemonBattleTransformerNet.get_action_rl
