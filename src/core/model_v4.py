"""엔티티 토큰 트랜스포머 v4 = v3 + 신념 활용 방식 변경 (model_v3.py의 나머지 구조/인터페이스는 그대로).
1) 숨은 기술을 "슬롯당 추측 기술 하나"가 아니라 마리당 확률 상위 12개 후보의 혼합으로 취급:
   후보의 세트 포함 확률 w(= 미공개 칸 수 x 확률, 최대 1)로 포켓몬 쌍(내 i, 상대 o)마다 교체 판단에 필요한 두 값을 계산:
   P_KO = 1 - Π(1 - w·KO) (숨은 기술 중 하나라도 확정 KO일 확률), 그럴듯한 최대 데미지 = w >= 0.15인 후보의 최대 데미지(확률 가중 없음)
   → 포켓몬 토큰 요약, 교체 후보 특징, 포켓몬<->포켓몬 어텐션 편향에 반영. (확률x크기로 곱하면 "30%의 확정 1타"가 0.3짜리 위협으로 뭉개지므로 분리)
   미공개 기술 토큰은 "모름" 표시만 유지하고, 후보 기술의 수치/임베딩 요약은 상대 포켓몬 토큰에 더함
2) 노력치/성격: 신념이 상대 능력치 배율(stat 출력, 표준 가정 252/무보정 대비)을 예측 → 매치업의 상대 능력치(데미지/KO/스피드)에 곱하고 상대 포켓몬 토큰에도 입력
   (팀 데이터 사전학습으로만 학습, 정책/가치 손실은 신념에 닿지 않음 — 출력 전부 detach)
3) my_pos=False: 내 쪽 팀 슬롯/기술 칸 위치 임베딩을 끔 → 내 팀 순서 순열에 대해 정확히 대칭(순서가 팀 지문 통로가 되는 것을 막음). 상대 쪽은 공개된 순서라 유지
arch: {"model": "v4"}"""
import torch
import torch.nn as nn

from src.core.belief import STAT_SCALE
from src.core.model_v3 import BASE_IN, EntityPokemonNetV3, N_TOK, _mx
from src.core.tensor_encoder import MOVE_NUM_DIM

K_GUESS = 12            # 마리당 후보 기술 수 (숨은 기술 재현율: 상위 4 0.73 / 상위 8 0.94 / 상위 12 0.98)
PLAUSIBLE = 0.15        # 이 이상의 세트 포함 확률이면 "그럴듯한" 후보
MON_EXTRA4 = 7 + 2 + 6  # v3의 7 + 숨은 기술 위협(P_KO, 그럴듯한 최대 데미지) + 상대 능력치 예측 6
CAND4 = 9 + 2           # v3의 9 + 후보 i에게 상대 활성의 숨은 기술 위협 2
MULT_RANGE = (0.6, 1.25)


class EntityPokemonNetV4(EntityPokemonNetV3):
    def __init__(self, my_pos: bool = True, **kw):
        super().__init__(**kw)
        self.cfg = {**{k: v for k, v in self.cfg.items() if k != "version"}, "model": "v4", "my_pos": my_pos}
        pos = torch.ones(12) if my_pos else torch.tensor([0.0] * 6 + [1.0] * 6)       # 내 쪽 슬롯/기술 칸 순서는 임의(팀 export 순서)라서 끌 수 있음. 상대 쪽 순서는 공개된 순서라 정보가 있어 유지
        self.register_buffer("pos_gate", pos, persistent=False)
        d, hw, ld = self.d, self.cfg["head_width"], self.cfg["latent_dim"]
        del self.guess_proj
        self.hid_proj = nn.Linear(1 + MOVE_NUM_DIM + 48, d)                 # 후보 기술 혼합(확률 합, 수치, 임베딩) -> 상대 포켓몬 토큰
        self.mon_proj = nn.Sequential(nn.Linear(BASE_IN + MON_EXTRA4 + 1, d), nn.LayerNorm(d))
        self.switch_score = nn.Sequential(nn.Linear(ld + d + CAND4, hw), nn.GELU(), nn.Linear(hw, 1))
        self.type_head = nn.Sequential(nn.Linear(ld + CAND4 + 2, 64), nn.GELU(), nn.Linear(64, 2))
        self.bias_hid = nn.ModuleList([nn.Linear(2, self.n_heads) for _ in self.layers])     # 숨은 기술 위협 -> 포켓몬<->포켓몬 어텐션 편향
        for lin in self.bias_hid:
            nn.init.normal_(lin.weight, std=0.5)
            nn.init.zeros_(lin.bias)

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

    def _matchup_feats4(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat):
        B, K = my_cat.size(0), K_GUESS
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, mult)
            D, O = M["def"], M["off"]                                              # 공개된 기술만 [B,6,24,4]
            Dp = self.matchup._pair(opp_cat, opp_num, guess["pseudo"], False, my_cat, my_num, True, att_mult=mult)   # [B,6(내),6K,4]
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

    def _bias4(self, D, O, H, layer):
        b = self._bias(D, O, layer)
        fh = self.bias_hid[layer](H).permute(0, 3, 1, 2)                           # [B,heads,6(내),6(상대)]
        b[:, :, 0:6, 6:12] = fh
        b[:, :, 6:12, 0:6] = fh.transpose(2, 3)
        return b

    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec):
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
                                                                             opp_move_num, guess, mult, stat)
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
            mask = (self._bias4(D, O, H, li).to(x.dtype) + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        opp_act = opp_team_num[:, :, 0].argmax(1)
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m}
