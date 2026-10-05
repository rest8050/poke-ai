"""엔티티 토큰 트랜스포머 (model_v2). 기존 입력 텐서를 그대로 쓰되, 한 턴을 토큰 64개로 펼쳐 전부 한꺼번에 어텐션:
  포켓몬 12 (내 6 + 상대 6) + 기술 48 (12마리 x 4) + 필드 3 + CLS 1.
- 기술도 별도 토큰 (주인 포켓몬의 쪽/슬롯/활성 여부 임베딩으로 소속을 알림) → '이 포켓몬이 어떤 기술을 가졌나'를 어텐션이 직접 읽음
- 출력 토큰을 압축하지 않고 헤드가 직접 읽음: 기술 점수 = 활성의 기술 토큰 + 활성 포켓몬 토큰, 교체 점수 = 후보 포켓몬 토큰,
  가치/상대 예측 = CLS(+히스토리). 히스토리는 턴마다의 CLS 출력을 인과 트랜스포머로 요약 (기존 model.py와 같은 방식)
- 인터페이스는 DeepPokemonBattleTransformerNet과 같음 (forward/forward_sequences/get_action, cfg + save_ckpt/model_from_ckpt)
선택: arch에 {"model": "entity"} (train_fp_distill.py --arch, model.build_model)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import MAX_TURNS, DeepPokemonBattleTransformerNet, PokemonEmbeddingLayer


class EntityPokemonNet(nn.Module):
    ACTION_DIM = 22

    def __init__(self, vocab_path: str = "data/vocab.json", d_model: int = 192, n_layers: int = 4, n_heads: int = 6,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 128, field_dim: int = 58, entity_features: bool = True, switch_ctx: bool = False):
        super().__init__()
        self.cfg = dict(model="entity", d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width, switch_ctx=switch_ctx)
        self.switch_ctx = switch_ctx
        d = self.d = d_model
        self.latent_dim = latent_dim
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        # 토큰 입력: 포켓몬 = 임베딩들(64+32+32+16+16+8) + 테라(16) + 배틀 수치(11) + 나머지 수치(9) = 204 / 기술 = 인코딩(48)+원 임베딩(48)+수치(46) = 142
        self.mon_proj = nn.Sequential(nn.Linear(204, d), nn.LayerNorm(d))
        self.move_proj = nn.Sequential(nn.Linear(142, d), nn.LayerNorm(d))
        self.field_proj = nn.Sequential(nn.Linear(field_dim, d), nn.GELU(), nn.Linear(d, 3 * d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.role_emb = nn.Embedding(4, d)      # 0 포켓몬 1 기술 2 필드 3 CLS
        self.side_emb = nn.Embedding(2, d)      # 0 내 쪽 1 상대 쪽
        self.slot_emb = nn.Embedding(6, d)      # 팀 슬롯
        self.mslot_emb = nn.Embedding(4, d)     # 기술 칸
        self.act_emb = nn.Embedding(2, d)       # 주인이 활성 포켓몬인가
        self.fld_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        # 히스토리: 턴별 CLS 출력의 인과 트랜스포머
        self.turn_proj = nn.Linear(d, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        hl = nn.TransformerEncoderLayer(history_dim, 4, history_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.history_seq = nn.TransformerEncoder(hl, hist_layers, enable_nested_tensor=False)
        self.history_out = nn.Linear(history_dim, history_dim)
        self.fusion = nn.Sequential(nn.Linear(d + history_dim, latent_dim * 2), nn.LayerNorm(latent_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.move_score = nn.Sequential(nn.Linear(latent_dim + 2 * d, head_width), nn.GELU(), nn.Linear(head_width, 2))
        # switch_ctx: 교체/기술 판단(type_head)과 교체 후보 점수가 상대 활성·내 활성·(후보와 상대 활성의 곱)을 직접 봄
        #   type_head 입력 = [latent, 내 활성, 상대 활성, 교체 가능 후보 중 최댓값 풀링] / 후보 점수 입력 = [latent, 후보, 상대 활성, 내 활성, 후보*상대 활성]
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + (4 if switch_ctx else 1) * d, head_width), nn.GELU(), nn.Linear(head_width, 1))
        self.type_head = nn.Linear(latent_dim + (3 * d if switch_ctx else 0), 2)
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.opp_action_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, self.ACTION_DIM))
        self.value_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

    # ---- 토큰 구성 + 트렁크 ----
    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec):
        B, dev, d = my_team_cat.size(0), my_team_cat.device, self.d
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        mon_in = torch.cat([E.species_proj(num[:, 11:18]), E._emb("item", cat[:, 0]), E._emb("ability", cat[:, 1]), E.type_embed(cat[:, 2]),
                            E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:]], -1)
        raw = E._emb("move", cat[:, 5:9])                                   # [B*12, 4, 48]
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn], -1)         # [B*12, 4, 142]
        side = torch.arange(12, device=dev) // 6                            # 0 내 쪽, 1 상대 쪽
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] + self.act_emb(active)             # [B,12,d]
        mon = self.mon_proj(mon_in).view(B, 12, d) + base + self.role_emb.weight[0]
        mv = self.move_proj(mv_in).view(B, 12, 4, d) + base[:, :, None] + self.mslot_emb.weight[None, None] + self.role_emb.weight[1]
        fld = self.field_proj(field_vec).view(B, 3, d) + self.fld_emb.weight[None] + self.role_emb.weight[2]
        cls = self.cls.expand(B, 1, d) + self.role_emb.weight[3]
        x = torch.cat([mon, mv.reshape(B, 48, d), fld, cls], 1)                                          # [B,64,d]
        mon_pad = (num[:, 11] == 0).view(B, 12)                                                          # 빈 슬롯 (HP 종족값 0)
        mv_pad = ((mvn.abs().sum(-1) == 0).view(B, 12, 4)) | mon_pad[:, :, None]                       # 안 드러난 기술칸
        pad = torch.cat([mon_pad, mv_pad.reshape(B, 48), mon_pad.new_zeros(B, 4)], 1)
        out = self.trunk(x, src_key_padding_mask=pad)
        my_mon = out[:, :6]
        a = my_team_num[:, :, 0].argmax(1)
        ar = torch.arange(B, device=dev)
        my_mv = out[:, 12:60].view(B, 12, 4, d)[:, :6]
        oa = opp_team_num[:, :, 0].argmax(1)
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, a], "active_mv": my_mv[ar, a], "opp_active_mon": out[:, 6:12][ar, oa]}

    def _history(self, turn_seq):
        B, T = turn_seq.size(0), turn_seq.size(1)
        pos = torch.arange(T, device=turn_seq.device).clamp(max=MAX_TURNS - 1)
        x = self.turn_proj(turn_seq) + self.turn_pos(pos)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=turn_seq.device), diagonal=1)
        return self.history_out(self.history_seq(x, mask=causal))

    def _heads(self, enc, history, action_mask=None, opp_action_mask=None):
        B = history.size(0)
        latent = self.fusion(torch.cat([enc["cls"], history], -1))
        mv_in = torch.cat([latent[:, None].expand(-1, 4, -1), enc["active_mv"], enc["active_mon"][:, None].expand(-1, 4, -1)], -1)
        sw_in = torch.cat([latent[:, None].expand(-1, 6, -1), enc["my_mon"]], -1)
        type_in = latent
        if self.switch_ctx:
            act, opp = enc["active_mon"], enc["opp_active_mon"]
            ok = action_mask[:, 8:14] if action_mask is not None else torch.ones(B, 6, dtype=torch.bool, device=latent.device)
            bench = enc["my_mon"].masked_fill(~ok[..., None], -1e4).max(1).values
            bench = torch.where(ok.any(1, keepdim=True), bench, torch.zeros_like(bench))          # 교체 가능한 후보가 없으면 0
            type_in = torch.cat([latent, act, opp, bench], -1)
            sw_in = torch.cat([sw_in, opp[:, None].expand(-1, 6, -1), act[:, None].expand(-1, 6, -1), enc["my_mon"] * opp[:, None]], -1)
        mv, sw = self.move_score(mv_in), self.switch_score(sw_in).squeeze(-1)
        detail = torch.cat([mv[..., 0], mv[..., 1], sw, latent.new_full((B, 8), -1e9)], 1)
        type_logits = self.type_head(type_in)
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
        return {"policy_logits": policy_logits, "opp_action_logits": opp, "value": self.value_head(latent)}

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None, opp_action_mask=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec)
        cur = enc["cls"].unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask, opp_action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None):
        enc = self._encode_turn(*obs)
        T = int(time_index.max().item()) + 1
        turn_seq = enc["cls"].new_zeros(n_seq, T, self.d)
        turn_seq[seq_index, time_index] = enc["cls"]
        return self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask, None)

    get_action = DeepPokemonBattleTransformerNet.get_action
    get_action_rl = DeepPokemonBattleTransformerNet.get_action_rl
