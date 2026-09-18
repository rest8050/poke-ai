import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


def load_vocab_sizes(vocab_path: str = "data/vocab.json"):
    defaults = {
        "num_species": 1500,
        "num_moves": 1050,
        "num_items": 700,
        "num_abilities": 400,
        "num_types": 30,
        "num_statuses": 16
    }
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
                defaults["num_species"] = len(vocab.get("species", {})) + 50
                defaults["num_moves"] = len(vocab.get("move", {})) + 50
                defaults["num_items"] = len(vocab.get("item", {})) + 50
                defaults["num_abilities"] = len(vocab.get("ability", {})) + 50
                defaults["num_types"] = len(vocab.get("type", {})) + 10
                defaults["num_statuses"] = len(vocab.get("status", {})) + 8
        except Exception:
            pass
    return defaults


def load_compatible(model: nn.Module, state: dict):
    """
    체크포인트 부분 로드.
    - shape가 같으면 그대로 복사
    - 입력 열만 늘어난 2D weight(끝에 새 피처를 붙인 Linear)는 기존 열 복사 + 새 열 0
      → 새 피처가 처음엔 무시되어 기존 정책이 그대로 유지되고, 학습하며 점차 사용
    return: (확장된 키, 버려진 키)
    """
    own = model.state_dict()
    expanded, skipped = [], []
    for k, v in state.items():
        if k not in own:
            skipped.append(k)
        elif own[k].shape == v.shape:
            own[k] = v
        elif v.dim() == 2 and own[k].shape[0] == v.shape[0] and own[k].shape[1] > v.shape[1]:
            w = torch.zeros_like(own[k])
            w[:, :v.shape[1]] = v
            own[k] = w
            expanded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    return expanded, skipped


class MoveEncoderBlock(nn.Module):
    """
    기술 임베딩(48d)과 기술 수치 데이터를 결합하는 인코더 (GELU)
    수치 44d: 위력, 명중률, PP비율, 우선도, 분류, 부가효과(구), 기술타입, 상대 액티브에게 실제 배율,
             + 부가효과 벡터 36칸 (tensor_encoder.MOVE_EFFECT_NAMES). 새 피처는 항상 끝에 추가.
    """
    def __init__(self, move_dim: int = 48, move_num_dim: int = 44):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(move_dim + move_num_dim, 64),
            nn.GELU(),
            nn.Linear(64, move_dim)
        )

    def forward(self, move_embs, move_numerics):
        # move_embs: [B, 4, 48], move_numerics: [B, 4, 44]
        combined = torch.cat([move_embs, move_numerics], dim=-1) # [B, 4, 92]
        out = self.fc(combined) # [B, 4, 48]
        return out


# 히스토리 트랜스포머의 턴 위치 임베딩 크기 (그 이상 긴 배틀은 마지막 위치로 고정)
MAX_TURNS = 256

# 스탯 기반 종(Species) 인코딩 차원
SPECIES_STAT_DIM = 7  # hp, atk, def, spa, spd, spe, weight (각 정규화)


class PokemonEmbeddingLayer(nn.Module):
    """
    스탯 수치 기반 종(Species) 인코딩 레이어.
    기존 nn.Embedding(ID 룩업) 방식 대신 base_stats 7개 수치를 Linear로 투영하여
    처음 보는 포켓몬에도 범용적으로 대응할 수 있도록 개선.
    """
    def __init__(self, 
                 vocab_path: str = "data/vocab.json",
                 species_dim: int = 64,
                 move_dim: int = 48,
                 item_dim: int = 32,
                 ability_dim: int = 32,
                 type_dim: int = 16,
                 status_dim: int = 8):
        super().__init__()
        
        sizes = load_vocab_sizes(vocab_path)

        # [핵심 변경] species_embed(ID룩업) → species_proj(스탯 수치 투영)
        # hp/255, atk/255, def/255, spa/255, spd/255, spe/255, weight/500 → 64d
        self.species_proj = nn.Sequential(
            nn.Linear(SPECIES_STAT_DIM, species_dim),
            nn.LayerNorm(species_dim),
            nn.GELU()
        )
        self.move_embed = nn.Embedding(sizes["num_moves"], move_dim, padding_idx=0)
        self.item_embed = nn.Embedding(sizes["num_items"], item_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(sizes["num_abilities"], ability_dim, padding_idx=0)
        self.type_embed = nn.Embedding(sizes["num_types"], type_dim, padding_idx=0)
        self.status_embed = nn.Embedding(sizes["num_statuses"], status_dim, padding_idx=0)
        
        # MoveEncoderBlock 연결
        self.move_encoder = MoveEncoderBlock(move_dim=move_dim)

    def forward(self, species_stats, item_id, ability_id, type1_id, type2_id, status_id, move_ids, move_numerics):
        """
        species_stats: [B, 7] float (hp/255, atk/255, def/255, spa/255, spd/255, spe/255, weight/500)
        move_ids: [B, 4]
        move_numerics: [B, 4, 44]
        """
        sp_emb = self.species_proj(species_stats)  # [B, 64]
        it_emb = self.item_embed(item_id)          # [B, 32]
        ab_emb = self.ability_embed(ability_id)    # [B, 32]
        t1_emb = self.type_embed(type1_id)         # [B, 16]
        t2_emb = self.type_embed(type2_id)         # [B, 16]
        st_emb = self.status_embed(status_id)      # [B, 8]
        
        # 기술 임베딩 + 수치 인코딩 결합
        raw_mv_emb = self.move_embed(move_ids)                        # [B, 4, 48]
        encoded_mv_emb = self.move_encoder(raw_mv_emb, move_numerics) # [B, 4, 48]
        mv_emb_flat = encoded_mv_emb.view(encoded_mv_emb.size(0), -1) # [B, 192]
        
        cat_features = torch.cat([sp_emb, it_emb, ab_emb, t1_emb, t2_emb, st_emb, mv_emb_flat], dim=-1)
        return cat_features # [B, 360]


class PositionalSlotEncoding(nn.Module):
    """
    팀 엔트리 슬롯(0~5번 슬롯) 고유 위치 정보 임베딩
    """
    def __init__(self, num_slots: int = 6, slot_dim: int = 16):
        super().__init__()
        self.slot_embed = nn.Embedding(num_slots, slot_dim)

    def forward(self, batch_size: int, device: torch.device):
        slot_ids = torch.arange(6, device=device).unsqueeze(0).repeat(batch_size, 1)
        return self.slot_embed(slot_ids) # [B, 6, 16]


class DeepPokemonBattleTransformerNet(nn.Module):
    """
    트랜스포머 AI (GELU 활성화 함수 개편 완료)
    """
    ACTION_DIM = 22

    def __init__(self, 
                 vocab_path: str = "data/vocab.json",
                 pokemon_embed_dim: int = 128,
                 history_dim: int = 256,
                 field_dim: int = 48):
        super().__init__()
        
        self.pokemon_embed_dim = pokemon_embed_dim
        self.action_dim = self.ACTION_DIM
        
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path)
        
        # 입력: 임베딩(360) + 배틀 수치(12) + 테라 타입(16) + 추론 플래그(4)
        # 새 피처는 반드시 끝에 붙임 (load_compatible이 기존 열을 보존하도록)
        self.pkmn_fc = nn.Sequential(
            nn.Linear(360 + 12 + 16 + 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, pokemon_embed_dim),
            nn.LayerNorm(pokemon_embed_dim)
        )
        
        encoder_layer_my = nn.TransformerEncoderLayer(
            d_model=pokemon_embed_dim, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu", batch_first=True
        )
        self.my_team_attention = nn.TransformerEncoder(encoder_layer_my, num_layers=2)
        
        encoder_layer_opp = nn.TransformerEncoderLayer(
            d_model=pokemon_embed_dim, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu", batch_first=True
        )
        self.opp_team_attention = nn.TransformerEncoder(encoder_layer_opp, num_layers=2)
        
        self.field_encoder = nn.Sequential(
            nn.Linear(field_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        
        # 히스토리: 배틀의 모든 턴 요약을 시퀀스로 보는 인과 트랜스포머 (이전 GRUCell 대체)
        # 턴 요약 = 아군액티브(128) + 상대액티브(128) + 아군팀요약(256) + 상대팀요약(256) + 필드(64) = 832
        turn_dim = pokemon_embed_dim * 2 + 256 * 2 + 64
        self.turn_proj = nn.Linear(turn_dim, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=history_dim, nhead=4, dim_feedforward=history_dim * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.history_seq = nn.TransformerEncoder(seq_layer, num_layers=2, enable_nested_tensor=False)
        # 출력 투영을 0으로 시작 → 처음엔 히스토리 = 0 벡터 (기존 체크포인트의 나머지 부분이 안정적으로 이어지도록)
        self.history_out = nn.Linear(history_dim, history_dim)
        nn.init.zeros_(self.history_out.weight)
        nn.init.zeros_(self.history_out.bias)

        # Fusion: 아군액티브(128) + 아군팀요약(256) + 상대액티브(128) + 상대팀요약(256) + 필드(64) + 히스토리(256) = 1088
        fusion_dim = pokemon_embed_dim + 256 + pokemon_embed_dim + 256 + 64 + history_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # 크로스 어텐션: 내 팀 슬롯(Query)이 상대 팀 전체(Key/Value)를 스캔
        self.cross_attn = nn.MultiheadAttention(pokemon_embed_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(pokemon_embed_dim)

        # 후보별 점수 헤드 (pointer 방식)
        # 기술 j: latent(256) + 액티브 기술 j 벡터(48) + 액티브 문맥(128) -> [일반, Z/테라]
        self.move_score = nn.Sequential(
            nn.Linear(256 + 48 + pokemon_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        # 교체 k: latent(256) + 크로스 어텐션 거친 k번 슬롯 벡터(128) -> 1
        self.switch_score = nn.Sequential(
            nn.Linear(256 + pokemon_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        # 계층 행동: 0=기술, 1=교체
        self.type_head = nn.Linear(256, 2)
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.opp_action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.ACTION_DIM)
        )
        self.opp_item_belief_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 50)
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self,
                my_team_cat, my_team_num, my_move_num,
                opp_team_cat, opp_team_num, opp_move_num,
                field_vec, history_state=None,
                action_mask=None, opp_action_mask=None):
        """
        한 턴 추론 (롤아웃/배포용). history_state = 이전 턴들의 요약 시퀀스 [B, t-1, 832] (첫 턴은 None).
        반환 history_state = 이번 턴까지의 시퀀스 → 다음 턴에 그대로 넘기면 됨.
        """
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num,
                                opp_team_cat, opp_team_num, opp_move_num, field_vec)
        cur = enc["turn_input"].unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        history = self._history(seq)[:, -1]
        out = self._heads(enc, history, action_mask, opp_action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None):
        """
        학습용: 여러 배틀의 모든 턴을 한 번에 계산. 각 턴은 같은 배틀의 이전 턴만 봄 (인과 마스크).
        obs: 한 턴 입력 7개 (각 [N, ...], N = 전체 턴 수)
        seq_index/time_index: [N] 각 턴의 배틀 번호(0..n_seq-1)와 배틀 내 턴 번호(0부터 빈틈 없이)
        """
        enc = self._encode_turn(*obs)
        turn_input = enc["turn_input"]
        T = int(time_index.max().item()) + 1
        turn_seq = turn_input.new_zeros(n_seq, T, turn_input.size(-1))
        turn_seq[seq_index, time_index] = turn_input
        history = self._history(turn_seq)[seq_index, time_index]
        return self._heads(enc, history, action_mask, None)

    def _history(self, turn_seq):
        """[B, T, 832] 턴 요약 시퀀스 → [B, T, 256] 각 시점의 히스토리 (미래 턴은 보지 않음)"""
        T = turn_seq.size(1)
        pos = torch.arange(T, device=turn_seq.device).clamp(max=MAX_TURNS - 1)
        x = self.turn_proj(turn_seq) + self.turn_pos(pos)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=turn_seq.device), diagonal=1)
        return self.history_out(self.history_seq(x, mask=causal))

    def _encode_turn(self,
                     my_team_cat, my_team_num, my_move_num,
                     opp_team_cat, opp_team_num, opp_move_num,
                     field_vec):
        """
        my_team_cat: [B, 6, 10]
        my_team_num: [B, 6, 23]
        my_move_num: [B, 6, 4, 44] (기술 수치)
        opp_team_cat: [B, 6, 10]
        opp_team_num: [B, 6, 23]
        opp_move_num: [B, 6, 4, 44]
        field_vec: [B, 48]
        """
        B = my_team_cat.size(0)
        device = my_team_cat.device
        
        # --- 패딩 마스크 생성 (base_stats HP가 0인 빈/미공개 슬롯은 True) ---
        # my_team_num[:, :, 12] = hp/255 (정규화된 기본 HP 스탯), 0이면 미사용 슬롯
        my_team_pad_mask = (my_team_num[:, :, 12] == 0.0)   # [B, 6] (Bool)
        opp_team_pad_mask = (opp_team_num[:, :, 12] == 0.0) # [B, 6] (Bool)

        # --- A. 내 팀 6마리 인코딩 ---
        my_pkmn_vectors = []
        my_active_vec = torch.zeros(B, self.pokemon_embed_dim, device=device)
        my_active_moves = torch.zeros(B, 4, 48, device=device)  # 액티브의 인코딩된 기술 4개

        for i in range(6):
            cat_i = my_team_cat[:, i, :]       # [B, 10] (item, ability, type1, type2, status, moves×4, tera)
            num_i = my_team_num[:, i, :]        # [B, 23] (battle 12 + species_stats 7 + 추론 플래그 4)
            mv_num_i = my_move_num[:, i, :, :]  # [B, 4, 44]

            # 배틀 수치 (앞 12): is_active, fainted, slot_pos, hp_frac, boosts×5, level, gimmick×2
            battle_num_i = num_i[:, :12]         # [B, 12]
            # 종 스탯 (뒤 7): hp/255, atk/255, def/255, spa/255, spd/255, spe/255, weight/500
            species_stats_i = num_i[:, 12:19]    # [B, 7]

            it, ab, t1, t2, st = cat_i[:, 0], cat_i[:, 1], cat_i[:, 2], cat_i[:, 3], cat_i[:, 4]
            mvs = cat_i[:, 5:9]

            cat_emb = self.embeddings(species_stats_i, it, ab, t1, t2, st, mvs, mv_num_i)  # [B, 360]
            tera_emb = self.embeddings.type_embed(cat_i[:, 9])  # [B, 16]
            full_i = torch.cat([cat_emb, battle_num_i, tera_emb, num_i[:, 19:23]], dim=-1)  # [B, 392]
            vec_i = self.pkmn_fc(full_i)

            # 빈 슬롯(HP stat == 0)인 경우 인코딩 벡터 0 처리
            is_empty = (species_stats_i[:, 0] == 0.0).unsqueeze(-1)  # [B, 1]
            vec_i = vec_i.masked_fill(is_empty, 0.0)

            my_pkmn_vectors.append(vec_i.unsqueeze(1))

            is_act = battle_num_i[:, 0].unsqueeze(-1)
            my_active_vec = my_active_vec + vec_i * is_act
            # cat_emb = sp64 + it32 + ab32 + t16 + t16 + st8 (=168) + 기술 4×48
            my_active_moves = my_active_moves + cat_emb[:, 168:].view(B, 4, 48) * is_act.unsqueeze(-1)

        my_team_stack = torch.cat(my_pkmn_vectors, dim=1) # [B, 6, 128]
        
        # --- B. 상대 팀 6마리 인코딩 ---
        opp_pkmn_vectors = []
        opp_active_vec = torch.zeros(B, self.pokemon_embed_dim, device=device)
        
        for i in range(6):
            cat_i = opp_team_cat[:, i, :]       # [B, 10]
            num_i = opp_team_num[:, i, :]        # [B, 23]
            mv_num_i = opp_move_num[:, i, :, :]  # [B, 4, 44]

            battle_num_i = num_i[:, :12]         # [B, 12]
            species_stats_i = num_i[:, 12:19]    # [B, 7]

            it, ab, t1, t2, st = cat_i[:, 0], cat_i[:, 1], cat_i[:, 2], cat_i[:, 3], cat_i[:, 4]
            mvs = cat_i[:, 5:9]

            cat_emb = self.embeddings(species_stats_i, it, ab, t1, t2, st, mvs, mv_num_i)
            tera_emb = self.embeddings.type_embed(cat_i[:, 9])  # [B, 16]
            full_i = torch.cat([cat_emb, battle_num_i, tera_emb, num_i[:, 19:23]], dim=-1)  # [B, 392]
            vec_i = self.pkmn_fc(full_i)

            is_empty = (species_stats_i[:, 0] == 0.0).unsqueeze(-1)
            vec_i = vec_i.masked_fill(is_empty, 0.0)

            opp_pkmn_vectors.append(vec_i.unsqueeze(1))

            is_act = battle_num_i[:, 0].unsqueeze(-1)
            opp_active_vec = opp_active_vec + vec_i * is_act

        opp_team_stack = torch.cat(opp_pkmn_vectors, dim=1) # [B, 6, 128]
        
        # --- C. Transformer Self-Attention (src_key_padding_mask 적용!) ---
        my_team_attended = self.my_team_attention(my_team_stack, src_key_padding_mask=my_team_pad_mask)
        opp_team_attended = self.opp_team_attention(opp_team_stack, src_key_padding_mask=opp_team_pad_mask)

        # --- C.2 크로스 어텐션 (내 팀 -> 상대 팀) ---
        # 상대 슬롯이 전부 마스킹되면 softmax가 NaN -> 그 샘플만 마스크 해제 (값은 0 벡터)
        cross_mask = opp_team_pad_mask & ~opp_team_pad_mask.all(dim=1, keepdim=True)
        cross, _ = self.cross_attn(my_team_attended, opp_team_attended, opp_team_attended,
                                   key_padding_mask=cross_mask, need_weights=False)
        my_team_attended = self.cross_norm(my_team_attended + cross)  # [B, 6, 128]
        my_active_ctx = (my_team_attended * my_team_num[:, :, 0:1]).sum(dim=1)  # [B, 128]

        # --- D. 마스크가 적용된 안전한 Mean + Max Pooling ---
        # D.1 내 팀 Masked Mean & Max Pooling
        my_valid_mask = (~my_team_pad_mask).unsqueeze(-1) # [B, 6, 1] True: 유효한 슬롯
        my_valid_counts = my_valid_mask.sum(dim=1).clamp(min=1.0) # [B, 1]
        
        my_team_masked_for_mean = my_team_attended.masked_fill(~my_valid_mask, 0.0)
        my_team_mean = my_team_masked_for_mean.sum(dim=1) / my_valid_counts # [B, 128]
        
        my_team_masked_for_max = my_team_attended.masked_fill(~my_valid_mask, -1e9)
        my_team_max = my_team_masked_for_max.max(dim=1).values.clamp(min=-100.0) # [B, 128]
        my_team_summary = torch.cat([my_team_mean, my_team_max], dim=-1) # [B, 256]
        
        # D.2 상대 팀 Masked Mean & Max Pooling
        opp_valid_mask = (~opp_team_pad_mask).unsqueeze(-1)
        opp_valid_counts = opp_valid_mask.sum(dim=1).clamp(min=1.0)
        
        opp_team_masked_for_mean = opp_team_attended.masked_fill(~opp_valid_mask, 0.0)
        opp_team_mean = opp_team_masked_for_mean.sum(dim=1) / opp_valid_counts
        
        opp_team_masked_for_max = opp_team_attended.masked_fill(~opp_valid_mask, -1e9)
        opp_team_max = opp_team_masked_for_max.max(dim=1).values.clamp(min=-100.0)
        opp_team_summary = torch.cat([opp_team_mean, opp_team_max], dim=-1) # [B, 256]
        
        # --- E. 필드 + 턴 요약 ---
        field_features = self.field_encoder(field_vec) # [B, 64]
        turn_input = torch.cat([my_active_vec, opp_active_vec, my_team_summary, opp_team_summary, field_features], dim=-1) # [B, 832]
        return {
            "turn_input": turn_input, "my_active_vec": my_active_vec, "my_team_summary": my_team_summary,
            "opp_active_vec": opp_active_vec, "opp_team_summary": opp_team_summary, "field_features": field_features,
            "my_team_attended": my_team_attended, "my_active_moves": my_active_moves, "my_active_ctx": my_active_ctx,
        }

    def _heads(self, enc, history, action_mask=None, opp_action_mask=None):
        """턴 인코딩 + 히스토리 [B, 256] → 정책/가치 등 출력 헤드"""
        my_active_vec, my_team_summary = enc["my_active_vec"], enc["my_team_summary"]
        opp_active_vec, opp_team_summary = enc["opp_active_vec"], enc["opp_team_summary"]
        field_features, my_team_attended = enc["field_features"], enc["my_team_attended"]
        my_active_moves, my_active_ctx = enc["my_active_moves"], enc["my_active_ctx"]
        B = my_active_vec.size(0)

        # --- F. 전황 통합 (Fusion) ---
        combined_all = torch.cat([
            my_active_vec, my_team_summary,
            opp_active_vec, opp_team_summary,
            field_features, history
        ], dim=-1) # [B, 1088]
        
        latent = self.fusion(combined_all) # [B, 256]
        
        # --- G. 출력 헤드 ---
        # G.1 후보별 세부 점수 (기술 0-3 일반, 4-7 Z/테라, 교체 8-13, 14-21 미사용)
        mv = self.move_score(torch.cat([
            latent.unsqueeze(1).expand(-1, 4, -1), my_active_moves,
            my_active_ctx.unsqueeze(1).expand(-1, 4, -1)
        ], dim=-1))                                                                   # [B, 4, 2]
        sw = self.switch_score(torch.cat([
            latent.unsqueeze(1).expand(-1, 6, -1), my_team_attended
        ], dim=-1)).squeeze(-1)                                                       # [B, 6]
        detail = torch.cat([mv[..., 0], mv[..., 1], sw, latent.new_full((B, 8), -1e9)], dim=1)  # [B, 22]

        # G.2 계층 결합: log π(a) = log π(유형) + log π(세부 | 유형)
        # 결과를 22칸 logit으로 돌려주므로 Categorical/학습 코드는 그대로 사용 가능
        type_logits = self.type_head(latent)                                          # [B, 2]
        if action_mask is not None:
            detail = detail.masked_fill(~action_mask, -1e9)
            no_move = ~(action_mask & ~self.is_switch).any(dim=1)
            no_switch = ~(action_mask & self.is_switch).any(dim=1)
            type_logits = type_logits.masked_fill(torch.stack([no_move, no_switch], dim=1), -1e9)
        lp_type = F.log_softmax(type_logits, dim=-1)
        lp_move = F.log_softmax(detail.masked_fill(self.is_switch, -1e9), dim=-1)
        lp_switch = F.log_softmax(detail.masked_fill(~self.is_switch, -1e9), dim=-1)
        policy_logits = torch.where(self.is_switch, lp_type[:, 1:2] + lp_switch, lp_type[:, 0:1] + lp_move)

        opp_action_logits = self.opp_action_head(latent)   # [B, 22]
        opp_item_logits = self.opp_item_belief_head(latent)# [B, 50]
        value = self.value_head(latent)                    # [B, 1]

        # --- H. 상대 행동 마스킹 ---
        if opp_action_mask is not None:
            opp_action_logits = opp_action_logits.masked_fill(~opp_action_mask, -1e9)
        
        return {
            "policy_logits": policy_logits,
            "opp_action_logits": opp_action_logits,
            "opp_item_logits": opp_item_logits,
            "value": value,
        }

    def get_action(self, my_team_cat, my_team_num, my_move_num, 
                   opp_team_cat, opp_team_num, opp_move_num, 
                   field_vec, history_state=None, action_mask=None):
        self.eval()
        with torch.no_grad():
            res = self.forward(my_team_cat, my_team_num, my_move_num,
                               opp_team_cat, opp_team_num, opp_move_num, 
                               field_vec, history_state, action_mask)
            
            p_logits = res["policy_logits"]
            p_probs = F.softmax(p_logits, dim=-1)
            action_idx = torch.argmax(p_probs, dim=-1).item()
            
            opp_p_probs = F.softmax(res["opp_action_logits"], dim=-1)
            val = res["value"].item()
            
        return action_idx, p_probs, opp_p_probs, val, res["history_state"]

    def get_action_rl(self, my_team_cat, my_team_num, my_move_num, 
                      opp_team_cat, opp_team_num, opp_move_num, 
                      field_vec, history_state=None, action_mask=None, deterministic=False):
        """
        PPO 강화학습을 위한 행동 샘플링 메서드.
        deterministic=True일 경우 가장 확률이 높은 행동을 무조건 선택 (완벽한 평가 모드)
        """
        self.eval()
        with torch.no_grad():
            res = self.forward(my_team_cat, my_team_num, my_move_num,
                               opp_team_cat, opp_team_num, opp_move_num, 
                               field_vec, history_state, action_mask)
            
            p_logits = res["policy_logits"]
            
            # Categorical 분포를 생성하기 전, 마스킹된 부분(-1e9)이 완벽히 무시되도록 처리
            dist = torch.distributions.Categorical(logits=p_logits)
            
            if deterministic:
                action = torch.argmax(p_logits, dim=-1)
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action)
            val = res["value"]
            
        return action.item(), log_prob, val, res["history_state"]


PokemonBattleNet = DeepPokemonBattleTransformerNet
