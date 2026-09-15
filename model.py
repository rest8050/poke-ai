import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


def load_vocab_sizes(vocab_path: str = "vocab.json"):
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


class MoveEncoderBlock(nn.Module):
    """
    기술 임베딩(48d)과 기술 수치 데이터(위력, 명중률, PP비율, 우선도, 물리/특수/변화 분류, 부가효과, 기술타입 - 7d)를 정밀 결합하는 인코더 (GELU)
    """
    def __init__(self, move_dim: int = 48, move_num_dim: int = 7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(move_dim + move_num_dim, 64),
            nn.GELU(),
            nn.Linear(64, move_dim)
        )

    def forward(self, move_embs, move_numerics):
        # move_embs: [B, 4, 48], move_numerics: [B, 4, 7]
        combined = torch.cat([move_embs, move_numerics], dim=-1) # [B, 4, 55]
        out = self.fc(combined) # [B, 4, 48]
        return out


class PokemonEmbeddingLayer(nn.Module):
    """
    기술 수치 데이터(MoveEncoderBlock)가 반영된 정적 사전 임베딩 레이어
    """
    def __init__(self, 
                 vocab_path: str = "vocab.json",
                 species_dim: int = 64,
                 move_dim: int = 48,
                 item_dim: int = 32,
                 ability_dim: int = 32,
                 type_dim: int = 16,
                 status_dim: int = 8):
        super().__init__()
        
        sizes = load_vocab_sizes(vocab_path)
        
        self.species_embed = nn.Embedding(sizes["num_species"], species_dim, padding_idx=0)
        self.move_embed = nn.Embedding(sizes["num_moves"], move_dim, padding_idx=0)
        self.item_embed = nn.Embedding(sizes["num_items"], item_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(sizes["num_abilities"], ability_dim, padding_idx=0)
        self.type_embed = nn.Embedding(sizes["num_types"], type_dim, padding_idx=0)
        self.status_embed = nn.Embedding(sizes["num_statuses"], status_dim, padding_idx=0)
        
        # MoveEncoderBlock 연결
        self.move_encoder = MoveEncoderBlock(move_dim=move_dim)

    def forward(self, species_id, item_id, ability_id, type1_id, type2_id, status_id, move_ids, move_numerics):
        """
        move_ids: [B, 4]
        move_numerics: [B, 4, 3] (base_power/200, accuracy, pp_ratio)
        """
        sp_emb = self.species_embed(species_id) # [B, 64]
        it_emb = self.item_embed(item_id)       # [B, 32]
        ab_emb = self.ability_embed(ability_id) # [B, 32]
        t1_emb = self.type_embed(type1_id)      # [B, 16]
        t2_emb = self.type_embed(type2_id)      # [B, 16]
        st_emb = self.status_embed(status_id)   # [B, 8]
        
        # 기술 임베딩 + 수치 인코딩 결합
        raw_mv_emb = self.move_embed(move_ids)                     # [B, 4, 48]
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
                 vocab_path: str = "vocab.json",
                 pokemon_embed_dim: int = 128,
                 history_dim: int = 64,
                 field_dim: int = 48):
        super().__init__()
        
        self.pokemon_embed_dim = pokemon_embed_dim
        self.action_dim = self.ACTION_DIM
        
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path)
        self.slot_positional_encoding = PositionalSlotEncoding(num_slots=6, slot_dim=16)
        
        self.pkmn_fc = nn.Sequential(
            nn.Linear(360 + 12 + 16, 256),
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
        
        self.history_gru = nn.GRUCell(input_size=pokemon_embed_dim * 4 + 64, hidden_size=history_dim)
        
        fusion_dim = pokemon_embed_dim * 6 + 64 + history_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.ACTION_DIM)
        )
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
        my_team_cat: [B, 6, 10]
        my_team_num: [B, 6, 12]
        my_move_num: [B, 6, 4, 6] (기술 수치)
        opp_team_cat: [B, 6, 10]
        opp_team_num: [B, 6, 12]
        opp_move_num: [B, 6, 4, 6]
        field_vec: [B, 48]
        """
        B = my_team_cat.size(0)
        device = my_team_cat.device
        
        slot_pos_emb = self.slot_positional_encoding(B, device)
        
        # --- 패딩 마스크 생성 (종 ID가 0인 빈/미공개 슬롯은 True) ---
        my_team_pad_mask = (my_team_cat[:, :, 0] == 0)   # [B, 6] (Bool)
        opp_team_pad_mask = (opp_team_cat[:, :, 0] == 0) # [B, 6] (Bool)

        # --- A. 내 팀 6마리 인코딩 ---
        my_pkmn_vectors = []
        my_active_vec = torch.zeros(B, self.pokemon_embed_dim, device=device)
        
        for i in range(6):
            cat_i = my_team_cat[:, i, :]
            num_i = my_team_num[:, i, :]
            mv_num_i = my_move_num[:, i, :, :] # [B, 4, 3]
            pos_i = slot_pos_emb[:, i, :]
            
            sp, it, ab, t1, t2, st = cat_i[:, 0], cat_i[:, 1], cat_i[:, 2], cat_i[:, 3], cat_i[:, 4], cat_i[:, 5]
            mvs = cat_i[:, 6:10]
            
            cat_emb = self.embeddings(sp, it, ab, t1, t2, st, mvs, mv_num_i) # [B, 360]
            full_i = torch.cat([cat_emb, num_i, pos_i], dim=-1)
            vec_i = self.pkmn_fc(full_i)
            
            # 빈 슬롯(sp == 0)인 경우 인코딩 벡터 0 처리
            is_empty = (sp == 0).unsqueeze(-1) # [B, 1]
            vec_i = vec_i.masked_fill(is_empty, 0.0)
            
            my_pkmn_vectors.append(vec_i.unsqueeze(1))
            
            is_act = num_i[:, 0].unsqueeze(-1)
            my_active_vec = my_active_vec + vec_i * is_act

        my_team_stack = torch.cat(my_pkmn_vectors, dim=1) # [B, 6, 128]
        
        # --- B. 상대 팀 6마리 인코딩 ---
        opp_pkmn_vectors = []
        opp_active_vec = torch.zeros(B, self.pokemon_embed_dim, device=device)
        
        for i in range(6):
            cat_i = opp_team_cat[:, i, :]
            num_i = opp_team_num[:, i, :]
            mv_num_i = opp_move_num[:, i, :, :]
            pos_i = slot_pos_emb[:, i, :]
            
            sp, it, ab, t1, t2, st = cat_i[:, 0], cat_i[:, 1], cat_i[:, 2], cat_i[:, 3], cat_i[:, 4], cat_i[:, 5]
            mvs = cat_i[:, 6:10]
            
            cat_emb = self.embeddings(sp, it, ab, t1, t2, st, mvs, mv_num_i)
            full_i = torch.cat([cat_emb, num_i, pos_i], dim=-1)
            vec_i = self.pkmn_fc(full_i)
            
            is_empty = (sp == 0).unsqueeze(-1)
            vec_i = vec_i.masked_fill(is_empty, 0.0)
            
            opp_pkmn_vectors.append(vec_i.unsqueeze(1))
            
            is_act = num_i[:, 0].unsqueeze(-1)
            opp_active_vec = opp_active_vec + vec_i * is_act

        opp_team_stack = torch.cat(opp_pkmn_vectors, dim=1) # [B, 6, 128]
        
        # --- C. Transformer Self-Attention (src_key_padding_mask 적용!) ---
        my_team_attended = self.my_team_attention(my_team_stack, src_key_padding_mask=my_team_pad_mask)
        opp_team_attended = self.opp_team_attention(opp_team_stack, src_key_padding_mask=opp_team_pad_mask)
        
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
        
        # --- E. 필드 및 GRU 히스토리 ---
        field_features = self.field_encoder(field_vec) # [B, 64]
        
        turn_input = torch.cat([my_active_vec, opp_active_vec, my_team_summary, field_features], dim=-1) # [B, 576]
        if history_state is None:
            history_state = torch.zeros(B, 64, device=device)
        new_history_state = self.history_gru(turn_input, history_state) # [B, 64]
        
        # --- F. 전황 통합 (Fusion) ---
        combined_all = torch.cat([
            my_active_vec, my_team_summary, 
            opp_active_vec, opp_team_summary, 
            field_features, new_history_state
        ], dim=-1) # [B, 896]
        
        latent = self.fusion(combined_all) # [B, 256]
        
        # --- G. 출력 헤드 ---
        policy_logits = self.policy_head(latent)           # [B, 22]
        opp_action_logits = self.opp_action_head(latent)   # [B, 22]
        opp_item_logits = self.opp_item_belief_head(latent)# [B, 50]
        value = self.value_head(latent)                    # [B, 1]
        
        # --- H. 행동 마스킹 ---
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
            
        if opp_action_mask is not None:
            opp_action_logits = opp_action_logits.masked_fill(~opp_action_mask, -1e9)
        
        return {
            "policy_logits": policy_logits,
            "opp_action_logits": opp_action_logits,
            "opp_item_logits": opp_item_logits,
            "value": value,
            "history_state": new_history_state
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
