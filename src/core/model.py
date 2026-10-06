import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

import numpy as np


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


def zero_unseen_id_rows(model: nn.Module, path: str = "data/entity_features.npz") -> int:
    """학습 팀 풀에 한 번도 안 나온 도구/특성/기술의 ID 임베딩 행(무작위 초기값)을 0으로 → 미등장 개체는 특징으로만 표현"""
    z, n = np.load(path), 0
    emb = model.embeddings
    with torch.no_grad():
        for kind in ("item", "ability", "move"):
            w = getattr(emb, f"{kind}_embed").weight
            unseen = torch.from_numpy(z[f"{kind}_seen"] == 0).to(w.device)
            unseen[0] = False
            w[unseen] = 0.0
            n += int(unseen.sum())
    return n


def load_compatible(model: nn.Module, state: dict):
    """
    체크포인트 부분 로드.
    - shape가 같으면 그대로 복사
    - 입력 열만 늘어난 2D weight(끝에 새 피처를 붙인 Linear)는 기존 열 복사 + 새 열 0
      → 새 피처가 처음엔 무시되어 기존 정책이 그대로 유지되고, 학습하며 점차 사용
    return: (확장된 키, 버려진 키)
    체크포인트에 깊은 헤드 키(move_deep.*)가 있으면 모델에도 자동으로 켬 (안 켜면 그 가중치가 조용히 버려져 성능만 떨어짐)
    """
    if hasattr(model, "enable_deep_heads") and any(k.startswith("move_deep.") for k in state):
        model.enable_deep_heads()
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
    수치 46d: 위력, 명중률, PP비율, 우선도, 분류, 부가효과(구), 기술타입, 상대 액티브에게 실제 배율,
             + 부가효과 벡터 37칸 + 선공권 우열 1칸 (tensor_encoder.MOVE_EFFECT_NAMES). 새 피처는 항상 끝에 추가.
    """
    def __init__(self, move_dim: int = 48, move_num_dim: int = 46):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(move_dim + move_num_dim, 64),
            nn.GELU(),
            nn.Linear(64, move_dim)
        )

    def forward(self, move_embs, move_numerics):
        # move_embs: [B, 4, 48], move_numerics: [B, 4, 46]
        combined = torch.cat([move_embs, move_numerics], dim=-1) # [B, 4, 94]
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
                 status_dim: int = 8,
                 feature_path: str = "data/entity_features.npz"):
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

        # 도구/특성/기술의 설명문 기반 고정 특징 표 (tools/build_entity_features.py) → 미등장 개체도 효과로 표현
        # 투영은 0으로 시작 → 기존 체크포인트의 동작이 처음엔 그대로. id_dropout은 학습 스크립트가 켬 (ID 없이도 특징만으로 두게)
        self.id_dropout = 0.0
        self.novel_p = 0.0  # 학습 중 이 확률로 실제 개체를 "처음 본 개체"(<unk> ID, 특징 0)로 바꿈 → novel 토큰이 학습됨
        with open(vocab_path, "r", encoding="utf-8") as f:
            _v = json.load(f)
        self.unk = {k: _v[k]["<unk>"] for k in ("item", "ability", "move")}
        self.has_feat = os.path.exists(feature_path)
        if self.has_feat:
            z = np.load(feature_path)
            for kind, dim in (("item", item_dim), ("ability", ability_dim), ("move", move_dim)):
                f = torch.from_numpy(z[f"{kind}_feat"]).float()
                assert f.size(0) == getattr(self, f"{kind}_embed").num_embeddings, f"{kind} 특징 표 크기가 vocab과 다름 → 표를 다시 생성"
                self.register_buffer(f"{kind}_feat", f, persistent=False)
                proj = nn.Linear(f.size(1), dim)
                nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
                setattr(self, f"{kind}_proj", proj)

    def _emb(self, kind, ids):
        if self.novel_p > 0:
            real = (ids > 0) & (ids < self.unk[kind])  # 빈 자리(0)/novel/hidden은 그대로
            ids = torch.where(real & (torch.rand(ids.shape, device=ids.device) < self.novel_p), torch.full_like(ids, self.unk[kind]), ids)
        e = getattr(self, f"{kind}_embed")(ids)
        if not self.has_feat:
            return e
        if self.id_dropout > 0:
            e = e.masked_fill((torch.rand(ids.shape, device=ids.device) < self.id_dropout).unsqueeze(-1), 0.0)
        return e + getattr(self, f"{kind}_proj")(getattr(self, f"{kind}_feat")[ids])

    def forward(self, species_stats, item_id, ability_id, type1_id, type2_id, status_id, move_ids, move_numerics):
        """
        species_stats: [B, 7] float (hp/255, atk/255, def/255, spa/255, spd/255, spe/255, weight/500)
        move_ids: [B, 4]
        move_numerics: [B, 4, 46]
        """
        sp_emb = self.species_proj(species_stats)  # [B, 64]
        it_emb = self._emb("item", item_id)          # [B, 32]
        ab_emb = self._emb("ability", ability_id)    # [B, 32]
        t1_emb = self.type_embed(type1_id)         # [B, 16]
        t2_emb = self.type_embed(type2_id)         # [B, 16]
        st_emb = self.status_embed(status_id)      # [B, 8]
        
        # 기술 임베딩 + 수치 인코딩 결합
        raw_mv_emb = self._emb("move", move_ids)                        # [B, 4, 48]
        encoded_mv_emb = self.move_encoder(raw_mv_emb, move_numerics) # [B, 4, 48]
        mv_emb_flat = encoded_mv_emb.view(encoded_mv_emb.size(0), -1) # [B, 192]
        
        cat_features = torch.cat([sp_emb, it_emb, ab_emb, t1_emb, t2_emb, st_emb, mv_emb_flat], dim=-1)
        return cat_features # [B, 360]


class ResHead(nn.Module):
    """
    기존 얕은 점수 헤드의 출력에 더하는 깊은 잔차 가지 (LayerNorm + 잔차 블록).
    마지막 층을 0으로 초기화해서 시작 시 출력이 0 → 기존 체크포인트의 동작이 그대로 유지되고, 학습하며 점차 기여함.
    """
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, blocks: int = 1):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(blocks)
        ])
        self.out = nn.Linear(width, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class TeamCrossBlock(nn.Module):
    """
    팀 매치업 한 라운드: 팀 내부 셀프어텐션(직전 라운드에서 상대를 보고 알게 된 것을 팀원끼리 재조율)
    -> 양방향 크로스어텐션(내 팀 <-> 상대 팀). 기존엔 셀프->크로스가 팀당 딱 1번, 크로스도 단방향(내->상대)이었음 —
    이러면 "상대를 보고 배운 걸 내 팀원끼리 다시 맞춰본다"가 불가능해서, 라운드를 반복해 그 경로를 만듦.
    """
    def __init__(self, dim: int, ff: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_my = nn.TransformerEncoderLayer(dim, heads, ff, dropout=dropout, activation="gelu", batch_first=True)
        self.self_opp = nn.TransformerEncoderLayer(dim, heads, ff, dropout=dropout, activation="gelu", batch_first=True)
        self.cross_my = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_opp = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_my = nn.LayerNorm(dim)
        self.norm_opp = nn.LayerNorm(dim)

    def forward(self, my, opp, my_pad, opp_pad):
        my = self.self_my(my, src_key_padding_mask=my_pad)
        opp = self.self_opp(opp, src_key_padding_mask=opp_pad)
        # 슬롯이 전부 마스킹되면 softmax가 NaN -> 그 샘플만 마스크 해제 (값은 0 벡터라 결과엔 영향 없음)
        opp_key_mask = opp_pad & ~opp_pad.all(dim=1, keepdim=True)
        my_key_mask = my_pad & ~my_pad.all(dim=1, keepdim=True)
        c_my, _ = self.cross_my(my, opp, opp, key_padding_mask=opp_key_mask)
        c_opp, _ = self.cross_opp(opp, my, my, key_padding_mask=my_key_mask)
        my = self.norm_my(my + c_my)
        opp = self.norm_opp(opp + c_opp)
        return my, opp


class DeepPokemonBattleTransformerNet(nn.Module):
    """
    트랜스포머 AI (GELU 활성화 함수 개편 완료)
    """
    ACTION_DIM = 22

    def __init__(self, 
                 vocab_path: str = "data/vocab.json",
                 pokemon_embed_dim: int = 128,
                 history_dim: int = 256,
                 field_dim: int = 58,  # 48 + 직전 턴 사건 10 (events.TURN_DIM)
                 latent_dim: int = 256,
                 num_layers: int = 2,
                 entity_features: bool = True,
                 deep_heads: bool = False,
                 cross_rounds: int = 2,
                 head_width: int = 128):
        super().__init__()
        self.cross_rounds = cross_rounds
        # 체크포인트 옆 .cfg.json에 저장할 구조 인자 (기본 구조와 다른 모델을 다시 만들 때 필요)
        self.cfg = dict(pokemon_embed_dim=pokemon_embed_dim, history_dim=history_dim, latent_dim=latent_dim,
                        num_layers=num_layers, cross_rounds=cross_rounds, head_width=head_width)

        self.pokemon_embed_dim = pokemon_embed_dim
        self.action_dim = self.ACTION_DIM
        team_dim = pokemon_embed_dim * 2  # 팀 요약 = mean + max 풀링
        
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        
        # 입력: 임베딩(360) + 배틀 수치(11) + 테라 타입(16) + 팀프리뷰만 봄(1)+상대 슬롯 증거(8)
        # 새 피처는 반드시 끝에 붙임 (load_compatible이 기존 열을 보존하도록)
        self.pkmn_fc = nn.Sequential(
            nn.Linear(360 + 11 + 16 + 9, team_dim),  # 마지막 9 = 아직 팀프리뷰로만 보임 1 + 상대 슬롯 증거 8
            nn.LayerNorm(team_dim),
            nn.GELU(),
            nn.Linear(team_dim, pokemon_embed_dim),
            nn.LayerNorm(pokemon_embed_dim)
        )

        
        self.field_encoder = nn.Sequential(
            nn.Linear(field_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        
        # 히스토리: 배틀의 모든 턴 요약을 시퀀스로 보는 인과 트랜스포머 (이전 GRUCell 대체)
        # 턴 요약 = 아군액티브(128) + 상대액티브(128) + 아군팀요약(256) + 상대팀요약(256) + 필드(64) = 832
        turn_dim = pokemon_embed_dim * 2 + team_dim * 2 + 64
        self.turn_proj = nn.Linear(turn_dim, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=history_dim, nhead=4, dim_feedforward=history_dim * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.history_seq = nn.TransformerEncoder(seq_layer, num_layers=num_layers, enable_nested_tensor=False)
        # 출력 투영을 0으로 시작 → 처음엔 히스토리 = 0 벡터 (기존 체크포인트의 나머지 부분이 안정적으로 이어지도록)
        self.history_out = nn.Linear(history_dim, history_dim)
        nn.init.zeros_(self.history_out.weight)
        nn.init.zeros_(self.history_out.bias)

        # Fusion: 아군액티브(128) + 아군팀요약(256) + 상대액티브(128) + 상대팀요약(256) + 필드(64) + 히스토리(256) = 1088
        fusion_dim = pokemon_embed_dim + team_dim + pokemon_embed_dim + team_dim + 64 + history_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # 팀-매치업 블록: 셀프어텐션(팀 내부 재조율) -> 양방향 크로스어텐션(상대 팀과 대조) 을 cross_rounds번 반복.
        # 라운드가 반복돼야 "상대를 보고 알게 된 것"을 팀원끼리 다시 조율할 수 있음 (기존엔 셀프->크로스 딱 1번뿐이었음)
        self.team_cross = nn.ModuleList([TeamCrossBlock(pokemon_embed_dim, team_dim) for _ in range(cross_rounds)])

        # 후보별 점수 헤드 (pointer 방식)
        # 기술 j: latent(256) + 액티브 기술 j 벡터(48) + 액티브 문맥(128) -> [일반, Z/테라]
        self.move_score = nn.Sequential(
            nn.Linear(latent_dim + 48 + pokemon_embed_dim, head_width),
            nn.GELU(),
            nn.Linear(head_width, 2)
        )
        # 교체 k: latent(256) + 크로스 어텐션 거친 k번 슬롯 벡터(128) -> 1
        self.switch_score = nn.Sequential(
            nn.Linear(latent_dim + pokemon_embed_dim, head_width),
            nn.GELU(),
            nn.Linear(head_width, 1)
        )
        # 깊은 잔차 가지: 기존 헤드 출력에 더함. 입력에 상대 액티브 벡터를 직접 추가 (latent 병목을 거치지 않고 상호작용을 만들 수 있게)
        self.deep_heads = False
        self.latent_dim = latent_dim
        if deep_heads:
            self.enable_deep_heads()
        # 계층 행동: 0=기술, 1=교체
        self.type_head = nn.Linear(latent_dim, 2)
        is_switch = torch.zeros(self.ACTION_DIM, dtype=torch.bool)
        is_switch[8:14] = True
        self.register_buffer("is_switch", is_switch, persistent=False)
        self.opp_action_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.ACTION_DIM)
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def enable_deep_heads(self):
        """깊은 잔차 가지를 켬 (이미 켜져 있으면 무시). 0 초기화라 켜도 출력은 그대로. load_compatible이 체크포인트에 이 가지의 키가 있으면 자동 호출"""
        if self.deep_heads:
            return
        dev = next(self.parameters()).device
        self.deep_heads = True
        self.move_deep = ResHead(self.latent_dim + 48 + 2 * self.pokemon_embed_dim, 2).to(dev)  # latent + 기술벡터 + 내 액티브 문맥 + 상대 액티브
        self.switch_deep = ResHead(self.latent_dim + 2 * self.pokemon_embed_dim, 1).to(dev)      # latent + 후보 슬롯 벡터 + 상대 액티브

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
        out = self._heads(enc, history, action_mask, None)
        return out

    def _history(self, turn_seq):
        """[B, T, 832] 턴 요약 시퀀스 → [B, T, 256] 각 시점의 히스토리 (미래 턴은 보지 않음)"""
        B, T = turn_seq.size(0), turn_seq.size(1)
        dev = turn_seq.device
        pos = torch.arange(T, device=dev).clamp(max=MAX_TURNS - 1)
        x = self.turn_proj(turn_seq) + self.turn_pos(pos)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=dev), diagonal=1)
        return self.history_out(self.history_seq(x, mask=causal))

    def _encode_turn(self,
                     my_team_cat, my_team_num, my_move_num,
                     opp_team_cat, opp_team_num, opp_move_num,
                     field_vec):
        """
        my_team_cat: [B, 6, 10]
        my_team_num: [B, 6, 23]
        my_move_num: [B, 6, 4, 46] (기술 수치)
        opp_team_cat: [B, 6, 10]
        opp_team_num: [B, 6, 23]
        opp_move_num: [B, 6, 4, 46]
        field_vec: [B, 48]
        """
        B = my_team_cat.size(0)
        device = my_team_cat.device
        
        # --- 패딩 마스크 생성 (base_stats HP가 0인 빈/미공개 슬롯은 True) ---
        # num[:, :, 11] = 기본 HP 종족값/255 (12는 공격). 0이면 미사용 슬롯 — 아래 is_empty(species_stats[:, 0])와 같은 기준
        my_team_pad_mask = (my_team_num[:, :, 11] == 0.0)   # [B, 6] (Bool)
        opp_team_pad_mask = (opp_team_num[:, :, 11] == 0.0) # [B, 6] (Bool)

        # --- A. 내 팀 6마리 인코딩 ---
        my_pkmn_vectors = []
        my_active_vec = torch.zeros(B, self.pokemon_embed_dim, device=device)
        my_active_moves = torch.zeros(B, 4, 48, device=device)  # 액티브의 인코딩된 기술 4개

        for i in range(6):
            cat_i = my_team_cat[:, i, :]       # [B, 10] (item, ability, type1, type2, status, moves×4, tera)
            num_i = my_team_num[:, i, :]        # [B, 27] (battle 11 + species_stats 7 + 팀프리뷰만 봄 1 + 상대 슬롯 증거 8, 내 쪽은 뒤 9칸 항상 0)
            mv_num_i = my_move_num[:, i, :, :]  # [B, 4, 46]

            # 배틀 수치 (앞 11): is_active, fainted, hp_frac, boosts×5, level, gimmick×2
            battle_num_i = num_i[:, :11]         # [B, 11]
            # 종 스탯 (뒤 7): hp/255, atk/255, def/255, spa/255, spd/255, spe/255, weight/500
            species_stats_i = num_i[:, 11:18]    # [B, 7]

            it, ab, t1, t2, st = cat_i[:, 0], cat_i[:, 1], cat_i[:, 2], cat_i[:, 3], cat_i[:, 4]
            mvs = cat_i[:, 5:9]

            cat_emb = self.embeddings(species_stats_i, it, ab, t1, t2, st, mvs, mv_num_i)  # [B, 360]
            tera_emb = self.embeddings.type_embed(cat_i[:, 9])  # [B, 16]
            full_i = torch.cat([cat_emb, battle_num_i, tera_emb, num_i[:, 18:]], dim=-1)  # [B, 396]
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
            num_i = opp_team_num[:, i, :]        # [B, 27] (battle 11 + species_stats 7 + 팀프리뷰만 봄 1 + 상대 슬롯 증거 8)
            mv_num_i = opp_move_num[:, i, :, :]  # [B, 4, 46]

            battle_num_i = num_i[:, :11]         # [B, 11]
            species_stats_i = num_i[:, 11:18]    # [B, 7]

            it, ab, t1, t2, st = cat_i[:, 0], cat_i[:, 1], cat_i[:, 2], cat_i[:, 3], cat_i[:, 4]
            mvs = cat_i[:, 5:9]

            cat_emb = self.embeddings(species_stats_i, it, ab, t1, t2, st, mvs, mv_num_i)
            tera_emb = self.embeddings.type_embed(cat_i[:, 9])  # [B, 16]
            full_i = torch.cat([cat_emb, battle_num_i, tera_emb, num_i[:, 18:]], dim=-1)  # [B, 396]
            vec_i = self.pkmn_fc(full_i)

            is_empty = (species_stats_i[:, 0] == 0.0).unsqueeze(-1)
            vec_i = vec_i.masked_fill(is_empty, 0.0)

            opp_pkmn_vectors.append(vec_i.unsqueeze(1))

        opp_team_stack = torch.cat(opp_pkmn_vectors, dim=1) # [B, 6, 128]
        opp_active_vec = (opp_team_stack * opp_team_num[:, :, 0:1]).sum(dim=1)  # [B, 128]
        # --- C. 팀 매치업: (팀 내부 셀프어텐션 -> 양방향 크로스어텐션) 을 cross_rounds번 반복 ---
        my_team_attended, opp_team_attended = my_team_stack, opp_team_stack
        for block in self.team_cross:
            my_team_attended, opp_team_attended = block(my_team_attended, opp_team_attended, my_team_pad_mask, opp_team_pad_mask)
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
        mv_in = torch.cat([
            latent.unsqueeze(1).expand(-1, 4, -1), my_active_moves,
            my_active_ctx.unsqueeze(1).expand(-1, 4, -1)
        ], dim=-1)
        sw_in = torch.cat([latent.unsqueeze(1).expand(-1, 6, -1), my_team_attended], dim=-1)
        mv = self.move_score(mv_in)                                                   # [B, 4, 2]
        sw = self.switch_score(sw_in)                                                 # [B, 6, 1]
        if self.deep_heads:
            opp_ctx = opp_active_vec
            mv = mv + self.move_deep(torch.cat([mv_in, opp_ctx.unsqueeze(1).expand(-1, 4, -1)], dim=-1))
            sw = sw + self.switch_deep(torch.cat([sw_in, opp_ctx.unsqueeze(1).expand(-1, 6, -1)], dim=-1))
        sw = sw.squeeze(-1)                                                           # [B, 6]
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
        value = self.value_head(latent)                    # [B, 1]

        # --- H. 상대 행동 마스킹 ---
        if opp_action_mask is not None:
            opp_action_logits = opp_action_logits.masked_fill(~opp_action_mask, -1e9)
        
        return {
            "policy_logits": policy_logits,
            "opp_action_logits": opp_action_logits,
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


def build_model(cfg: dict) -> nn.Module:
    """구조 인자로 모델 생성: {"model": "entity"}면 model_v2.EntityPokemonNet, 아니면 기존 구조"""
    cfg = dict(cfg)
    for k, default in (("history_mode", "flat"), ("fusion_pair", False), ("fusion_res", False), ("switch_skip", False)):  # 제거된 실험 옵션: 기본값이던 체크포인트만 로드 가능
        if cfg.pop(k, default) != default:
            raise ValueError(f"제거된 옵션 {k}를 쓰는 체크포인트는 더 이상 로드할 수 없음")
    kind = cfg.pop("model", "v1")
    if kind == "entity":
        from src.core.model_v2 import EntityPokemonNet
        return EntityPokemonNet(**cfg)
    if kind == "v3":
        ver = cfg.pop("version", 1)
        if ver >= 3:                                     # 확정 설정이 기본인 현재 구조 (자기 완결형 신념, 옵션 없음)
            from src.core.model_v3 import EntityPokemonNetV3
            return EntityPokemonNetV3(**cfg)
        from src.core.model_v3_legacy import EntityPokemonNetV3Legacy      # 옛 구조: 호환 전용, 퇴역 시 삭제
        if ver == 2:                                     # 트렁크 임베딩을 공유하던 신념(v3_pilot): 옛 구조 + belief 켜기, 벤치 위협은 공개 기술만
            m = EntityPokemonNetV3Legacy(belief=True, **cfg)
            m.bench_from_guess = False
            return m
        return EntityPokemonNetV3Legacy(**cfg)
    return DeepPokemonBattleTransformerNet(**cfg)


def save_ckpt(model: nn.Module, path: str):
    """가중치 저장 + 구조 인자를 <path>.cfg.json에 기록 (기본 구조 모델도 기록해 둠)"""
    torch.save(model.state_dict(), path)
    with open(path + ".cfg.json", "w", encoding="utf-8") as f:
        json.dump(getattr(model, "cfg", {}), f)


def model_from_ckpt(path: str, map_location="cpu") -> nn.Module:
    """체크포인트 옆 .cfg.json이 있으면 그 구조로, 없으면 기본 구조로 모델을 만들어 가중치 로드.
    cfg.json 없는 체크포인트(예: team_cross 이전의 all11)는 지금 기본 구조로 지어지는데, 그 체크포인트에
    없는 새 모듈(team_cross 등)은 load_compatible이 무작위로 남겨둠 — h2h/eval/KL 기준(ref)으로 이걸
    그대로 쓰면 "학습된 척하는 무작위 조각"이 섞여 들어가는 사고가 난 적 있어 경고를 남김."""
    cfg_path = path + ".cfg.json"
    has_cfg = os.path.exists(cfg_path)
    cfg = json.load(open(cfg_path, encoding="utf-8")) if has_cfg else {}
    if not has_cfg:
        print(f"  ⚠️ {path}에 .cfg.json 없음 → 기본 구조로 생성. 이 체크포인트가 모르는 새 모듈(예: team_cross)은 "
              f"무작위로 남습니다 — 기준/비교 모델로 쓴다면 cfg.json 있는(구조가 맞는) 체크포인트를 쓰세요.")
    if cfg.get("model") == "v3" and "version" not in cfg:     # version 없음 = v3_full 등 옛 구조
        cfg.setdefault("unrevealed_tokens", False)       # 이 옵션들 이전에 학습한 v3 체크포인트(v3_full 등)는 미공개 기술 칸을 마스킹하던 구조
        cfg.setdefault("opp_species", False)             # 상대 종 ID 임베딩도 없던 구조
    model = build_model(cfg)
    load_compatible(model, torch.load(path, map_location=map_location))
    return model

