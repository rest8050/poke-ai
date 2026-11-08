"""정책/가치 네트워크 (엔티티 토큰 모델 v8; v6 구조는 퇴역해 제거됨) = v7(= v6 + 특성 EXEC 규칙 + 합법 요약 슬롯 순서 통로 제거) + 숨은 특성/도구 기대 임베딩 + 융합층의 활성 토큰 직접 입력.
v7 대비 바뀐 점 (나머지는 v7과 같음):
A) 상대 쪽 숨겨진(공개 안 된) 특성/도구는 고정된 "숨김" 임베딩 대신 세트 추정 분포의 기대 임베딩 q @ 임베딩 표를 포켓몬 토큰에 넣음. 공개된 값/내 쪽/빈 칸은 원래 임베딩 그대로
   (표는 학습용 난수 novel_p/id_dropout 없이 embed.weight + proj(feat)로 결정론적으로 만들고, q는 detach라서 기울기는 임베딩 표로만 감)
D) 융합층 입력에 내 활성 / 상대 활성 포켓몬 토큰을 CLS와 함께 직접 넣음 (기술/교체 판단이 CLS 하나로만 전달받던 정보를 직접 봄): 융합 입력 d + history + G_DIM -> + 2d
H) 종류 헤드(기술/교체)의 은닉 폭을 옛 고정값 64에서 head_width(기본 192)로: 행동 점수 헤드와 같은 폭
v6 대비 바뀐 점 (v7, 나머지는 v6와 같음):
1) 매치업을 MatchupV7로: 정밀기술/두꺼운지방/옹골참, 부유류 면역을 상대 특성 확률(세트 추정) 기대값으로, 스피드 규칙(랭크·마비·스카프·순풍·날씨 특성·트릭룸)과 장난꾸러기 우선도로
   만든 선공 기대값 fm을 기술 토큰(양쪽)에, 위협 후보의 상대 능력 배율 entry_opp 3개를 교체 후보 특징에, 재생력 회복량 exit_heal을 전역 요약에 추가
2) 합법 요약에서 교체 6칸 비트를 뺌 (슬롯 순서대로 들어가 my_pos=False의 순열 대칭을 깨던 통로. 슬롯별 교체 가능 여부는 포켓몬 토큰의 플래그가 이미 줌): LEG_G 25 -> 19
상대 특성 분포는 세트 추정(ability) 로짓에서 빈칸/<unk>/숨김을 뺀 softmax, 공개된 것은 원핫. 상대 스카프 확률도 세트 추정(item)에서.
arch: {"model": "v8", "my_pos": false}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import warnings

import numpy as np

from src.core.belief import STAT_SCALE, SetBelief, build_move_table
from src.core.matchup_v7 import MatchupV7
from src.core.tensor_encoder import EFFECT_DIM, EFFECT_START, FIELD_DIM, MOVE_NUM_DIM, NUM_DIM


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


# 3차원 어텐션 마스크(헤드별 편향)는 추론용 빠른 커널에서 학습 경로와 값이 달라짐(CLS 기준 2e-3) → 학습과 같은 일반 경로만 쓰도록 끔
torch.backends.mha.set_fastpath_enabled(False)

N_TOK = 64            # 포켓몬 12 + 기술 48 + 필드 3 + CLS 1
MV_EXTRA = 4            # 기술 토큰 추가 수치: 배율, 데미지, KO + 선공 기대값 fm
BASE_IN = 177 + NUM_DIM         # 포켓몬 토큰 기본 입력 차원 (종족값 투영 64 + 도구/특성/타입/상태/테라 임베딩 + 배틀 수치 + 이벤트 증거)
K_GUESS = 12          # 마리당 후보 기술 수 (숨은 기술 재현율: 상위 4 0.73 / 상위 8 0.94 / 상위 12 0.98)
PLAUSIBLE = 0.15      # 이 이상의 세트 포함 확률이면 "그럴듯한" 후보
MULT_RANGE = (0.6, 1.25)
LEG_G = 8 + 6 + 3 + 2    # 합법 비트 8(기술 4 + 테라 4) + 플래그 6 + 사용 가능한 기술만의 공격 요약 3 + 고정으로 잃은 화력 2
MON_X = 7 + 2 + 6 + 1    # 포켓몬 토큰 추가 수치: 매치업 요약 7 + 숨은 기술 위협 2 + 상대 능력치 예측 6 + 진입 피해 비율 1
CAND_N = 9 + 2 + 1 + 3   # 교체 후보 특징: 요약 9 + 상대 활성의 숨은 기술 위협 2 + 진입 피해 비율 1 + 위협 효과 3
G_DIM = LEG_G + EFFECT_DIM + 1   # 전역 요약: 합법성 요약 + 양쪽 활성의 효과 열 + 재생력 회복량


class PairGLU(nn.Module):
    def __init__(self, d, k, n_heads):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.v, self.g, self.rel = nn.Linear(d, k), nn.Linear(d, k), nn.Embedding(3, k)
        self.to_bias, self.to_tok = nn.Linear(k, n_heads), nn.Linear(k, d)
        for lin in (self.to_bias, self.to_tok):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        owner = torch.arange(48) // 4                                   # 기술 토큰의 주인 포켓몬 (내 0..5, 상대 6..11)
        mon = torch.arange(12)
        rel = torch.where(owner[:, None] == mon[None], 0, torch.where(owner[:, None] // 6 == mon[None] // 6, 1, 2))
        self.register_buffer("rel_idx", rel, persistent=False)

    def forward(self, x, mon_pad):
        """x [B,64,d] 해당 층 입력, mon_pad [B,12] 빈 칸. 반환 (편향 [B,H,64,64] 전체 크기, 기술 토큰에 더할 값 [B,48,d])"""
        h = self.ln(x)
        v = self.v(h[:, 12:60])[:, :, None]                             # [B,48,1,k]
        g = self.g(h[:, :12])[:, None] + self.rel(self.rel_idx)[None]    # [B,48,12,k]
        z = v * torch.sigmoid(g)
        pb = self.to_bias(z).permute(0, 3, 1, 2)                         # [B,H,48,12]
        full = F.pad(pb, (0, N_TOK - 12, 12, N_TOK - 60))                # 행 12..59, 열 0..11
        keep = (~mon_pad).to(z.dtype)[:, None, :, None]
        return full + full.transpose(2, 3), self.to_tok((z * keep).sum(2) / 12.0)


def _mx(x):           # [..., 4(기술), 4(채널)] 중 앞 3채널(배율, 데미지, KO)의 기술별 최댓값 → [..., 3]
    return x[..., :3].amax(-2)


class EntityPokemonNetV8(nn.Module):
    ACTION_DIM = 22
    uses_legal = True

    def __init__(self, vocab_path: str = "data/vocab.json", my_pos: bool = True, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0, history_dim: int = 256, hist_layers: int = 2, latent_dim: int = 256,
                 head_width: int = 192, entity_features: bool = True,
                 species_dim: int = 32, belief_hidden: int = 256, belief_species_dropout: float = 0.3,
                 mon_moves: bool = False, pair_glu: int = 0, pair_mode: str = "glu",
                 forced_legal_zero: bool = False):
        super().__init__()
        self.cfg = dict(model="v8", d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout,
                        history_dim=history_dim, hist_layers=hist_layers, latent_dim=latent_dim, head_width=head_width,
                        species_dim=species_dim, belief_hidden=belief_hidden, belief_species_dropout=belief_species_dropout, my_pos=my_pos)
        self.mon_moves = mon_moves
        if mon_moves:                 # 포켓몬 토큰에 자기 기술 가방(공개된 기술 임베딩 합 + 숨은 칸은 세트 추정 기대값)을 직접 입력. 켰을 때만 cfg에 기록
            self.cfg["mon_moves"] = True
            self.bag_proj = nn.Linear(48, d_model)
        self.pair_k = pair_glu
        assert pair_mode == "glu", "pair_mode는 glu만 지원 (bilin/add는 효과 없어 제거)"
        if pair_glu:                  # 기술 x 포켓몬 쌍 모듈(이 파일의 PairGLU, 층마다 하나): 켰을 때만 cfg에 기록
            self.cfg["pair_glu"], self.cfg["pair_mode"] = pair_glu, pair_mode
            self.pair = nn.ModuleList([PairGLU(d_model, pair_glu, n_heads) for _ in range(n_layers)])
        self.forced_legal_zero = forced_legal_zero    # 합법 기술이 0개(강제 교체)면 "기술 고정"·"잃은 화력" 요약을 0으로. 옛 체크포인트는 꺼진 채(예전 계산) 학습됨
        if forced_legal_zero:         # 켰을 때만 cfg에 기록
            self.cfg["forced_legal_zero"] = True
        d = self.d = d_model
        field_dim = FIELD_DIM
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        pos = torch.ones(12) if my_pos else torch.tensor([0.0] * 6 + [1.0] * 6)       # 내 쪽 슬롯/기술 칸 순서는 임의(팀 export 순서)라서 끌 수 있음. 상대 쪽 순서는 공개된 순서라 정보가 있어 유지
        self.register_buffer("pos_gate", pos, persistent=False)
        self.embeddings = PokemonEmbeddingLayer(vocab_path=vocab_path, feature_path="data/entity_features.npz" if entity_features else "")
        self.matchup = MatchupV7(vocab_path)
        self.belief = SetBelief(vocab_path, species_dim, belief_hidden, belief_species_dropout)
        self.register_buffer("move_table", build_move_table(vocab_path), persistent=False)
        self.unk_move = nn.Parameter(torch.zeros(d))                        # 미공개 기술 칸 표시
        self.hid_proj = nn.Linear(1 + MOVE_NUM_DIM + 48, d)                 # 후보 기술 혼합(확률 합, 수치, 임베딩) -> 상대 포켓몬 토큰
        self.mon_proj = nn.Sequential(nn.Linear(BASE_IN + MON_X + 1 + 1, d), nn.LayerNorm(d))      # + 공개된 기술 수 + 교체 가능 플래그
        self.move_proj = nn.Sequential(nn.Linear(142 + MV_EXTRA + 1, d), nn.LayerNorm(d))          # + 사용 가능 플래그
        self.field_proj = nn.Sequential(nn.Linear(field_dim, d), nn.GELU(), nn.Linear(d, 3 * d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.role_emb, self.side_emb, self.slot_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(6, d)
        self.mslot_emb, self.act_emb, self.fld_emb = nn.Embedding(4, d), nn.Embedding(2, d), nn.Embedding(3, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, n_heads, d * ff_mult, dropout=dropout, activation="gelu",
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        # 층별 어텐션 편향: 입력 4채널(배율, 데미지, KO, 공격기) -> 헤드별 값 / 숨은 기술 위협 2채널 -> 헤드별 값. 작게 시작
        n_bias = n_layers
        self.bias_def = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_bias)])
        self.bias_off = nn.ModuleList([nn.Linear(4, n_heads) for _ in range(n_bias)])
        self.bias_hid = nn.ModuleList([nn.Linear(2, n_heads) for _ in range(n_bias)])
        for lin in list(self.bias_def) + list(self.bias_off) + list(self.bias_hid):
            nn.init.normal_(lin.weight, std=0.5)
            nn.init.zeros_(lin.bias)
        n_mv = len(json.load(open(vocab_path, encoding="utf-8"))["move"])
        self.n_mv = n_mv
        self.turn_proj = nn.Linear(3 * d + field_dim, history_dim)
        self.turn_pos = nn.Embedding(MAX_TURNS, history_dim)
        hl = nn.TransformerEncoderLayer(history_dim, 4, history_dim * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.history_seq = nn.TransformerEncoder(hl, hist_layers, enable_nested_tensor=False)
        self.history_out = nn.Linear(history_dim, history_dim)
        fuse_in = history_dim + G_DIM + 3 * d
        self.fusion =nn.Sequential(nn.Linear(fuse_in, latent_dim * 2), nn.LayerNorm(latent_dim * 2), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(latent_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.move_score = nn.Sequential(nn.Linear(latent_dim + 2 * d, head_width), nn.GELU(), nn.Linear(head_width, 2))
        self.switch_score = nn.Sequential(nn.Linear(latent_dim + d + CAND_N + G_DIM + 1, head_width), nn.GELU(), nn.Linear(head_width, 1))
        self.type_head = nn.Sequential(nn.Linear(latent_dim + CAND_N + 2 + G_DIM, head_width), nn.GELU(), nn.Linear(head_width, 2))
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
        """세트 추정의 기술 확률 -> 마리당 상위 K 후보. idx/w [B,6,K] (w = 세트 포함 확률 근사, 미공개 칸이 없으면 0), pseudo [B,6,K,46] (기울기 없음)"""
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

    @staticmethod
    def _legal_summary(leg, off_act, n_rev, forced_zero=False):
        """leg [B,22] (0/1), off_act [B,4,4] 활성 포켓몬의 기술 4칸 x (배율, 데미지, KO, 공격기) vs 상대 활성, n_rev [B] 공개된 활성 기술 수 -> [B, LEG_G]. 교체는 칸별 비트가 아니라 개수만 (슬롯 순서 통로 제거)"""
        mv = leg[:, :4]
        n_normal, n_attack, n_sw = mv.sum(1), (mv * off_act[..., 3]).sum(1), leg[:, 8:14].sum(1)
        one_move = ((n_normal == 1) & (n_rev >= 2)).float()
        lock = (n_rev - n_normal).clamp_min(0) / 4.0                                     # 기술이 몇 개 막혀 있나 (고정·도발·PP 소진 등)
        full = off_act[..., :3].amax(1)
        legal_off = torch.where(mv[..., None] > 0, off_act[..., :3], torch.full_like(off_act[..., :3], -1.0)).amax(1).clamp_min(0)
        lost = torch.stack([(full[:, 1] - legal_off[:, 1]).clamp_min(0), (full[:, 2] - legal_off[:, 2]).clamp_min(0)], -1)
        if forced_zero:   # 기술이 하나도 합법이 아닌 결정(강제 교체)은 고정/화력 손실이 아님: 실제 구애 고정 같은 신호와 섞이지 않게 0
            has = (n_normal > 0).to(lock.dtype)
            lock, lost = lock * has, lost * has[:, None]
        return torch.cat([leg[:, :8], torch.stack([n_normal / 4.0, n_attack / 4.0, n_sw / 5.0, leg[:, 4:8].amax(1), one_move, lock], -1), legal_off, lost], -1)

    @staticmethod
    def _prob(ids, logits):
        """상대 특성/도구 분포 [B,6,V]: 공개된 값은 원핫, 아니면 세트 추정 softmax (빈칸/<unk>/숨김 제외). ids [B,6], logits [B,6,V] (V = <unk>+2)"""
        lg = logits.detach().float().clone()
        unk = lg.size(-1) - 2
        lg[..., 0] = -1e9
        lg[..., unk:] = -1e9
        known = (ids >= 1) & (ids < unk)
        return torch.where(known[..., None], F.one_hot(ids.clamp(0, lg.size(-1) - 1), lg.size(-1)).float(), torch.softmax(lg, -1))

    def _expected_emb(self, kind, ids, q, B):
        """포켓몬 토큰 12개의 특성/도구 임베딩 [B*12, D]. 상대 쪽(뒤 6마리)에서 숨겨진 값(<unk>+1)만 분포 q [B,6,V]의 기대 임베딩 q @ 표로 바꾸고, 나머지는 원래 임베딩.
        표는 학습용 난수(novel_p, id_dropout)를 뺀 embed.weight + proj(feat)"""
        E = self.embeddings
        e = E._emb(kind, ids).view(B, 12, -1)
        w = getattr(E, f"{kind}_embed").weight
        if E.has_feat:
            w = w + getattr(E, f"{kind}_proj")(getattr(E, f"{kind}_feat"))
        hidden = (ids.view(B, 12)[:, 6:] == E.unk[kind] + 1)[..., None]
        expected = (q @ w[: q.size(-1)].to(q.dtype)).to(e.dtype)
        return torch.cat([e[:, :6], torch.where(hidden, expected, e[:, 6:])], 1).view(B * 12, -1)

    def _matchup_feats(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, guess, mult, stat, field_vec, q_opp, p_scarf):
        B, K = my_cat.size(0), K_GUESS
        ar = torch.arange(B, device=my_cat.device)
        with torch.no_grad():
            M = self.matchup(my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, field_vec, q_opp, p_scarf, mult)
            D, O, ent, entry_opp = M["def"], M["off"], M["entry_frac"], M["entry_opp"]    # 공개된 기술만 [B,6,24,4]
            Dp = self.matchup._pair(opp_cat, opp_num, guess["pseudo"], False, q_opp, my_cat, my_num, True, self.matchup._onehot(my_cat),
                                    att_mult=mult, entry=ent, cand_f=(entry_opp[..., 0], entry_opp[..., 1]), mid=guess["idx"], field=field_vec)     # [B,6(내),6K,4]
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
        mv_extra = torch.cat([mv_extra, torch.cat([M["fm_my"], M["fm_opp"]], 1)[..., None]], -1)                       # [B,12,4,4] + 선공 기대값
        bench_thr = DS[..., :3].amax((2, 3))[..., 1:]                                                                   # 후보 i가 상대 전체의 공개 기술에 맞는 최대 데미지/KO
        cand = torch.cat([_mx(def_opp_act), _mx(off_vs_opp_act), M["spd_my"][..., None], bench_thr, h_my], -1)         # [B,6,11]
        mon_extra = torch.cat([mon_extra, torch.cat([ent, torch.zeros_like(ent)], 1)[..., None]], -1)      # 진입 피해 비율 (상대 쪽 0)
        cand = torch.cat([cand, ent[..., None], entry_opp], -1)                                                         # [B,6,15]
        return D, O, H, mon_extra, mv_extra, cand, ar, act_m, M["exit_heal"]

    def _encode_turn(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec, legal=None):
        field_vec = self._fit_field(field_vec)
        B, dev, d = my_team_cat.size(0), my_team_cat.device, self.d
        cat = torch.cat([my_team_cat[..., :10], opp_team_cat[..., :10]], 1).reshape(B * 12, 10)
        num = torch.cat([my_team_num, opp_team_num], 1).reshape(B * 12, -1)
        mvn = torch.cat([my_move_num, opp_move_num], 1).reshape(B * 12, 4, -1)
        E = self.embeddings
        belief_logits = self.belief(opp_team_cat, opp_team_num)
        q_opp, q_item = self._prob(opp_team_cat[..., 1], belief_logits["ability"]), self._prob(opp_team_cat[..., 0], belief_logits["item"])
        p_scarf = q_item[..., self.matchup.scarf]
        parts = [E.species_proj(num[:, 11:18]), self._expected_emb("item", cat[:, 0], q_item, B), self._expected_emb("ability", cat[:, 1], q_opp, B),
                 E.type_embed(cat[:, 2]), E.type_embed(cat[:, 3]), E.status_embed(cat[:, 4]), E.type_embed(cat[:, 9]), num[:, :11], num[:, 18:]]
        raw = E._emb("move", cat[:, 5:9])
        revealed = (mvn.abs().sum(-1) > 0)
        n_rev = revealed.float().sum(-1, keepdim=True) / 4.0
        stat = belief_logits["stat"].detach().float()
        mult = torch.exp(STAT_SCALE * stat).clamp(*MULT_RANGE)
        guess = self._top_guess(belief_logits, opp_team_cat, opp_move_num, opp_team_num)
        D, O, H, mon_extra, mv_extra, cand, ar, act_m, exit_heal = self._matchup_feats(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num,
                                                                             opp_move_num, guess, mult, stat, field_vec, q_opp, p_scarf)
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
        leg_g = torch.cat([self._legal_summary(leg, off_act, n_rev_act, self.forced_legal_zero), field_vec[:, EFFECT_START:EFFECT_START + EFFECT_DIM], exit_heal[:, None]], -1)      # 합법성 요약 + 양쪽 활성의 효과 열 + 재생력 회복량
        mon_in = torch.cat(parts + [mon_extra.reshape(B * 12, MON_X), n_rev, mon_flag.reshape(B * 12, 1)], -1)
        mv_in = torch.cat([E.move_encoder(raw, mvn), raw, mvn, mv_extra.reshape(B * 12, 4, MV_EXTRA), mv_flag.reshape(B * 12, 4, 1)], -1)
        side = torch.arange(12, device=dev) // 6
        slot = torch.arange(12, device=dev) % 6
        active = (num[:, 0] > 0.5).long().view(B, 12)
        base = self.side_emb(side)[None] + self.slot_emb(slot)[None] * self.pos_gate[None, :, None] + self.act_emb(active)
        g = torch.cat([guess["w"].sum(-1, keepdim=True), (guess["w"][..., None] * guess["pseudo"]).sum(2),
                       (guess["w"][..., None] * E._emb("move", guess["idx"]).detach()).sum(2)], -1)
        mon = self.mon_proj(mon_in).view(B, 12, d) + base + self.role_emb.weight[0]
        mon = mon + torch.cat([torch.zeros_like(mon[:, :6]), self.hid_proj(g)], 1)
        if self.mon_moves:      # 기술 가방: 양쪽 같은 경로. 내 쪽은 전부 공개라 숨은 항이 0, 상대는 공개되면 실제 임베딩이 들어가고 숨은 칸 기대값은 그만큼 줄어듦
            bag = (raw * revealed[..., None]).sum(1).view(B, 12, -1)
            bag = bag + torch.cat([torch.zeros_like(bag[:, :6]), g[..., -48:].to(bag.dtype)], 1)
            mon = mon + self.bag_proj(bag)
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
            bias = self._bias(D, O, H, li).to(x.dtype)
            if self.pair_k:
                pb, pt = self.pair[li](x, mon_pad)
                bias = bias + pb.to(x.dtype)
                x = torch.cat([x[:, :12], x[:, 12:60] + pt.to(x.dtype), x[:, 60:]], 1)
            mask = (bias + pad_f).reshape(B * self.n_heads, N_TOK, N_TOK)
            x = layer(x, src_mask=mask)
        out = self.norm(x)
        my_mon = out[:, :6]
        return {"cls": out[:, 63], "my_mon": my_mon, "active_mon": my_mon[ar, act_m], "opp_active_mon": out[:, 6:12][ar, opp_act],
                "active_mv": out[:, 12:60].view(B, 12, 4, d)[:, :6][ar, act_m], "belief_logits": belief_logits, "cand": cand,
                "field": field_vec, "act_m": act_m, "leg_g": leg_g, "leg_sw": leg[:, 8:14]}

    def _heads(self, enc, history, action_mask=None):
        B = history.size(0)
        lg = enc["leg_g"].to(history.dtype)
        latent = self.fusion(torch.cat([enc["cls"], history, lg, enc["active_mon"], enc["opp_active_mon"]], -1))
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
        if action_mask is not None:
            no_move = ~(action_mask & ~self.is_switch).any(dim=1)
            no_switch = ~(action_mask & self.is_switch).any(dim=1)
            type_logits = type_logits.masked_fill(torch.stack([no_move, no_switch], dim=1), -1e9)
        lp_type = F.log_softmax(type_logits, -1)
        lp_move = F.log_softmax(detail.masked_fill(self.is_switch, -1e9), -1)
        lp_switch = F.log_softmax(detail.masked_fill(~self.is_switch, -1e9), -1)
        policy_logits = torch.where(self.is_switch, lp_type[:, 1:2] + lp_switch, lp_type[:, 0:1] + lp_move)
        bl = enc["belief_logits"]
        out = {"policy_logits": policy_logits, "latent": latent, "value": self.value_head(latent),
               "hid_move": bl["move"], "hid_item": bl["item"], "hid_ability": bl["ability"]}
        return out

    def forward(self, my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                history_state=None, action_mask=None, legal_mask=None):
        enc = self._encode_turn(my_team_cat, my_team_num, my_move_num, opp_team_cat, opp_team_num, opp_move_num, field_vec,
                                legal=legal_mask if legal_mask is not None else action_mask)
        cur = self._turn_vec(enc).unsqueeze(1)
        seq = cur if history_state is None else torch.cat([history_state.to(cur.dtype), cur], dim=1)
        out = self._heads(enc, self._history(seq)[:, -1], action_mask)
        out["history_state"] = seq
        return out

    def forward_sequences(self, obs, seq_index, time_index, n_seq, action_mask=None, legal_mask=None):
        enc = self._encode_turn(*obs, legal=legal_mask if legal_mask is not None else action_mask)
        tv = self._turn_vec(enc)
        T = int(time_index.max().item()) + 1
        turn_seq = tv.new_zeros(n_seq, T, tv.size(-1))
        turn_seq[seq_index, time_index] = tv
        out = self._heads(enc, self._history(turn_seq)[seq_index, time_index], action_mask)
        return out

    def get_action(self, my_team_cat, my_team_num, my_move_num, 
                   opp_team_cat, opp_team_num, opp_move_num, 
                   field_vec, history_state=None, action_mask=None, legal_mask=None, prev_move=None):
        """legal_mask: 서버 요청 기반 순수 합법 마스크 (합법성을 입력 특징으로 쓰는 모델(uses_legal)만 받음, action_mask는 로짓 마스킹용)
        prev_move: 직전에 쓴 기술 id [B,2] (BattleTensorEncoder.prev_move). 직전 기술을 입력으로 학습한 모델(prev_act)만 받음"""
        self.eval()
        with torch.no_grad():
            kw = {"legal_mask": legal_mask} if getattr(self, "uses_legal", False) else {}
            if getattr(self, "prev_act", False):
                if prev_move is None and not getattr(self, "_warned_prev", False):      # 학습 때 쓴 입력이 비면 조용히 성능이 떨어지므로 한 번 알림
                    self._warned_prev = True
                    warnings.warn("prev_act 모델인데 prev_move가 없음: 직전 기술을 '없음'으로 둠 (encoder.prev_move(battle)를 넘길 것)")
                kw["prev_move"] = prev_move
            res = self.forward(my_team_cat, my_team_num, my_move_num,
                               opp_team_cat, opp_team_num, opp_move_num, 
                               field_vec, history_state, action_mask, **kw)
        
            p_logits = res["policy_logits"]
            p_probs = F.softmax(p_logits, dim=-1)
            action_idx = torch.argmax(p_probs, dim=-1).item()
        
            val = res["value"].item()
        
        return action_idx, p_probs, val, res["history_state"]


def build_model(cfg: dict) -> nn.Module:
    """구조 인자로 모델 생성: model 키가 v8인 엔티티 토큰 모델."""
    cfg = dict(cfg)
    for k, default in (("history_mode", "flat"), ("fusion_pair", False), ("fusion_res", False), ("switch_skip", False),
                       ("hist_aux", False), ("prev_act", False), ("pm_fusion", False), ("hist_only", False), ("loop_n", 0), ("loop_layers", 2),
                       ("value_gammas", []), ("value_bins", 51), ("pair_scope", "all"), ("pair_feat", False), ("pair_mon", 0), ("bench_off", False),
                       ("tera_feat", False), ("tera_def", False), ("tera_gate", False), ("type_lse", False), ("value_detach", False)):  # 제거된 실험 옵션: 기본값이던 체크포인트만 로드 가능
        if cfg.pop(k, default) != default:
            raise ValueError(f"제거된 옵션 {k}를 쓰는 체크포인트는 더 이상 로드할 수 없음")
    if cfg.get("mon_moves", True) is not True:
        raise ValueError('제거된 옵션 mon_moves="mlp"를 쓰는 체크포인트는 더 이상 로드할 수 없음')
    kind = cfg.pop("model", None)
    if kind == "v8":
        return EntityPokemonNetV8(**cfg)
    raise ValueError(f"지원하지 않는 구조: {kind} (v1/v6/v9 구조는 제거됨)")


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
    model = build_model(cfg)
    load_compatible(model, torch.load(path, map_location=map_location))
    return model

