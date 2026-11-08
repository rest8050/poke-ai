import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import warnings

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
    """구조 인자로 모델 생성: model 키가 v8(model_v8.py)/v6(model_v6.py, 서버 탑재)인 엔티티 토큰 모델."""
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
    if kind == "v8":                                     # v6 + 특성 EXEC 규칙(matchup_v7) + 합법 요약 슬롯 순서 통로 제거 + 숨은 특성/도구 기대 임베딩 + 융합층 활성 토큰 입력
        from src.core.model_v8 import EntityPokemonNetV8
        return EntityPokemonNetV8(**cfg)
    if kind == "v6":                                     # 서버 탑재 구조
        from src.core.model_v6 import EntityPokemonNetV6
        return EntityPokemonNetV6(**cfg)
    raise ValueError(f"지원하지 않는 구조: {kind} (v1/v9 구조는 제거됨)")


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

