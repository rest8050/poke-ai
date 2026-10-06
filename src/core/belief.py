"""상대 세트 신념(belief) 모듈: 상대 포켓몬의 숨겨진 기술/도구/특성을 예측. 종 ID는 여기에서만 쓰임(트렁크에는 안 들어감).
- 입력(모두 detach): 상대 포켓몬 토큰 입력 조각(종족값 투영, 도구/특성/타입/상태 임베딩, 배틀 수치, 이벤트 증거) + 공개 기술 임베딩 합 + 공개 기술 수
- 학습: 숨김정보 손실(src/training/hidden_labels.py)만. 정책/가치 손실은 이 모듈에 닿지 않음(출력도 detach해서 사용) → 종 조합으로 승패를 외울 수 없음
- 출력 활용: 확률 상위 기술을 미공개 칸에 "추측 기술"로 배정 → 그 기술의 위력/타입/분류로 매치업 위협(기대 데미지/KO)을 계산해 미공개 토큰, 어텐션 편향, 교체 헤드 입력에 반영 (model_v3)"""
import json

import torch
import torch.nn as nn

from src.core.tensor_encoder import MOVE_EFFECT_DIM, MOVE_NUM_DIM, _move_obj, extract_move_features, move_effect_features


def build_move_table(vocab_path="data/vocab.json"):
    """기술 ID -> 수치 특징 행 [V, MOVE_NUM_DIM] (인코더가 공개된 기술에 넣는 것과 같은 값: 위력/명중/PP/우선도/분류/부가효과/타입 + 효과 벡터).
    ponytail: 엔진 데이터(poke_env)에서 한 번 계산(0.1초)해서 버퍼로 둠. 상대 의존 값(상성 배율, 선공 우열)은 0"""
    v = json.load(open(vocab_path, encoding="utf-8"))["move"]
    tab = torch.zeros(v["<unk>"] + 2, MOVE_NUM_DIM)
    for name, i in v.items():
        if name.startswith("<"):
            continue
        m = _move_obj(name)
        if m is None:
            continue
        tab[i, :7] = torch.tensor(extract_move_features(m, None))
        tab[i, 8:8 + MOVE_EFFECT_DIM] = torch.tensor(move_effect_features(name))
    return tab


class SetBelief(nn.Module):
    def __init__(self, in_dim, vocab_path="data/vocab.json", species_dim=32, hidden=256, species_dropout=0.3):
        super().__init__()
        vv = json.load(open(vocab_path, encoding="utf-8"))
        self.unk_species = vv["species"]["<unk>"]
        self.species_dropout = species_dropout
        self.species_emb = nn.Embedding(max(vv["species"].values()) + 3, species_dim)
        nn.init.normal_(self.species_emb.weight, std=0.1)
        self.net = nn.Sequential(nn.Linear(in_dim + species_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.move = nn.Linear(hidden, vv["move"]["<unk>"] + 2)       # id 0..unk+1
        self.item = nn.Linear(hidden, vv["item"]["<unk>"] + 2)
        self.ability = nn.Linear(hidden, vv["ability"]["<unk>"] + 2)

    def forward(self, x, species_id):
        """x [B,6,F] (detach된 입력), species_id [B,6] -> {"move","item","ability"} 로짓 [B,6,V]"""
        sid = species_id.long().clamp(0, self.species_emb.num_embeddings - 1)
        if self.training and self.species_dropout > 0:          # 처음 보는 종도 공개 정보/종족값만으로 예측하도록 종을 가림
            sid = torch.where(torch.rand(sid.shape, device=sid.device) < self.species_dropout, torch.full_like(sid, self.unk_species), sid)
        h = self.net(torch.cat([x, self.species_emb(sid)], -1))
        return {"move": self.move(h), "item": self.item(h), "ability": self.ability(h)}
