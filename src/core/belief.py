"""상대 세트 신념(belief) 모듈: 상대 포켓몬의 숨겨진 기술/도구/특성을 예측. 종 ID는 여기에서만 쓰임(트렁크에는 안 들어감).
- 입력: 상대 팀 텐서(ID, 종족값, 배틀 상황 증거)를 자기 임베딩으로 직접 읽음 (트렁크와 공유하는 것 없음)
- 학습: 팀 데이터 사전학습(src/training/pretrain_belief.py, 완전한 세트 라벨) 후 배틀 학습에서는 숨김정보 손실(src/training/hidden_labels.py)만 낮은 학습률로. 정책/가치 손실은 이 모듈에 닿지 않음(출력도 detach해서 사용) → 종 조합으로 승패를 외울 수 없음
- 출력 활용: 확률 상위 기술을 미공개 칸에 "추측 기술"로 배정 → 그 기술의 위력/타입/분류로 매치업 위협(기대 데미지/KO)을 계산해 미공개 토큰, 어텐션 편향, 교체 헤드 입력에 반영 (model_v3)"""
import json

import torch
import torch.nn as nn

from src.core.tensor_encoder import MOVE_EFFECT_DIM, MOVE_NUM_DIM, NUM_DIM, _move_obj, extract_move_features, move_effect_features


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


STAT_SCALE = 0.1      # stat 출력 1 = 능력치 약 10% 차이 (성격 +-10%, 노력치 0~252 약 -10~-25%)


class SetBelief(nn.Module):
    """자기 완결형 상대 세트 신념 모듈: 트렁크/공유 임베딩에 의존하지 않고 자기만의 임베딩으로 입력 텐서(ID, 종족값)를 직접 읽음 → 따로 사전학습/저장/고정 가능.
    입력: opp_cat [B,6,11] (0 도구, 1 특성, 2~3 타입, 5~8 공개 기술, 9 테라 타입, 10 종), opp_num [B,6,NUM_DIM] (11~17 종족값/몸무게, 나머지는 배틀 상황 증거: 선택 입력)
    출력: {"move","item","ability"} 로짓 [B,6,V] (숨은 기술/도구/특성 분포), "stat" [B,6,6] (노력치/성격이 정하는 능력치 배율 exp(STAT_SCALE x stat), 순서 hp/atk/def/spa/spd/spe,
    기준은 매치업의 표준 가정 = 252노력치/무보정). 팀 데이터로 사전학습하면 배틀 상황 입력(ctx)은 기본값이라 미세조정에서 처음 배움. stat은 팀 데이터에만 정답이 있어 배틀 학습에서는 손실이 없음"""

    def __init__(self, vocab_path="data/vocab.json", species_dim=32, hidden=256, species_dropout=0.3):
        super().__init__()
        vv = json.load(open(vocab_path, encoding="utf-8"))
        self.unk_species = vv["species"]["<unk>"]
        self.unk_move = vv["move"]["<unk>"]
        self.species_dropout = species_dropout
        self.cfg = dict(species_dim=species_dim, hidden=hidden, species_dropout=species_dropout)
        self.species_emb = nn.Embedding(max(vv["species"].values()) + 3, species_dim)
        nn.init.normal_(self.species_emb.weight, std=0.1)
        self.stat_proj = nn.Sequential(nn.Linear(7, 32), nn.GELU())
        self.type_emb = nn.Embedding(vv["type"]["<unk>"] + 2, 16)
        self.item_emb = nn.Embedding(vv["item"]["<unk>"] + 2, 32)
        self.ability_emb = nn.Embedding(vv["ability"]["<unk>"] + 2, 32)
        self.move_emb = nn.Embedding(vv["move"]["<unk>"] + 2, 48)
        in_dim = species_dim + 32 + 3 * 16 + 32 + 32 + 48 + 1 + (11 + NUM_DIM - 18)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.move = nn.Linear(hidden, vv["move"]["<unk>"] + 2)       # id 0..unk+1
        self.item = nn.Linear(hidden, vv["item"]["<unk>"] + 2)
        self.ability = nn.Linear(hidden, vv["ability"]["<unk>"] + 2)
        self.stat = nn.Linear(hidden, 6)

    def forward(self, opp_cat, opp_num):
        cat = opp_cat.long()
        sid = cat[..., 10].clamp(0, self.species_emb.num_embeddings - 1)
        if self.training and self.species_dropout > 0:          # 처음 보는 종도 공개 정보/종족값만으로 예측하도록 종을 가림
            sid = torch.where(torch.rand(sid.shape, device=sid.device) < self.species_dropout, torch.full_like(sid, self.unk_species), sid)
        mv = cat[..., 5:9]
        valid = (mv >= 1) & (mv < self.unk_move)                 # 공개된 기술 (0 빈칸, unk/숨김 제외)
        moves = (self.move_emb(mv.clamp(0, self.move_emb.num_embeddings - 1)) * valid[..., None]).sum(2)
        types = torch.cat([self.type_emb(cat[..., k].clamp(0, self.type_emb.num_embeddings - 1)) for k in (2, 3, 9)], -1)
        x = torch.cat([self.species_emb(sid), self.stat_proj(opp_num[..., 11:18]), types,
                       self.item_emb(cat[..., 0].clamp(0, self.item_emb.num_embeddings - 1)),
                       self.ability_emb(cat[..., 1].clamp(0, self.ability_emb.num_embeddings - 1)),
                       moves, valid.sum(-1, keepdim=True).float() / 4.0, opp_num[..., :11], opp_num[..., 18:]], -1)
        h = self.net(x)
        return {"move": self.move(h), "item": self.item(h), "ability": self.ability(h), "stat": self.stat(h)}
