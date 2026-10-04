"""도구/특성/기술 특징 표: 0 초기화 투영이면 기존 동작과 동일 / 미등장 ID 행 0 처리 / ID 드롭아웃. 실행: python tests/test_entity_features.py"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from src.core.model import DeepPokemonBattleTransformerNet, zero_unseen_id_rows

torch.manual_seed(0)
m = DeepPokemonBattleTransformerNet().eval()
assert m.embeddings.has_feat, "data/entity_features.npz 없음 → python tools/build_entity_features.py"
z = np.load("data/entity_features.npz")
seen = torch.from_numpy(np.flatnonzero(z["item_seen"] > 0)[:8]); unseen = torch.from_numpy(np.flatnonzero(z["item_seen"] == 0)[:8])

with torch.no_grad():
    a = m.embeddings._emb("item", torch.cat([seen, unseen]))
    m.embeddings.has_feat = False
    b = m.embeddings._emb("item", torch.cat([seen, unseen]))
    m.embeddings.has_feat = True
assert torch.equal(a, b), "0 초기화 투영인데 출력이 달라짐"

n = zero_unseen_id_rows(m)
with torch.no_grad():
    assert torch.all(m.embeddings.item_embed(unseen) == 0) and torch.equal(m.embeddings.item_embed(seen), b[:len(seen)])
    nz = m.embeddings.item_proj.weight.data.normal_(); assert m.embeddings._emb("item", unseen).abs().sum() > 0, "미등장 개체가 특징으로 표현되지 않음"
    m.embeddings.id_dropout = 1.0
    assert torch.allclose(m.embeddings._emb("item", seen), m.embeddings.item_proj(m.embeddings.item_feat[seen])), "ID 드롭아웃 100%면 특징만 남아야 함"
print("OK: 0 처리한 행", n)
