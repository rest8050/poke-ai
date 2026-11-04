"""모델 v9(사슬 헤드) self-check: PYTHONPATH=. python tests/test_model_v9.py
1) 출력 형태와 Q 경로 항등식 Q = Σ p q̂, 2) forced 변형은 policy_logits == q_logits(합법 밖 -1e9), aux는 기존 정책 그대로,
3) r 헤드는 실제 둔 쌍(a,o)의 은닉만으로 Δ를 냄, 4) 손실이 사슬 파라미터까지 기울기를 보냄(몸통 포함), 5) 체크포인트 왕복"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, ".")
from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.training.chain_targets import N_CLS, N_ACT
from tests.test_model_v8 import random_obs

torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
small = {"d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False}
L = 6
obs = random_obs(L, gen)
mask = torch.zeros(L, 22, dtype=torch.bool); mask[:, [0, 1, 2, 3, 5, 9, 10, 11]] = True
seq, tim = torch.zeros(L, dtype=torch.long), torch.arange(L)
for variant in ("aux", "forced"):
    m = build_model({"model": "v9", "chain_variant": variant, **small}).eval()
    assert type(m).__name__ == "EntityPokemonNetV9" and m.cfg["model"] == "v9"
    with torch.no_grad():
        out = m.forward_sequences(obs, seq, tim, 1, action_mask=mask)
    R = m.cfg["chain_r"]
    assert out["opp_logits"].shape == (L, N_CLS) and out["q_hat"].shape == (L, N_ACT, N_CLS) and out["chain_h"].shape == (L, N_ACT, N_CLS, R)
    assert out["q_pi"].shape == (L, 22) and out["q_logits"].shape == (L, 22) and out["policy_logits"].shape == (L, 22)
    q = (out["q_hat"] * out["opp_logits"].softmax(-1)[:, None]).sum(-1)
    assert torch.allclose(out["q_pi"][:, :N_ACT], q, atol=1e-6), "Q = Σ p q̂"
    assert torch.allclose(out["q_logits"][~mask], torch.tensor(-1e9 - torch.logsumexp(torch.zeros(1), 0).item())) or (out["q_logits"][~mask] < -1e8).all()
    assert torch.allclose(out["q_logits"].exp().sum(-1), torch.ones(L), atol=1e-5), "마스크 후 정규화"
    if variant == "forced":
        assert torch.equal(out["policy_logits"], out["q_logits"]) and "policy_logits_base" in out
    else:
        assert "policy_logits_base" not in out and not torch.equal(out["policy_logits"], out["q_logits"])
    a = torch.tensor([0, 1, 9, 5, 10, 3]); o = torch.tensor([0, 2, 9, 4, 1, 7])
    h = out["chain_h"][torch.arange(L), a, o]
    r = m.outcome(h)
    assert r["dhp"].shape == (L, 12) and r["faint"].shape == (L, 12) and r["act_my"].shape == (L, 6) and r["act_opp"].shape == (L, 6) and r["boost"].shape == (L, 10)

    m.train()                                                   # 기울기: 사슬 손실이 몸통(임베딩/층)까지 도달
    out = m.forward_sequences(obs, seq, tim, 1, action_mask=mask)
    h = out["chain_h"][torch.arange(L), a, o]
    loss = out["opp_logits"].logsumexp(-1).mean() + sum(v.float().pow(2).mean() for v in m.outcome(h).values()) + out["q_pi"].mean()
    loss.backward()
    for n in ("opp_head.0.weight", "r_out.weight", "q_out.weight", "act_mv.weight", "w_opp.weight", "fusion.0.weight", "layers.0.linear1.weight"):
        g = dict(m.named_parameters())[n].grad
        assert g is not None and g.abs().sum() > 0, n

    m.eval()
    path = os.path.join(tempfile.mkdtemp(), "v9.pt")
    save_ckpt(m, path)
    m2 = model_from_ckpt(path).eval()
    with torch.no_grad():
        o1, o2 = m.forward_sequences(obs, seq, tim, 1, action_mask=mask), m2.forward_sequences(obs, seq, tim, 1, action_mask=mask)
    assert type(m2).__name__ == "EntityPokemonNetV9" and torch.allclose(o1["policy_logits"], o2["policy_logits"]), "체크포인트 왕복"
print("model v9 OK")
