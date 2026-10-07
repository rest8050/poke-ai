"""모델 v5 self-check (v4 전체 항목은 test_model_v4.py가 그대로 확인): PYTHONPATH=. python tests/test_model_v5.py
1) 롤아웃 == 학습(forward_sequences), 2) 의도 분포 정규화/마스크, 3) 선택지 피처의 의미(상대가 교체 확률 0이면 받을 피해만, 1이면 내 피해만),
4) 기울기 분리(정책 손실은 의도 헤드로 안 가고, 의도 손실은 트렁크로 감), 5) 체크포인트 왕복"""
import os
import tempfile

import torch

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.model_v5 import F_OPT, N_SLOT
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):
    my_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    opp_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    my_num = torch.rand(n, 6, NUM_DIM, generator=gen); my_num[:, :, 0] = 0; my_num[:, 0, 0] = 1
    opp_num = torch.rand(n, 6, NUM_DIM, generator=gen); opp_num[:, :, 0] = 0; opp_num[:, 1, 0] = 1
    my_mv, opp_mv = torch.rand(n, 6, 4, 46, generator=gen), torch.rand(n, 6, 4, 46, generator=gen)
    opp_num[:, 4:, 11:19] = 0                   # 상대 5~6번째 슬롯은 비어 있음
    opp_mv[:, 4:] = 0
    opp_mv[:, :4, 2:] = 0                       # 3~4번째 기술칸은 아직 안 드러남
    return [my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, torch.rand(n, FIELD_DIM, generator=gen)]


torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
cfg = {"model": "v5", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "my_pos": False}
model = build_model(cfg).eval()
assert type(model).__name__ == "EntityPokemonNetV5"
lengths = [5, 3]
obs = [random_obs(L, gen) for L in lengths]
for o in obs:
    o[4][:, :, 1] = 0                           # 상대 기절 없음 (교체 대상이 있어야 분포 검사가 의미 있음)
masks = []
for L in lengths:
    m = torch.zeros(L, 22, dtype=torch.bool); m[:, [0, 2, 5, 9, 11]] = True; masks.append(m)

with torch.no_grad():
    # 1) 롤아웃 == 시퀀스 학습
    step = []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = model(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1])
            hs = out["history_state"]; step.append(out["policy_logits"][0])
    flat = [torch.cat([o[k] for o in obs]) for k in range(7)]
    seq_index = torch.cat([torch.full((L,), b) for b, L in enumerate(lengths)])
    time_index = torch.cat([torch.arange(L) for L in lengths])
    seq_out = model.forward_sequences(flat, seq_index, time_index, len(lengths), torch.cat(masks))
    assert torch.allclose(seq_out["policy_logits"], torch.stack(step), atol=1e-4)
    it = seq_out["intent"]
    N = len(seq_index)
    assert it["kind"].shape == (N, 2) and it["move_lp"].shape == (N, N_SLOT + 1) and it["tgt_lp"].shape == (N, 6) and it["slot_ids"].shape == (N, N_SLOT)

    # 2) 분포 정규화/마스크: 기술 칸 합 1, 교체 대상은 비어 있거나 활성인 칸에 0
    assert torch.allclose(it["move_lp"].exp().sum(-1), torch.ones(N), atol=1e-5)
    assert torch.allclose(it["tgt_lp"].exp().sum(-1), torch.ones(N), atol=1e-5)
    p_tgt = it["tgt_lp"].exp()
    assert (p_tgt[:, 4:] < 1e-6).all() and (p_tgt[:, 1] < 1e-6).all()          # 5~6번째는 빈 슬롯, 1번째 슬롯이 상대 활성

    # 3) 선택지 피처: 상대가 교체하지 않을 확률 1이면 "교체 시 내 피해" 0, 반대면 "받을 피해" 0
    enc = model._encode_turn(*flat)
    lat = torch.randn(N, 64)
    base = model._intent(enc, lat)
    for sw_logit, zero_cols, live_cols in ((-60.0, (2, 3), (0, 1)), (60.0, (0, 1), (2, 3))):
        intent = dict(base); kind = base["kind"].clone(); kind[:, 0] = sw_logit; intent["kind"] = kind
        feat, g = model._option_features(enc, intent)
        assert feat.shape == (N, 6, F_OPT) and g.shape == (N, 2)
        assert feat[..., list(zero_cols)].abs().max() < 1e-4
        assert feat[..., list(live_cols)].abs().max() > 0 and (feat >= 0).all()
    assert abs(g[:, 0].mean().item() - 1.0) < 1e-4                              # 마지막 반복(60.0)의 상대 교체 확률 ~1

# 4) 기울기 분리
model.train()
out = model.forward_sequences(flat, seq_index, time_index, len(lengths), torch.cat(masks))
out["policy_logits"].logsumexp(-1).sum().backward(retain_graph=True)
heads = [model.intent_kind, model.intent_move, model.intent_tgt]
assert all(p.grad is None or p.grad.abs().max() == 0 for h in heads for p in h.parameters()), "정책 손실이 의도 헤드로 샘"
model.zero_grad()
(out["intent"]["kind"].sum() + out["intent"]["move_lp"][:, 0].sum() + out["intent"]["tgt_lp"][:, 0].sum()).backward()
assert model.cls.grad is not None and model.cls.grad.abs().max() > 0, "의도 손실이 트렁크로 안 감"

# 5) 체크포인트 왕복
model.eval()
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "m.pt")
    save_ckpt(model, path)
    m2 = model_from_ckpt(path).eval()
    assert type(m2).__name__ == "EntityPokemonNetV5" and m2.cfg == model.cfg
    with torch.no_grad():
        a = model.forward_sequences(flat, seq_index, time_index, len(lengths), torch.cat(masks))["policy_logits"]
        b = m2.forward_sequences(flat, seq_index, time_index, len(lengths), torch.cat(masks))["policy_logits"]
    assert torch.allclose(a, b, atol=1e-5)
print("ok")
