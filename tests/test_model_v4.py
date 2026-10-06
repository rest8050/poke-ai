"""엔티티 토큰 모델 v4 self-check (test_model_v3 항목 + v4 전용 6~8): PYTHONPATH=. python tests/test_model_v4.py
1) 롤아웃(한 턴씩) == 학습(forward_sequences), 2) 인과성, 3) 빈 슬롯 입력은 출력에 영향 없음 / 공개된 기술 수와 미공개 칸 내용은 영향,
4) 기술/후보 정보와 상대 기술 위력이 교체 점수에 반영, 5) 신념: 상대 종 ID는 신념에만 쓰이고 내 쪽 종 ID는 무시, 정책/가치 손실은 신념에 안 닿고 숨김정보 손실은 신념 밖에 안 닿음,
6) 기술 속성 표, 7) 체크포인트 왕복, 8) 상위 8개 후보 구조(가중치 범위/공개 후 0), 숨은 기술 예측과 노력치 예측이 정책에 반영"""
import json
import os
import tempfile

import torch

from src.core.belief import build_move_table
from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM


def random_obs(n, gen):
    my_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    opp_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    my_num = torch.rand(n, 6, NUM_DIM, generator=gen); my_num[:, :, 0] = 0; my_num[:, 0, 0] = 1
    opp_num = torch.rand(n, 6, NUM_DIM, generator=gen); opp_num[:, :, 0] = 0; opp_num[:, 1, 0] = 1
    my_mv, opp_mv = torch.rand(n, 6, 4, 46, generator=gen), torch.rand(n, 6, 4, 46, generator=gen)
    opp_num[:, 4:, 11:19] = 0                   # 상대 5~6번째 슬롯은 통째로 비어 있음 (HP 종족값 0 = 패딩)
    opp_mv[:, 4:] = 0
    opp_mv[:, :4, 2:] = 0                       # 앞 4마리도 3~4번째 기술칸은 아직 안 드러남 (수치 전부 0)
    return [my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, torch.rand(n, FIELD_DIM, generator=gen)]


torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
model = build_model({"model": "v4", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64}).eval()
print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (테스트용 소형)")

lengths = [5, 9, 1, 3]
obs = [random_obs(L, gen) for L in lengths]
masks = []
for L in lengths:
    m = torch.zeros(L, 22, dtype=torch.bool); m[:, [0, 2, 5, 9, 11]] = True; masks.append(m)
Z = lambda L: torch.zeros(L, dtype=torch.long)
seq = lambda o, L=5, i=0: model.forward_sequences(o, Z(L), torch.arange(L), 1, masks[i])

with torch.no_grad():
    # 1) 롤아웃 == 시퀀스 학습
    step_logits, step_values = [], []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = model(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1])
            hs = out["history_state"]
            assert hs.shape == (1, t + 1, model.turn_proj.in_features) and out["policy_logits"].shape == (1, 22)
            step_logits.append(out["policy_logits"][0]); step_values.append(out["value"][0])
    step_logits, step_values = torch.stack(step_logits), torch.stack(step_values)
    flat = [torch.cat([o[k] for o in obs]) for k in range(7)]
    seq_index = torch.cat([torch.full((L,), b) for b, L in enumerate(lengths)])
    time_index = torch.cat([torch.arange(L) for L in lengths])
    mask = torch.cat(masks)
    perm = torch.randperm(len(seq_index)); inv = torch.argsort(perm)
    seq_out = model.forward_sequences([x[perm] for x in flat], seq_index[perm], time_index[perm], len(lengths), mask[perm])
    assert torch.allclose(seq_out["policy_logits"][inv], step_logits, atol=1e-4), (seq_out["policy_logits"][inv] - step_logits).abs().max()
    assert torch.allclose(seq_out["value"][inv], step_values, atol=1e-5)

    # 2) 인과성: 미래 턴을 바꿔도 과거 턴 출력 그대로
    obs2 = [x.clone() for x in obs[1]]
    obs2[1][6:] = torch.rand_like(obs2[1][6:])
    a = model.forward_sequences(obs[1], Z(9), torch.arange(9), 1, masks[1])["policy_logits"]
    b2 = model.forward_sequences(obs2, Z(9), torch.arange(9), 1, masks[1])["policy_logits"]
    assert torch.allclose(a[:6], b2[:6], atol=1e-5) and not torch.allclose(a[6:], b2[6:], atol=1e-3)

    base_o = seq(obs[0])["policy_logits"]
    # 3) 빈 슬롯(포켓몬 자체가 없음)의 입력은 출력에 영향 없음. 공개된 기술 수와 미공개 칸 내용은 출력에 반영
    y = [t.clone() for t in obs[0]]
    y[3][:, 4:, :] = torch.randint(1, 8, y[3][:, 4:, :].shape)
    assert torch.allclose(base_o, seq(y)["policy_logits"], atol=1e-5), "빈 슬롯 입력이 출력에 샘"
    u = [t.clone() for t in obs[0]]
    u[5][:, 1, 1:, :] = 0                                           # 상대 활성이 기술을 1개만 공개한 상태
    assert (seq(u)["policy_logits"] - base_o).abs().max() > 1e-4, "공개된 기술 수가 출력에 반영돼야 함"

    # 4) 내 활성의 기술/벤치 후보의 기술, 상대 활성의 기술 위력이 점수에 반영
    z = [t.clone() for t in obs[0]]
    z[2][:, 0, 1, :] = torch.rand_like(z[2][:, 0, 1, :])
    assert (seq(z)["policy_logits"] - base_o).abs().max() > 1e-4
    w = [t.clone() for t in obs[0]]
    w[2][:, 3, 0, :] = torch.rand_like(w[2][:, 3, 0, :])
    assert (seq(w)["policy_logits"] - base_o).abs().max() > 1e-4

    def alive_obs(power):                                          # 기절 플래그를 0으로 고정해 후보가 살아 있게 함
        q = [t.clone() for t in obs[0]]
        q[1][:, :, 1] = 0; q[4][:, :, 1] = 0
        q[5][:, 1, 0, 4] = 0.33; q[5][:, 1, 0, 0] = power
        return seq(q)["policy_logits"]
    assert (alive_obs(0.2)[:, 8:14] - alive_obs(0.9)[:, 8:14]).abs().max() > 1e-4

    # 5) 신념: 출력 형태, 상대 종 ID는 신념에 반영, 내 쪽 종 ID는 무시
    out_b = seq(obs[0])
    assert out_b["hid_move"].shape[:2] == (5, 6) and out_b["hid_item"].shape[:2] == (5, 6) and out_b["hid_ability"].shape[:2] == (5, 6)
    sb = [t.clone() for t in obs[0]]; sb[3][:, 1, 10] = obs[0][3][:, 1, 10] + 3
    assert (seq(sb)["hid_move"] - out_b["hid_move"]).abs().max() > 1e-4, "신념이 상대 종 ID를 써야 함"
    sm = [t.clone() for t in obs[0]]; sm[0][:, :, 10] = obs[0][0][:, :, 10] + 5
    assert torch.allclose(seq(sm)["policy_logits"], base_o, atol=1e-5), "내 쪽 종 ID는 무시"

# 5b) 기울기 분리: 정책/가치 손실은 신념에 닿지 않고, 숨김정보 손실은 신념 밖에 닿지 않음 (학습 모드)
model.train(); model.zero_grad()
g1 = seq(obs[0])
(g1["policy_logits"][masks[0]].sum() + g1["value"].sum()).backward()
assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.belief.parameters()), "정책 손실이 신념에 닿음"
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layers.parameters())
model.zero_grad()
g2 = seq(obs[0])
(g2["hid_move"].sum() + g2["hid_item"].sum() + g2["hid_ability"].sum()).backward()
assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in model.named_parameters() if not n.startswith("belief.")), "숨김정보 손실이 신념 밖에 닿음"
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.belief.parameters())
model.zero_grad(); model.eval()

# 6) 기술 속성 표: 지진 = 위력 100(0.5), 물리(0.33), 땅 타입(9/20)
t = build_move_table(); i = json.load(open("data/vocab.json", encoding="utf-8"))["move"]["earthquake"]
assert abs(float(t[i, 0]) - 0.5) < 1e-6 and abs(float(t[i, 4]) - 0.33) < 1e-6 and abs(float(t[i, 6]) - 9 / 20) < 1e-6

# 7) 체크포인트 왕복 (cfg로 같은 구조 복원)
path = os.path.join(tempfile.mkdtemp(), "m.pt")
save_ckpt(model, path)
m2 = model_from_ckpt(path).eval()
assert type(m2).__name__ == "EntityPokemonNetV4" and m2.cfg == model.cfg
with torch.no_grad():
    assert torch.allclose(m2.forward_sequences(obs[0], Z(5), torch.arange(5), 1, masks[0])["policy_logits"], base_o, atol=1e-6)

# 8) v4 전용: 후보 구조, 숨은 기술/노력치 예측이 정책에 반영
from src.core.model_v4 import K_GUESS
with torch.no_grad():
    o = obs[0]
    bl = model.belief(o[3], o[4])
    gs = model._top_guess(bl, o[3], o[5], o[4])
    assert gs["idx"].shape == (5, 6, K_GUESS) and gs["w"].shape == (5, 6, K_GUESS) and gs["pseudo"].shape[:3] == (5, 6, K_GUESS)
    assert gs["w"].max() <= 1.0 + 1e-6 and (gs["w"][:, 4:] == 0).all(), "빈 슬롯은 후보 가중치 0"           # 상대 5~6번째 슬롯은 통째로 비어 있음
    full = [t.clone() for t in o]; full[5][:, :4, :, :] = torch.rand_like(full[5][:, :4, :, :]) + 0.1          # 4칸 모두 공개 → 후보 가중치 0
    assert (model._top_guess(model.belief(full[3], full[4]), full[3], full[5], full[4])["w"][:, :4] == 0).all()
    base = seq(o)["policy_logits"]
    strong = json.load(open("data/vocab.json", encoding="utf-8"))["move"]["earthquake"]
    saved_b, saved_s = model.belief.move.bias.clone(), model.belief.stat.bias.clone()
    model.belief.move.bias[strong] += 12.0                                    # 숨은 기술 예측이 바뀌면 정책이 달라져야 함
    assert (seq(o)["policy_logits"] - base).abs().max() > 1e-5, "숨은 기술 예측이 정책에 반영돼야 함"
    model.belief.move.bias.copy_(saved_b); model.belief.stat.bias += 4.0     # 상대 능력치 예측이 바뀌면 정책이 달라져야 함
    assert (seq(o)["policy_logits"] - base).abs().max() > 1e-5, "능력치 예측이 정책에 반영돼야 함"
    model.belief.stat.bias.copy_(saved_s)
print("model v4 OK: 롤아웃 == 시퀀스, 인과성, 패딩/미공개 칸, 매치업 반영, 신념 격리, 기술 표, 체크포인트 왕복, 상위 8개 후보, 숨은 기술/능력치 예측 반영")
