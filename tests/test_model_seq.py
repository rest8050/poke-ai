"""
인과 트랜스포머 히스토리 self-check: python tests/test_model_seq.py (프로젝트 루트에서 PYTHONPATH=. 로)
1) 롤아웃(한 턴씩 + history_state) == 학습(forward_sequences 한 번에)  → PPO 확률 비율이 맞으려면 필수
2) 인과성: 미래 턴을 바꿔도 과거 턴 출력은 그대로
3) 다른 배틀끼리 섞이지 않음
"""
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM
import torch

from src.core.model import DeepPokemonBattleTransformerNet


def random_obs(n, gen):
    my_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    opp_cat = torch.randint(1, 8, (n, 6, 11), generator=gen)
    my_num = torch.rand(n, 6, NUM_DIM, generator=gen); my_num[:, :, 0] = 0; my_num[:, 0, 0] = 1
    opp_num = torch.rand(n, 6, NUM_DIM, generator=gen); opp_num[:, :, 0] = 0; opp_num[:, 1, 0] = 1
    opp_num[:, 4:, 11:19] = 0  # 미공개 상대 슬롯 (종족값 전부 0 = 패딩)
    return [my_cat, my_num, torch.rand(n, 6, 4, 46, generator=gen), opp_cat, opp_num,
            torch.rand(n, 6, 4, 46, generator=gen), torch.rand(n, FIELD_DIM, generator=gen)]


torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
model = DeepPokemonBattleTransformerNet().eval()
torch.nn.init.normal_(model.history_out.weight, std=0.2)  # 0 초기화면 히스토리가 항상 0이라 검사가 무의미
torch.nn.init.normal_(model.history_out.bias, std=0.2)

lengths = [5, 9, 1, 3]
obs = [random_obs(L, gen) for L in lengths]
masks = []
for L in lengths:
    m = torch.zeros(L, 22, dtype=torch.bool)
    m[:, [0, 2, 5, 9, 11]] = True
    masks.append(m)

with torch.no_grad():
    # 1) 롤아웃 방식
    step_logits, step_values = [], []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = model(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1])
            hs = out["history_state"]
            assert hs.shape == (1, t + 1, 832)
            step_logits.append(out["policy_logits"][0]); step_values.append(out["value"][0])
    step_logits, step_values = torch.stack(step_logits), torch.stack(step_values)

    # 학습 방식: 배틀을 섞은 순서로 넣어도 같아야 함
    flat = [torch.cat([o[k] for o in obs]) for k in range(7)]
    seq_index = torch.cat([torch.full((L,), b) for b, L in enumerate(lengths)])
    time_index = torch.cat([torch.arange(L) for L in lengths])
    mask = torch.cat(masks)
    perm = torch.randperm(len(seq_index))
    seq_out = model.forward_sequences([x[perm] for x in flat], seq_index[perm], time_index[perm], len(lengths), mask[perm])
    inv = torch.argsort(perm)
    assert torch.allclose(seq_out["policy_logits"][inv], step_logits, atol=1e-4), \
        (seq_out["policy_logits"][inv] - step_logits).abs().max()
    assert torch.allclose(seq_out["value"][inv], step_values, atol=1e-5)

    # 2) 인과성: 배틀 0의 3번째 턴 입력을 바꾸면 0~2턴은 그대로, 3턴 이후는 바뀜
    flat2 = [x.clone() for x in flat]
    flat2[6][3] += 1.0
    out2 = model.forward_sequences(flat2, seq_index, time_index, len(lengths), mask)
    base = model.forward_sequences(flat, seq_index, time_index, len(lengths), mask)
    assert torch.allclose(out2["value"][:3], base["value"][:3], atol=1e-6)
    assert (out2["value"][3:5] - base["value"][3:5]).abs().max() > 1e-5
    # 3) 다른 배틀(1번 이후)은 영향 없음
    assert torch.allclose(out2["value"][5:], base["value"][5:], atol=1e-6)

# 4) 역전파: 히스토리 트랜스포머까지 gradient가 흐름
model.train(False)
out = model.forward_sequences(flat, seq_index, time_index, len(lengths), mask)
out["value"].sum().backward()
assert model.history_seq.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
assert model.turn_pos.weight.grad.abs().sum() > 0
print("model seq OK: 롤아웃 == 학습 시퀀스, 인과성, 배틀 간 분리, gradient")

# 5) 깊은 헤드(ResHead): 무작위 가중치로 켜도 롤아웃 == 학습 시퀀스, 그리고 실제로 출력에 영향을 줌
deep = DeepPokemonBattleTransformerNet(deep_heads=True).eval()
deep.load_state_dict(model.state_dict(), strict=False)
for p in list(deep.move_deep.parameters()) + list(deep.switch_deep.parameters()):
    torch.nn.init.normal_(p, std=0.05)
with torch.no_grad():
    d_step = []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = deep(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1])
            hs = out["history_state"]
            d_step.append(out["policy_logits"][0])
    d_step = torch.stack(d_step)
    d_seq = deep.forward_sequences(flat, seq_index, time_index, len(lengths), mask)
    assert torch.allclose(d_seq["policy_logits"], d_step, atol=1e-4), (d_seq["policy_logits"] - d_step).abs().max()
    assert (d_seq["policy_logits"] - base["policy_logits"]).abs().max() > 1e-3
print("deep heads OK: 롤아웃 == 학습 시퀀스, 가지가 출력에 영향")

