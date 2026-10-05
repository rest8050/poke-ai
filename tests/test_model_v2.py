"""엔티티 토큰 모델(model_v2) self-check: PYTHONPATH=. python tests/test_model_v2.py
1) 토큰 64개 구성/출력 형태, 2) 롤아웃(한 턴씩) == 학습(forward_sequences), 3) 인과성, 4) 패딩 토큰(빈 슬롯/안 드러난 기술)이 출력에 영향 없음,
5) 기술/포켓몬 정보가 정말 해당 헤드 출력에 영향, 6) 체크포인트 왕복(cfg로 같은 구조 복원)"""
import os
import tempfile

import torch

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
    opp_mv[:, :4, 2:] = 0                       # 앞 4마리도 3~4번째 기술칸은 아직 안 드러남 (수치 전부 0 = 패딩)
    return [my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, torch.rand(n, FIELD_DIM, generator=gen)]


torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
import sys
SW = "--switch-ctx" in sys.argv
model = build_model({"model": "entity", "d_model": 64, "n_layers": 2, "n_heads": 4, "history_dim": 64, "latent_dim": 64, "switch_ctx": SW}).eval()
print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (테스트용 소형)")

lengths = [5, 9, 1, 3]
obs = [random_obs(L, gen) for L in lengths]
masks = []
for L in lengths:
    m = torch.zeros(L, 22, dtype=torch.bool); m[:, [0, 2, 5, 9, 11]] = True; masks.append(m)

with torch.no_grad():
    # 1)+2) 롤아웃 == 시퀀스 학습
    step_logits, step_values = [], []
    for b, L in enumerate(lengths):
        hs = None
        for t in range(L):
            out = model(*[x[t:t + 1] for x in obs[b]], history_state=hs, action_mask=masks[b][t:t + 1])
            hs = out["history_state"]
            assert hs.shape == (1, t + 1, 64) and out["policy_logits"].shape == (1, 22)
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

    # 3) 인과성: 미래 턴을 바꿔도 과거 턴 출력 그대로
    obs2 = [x.clone() for x in obs[1]]
    obs2[1][6:] = torch.rand_like(obs2[1][6:])
    a = model.forward_sequences(obs[1], torch.zeros(9, dtype=torch.long), torch.arange(9), 1, masks[1])["policy_logits"]
    b2 = model.forward_sequences(obs2, torch.zeros(9, dtype=torch.long), torch.arange(9), 1, masks[1])["policy_logits"]
    assert torch.allclose(a[:6], b2[:6], atol=1e-5) and not torch.allclose(a[6:], b2[6:], atol=1e-3)

    # 4) 패딩 토큰 내용이 바뀌어도 출력 불변 (상대의 빈 슬롯 5~6번째의 포켓몬/기술 입력과 안 드러난 기술칸)
    x = [t.clone() for t in obs[0]]
    y = [t.clone() for t in obs[0]]
    y[3][:, 4:, :] = torch.randint(1, 8, y[3][:, 4:, :].shape)     # 빈 슬롯의 범주 입력
    y[3][:, :4, 7] = torch.randint(1, 8, y[3][:, :4, 7].shape)    # 안 드러난 기술칸(3번째)의 기술 ID는 패딩이라 무시돼야 함
    o1 = model.forward_sequences(x, torch.zeros(5, dtype=torch.long), torch.arange(5), 1, masks[0])["policy_logits"]
    o2 = model.forward_sequences(y, torch.zeros(5, dtype=torch.long), torch.arange(5), 1, masks[0])["policy_logits"]
    assert torch.allclose(o1, o2, atol=1e-5), "패딩 슬롯 입력이 출력에 샘: " + str((o1 - o2).abs().max())

    # 5) 내 활성의 기술 토큰/후보 포켓몬 정보가 해당 점수에 실제로 영향
    z = [t.clone() for t in obs[0]]
    z[2][:, 0, 1, :] = torch.rand_like(z[2][:, 0, 1, :])             # 내 활성(0번)의 2번째 기술 수치 변경
    o3 = model.forward_sequences(z, torch.zeros(5, dtype=torch.long), torch.arange(5), 1, masks[0])["policy_logits"]
    assert (o3 - o1).abs().max() > 1e-4
    w = [t.clone() for t in obs[0]]
    w[2][:, 3, 0, :] = torch.rand_like(w[2][:, 3, 0, :])             # 내 벤치(3번)의 기술 수치 변경 -> 교체 후보 3의 점수에 반영돼야 함
    o4 = model.forward_sequences(w, torch.zeros(5, dtype=torch.long), torch.arange(5), 1, masks[0])["policy_logits"]
    assert (o4 - o1).abs().max() > 1e-4

# 6) 체크포인트 왕복
path = os.path.join(tempfile.mkdtemp(), "m.pt")
save_ckpt(model, path)
m2 = model_from_ckpt(path).eval()
assert type(m2).__name__ == "EntityPokemonNet" and m2.cfg == model.cfg
with torch.no_grad():
    assert torch.allclose(m2.forward_sequences(obs[0], torch.zeros(5, dtype=torch.long), torch.arange(5), 1, masks[0])["policy_logits"], o1, atol=1e-6)
print("entity model OK: 롤아웃 == 시퀀스, 인과성, 패딩 불변, 기술/후보 영향, 체크포인트 왕복")
