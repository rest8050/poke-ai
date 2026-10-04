"""유형별 토큰 히스토리(history_mode="typed") 검증: (1) 기본 구조 체크포인트 호환 유지 (2) 시퀀스 계산 == 턴별 추론 (3) 미래 턴이 과거 출력에 영향 없음.
실행: python -m tests.test_typed_history"""
import numpy as np
import torch

from src.core.model import DeepPokemonBattleTransformerNet as M, load_compatible, model_from_ckpt, save_ckpt

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
torch.manual_seed(0)

# (1) 기본 구조(flat)로 예전 체크포인트를 읽으면: move_encoder.fc.0.weight는 조건부 선공권/선공권 우열 피처
# 추가로 입력 열이 44->46으로 늘어난 것이라 "확장"은 정상. team_cross(셀프->양방향 크로스 반복)는 완전히 새
# 구조라 옛 my/opp_team_attention·cross_attn·cross_norm 키는 이름이 안 맞아 버려지는 게 정상(무작위 시작).
m0 = M()
exp, skipped = load_compatible(m0, torch.load("checkpoints/supervised_v2_fp_all11.pt", map_location="cpu"))
assert exp == ["embeddings.move_encoder.fc.0.weight"], exp
assert skipped and all(k.startswith(("my_team_attention.", "opp_team_attention.", "cross_attn.", "cross_norm.")) for k in skipped), skipped

# 실제 데이터 한 판
z = np.load("data/dagger/dg3_ho.npz")
lens = z["lens"]
T = int(min(lens[0], 12))
obs = [torch.tensor(z[k][:T]) for k in KEYS]
mask = torch.tensor(z["action_mask"][:T])
# 이 npz는 조건부 선공권/선공권 우열 피처(44→46) 추가 전에 만들어짐 → 기술 수치 뒤에 0열 2개를 덧붙여 새 차원에 맞춤
# (이 테스트는 실제 특징값이 아니라 flat/typed 계산 일관성만 보므로 0 패딩으로 충분)
for i in (2, 5):
    obs[i] = torch.cat([obs[i], torch.zeros(*obs[i].shape[:-1], 2, dtype=obs[i].dtype)], dim=-1)

for mode in ("flat", "typed"):
    m = M(pokemon_embed_dim=64, history_dim=96, latent_dim=96, num_layers=2, history_mode=mode).eval()
    # history_out은 0 초기화라 그대로면 히스토리가 항상 0 → 검증이 무의미. 무작위 값으로 바꿈
    torch.nn.init.normal_(m.history_out.weight, std=0.3)
    with torch.no_grad():
        full = m.forward_sequences(obs, torch.zeros(T, dtype=torch.long), torch.arange(T), 1, mask)
        # (2) 턴별 추론 == 시퀀스 계산
        hs, outs = None, []
        for t in range(T):
            o = m.forward(*[x[t:t + 1] for x in obs], history_state=hs, action_mask=mask[t:t + 1])
            hs = o["history_state"]
            outs.append(o["policy_logits"])
        step = torch.cat(outs)
        assert torch.isfinite(full["policy_logits"][mask]).all()
        assert torch.allclose(full["policy_logits"][mask], step[mask], atol=1e-4), (mode, (full["policy_logits"] - step).abs().max())
        # (3) 인과성: 마지막 3턴을 바꿔도 앞부분 출력은 같아야 함
        obs2 = [x.clone() for x in obs]
        for x in obs2:
            x[T - 3:] = x[T - 3:].flip(0) * 1.0 if x.dtype.is_floating_point else x[T - 3:]
        obs2[6][T - 3:] += 5.0  # field_vec 크게 변경
        alt = m.forward_sequences(obs2, torch.zeros(T, dtype=torch.long), torch.arange(T), 1, mask)
        assert torch.allclose(full["policy_logits"][:T - 3][mask[:T - 3]], alt["policy_logits"][:T - 3][mask[:T - 3]], atol=1e-4), mode
    print(f"{mode}: 파라미터 {sum(p.numel() for p in m.parameters())/1e6:.2f}M | 시퀀스=턴별 추론 OK | 인과성 OK")

# (4) 저장/불러오기: 사이드카 cfg로 같은 구조가 복원되는지
import os, tempfile
m = M(pokemon_embed_dim=64, history_dim=96, latent_dim=96, num_layers=2, history_mode="typed")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "x.pt"); save_ckpt(m, p)
    r = model_from_ckpt(p)
    assert r.history_mode == "typed" and r.cfg == m.cfg
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), r.state_dict().values()))
print("ALL OK")
