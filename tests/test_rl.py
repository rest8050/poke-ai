"""RL 손실(PPO 클립 + KL 앵커) self-check: PYTHONPATH=. python tests/test_rl.py"""
import numpy as np
import torch
import torch.nn.functional as F

from src.training.train_rl import phi_advantage, policy_terms

mask = torch.zeros(3, 22, dtype=torch.bool)
mask[:, :3] = True
mask[2, 1:3] = False        # 3번째 결정은 선택지가 하나뿐
a = torch.tensor([1, 0, 0])


def run(logits, old_lp, adv, ref_lp, clip=0.2, kl=0.0, act=a):
    lg = logits.clone().requires_grad_(True)
    lp_all = F.log_softmax(lg.masked_fill(~mask, -1e9), -1)
    loss, st = policy_terms(lp_all, act, mask, old_lp, adv, ref_lp, clip, kl)
    loss.backward()
    return loss, st, lg.grad


logits = torch.zeros(3, 22)
lp_all0 = F.log_softmax(logits.masked_fill(~mask, -1e9), -1)
old = lp_all0.gather(-1, a[:, None]).squeeze(-1)

# 1) 비율 1에서: advantage 양수인 행동의 logit은 오르고(기울기 음수), 음수면 내려감. 선택지 하나뿐인 결정은 기울기 0
loss, st, g = run(logits, old, torch.tensor([1.0, -1.0, 1.0]), lp_all0)
assert g[0, 1] < 0 and g[1, 0] > 0 and g[2].abs().sum() == 0 and torch.isfinite(g).all()
assert abs(st["clipfrac"]) == 0 and abs(st["kl_old"]) < 1e-6 and abs(st["kl_ref"]) < 1e-6

# 2) 이미 클립 밖으로 올라간 행동(비율 > 1+ε, advantage > 0)은 더 올리는 기울기가 없음, advantage < 0이면 되돌리는 기울기는 있음
old_low = old - 0.5
_, st, g = run(logits, old_low, torch.tensor([1.0, 1.0, 0.0]), lp_all0)
assert g[0].abs().sum() == 0 and st["clipfrac"] > 0
_, _, g = run(logits, old_low, torch.tensor([-1.0, 1.0, 0.0]), lp_all0)
assert g[0, 1] > 0

# 3) KL 앵커: 시작 정책과 같으면 0, 다르면 양수이고 기울기가 시작 정책 쪽으로 당김 (advantage 0이어도)
zero = torch.zeros(3)
_, st, g = run(logits, old, zero, lp_all0, kl=1.0)
assert abs(st["kl_ref"]) < 1e-6 and g.abs().sum() < 1e-6
shifted = logits.clone(); shifted[0, 0] += 2.0
ref_lp = F.log_softmax(shifted.masked_fill(~mask, -1e9), -1)          # 시작 정책은 행동 0을 더 선호
_, st, g = run(logits, old, zero, ref_lp, kl=1.0)
assert st["kl_ref"] > 0 and g[0, 0] < 0     # 현재 정책이 시작 정책 쪽(행동 0 쪽)으로 당겨짐

# 4) Φ 비평가 advantage: λ=0이면 한 턴 변화 γΦ' − Φ (마지막은 승패 − Φ), λ=1이면 처음 값이 γ^(T-1)·승패 − Φ_0 (중간 Φ와 무관)
G = 0.995
phi = np.array([0.1, -0.2, 0.5, 0.3, -0.4])
one = phi_advantage(phi, 1.0, 0.0)
assert np.allclose(one[:-1], G * phi[1:] - phi[:-1]) and abs(one[-1] - (1.0 - phi[-1])) < 1e-12
full = phi_advantage(phi, -1.0, 1.0)
assert abs(full[0] - (G ** 4 * -1.0 - phi[0])) < 1e-12 and abs(full[-1] - (-1.0 - phi[-1])) < 1e-12
assert len(phi_advantage(np.array([0.3]), 1.0, 0.5)) == 1                                 # 한 결정짜리 판
print("OK")
