"""train_fp_distill.pg_loss: 실제 승패(advantage=value_target) 기반 정책 그라디언트 보조 손실.
행동이 매칭 안 된(action_taken=-1) 결정은 무시하고, 나머지는 배치 내 정규화 후 -logp(둔 수)*advantage 평균."""
import torch

from src.training.train_fp_distill import pg_loss


def test_all_unmatched_returns_zero():
    logp = torch.log_softmax(torch.randn(3, 22), dim=-1)
    b = {"action_taken": torch.tensor([-1, -1, -1]), "value_target": torch.tensor([0.5, -0.3, 0.1])}
    assert pg_loss(logp, b, b["value_target"]).item() == 0.0


def test_unmatched_rows_excluded_from_normalization_and_loss():
    logp = torch.log_softmax(torch.zeros(3, 4), dim=-1)  # 균등분포, logp(모든 행동)=log(0.25)
    b = {"action_taken": torch.tensor([0, -1, 1]), "value_target": torch.tensor([1.0, 999.0, -1.0])}
    loss = pg_loss(logp, b, b["value_target"])
    # 유효 2개(advantage 1.0, -1.0)만 정규화: 평균0, 표준편차1 -> adv_norm = [+1,-1]. logp_a 둘 다 log(0.25)로 동일
    # loss = -mean(logp_a * adv_norm) = -log(0.25)*(1 + -1)/2 = 0
    assert abs(loss.item()) < 1e-5


def test_positive_advantage_pushes_up_negative_pushes_down():
    # 표본0: advantage 양수(그 수가 이겼음) -> action0 로짓을 올리는 방향(그라디언트 음수)이어야 함
    # 표본1: advantage 음수(그 수가 졌음) -> action1 로짓을 내리는 방향(그라디언트 양수)이어야 함
    logits = torch.zeros(2, 3, requires_grad=True)
    logp = torch.log_softmax(logits, dim=-1)
    b = {"action_taken": torch.tensor([0, 1]), "value_target": torch.tensor([5.0, -5.0])}
    loss = pg_loss(logp, b, b["value_target"])
    loss.backward()
    assert logits.grad[0, 0].item() < 0
    assert logits.grad[1, 1].item() > 0


if __name__ == "__main__":
    test_all_unmatched_returns_zero()
    test_unmatched_rows_excluded_from_normalization_and_loss()
    test_positive_advantage_pushes_up_negative_pushes_down()
    print("ok")
