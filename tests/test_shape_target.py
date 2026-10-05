"""교사 타깃 정리 self-check: PYTHONPATH=. python tests/test_shape_target.py"""
import torch

from src.training.train_fp_distill import shape_target

t = torch.tensor([[0.5, 0.3, 0.1, 0.06, 0.04, 0.0], [0.02, 0.02, 0.02, 0.02, 0.02, 0.9]])
assert torch.equal(shape_target(t, 0.0, 1.0), t)                       # 기본값 = 원본
p = shape_target(t, 0.05, 1.0)
assert torch.allclose(p.sum(-1), torch.ones(2)) and p[0, 4] == 0 and p[0, :4].min() > 0 and p[1, 5] > 0.98
s = shape_target(t, 0.0, 0.5)
assert torch.allclose(s.sum(-1), torch.ones(2)) and s[0, 0] > t[0, 0] and (s.argmax(-1) == t.argmax(-1)).all()
x = torch.tensor([[0.04, 0.03, 0.03]]) / 0.1                            # 전부 임계 미만이어도 최상위는 남음
assert shape_target(x, 0.5, 1.0).argmax() == 0 and shape_target(x, 0.5, 1.0).sum().item() == 1.0
print("shape_target OK")
