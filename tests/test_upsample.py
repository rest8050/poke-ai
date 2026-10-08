"""train_fp_distill.upsample_train self-check: PYTHONPATH=. python tests/test_upsample.py
지정 파일 소속 학습 배틀만 (배수-1)번 더 들어가고, 다른 파일/검증은 그대로"""
import os
import tempfile

import numpy as np

from src.training.train_fp_distill import upsample_train

with tempfile.TemporaryDirectory() as td:
    a, b = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
    np.savez(a, lens=np.array([3, 4, 5]))        # 배틀 0~2
    np.savez(b, lens=np.array([2, 2]))           # 배틀 3~4
    train = [0, 2, 3, 4]                         # 1은 검증
    out = upsample_train(train, [a, b], {b: 3})
    assert sorted(out) == [0, 2, 3, 3, 3, 4, 4, 4], out
    assert upsample_train(train, [a, b], {a: 2}) == [0, 2, 3, 4, 0, 2]
print("ok")
