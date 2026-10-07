"""교사 가치 로딩: PYTHONPATH=. python tests/test_teacher_value.py
교사 가치가 든 새 형식 npz와 없는 옛 형식 npz를 섞어도 로드되고, 옛 파일 구간은 NaN(= 할인된 결과만 사용)이어야 함"""
import os
import tempfile

import numpy as np

from src.core.tensor_encoder import MOVE_NUM_DIM
from src.training.replay_dataset import OBS_KEYS
from src.training.train_fp_distill import KEYS, load


def make(path, n_battles, lens, with_tv):
    N = sum(lens)
    shapes = {"my_team_cat": (N, 6, 11), "my_team_num": (N, 6, 27), "my_move_num": (N, 6, 4, MOVE_NUM_DIM), "opp_team_cat": (N, 6, 11),
              "opp_team_num": (N, 6, 27), "opp_move_num": (N, 6, 4, MOVE_NUM_DIM), "field_vec": (N, 58), "action_mask": (N, 22), "target": (N, 22), "action_taken": (N,)}
    d = {k: np.zeros(shapes[k], np.int64 if k in ("my_team_cat", "opp_team_cat", "action_taken") else np.float32) for k in KEYS}
    d["lens"], d["outcome"] = np.array(lens), np.ones(n_battles, np.float32)
    if with_tv:
        d["teacher_value"] = np.linspace(0, 1, N).astype(np.float32)
    np.savez(path, **d)


tmp = tempfile.mkdtemp()
make(os.path.join(tmp, "new.npz"), 2, [3, 2], True)
make(os.path.join(tmp, "old.npz"), 1, [4], False)
data, offsets, lens, files = load(os.path.join(tmp, "*.npz"))
tv = data["teacher_value"]
assert len(tv) == 9 and np.isnan(tv).sum() == 4 and np.isfinite(tv).sum() == 5, tv
print("teacher value load OK")
