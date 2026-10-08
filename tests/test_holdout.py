"""홀드아웃 통합(src/evaluation/holdout.py) self-check: PYTHONPATH=. python tests/test_holdout.py
그룹 추가/교체/선택 읽기가 원본 배틀과 같은 내용을 돌려주는지, 옛 형식(기술 칸 44)이 46으로 채워지는지 확인"""
import os
import tempfile

import numpy as np

import src.evaluation.holdout as H


def make(path, lens, mv_dim, seed):
    rng = np.random.RandomState(seed)
    n = int(sum(lens))
    d = {"my_team_cat": rng.randint(0, 9, (n, 6, 11)), "my_team_num": rng.rand(n, 6, 5).astype(np.float32), "my_move_num": rng.rand(n, 6, 4, mv_dim).astype(np.float32),
         "opp_team_cat": rng.randint(0, 9, (n, 6, 11)), "opp_team_num": rng.rand(n, 6, 5).astype(np.float32), "opp_move_num": rng.rand(n, 6, 4, mv_dim).astype(np.float32),
         "field_vec": rng.rand(n, 3).astype(np.float32), "action_mask": rng.rand(n, 22) > 0.5, "target": rng.rand(n, 22).astype(np.float32), "lens": np.array(lens)}
    np.savez(path, **d)
    return d


with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "all.npz")
    a, b = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
    da, db = make(a, [3, 2], 44, 0), make(b, [4], 46, 1)
    H.write_group("a", a, out); names, ns, nd = H.write_group("b", b, out)
    assert names == ["a", "b"] and ns == 1 and nd == 4
    seqs = H.load_holdout(("a", "b"), out)
    assert [s[0] for s in seqs] == ["a", "a", "b"] and [len(s[2]) for s in seqs] == [3, 2, 4]
    assert seqs[0][1]["my_move_num"].shape[-1] == 46 and np.allclose(seqs[0][1]["my_move_num"][..., :44], da["my_move_num"][:3])     # 옛 형식은 뒤를 0으로 채움
    assert np.all(seqs[0][1]["my_move_num"][..., 44:] == 0) and np.allclose(seqs[2][3], db["target"])
    assert [s[0] for s in H.load_holdout(("b",), out)] == ["b"]
    # 교체: 같은 이름의 그룹만 바뀌고 나머지와 번호는 그대로
    c = os.path.join(td, "c.npz"); dc = make(c, [2, 2], 46, 2)
    H.write_group("a", c, out)
    seqs = H.load_holdout(("a", "b"), out)
    assert sorted(len(s[2]) for s in seqs if s[0] == "a") == [2, 2] and [len(s[2]) for s in seqs if s[0] == "b"] == [4]
print("ok")
