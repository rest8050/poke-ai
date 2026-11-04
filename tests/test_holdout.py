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
    # 히스토리 위치(pos): 옛 파일은 배틀마다 0,1,2,..., pos가 든 파일은 그대로 (빠진 결정 지점이 있으면 번호가 건너뜀), 교체해도 짝이 맞음
    assert list(H.decision_pos({"lens": np.array([3, 2])}, [3, 2])) == [0, 1, 2, 0, 1]
    e = os.path.join(td, "e.npz"); de = make(e, [3], 46, 3)
    np.savez(e, **de, pos=np.array([0, 2, 3]))
    H.write_group("e", e, out)
    with np.load(out) as f:
        z = {k: f[k] for k in f.files}
    names = [str(n) for n in z["src_names"]]
    src = np.repeat(z["src"], z["lens"])
    assert list(z["pos"][src == names.index("e")]) == [0, 2, 3] and list(z["pos"][src == names.index("b")]) == [0, 1, 2, 3]
    H.write_group("b", b, out)
    with np.load(out) as f:
        z = {k: f[k] for k in f.files}
    src = np.repeat(z["src"], z["lens"])
    assert list(z["pos"][src == names.index("e")]) == [0, 2, 3] and len(z["pos"]) == int(z["lens"].sum())
print("ok")
