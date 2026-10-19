"""npz 무손실 압축: 무압축 npz를 savez_compressed로 다시 쓰고, 모든 배열이 같을 때만 원본을 교체한다.
사용: python tools/compress_npz.py data/fp_selfplay/ms100/*.npz data/dagger/*.npz ...  (학습·수집이 돌고 있지 않을 때)"""
import os
import sys
import zipfile

import numpy as np


def compress(path):
    if all(i.compress_type != zipfile.ZIP_STORED for i in zipfile.ZipFile(path).infolist()):
        return None
    tmp = path + ".tmp.npz"
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    np.savez_compressed(tmp, **d)
    with np.load(tmp) as y:
        assert set(y.files) == set(d), path
        assert all(np.array_equal(d[k], y[k]) for k in d), f"배열 불일치: {path}"
    before, after = os.path.getsize(path), os.path.getsize(tmp)
    os.replace(tmp, path)
    return before, after


if __name__ == "__main__":
    tot = [0, 0]
    for p in sys.argv[1:]:
        r = compress(p)
        if r:
            tot[0] += r[0]; tot[1] += r[1]
            print(f"{p}: {r[0] / 1e9:.2f} -> {r[1] / 1e9:.3f} GB", flush=True)
    print(f"합계 {tot[0] / 1e9:.2f} -> {tot[1] / 1e9:.2f} GB")
