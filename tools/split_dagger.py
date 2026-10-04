"""DAgger npz를 배틀 단위로 학습/검증(홀드아웃)으로 나눔. 사용: split_dagger.py <입력 npz> <학습 출력> <홀드아웃 출력> [학습 복제 배수=1] [홀드아웃 비율=0.2]"""
import random
import sys

import numpy as np

src, out_tr, out_ho = sys.argv[1:4]
rep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
z = np.load(src)
lens = z["lens"]
off = np.concatenate([[0], np.cumsum(lens)])
ids = list(range(len(lens)))
random.Random(0).shuffle(ids)
n_ho = max(1, int(len(ids) * frac))
ho, tr = sorted(ids[:n_ho]), sorted(ids[n_ho:])
per_step = [k for k in z.files if k not in ("lens", "outcome")]
data = {k: z[k] for k in per_step}  # NpzFile은 접근할 때마다 다시 읽으므로 한 번만 읽음
outcome = z["outcome"]


def take(sel, times):
    d = {k: np.concatenate([data[k][off[b]:off[b + 1]] for _ in range(times) for b in sel]) for k in per_step}
    d["lens"] = np.array([lens[b] for _ in range(times) for b in sel])
    d["outcome"] = np.array([outcome[b] for _ in range(times) for b in sel], np.float32)
    return d


np.savez(out_tr, **take(tr, rep))
np.savez(out_ho, **take(ho, 1))
print(f"학습 {len(tr)}판 x{rep} / 검증 {len(ho)}판 (결정 {int(sum(lens[b] for b in ho))}개)")
