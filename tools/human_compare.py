"""사람 데이터 BC 두 모델의 테스트 NLL 쌍비교 (판 단위 부트스트랩 95% 구간). 양수 = 두 번째 모델이 더 좋음
    python tools/human_compare.py h0 h1   (logs/human_<tag>_test.npz, src/training/train_human.py 출력)"""
import sys

import numpy as np

a, b = (np.load(f"logs/human_{t}_test.npz") for t in sys.argv[1:3])
assert (a["pos"] == b["pos"]).all()
ok = ~np.isnan(a["nll"]) & ~np.isnan(b["nll"])
d = (a["nll"] - b["nll"])[ok]
ug, inv = np.unique(a["game"][ok], return_inverse=True)
gs, gn = np.bincount(inv, d, len(ug)), np.bincount(inv, None, len(ug))
rng = np.random.RandomState(0)
boot = [gs[i].sum() / gn[i].sum() for i in (rng.randint(0, len(ug), len(ug)) for _ in range(1000))]
print(f"NLL({sys.argv[1]}) - NLL({sys.argv[2]}) = {d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  결정 {ok.sum()}, 판 {len(ug)}  (양수 = {sys.argv[2]}가 더 좋음)")
