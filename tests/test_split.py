"""학습/검증 분할 self-check: PYTHONPATH=. python tests/test_split.py
거울 짝(같은 배틀의 양쪽 시점)이 학습/검증으로 갈라지지 않고, 검증 크기가 대략 val_frac"""
import numpy as np

from src.training.train_fp_distill import split_ids

S = 400
rng = np.random.default_rng(0)
teams = [rng.permutation(np.arange(1, 400))[:6] for _ in range(S)]       # 배틀 S/2개 x 양쪽 시점
my, opp = np.zeros((S, 6, 11), dtype=np.int64), np.zeros((S, 6, 11), dtype=np.int64)
for b in range(S // 2):                                                    # 시퀀스 2b = 팀 A 시점, 2b+1 = 팀 B 시점 (거울)
    my[2 * b, :, 10], opp[2 * b, :, 10] = teams[2 * b], teams[2 * b + 1]
    my[2 * b + 1, :, 10], opp[2 * b + 1, :, 10] = teams[2 * b + 1][::-1], teams[2 * b]   # 슬롯 순서가 달라도 같은 짝
perm = rng.permutation(S)                                                  # 파일 안 순서는 섞여 있음
data = {"my_team_cat": my[perm], "opp_team_cat": opp[perm]}
offsets = np.arange(S + 1)                                                 # 시퀀스당 결정 1개 (분할 로직은 첫 행만 봄)
val, train = split_ids(data, offsets, 0.1)
assert set(val).isdisjoint(train) and len(val) + len(train) == S
inv = np.argsort(perm)                                                     # 원래 인덱스로 되돌려 짝 확인
val_orig = {int(perm[i]) for i in val}
assert all((o ^ 1) in val_orig for o in val_orig), "거울 짝이 갈라짐"
assert 0.07 <= len(val) / S <= 0.14, len(val)
print(f"split OK: 검증 {len(val)}/{S}, 거울 짝 유지")
