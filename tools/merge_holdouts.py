"""홀드아웃 npz를 통합 파일(data/holdout/holdout_all.npz)에 그룹으로 넣음 (같은 이름이면 교체). 사용: python tools/merge_holdouts.py 이름=경로 [이름=경로 ...]
현재 그룹: dg2 dg3 dg4 dg5(DAgger 라운드의 학생 상태) fp(교사 상태). 새 라운드 예: python tools/merge_holdouts.py dg6=data/dagger/dg6_ho.npz"""
import sys

sys.path.insert(0, ".")
from src.evaluation.holdout import write_group

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    names, n_seq, n_dec = write_group(name, path)
    print(f"{name}: {n_seq}배틀 / {n_dec}결정 추가 → 그룹 {names}")
