"""브리지 모델 교체 self-check: PYTHONPATH=. python tests/test_bridge_swap.py
포인터 파일을 바꾸면 새 세션은 새 모델, 진행 중이던 세션은 이전 모델을 유지 / 읽기 실패 시 이전 모델 유지 / 구조가 다른 모델(v3)도 cfg로 복원"""
import os
import tempfile

from src.core import bridge_server as bs

ck = os.path.abspath("checkpoints")
tmp = tempfile.mkdtemp()
ptr = os.path.join(tmp, "CURRENT")
bs.POINTER = ptr
write = lambda t: open(ptr, "w").write(t + "\n")

write(os.path.join(ck, "supervised_v2_fp_v1_full_r3.pt"))
bs.holder.refresh(force=True)
old_model, old_name = bs.holder.model, bs.holder.name
assert old_name == "supervised_v2_fp_v1_full_r3.pt" and type(old_model).__name__ == "DeepPokemonBattleTransformerNet"
s_old = bs.Session("p1", bs.holder.model, bs.holder.name)

write(os.path.join(ck, "supervised_v2_fp_v3_full.pt"))
assert bs.holder.refresh() is old_model                 # 5초 안에는 다시 확인하지 않음
bs.holder.refresh(force=True)
assert bs.holder.name == "supervised_v2_fp_v3_full.pt" and type(bs.holder.model).__name__ == "EntityPokemonNetV3"
s_new = bs.Session("p1", bs.holder.model, bs.holder.name)
assert s_old.model is old_model and s_new.model is bs.holder.model and s_old.model is not s_new.model   # 진행 중 세션은 그대로

write(os.path.join(tmp, "없는파일.pt"))                     # 읽기 실패 -> 이전(v3) 모델 유지
keep = bs.holder.model
bs.holder.refresh(force=True)
assert bs.holder.model is keep and bs.holder.name == "supervised_v2_fp_v3_full.pt"

write("supervised_v2_fp_v1_full_r3.pt")                       # 파일 이름만 쓰면 포인터와 같은 폴더(tmp)에서 찾음 -> 없으니 실패, 유지
bs.holder.refresh(force=True)
assert bs.holder.name == "supervised_v2_fp_v3_full.pt"
h = bs.health()
assert h["ckpt"] == "supervised_v2_fp_v3_full.pt" and "pointer" in h
print("bridge swap OK:", h)
