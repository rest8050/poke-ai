"""신념 사전학습 self-check: PYTHONPATH=. python tests/test_pretrain_belief.py
1) 부분 관측: 공개된 기술만 입력에 보이고 나머지는 숨김 ID, 정답은 안 드러난 기술(완전한 세트 - 공개분) 2) 손실/통계 형태
3) 팀 파싱(별명, 성별, 종 이름) 4) 옛 구조(version 2: 트렁크 임베딩을 공유하던 신념) 체크포인트 라우팅이 같은 출력"""
import os

import numpy as np
import torch

from poke_env.data import GenData
from src.core.belief import SetBelief
from src.core.model import model_from_ckpt
from src.core.tensor_encoder import VocabManager
from src.training.pretrain_belief import belief_loss, make_batch, parse_team, to_arrays

vm = VocabManager.get_instance("data/vocab.json")
dex = GenData.from_gen(9).pokedex
export = """Smogon Nick (Garchomp) (M) @ Rocky Helmet
Ability: Rough Skin
Tera Type: Steel
EVs: 252 HP
- Earthquake
- Stealth Rock
- Dragon Tail
- Spikes

Rotom-Wash @ Leftovers
Ability: Levitate
- Hydro Pump
- Volt Switch
"""
mons = parse_team(export, vm, dex)
assert len(mons) == 2 and mons[0]["species"] == vm.get_id("species", "garchomp") and len(mons[0]["moves"]) == 4 and len(mons[1]["moves"]) == 2
assert mons[0]["item"] == vm.get_id("item", "rockyhelmet") and mons[0]["types"] == [vm.get_id("type", "dragon"), vm.get_id("type", "ground")]

from src.training.pretrain_belief import stat_logratio
base6 = [108, 130, 95, 80, 85, 102]                                           # 한카리아스
assert max(abs(x) for x in stat_logratio(base6, [252] * 6, [31] * 6, None)) < 1e-9, "252노력치/무보정은 표준 가정과 같아야 함"
lr = stat_logratio(base6, [0] * 6, [31] * 6, "adamant")                       # 노력치 0, 공격 +10% 특공 -10%
assert lr[0] < 0 and lr[5] < 0 and lr[1] < lr[2] + 0.2 and lr[3] < lr[2]
assert abs(mons[0]["stat"][0] - stat_logratio([108, 130, 95, 80, 85, 102], [252, 0, 0, 0, 0, 0], [31] * 6, None)[0]) < 1e-9   # 내보내기의 "EVs: 252 HP" 파싱
assert mons[0]["stat"][1] < -0.15, "노력치 0 공격 = 표준(252) 대비 크게 낮음"
dev = torch.device("cpu")
A, N, hid = to_arrays([mons] * 50, vm, dev)
gen = torch.Generator().manual_seed(0)
unk_move = vm.unk_id("move")
idx = torch.arange(N)
cat, num, tgt = make_batch(A, idx, hid, unk_move + 1, gen)
shown = cat[:, 0, 5:9]
for i in range(N):
    for j in range(4):
        if tgt["revealed"][i, j]:
            assert shown[i, j] == tgt["moves"][i, j] > 0
        else:
            assert shown[i, j] == unk_move + 1                        # 안 드러난 칸은 숨김 ID
assert (tgt["revealed"].sum(-1) <= (tgt["moves"] > 0).sum(-1)).all()
assert (cat[:, 0, 0] == hid["item"]).sum() > 0 and (cat[:, 0, 0] != hid["item"]).sum() > 0   # 도구는 일부만 공개
model = SetBelief()
loss, st = belief_loss(model(cat, num), tgt, unk_move)
assert torch.isfinite(loss) and "move_top4" in st
loss.backward()
# 옛 구조 체크포인트(version 2) 라우팅: v3_pilot이 있으면 로드해서 dg3_ho 일치율이 로그와 같은지 (0.476)
CK = "checkpoints/supervised_v2_fp_v3_pilot.pt"
if os.path.exists(CK):
    m = model_from_ckpt(CK).eval()
    assert type(m).__name__ == "EntityPokemonNetV3Legacy" and m.belief is not None and m.bench_from_guess is False
    z = np.load("data/dagger/dg3_ho.npz"); keys = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
    n = int(z["lens"][:20].sum())
    obs = [torch.tensor(z[k][:n]) for k in keys]
    for i in (2, 5):
        obs[i] = torch.cat([obs[i], obs[i].new_zeros(*obs[i].shape[:-1], 46 - obs[i].size(-1))], -1)
    sq = torch.cat([torch.full((int(L),), b) for b, L in enumerate(z["lens"][:20])]); tq = torch.cat([torch.arange(int(L)) for L in z["lens"][:20]])
    am = torch.tensor(z["action_mask"][:n])
    out = m.forward_sequences(obs, sq, tq, 20, am)
    agree = (out["policy_logits"].argmax(-1) == torch.tensor(z["target"][:n]).argmax(-1)).float().mean().item()
    assert 0.3 < agree < 0.8, agree
print("pretrain belief OK")
