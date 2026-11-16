"""매치업 보정 self-check: PYTHONPATH=. python tests/test_matchup_items_boost.py
1) 숨은 도구는 확률 기대 배율 (구애머리띠 0.5 -> 물리 데미지 x1.25, 공개 도구면 확률 무시), 2) 돌격조끼 확률은 특수 데미지를 줄임,
3) 활성의 쿼크차지 스피드 칸 -> 스피드 x1.5 (선공 뒤집힘), 공격 칸 -> 데미지 x1.3, 벤치에는 안 붙음, 4) 상대가 방어(+4)를 드러내도 내 기술의 fm은 스피드 기준"""
import torch
import torch.nn.functional as F

from src.core.tensor_encoder import EXT_START, FIELD_DIM, NAME_INDEX, SIDE_EXT
from tests.test_matchup_v7 import A, IT, M, active, team

HID = IT["<unk>"] + 1


def run(my, opp, field=None, p_item=None, hide_opp_item=False):
    mc, mn, mm = team(my, True)
    oc, on, om = team(opp, False)
    if hide_opp_item:
        oc[0, 0, 0] = HID
    fv = torch.zeros(1, FIELD_DIM)
    for k in field or ():
        fv[0, k] = 1.0
    with torch.no_grad():
        return M(mc, mn, mm, oc, on, om, fv, F.one_hot(oc[..., 1], A).float(), torch.zeros(1, 6), opp_item_p=p_item)


phys, spec = [(40, 0, "phys", "dragon")], [(40, 0, "spec", "dragon")]        # 약한 등배 기술: 데미지 칸의 1.5 상한에 안 걸리게
p = lambda k, v: torch.zeros(1, 6, 4).index_fill_(2, torch.tensor([k]), v)
hit_me = lambda **kw: float(run([active()], [active(moves=phys)], **kw)["def"][0, 0, 0, 1])           # 상대 물리기가 내 활성에 주는 데미지 칸
base = hit_me(hide_opp_item=True)
assert abs(hit_me(hide_opp_item=True, p_item=p(0, 0.5)) / base - 1.25) < 0.02, "숨은 구애머리띠 확률 0.5 -> x1.25"
assert abs(hit_me(p_item=p(0, 0.5)) - hit_me()) < 1e-6, "도구 칸이 숨김이 아니면 확률을 안 씀"
assert abs(hit_me(hide_opp_item=True, p_item=p(2, 1.0)) / base - 5324 / 4096) < 0.02, "숨은 생명의구슬"
hit_opp = lambda mv, **kw: float(run([active(moves=mv)], [active()], **kw)["off"][0, 0, 0, 1])       # 내 기술이 상대 활성에 주는 데미지 칸
assert hit_opp(spec, hide_opp_item=True, p_item=p(3, 1.0)) < 0.75 * hit_opp(spec, hide_opp_item=True), "숨은 돌격조끼는 특수 데미지를 줄임"
assert abs(hit_opp(phys, hide_opp_item=True, p_item=p(3, 1.0)) - hit_opp(phys, hide_opp_item=True)) < 1e-6, "돌격조끼는 물리와 무관"

q = lambda stat, opp=False: EXT_START + (SIDE_EXT if opp else 0) + NAME_INDEX["quarkdrive" + stat]
mine = [active(stats=(100, 100, 100, 100, 100, 250), moves=phys)]                                       # 내 250 대 상대 종족값 100 -> 299
fm = lambda field: float(run(mine, [active()], field)["fm_my"][0, 0, 0])
assert fm(None) == -1.0 and fm([q("spe")]) == 1.0, "쿼크차지 스피드 x1.5: 375 > 299"
assert fm([q("spe"), q("spe", True)]) == -1.0, "상대도 켜지면 448 > 375"
assert 1.2 < hit_opp(phys, field=[q("atk")]) / hit_opp(phys) <= 1.3 and hit_opp(phys, field=[q("spa")]) == hit_opp(phys), "쿼크차지 공격 x1.3 (데미지 식의 +2 때문에 비는 1.3보다 조금 작음), 특공 칸은 물리와 무관"
bench = run([dict(moves=phys), active()], [active()], [q("atk")])["off"][0, 0, 0, 1]
assert abs(float(bench) - float(run([dict(moves=phys), active()], [active()])["off"][0, 0, 0, 1])) < 1e-6, "벤치에는 안 붙음"

protect = [active(stats=(100, 100, 100, 100, 100, 50), moves=[(0, 4, "status", "normal")])]            # 느린 상대가 방어를 드러냄
o = run([active(stats=(100, 100, 100, 100, 100, 400), moves=phys)], protect)
assert float(o["fm_my"][0, 0, 0]) == 1.0 and float(o["fm_opp"][0, 0, 0]) == 1.0, "내 기술은 스피드 기준 먼저, 방어 칸 자신은 +1"
print("matchup items/boost OK")
