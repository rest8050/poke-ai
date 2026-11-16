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
p = lambda k, v: torch.zeros(1, 6, 7).index_fill_(2, torch.tensor([k]), v)
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
# 2026-11-17 규칙 수정: 가면은 오거폰만, 탁쳐서떨구기는 숨은 도구에도 x1.5, 기합의띠, 미스트필드 드래곤 반감, 총대장, KO는 난수 비율, 상대 쪽 장판
ch = lambda o, k, **kw: float(run(o, kw.pop("opp", [active()]), **kw)["off"][0, 0, 0, k])           # 내 활성의 첫 기술이 상대 활성에 주는 칸 k (1 데미지, 2 KO)
mask = lambda sp: ch([active(moves=phys, item="wellspringmask", species=sp)], 1) / hit_opp(phys)
assert abs(mask("bellibolt") - 1.0) < 1e-6 and 1.15 < mask("ogerponwellspring") <= 1.2, "가면 x1.2는 오거폰만"
knock = [(65, 0, "phys", "dark", "knockoff")]
k_none, k_hid = ch([active(moves=knock)], 1), ch([active(moves=knock)], 1, hide_opp_item=True)
assert 1.4 < k_hid / k_none <= 1.5, "탁쳐서떨구기: 숨은 도구도 들고 있는 것으로 봄"
strong = [(200, 0, "phys", "dragon")]
ko = lambda **m: ch([active(moves=strong, stats=(100, 900, 100, 100, 100, 100))], 2, opp=[active(**m)])
assert ko() == 1.0 and ko(item="focussash") == 0.0 and ko(item="focussash", hp=0.9) == 1.0, "기합의띠: 만피에서만 버팀"
assert ko(item="focussash") == 0.0 and float(run([active(moves=strong, stats=(100, 900, 100, 100, 100, 100))], [active(item="focussash")], field=[40])["off"][0, 0, 0, 2]) == 0.0, "활성은 장판 피해 없음"
bench_ko = lambda field: float(run([active(moves=strong, stats=(100, 900, 100, 100, 100, 100))], [active(), dict(item="focussash")], field=field)["off"][0, 1, 0, 2])
assert bench_ko(None) == 0.0 and bench_ko([40]) == 1.0, "상대 교체 후보는 스텔스록(상대 쪽 칸 40)에 깎여 띠가 깨짐"
weak = [(40, 0, "phys", "dragon")]
part = ch([active(moves=weak)], 2, opp=[active(hp=0.03)])
assert 0.0 < part < 1.0, f"KO 칸은 난수 비율 (애매한 구간에서 0과 1 사이): {part}"
assert 0.5 <= ch([active(moves=phys)], 1, field=[18]) / hit_opp(phys) < 0.6, "미스트필드: 드래곤 기술 반감 (데미지 식의 +2 때문에 비는 0.5보다 조금 큼)"
gambit = lambda n: float(run([active(moves=phys, ab="supremeoverlord")] + [dict(hp=0.0, fainted=True)] * n, [active()])["off"][0, 0, 0, 1])
assert 1.25 < gambit(3) / gambit(0) <= 1.3, "총대장: 쓰러진 아군 3마리 x1.3"
assert 1.4 < ch([active(moves=phys, ab="dragonsmaw")], 1) / hit_opp(phys) <= 1.5, "용의턱"
# 숨은 도구 확률: 진화의휘석(진화 전 종만), 기합의띠, 두꺼운부츠(상대 교체 후보의 장판 피해)
evo = lambda sp, **kw: float(run([active(moves=phys)], [active(species=sp)], hide_opp_item=True, **kw)["off"][0, 0, 0, 1])
assert evo("chansey", p_item=p(4, 1.0)) < 0.75 * evo("chansey") and abs(evo("blissey", p_item=p(4, 1.0)) - evo("blissey")) < 1e-6, "숨은 진화의휘석은 진화 전 종에만"
big = [active(moves=strong, stats=(100, 900, 100, 100, 100, 100))]
sash_ko = lambda v: float(run(big, [active()], hide_opp_item=True, p_item=p(5, v))["off"][0, 0, 0, 2])
assert sash_ko(0.0) == 1.0 and abs(sash_ko(0.3) - 0.7) < 1e-6, "숨은 기합의띠 확률 0.3 -> KO 확률 0.7"
def boots_ko(v):      # 상대 벤치(슬롯 1, 도구 숨김, 띠 확률 1)가 스텔스록에 깎이면 띠가 깨짐. 부츠 확률만큼은 안 깎임
    mc, mn, mm = team(big, True)
    oc, on, om = team([active(), dict()], False)
    oc[0, 1, 0] = HID
    fv = torch.zeros(1, FIELD_DIM); fv[0, 40] = 1.0
    pi = torch.zeros(1, 6, 7); pi[..., 5], pi[..., 6] = 0.5, v
    with torch.no_grad():
        return float(M(mc, mn, mm, oc, on, om, fv, F.one_hot(oc[..., 1], A).float(), torch.zeros(1, 6), opp_item_p=pi)["off"][0, 1, 0, 2])
assert boots_ko(0.0) == 1.0 and boots_ko(1.0) == 0.5, "부츠가 확실하면 장판 피해가 없어 띠(확률 0.5)가 살아 있음"
print("matchup items/boost OK")
