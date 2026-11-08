"""MatchupV7 self-check: PYTHONPATH=. python tests/test_matchup_v7.py
1) 정밀기술 위력 경계, 2) 두꺼운지방/부유 배율, 3) 옹골참 만피 KO, 4) 스피드(날씨 특성/마비/순풍/랭크/스카프/트릭룸),
5) 우선도와 장난꾸러기(악 타입 실패), 6) 상대 특성 확률 기대값, 7) 위협 교체 후보 배율, 8) 재생력 회복량, 9) 상대 기술칸의 선공 기대값"""
import json

import torch
import torch.nn.functional as F

from src.core.matchup_v7 import MatchupV7
from src.core.tensor_encoder import FIELD_DIM, NUM_DIM

V = json.load(open("data/vocab.json"))
TY, AB, IT, ST = V["type"], V["ability"], V["item"], V["status"]
CATEGORY = {"phys": 0.33, "spec": 0.67, "status": 1.0}
M = MatchupV7()
A = M.f_def.size(0)


def team(mons, mine):
    """mons: 최대 6개의 dict. 키: active, types, ab, item, status, hp, stats(hp,atk,def,spa,spd,spe), stage(spe 랭크 등 {인덱스: 값}), moves[(위력, 우선도, 분류, 타입)]"""
    cat = torch.zeros(1, 6, 11, dtype=torch.long)
    num = torch.zeros(1, 6, NUM_DIM)
    mv = torch.zeros(1, 6, 4, 46)
    for i, m in enumerate(mons):
        t = m.get("types", ("normal",))
        cat[0, i, 2], cat[0, i, 3] = TY[t[0]], (TY[t[1]] if len(t) > 1 else 0)
        cat[0, i, 1] = AB[m["ab"]] if m.get("ab") else 1
        cat[0, i, 0] = IT[m["item"]] if m.get("item") else 0
        cat[0, i, 4] = ST[m["status"]] if m.get("status") else 0
        num[0, i, 0] = 1.0 if m.get("active") else 0.0
        num[0, i, 2] = m.get("hp", 1.0)
        s = m.get("stats", (100, 100, 100, 100, 100, 100))
        num[0, i, 11:17] = torch.tensor([x / 255.0 for x in s])
        for k, v in (m.get("stage") or {}).items():                                              # 랭크 칸: 3 공, 4 방, 5 특공, 6 특방, 7 스피드 (랭크/6)
            num[0, i, k] = v / 6.0
        if m.get("species"):
            cat[0, i, 10] = V["species"][m["species"]]
        if m.get("tera"):
            num[0, i, 9], cat[0, i, 9] = 1.0, TY[m["tera"]]
        for j, (bp, pri, kind, mtype, *name) in enumerate(m.get("moves", ())):                  # 5번째 원소(선택): 기술 이름 -> 기술 id 칸 (기술별 규칙)
            mv[0, i, j, 0], mv[0, i, j, 3], mv[0, i, j, 4], mv[0, i, j, 6] = bp / 200.0, pri / 7.0, CATEGORY[kind], TY[mtype] / 20.0
            if name:
                cat[0, i, 5 + j] = V["move"][name[0]]
    return cat, num, mv


def run(my, opp, field=None, q_opp=None, scarf=None):
    mc, mn, mm = team(my, True)
    oc, on, om = team(opp, False)
    fv = torch.zeros(1, FIELD_DIM)
    for k, v in (field or {}).items():
        fv[0, k] = v
    q = F.one_hot(oc[..., 1], A).float() if q_opp is None else q_opp
    with torch.no_grad():
        return M(mc, mn, mm, oc, on, om, fv, q, torch.zeros(1, 6) if scarf is None else scarf)


def active(**kw):
    return dict(active=True, **kw)


def damage(out, o=0, j=0, ch=1):
    return float(out["off"][0, o, j, ch])        # 내 포켓몬 0의 기술 j가 상대 포켓몬 o에게: 채널 1 = 데미지 비율


# 공통 상황: 내 활성이 물리 공격기(위력 bp)로 상대 활성을 때림. 능력치는 데미지가 상한에 안 닿게 낮게
def hit(bp, my_ab=None, opp_ab=None, mtype="normal", opp_types=("grass",), opp_hp=1.0, kind="phys"):
    out = run([active(ab=my_ab, stats=(300, 60, 60, 60, 60, 100), moves=[(bp, 0, kind, mtype)])],
              [active(ab=opp_ab, types=opp_types, hp=opp_hp, stats=(100, 80, 80, 80, 80, 80))])
    return out

# 1) technician: 위력 40 x1.5 = 60 (경계), 위력 65는 변화 없음
assert abs(damage(hit(40, my_ab="technician")) - damage(hit(60))) < 1e-6
assert abs(damage(hit(60, my_ab="technician")) - damage(hit(90))) < 1e-6
assert abs(damage(hit(65, my_ab="technician")) - damage(hit(65))) < 1e-6
assert damage(hit(40, my_ab="technician")) > damage(hit(40))

# 2) thickfat: 불꽃/얼음 x0.5, 물은 그대로. 부유는 땅 무효, 확률로 주면 기대값
fire, water = hit(80, mtype="fire", opp_types=("normal",)), hit(80, mtype="water", opp_types=("normal",))
tf_fire, tf_water = hit(80, opp_ab="thickfat", mtype="fire", opp_types=("normal",)), hit(80, opp_ab="thickfat", mtype="water", opp_types=("normal",))
assert abs(damage(tf_fire) - 0.5 * damage(fire)) < 1e-6 and abs(damage(tf_water) - damage(water)) < 1e-6
assert damage(hit(80, mtype="ground", opp_types=("normal",), opp_ab="levitate")) == 0 and damage(hit(80, mtype="ground", opp_types=("normal",))) > 0

# 3) sturdy: 만피에서는 KO 없음, HP가 조금이라도 깎였으면 KO. 확률 0.5이면 KO 0.5
KO_ME, KO_OPP = (300, 400, 60, 60, 60, 100), (20, 20, 20, 20, 20, 80)                                   # 한 방에 확실히 쓰러지는 조합
big = lambda hp, ab=None: run([active(stats=KO_ME, moves=[(250, 0, "phys", "normal")])],
                              [active(ab=ab, types=("normal",), hp=hp, stats=KO_OPP)])["off"][0, 0, 0, 2].item()
assert big(1.0) == 1.0 and big(1.0, "sturdy") == 0.0 and big(0.9, "sturdy") == 1.0
mc, mn, mm = team([active(stats=KO_ME, moves=[(250, 0, "phys", "normal")])], True)
oc, on, om = team([active(types=("normal",), stats=KO_OPP)], False)
q = F.one_hot(oc[..., 1], A).float(); q[0, 0] = 0; q[0, 0, AB["sturdy"]] = 0.5; q[0, 0, AB["levitate"]] = 0.5
with torch.no_grad():
    assert abs(float(M(mc, mn, mm, oc, on, om, torch.zeros(1, FIELD_DIM), q, torch.zeros(1, 6))["off"][0, 0, 0, 2]) - 0.5) < 1e-6

# 4) 스피드: 내 스피드 200, 상대 종족값 100 -> 299. fm_my[0,0,0] = +1 먼저 / -1 나중
def first(my_extra=None, field=None, opp_extra=None, move=(50, 0, "phys", "normal")):
    my = active(stats=(100, 100, 100, 100, 100, 200), moves=[move], **(my_extra or {}))
    opp = active(**{"stats": (100, 100, 100, 100, 100, 100), "types": ("normal",), **(opp_extra or {})})
    return float(run([my], [opp], field)["fm_my"][0, 0, 0])
assert first() == -1.0
assert first({"ab": "chlorophyll"}, {0: 1.0}) == 1.0 and first({"ab": "chlorophyll"}, {2: 1.0}) == -1.0     # 쾌청에서만
assert first({"ab": "swiftswim"}, {2: 1.0}) == 1.0 and first({"ab": "swiftswim"}, {0: 1.0}) == -1.0         # 비에서만
assert first({"item": "choicescarf"}) == 1.0                                                                # x1.5 = 300 > 299
assert first({"stage": {7: 1}}) == 1.0                                                                      # +1 랭크 x1.5
assert first(field={26: 1.0}) == 1.0                                                                        # 내 쪽 순풍
slow = lambda **kw: float(run([active(stats=(100, 100, 100, 100, 100, 400), moves=[(50, 0, "phys", "normal")], **kw)],
                              [active(stats=(100, 100, 100, 100, 100, 100), types=("normal",))])["fm_my"][0, 0, 0])
assert slow() == 1.0 and slow(status="par") == -1.0                                                          # 마비 x0.5: 400 -> 200 < 299
assert float(run([active(stats=(100, 100, 100, 100, 100, 400), moves=[(50, 0, "phys", "normal")])],
                 [active(stats=(100, 100, 100, 100, 100, 100), types=("normal",))], {10: 1.0})["fm_my"][0, 0, 0]) == -1.0     # 트릭룸 반전
# 상대 쪽 순풍은 상대 스피드 x2 -> 299*2 = 598 > 400
assert float(run([active(stats=(100, 100, 100, 100, 100, 400), moves=[(50, 0, "phys", "normal")])],
                 [active(stats=(100, 100, 100, 100, 100, 100), types=("normal",))], {38: 1.0})["fm_my"][0, 0, 0]) == -1.0

# 5) 우선도: 등급이 다르면 스피드 무관. 장난꾸러기 변화기 +1, 악 타입 대상이면 실패(0), 공격기에는 안 붙음
assert first(move=(50, 1, "phys", "normal")) == 1.0
assert first({"ab": "prankster"}, move=(0, 0, "status", "normal")) == 1.0
assert first({"ab": "prankster"}, opp_extra={"types": ("dark",)}, move=(0, 0, "status", "normal")) == 0.0
assert first({"ab": "prankster"}, move=(50, 0, "phys", "normal")) == -1.0
fast_opp = lambda pri: float(run([active(stats=(100, 100, 100, 100, 100, 400), moves=[(50, 0, "phys", "normal")])],
                                 [active(stats=(100, 100, 100, 100, 100, 100), types=("normal",), moves=[(50, pri, "phys", "normal")])])["fm_my"][0, 0, 0])
assert fast_opp(0) == 1.0 and fast_opp(1) == -1.0                                                           # 상대가 우선기를 드러냄

# 6) 상대 특성 확률: 엽록소 0.5(쾌청) -> 내 400은 299를 이기고 598에는 짐 -> 기대값 0
mc, mn, mm = team([active(stats=(100, 100, 100, 100, 100, 400), moves=[(50, 0, "phys", "normal")])], True)
oc, on, om = team([active(types=("normal",), stats=(100, 100, 100, 100, 100, 100))], False)
q = torch.zeros(1, 6, A); q[0, 0, AB["chlorophyll"]] = 0.5; q[0, 0, AB["levitate"]] = 0.5
fv = torch.zeros(1, FIELD_DIM); fv[0, 0] = 1.0
with torch.no_grad():
    o = M(mc, mn, mm, oc, on, om, fv, q, torch.zeros(1, 6))
assert abs(float(o["fm_my"][0, 0, 0])) < 1e-6 and float(o["spd_opp"][0, 0]) > 0

# 7) 위협: 교체 후보(슬롯 1)가 위협이면 상대 활성 공격 -1 (x2/3). 상대 특성 결과 표
def intim(opp_ab, cand_ab="intimidate", cand_active=False):
    my = [active(stats=(100, 100, 100, 100, 100, 100)), dict(ab=cand_ab, stats=(300, 60, 60, 60, 60, 60))]
    if cand_active:
        my = [dict(stats=(100, 100, 100, 100, 100, 100)), active(ab=cand_ab, stats=(300, 60, 60, 60, 60, 60))]
    out = run(my, [active(ab=opp_ab, stats=(100, 100, 100, 100, 100, 100), moves=[(80, 0, "phys", "normal")])])
    return out["entry_opp"][0, 1], out
base_f, _ = intim("levitate")
assert torch.allclose(base_f, torch.tensor([2 / 3, 1.0, 1.0]), atol=1e-6)
assert torch.allclose(intim("clearbody")[0], torch.ones(3)) and torch.allclose(intim("innerfocus")[0], torch.ones(3))
assert torch.allclose(intim("defiant")[0], torch.tensor([1.5, 1.0, 1.0]))                                    # -1 후 +2 = +1 랭크
assert torch.allclose(intim("competitive")[0], torch.tensor([2 / 3, 2.0, 1.0]))
assert torch.allclose(intim("rattled")[0], torch.tensor([2 / 3, 1.0, 1.5]))
assert torch.allclose(intim("levitate", cand_ab="levitate")[0], torch.ones(3)) and torch.allclose(intim("levitate", cand_active=True)[0], torch.ones(3))
q = torch.zeros(1, 6, A); q[0, 0, AB["clearbody"]] = 0.5; q[0, 0, AB["levitate"]] = 0.5
out = run([active(stats=(100, 100, 100, 100, 100, 100)), dict(ab="intimidate", stats=(300, 60, 60, 60, 60, 60))],
          [active(stats=(100, 100, 100, 100, 100, 100), moves=[(80, 0, "phys", "normal")])], q_opp=q)
assert abs(float(out["entry_opp"][0, 1, 0]) - (0.5 * 1.0 + 0.5 * 2 / 3)) < 1e-6
# 후보의 데미지 입력에도 반영: 위협 후보가 받는 상대 물리 기술 데미지가 같은 조건의 비위협 후보보다 작음
dmg_i = intim("levitate")[1]["def"][0, 1, 0, 1]
dmg_n = intim("levitate", cand_ab="levitate")[1]["def"][0, 1, 0, 1]
assert dmg_i < dmg_n

# 8) 재생력: HP 100% -> 0, 50% -> 1/3, 90% -> 0.1. 다른 특성은 0
heal = lambda hp, ab: float(run([active(ab=ab, hp=hp)], [active()])["exit_heal"][0])
assert heal(1.0, "regenerator") == 0.0 and abs(heal(0.5, "regenerator") - 1 / 3) < 1e-6 and abs(heal(0.9, "regenerator") - 0.1) < 1e-6 and heal(0.5, "levitate") == 0.0

# 9) 상대 드러난 기술칸의 선공 기대값 (내 활성 기준)
opp_fm = lambda pri, spe=20, extra=None: float(run([active(stats=(100, 100, 100, 100, 100, 200))],
                                                    [active(stats=(100, 100, 100, 100, 100, spe), moves=[(50, pri, "phys", "normal")], **(extra or {}))])["fm_opp"][0, 0, 0])
assert opp_fm(0) == -1.0 and opp_fm(1) == 1.0                                                                # 우선도 1이면 느려도 먼저 (내 활성 최고 우선도 0)
assert float(run([active(stats=(100, 100, 100, 100, 100, 200))], [active(stats=(100, 100, 100, 100, 100, 100), moves=[(50, 0, "phys", "normal")])])["fm_opp"][0, 0, 1]) == 0.0   # 안 드러난 칸은 0
# 10) 다단히트: 효과 특징[2] = (기대 타수 - 1)/4 이면 데미지 비율이 기대 타수배 (3.1타 = 0.525), 정밀기술 경계는 한 방 위력 기준 그대로
def multi(frac, bp=50):
    mc, mn, mm = team([active()], True)
    oc, on, om = team([active()], False)
    mm[0, 0, 0, 0], mm[0, 0, 0, 3], mm[0, 0, 0, 4], mm[0, 0, 0, 6], mm[0, 0, 0, 10] = bp / 200.0, 0, CATEGORY["phys"], TY["normal"] / 20.0, frac
    with torch.no_grad():
        out = M(mc, mn, mm, oc, on, om, torch.zeros(1, FIELD_DIM), F.one_hot(oc[..., 1], A).float(), torch.zeros(1, 6))
    return float(out["off"][0, 0, 0, 1])
single, triple = multi(0.0), multi(0.5)
assert abs(triple / single - 3.0) < 1e-3, (single, triple)

# 11) 데미지 보정 (소스: Showdown, tools/matchup_check.py로 실측 대조). 내 활성(공격 60, 방어 60)이 상대 활성(HP 404, 방어/특방 259)을 때림. 기본 기술 자속 x1.5
ME = (300, 60, 60, 60, 60, 100)
def dm(att=None, dfn=None, move=(80, 0, "phys", "normal"), field=None, stats=ME):
    out = run([active(stats=stats, moves=[move], **(att or {}))], [active(types=("normal",), stats=(100, 80, 80, 80, 80, 80), **(dfn or {}))], field)
    return float(out["off"][0, 0, 0, 1])
def want(bp=80, a=60, d=259, mult=1.5):
    return min(((42 * bp * a / d) / 50 + 2) * mult * 0.925 / 404, 1.5) / 1.5
close = lambda x, y: abs(x - y) < 1e-5
assert close(dm(), want())
assert close(dm({"item": "lifeorb"}), want(mult=1.5 * 5324 / 4096))
assert close(dm({"item": "choiceband"}), want(a=90)) and close(dm({"item": "choiceband"}, move=(80, 0, "spec", "normal")), want())
assert close(dm({"item": "choicespecs"}, move=(80, 0, "spec", "normal")), want(a=90)) and close(dm({"item": "choicespecs"}), want())
assert close(dm(dfn={"item": "assaultvest"}, move=(80, 0, "spec", "normal")), want(d=259 * 1.5)) and close(dm(dfn={"item": "assaultvest"}), want())
assert close(dm(dfn={"ab": "multiscale"}), want(mult=0.75)) and close(dm(dfn={"ab": "multiscale", "hp": 0.9}), want())
assert close(dm(field={32: 1.0}), want(mult=0.75)) and close(dm(field={34: 1.0}), want()) and close(dm(field={34: 1.0}, move=(80, 0, "spec", "normal")), want(mult=0.75))
assert close(dm({"ab": "adaptability"}), want(mult=2.0)) and close(dm({"ab": "sharpness"}, move=(80, 0, "phys", "normal", "psychocut")), want(bp=120))
fire = lambda **kw: dm(move=(80, 0, "spec", "fire"), **kw)
assert close(fire(field={0: 1.0}), want(bp=120, mult=1.0)) and close(fire(field={2: 1.0}), want(bp=40, mult=1.0)) and close(fire(field={0: 1.0}, att={"ab": "cloudnine"}), want(mult=1.0))
assert close(dm(move=(80, 0, "phys", "normal", "bodypress"), stats=(300, 60, 200, 60, 60, 100)), want(a=200)) and close(dm(move=(80, 0, "phys", "normal", "foulplay")), want(a=259))
assert close(dm(dfn={"item": "leftovers"}, move=(80, 0, "phys", "normal", "knockoff")), want(bp=120)) and close(dm(move=(80, 0, "phys", "normal", "knockoff")), want())
# 테라: 원래 타입과 같은 타입으로 테라하면 자속 2배(관측 타입 칸은 테라 타입으로 덮이니 원래 타입은 종 표에서), 아니면 1.5, 위력 60 미만 기술은 60으로
snor = {"species": "snorlax", "tera": "normal"}
assert close(dm(snor), want(mult=2.0)) and close(dm(snor, move=(40, 0, "phys", "normal")), want(bp=60, mult=2.0))
assert close(dm({"species": "snorlax", "tera": "fire"}, move=(80, 0, "phys", "fire")), want(mult=1.5)) and close(dm({"species": "snorlax", "tera": "fire"}), want())
assert close(dm({**snor, "ab": "adaptability"}), want(mult=2.25))
print("matchup v7 OK")
