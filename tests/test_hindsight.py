"""replay_dataset._hindsight(로그 전체에서 드러난 내 포켓몬 기술/도구/특성/테라) 회귀 테스트.
오늘(2026-10-30) 이 함수의 도구 주인 규칙이 틀려 사람 BC 데이터의 내 도구 입력 약 7.5%가 틀렸고, 첫 수정은 특성 소유자를 깨뜨렸음 → 합성 로그 사례 + 정답을 아는 판(d7 rec)으로 고정
사용: PYTHONPATH=. python tests/test_hindsight.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.training.replay_dataset import _hindsight


def parse(text):
    return [s for s in (l.split("|") for l in text.strip().split("\n")) if len(s) > 1]


def synthetic():
    head = "|switch|p2a: Klefki|Klefki, M|100/100\n|switch|p1a: Neko|Neko, M|100/100\n"
    # 1) Knock Off: 도구를 잃은 쪽(Klefki, s[2])의 도구이고 [of] 공격자(Neko)의 것이 아님
    h2 = _hindsight(parse(head + "|move|p1a: Neko|Knock Off|p2a: Klefki\n|-enditem|p2a: Klefki|Light Clay|[from] move: Knock Off|[of] p1a: Neko"), "p2")
    h1 = _hindsight(parse(head + "|move|p1a: Neko|Knock Off|p2a: Klefki\n|-enditem|p2a: Klefki|Light Clay|[from] move: Knock Off|[of] p1a: Neko"), "p1")
    assert h2["klefki"]["item"] == "lightclay", h2["klefki"]
    assert h1["neko"]["item"] is None, h1["neko"]
    # 2) Trick으로 생긴 -item은 원래 도구가 아님 (원래 도구는 -enditem으로 먼저 드러난 경우만 인정)
    h = _hindsight(parse(head + "|-item|p2a: Klefki|Choice Scarf|[from] move: Trick"), "p2")
    assert h["klefki"]["item"] is None, h["klefki"]
    # 3) Frisk: 드러나는 도구는 대상(s[2])의 것, 특성 Frisk는 [of] (Frisk를 가진 쪽)의 것
    text = head + "|-item|p2a: Klefki|Leftovers|[from] ability: Frisk|[of] p1a: Neko|[identify]"
    assert _hindsight(parse(text), "p2")["klefki"]["item"] == "leftovers"
    assert _hindsight(parse(text), "p1")["neko"]["ability"] == "frisk"
    assert _hindsight(parse(text), "p2")["klefki"]["ability"] is None
    # 4) 기술: 다른 기술에서 불려 나온 것([from])은 제외, 4개까지
    h = _hindsight(parse(head + "|move|p2a: Klefki|Spikes|p1a: Neko\n|move|p2a: Klefki|Metronome|p1a: Neko\n|move|p2a: Klefki|Thunder Wave|p1a: Neko|[from]move: Metronome"), "p2")
    assert h["klefki"]["moves"] == ["spikes", "metronome"], h["klefki"]["moves"]


def against_truth():
    """정답을 아는 판(d7 rec의 실제 request)에서: 드러난 도구/특성이 진짜와 같은 비율, 드러난 기술이 실제 세트에 있는 비율"""
    from fp_fidelity import degrade, load_recs, requests, species_of
    from human_to_fp import _norm
    from proxy_accuracy import RECS, parse as plog
    tot = {"item": [0, 0], "ability": [0, 0], "move": [0, 0]}
    for name in RECS:
        path = f"data/dagger/raw/d7/{name}.jsonl.gz"
        if not os.path.exists(path):
            return print("  (정답 데이터 없음: 건너뜀)")
        for rec in load_recs(path):
            log, side = degrade(rec)
            hind = _hindsight(plog(log), side)
            truth = {species_of(p["details"]): p for p in requests(rec)[0][1]["side"]["pokemon"]}
            for sp, h in hind.items():
                if sp not in truth:
                    continue
                for f, key in (("item", "item"), ("ability", "baseAbility")):
                    if h[f]:
                        tot[f][0] += 1
                        tot[f][1] += _norm(truth[sp][key]) == h[f]
                for m in h["moves"]:
                    tot["move"][0] += 1
                    tot["move"][1] += m in [_norm(x) for x in truth[sp]["moves"]]
    for f, floor in (("item", 0.99), ("ability", 0.985), ("move", 0.99)):
        n, ok = tot[f]
        print(f"  {f}: {ok}/{n} = {ok / n:.4f} (기준 {floor})")
        assert ok / n >= floor, f"{f} 정확도가 기준 미달"


if __name__ == "__main__":
    synthetic()
    print("합성 사례 OK")
    against_truth()
    print("hindsight OK")
