"""
상대 세트 사전 확률 (정보 추론).

데이터: 로컬 Showdown의 gen9 세트 (다운로드 불필요)
- factory-sets.json["OU"]: OU 실전 세트 (도구/특성/테라/기술, 가중치 포함)
- sets.json: 랜덤배틀 세트 (특성/테라/기술 후보만, 도구 없음) → OU 세트에 없는 종의 폴백

공개된 기술/도구/특성과 모순되는 세트는 걸러내고 남은 세트로 가장 유력한 값을 고른다.
ponytail: 사용률 통계(Smogon chaos) 대신 세트 목록 빈도, 실제 래더 분포가 필요하면 chaos JSON으로 교체
"""
import json
import os
import re
from collections import Counter
from functools import lru_cache

SETS_DIR = "pokemon-showdown/data/random-battles/gen9"


def _id(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(getattr(x, "name", x) or "").lower())


@lru_cache(maxsize=1)
def _sets():
    """species id -> [(weight, items, abilities, teras, moves)]"""
    out = {}
    try:
        with open(os.path.join(SETS_DIR, "factory-sets.json"), encoding="utf-8") as f:
            for sp, entry in json.load(f)["OU"].items():
                for s in entry["sets"]:
                    out.setdefault(_id(sp), []).append((
                        s.get("weight", 1), [_id(x) for x in s.get("item", [])], [_id(x) for x in s.get("ability", [])],
                        [_id(x) for x in s.get("teraType", [])], {_id(m) for slot in s["moves"] for m in slot}))
        with open(os.path.join(SETS_DIR, "sets.json"), encoding="utf-8") as f:
            for sp, entry in json.load(f).items():
                if _id(sp) in out:
                    continue
                for s in entry.get("sets", []):
                    out.setdefault(_id(sp), []).append((
                        1, [], [_id(x) for x in s.get("abilities", [])],
                        [_id(x) for x in s.get("teraTypes", [])], {_id(m) for m in s.get("movepool", [])}))
    except (OSError, KeyError, ValueError) as e:
        print(f"⚠️ 세트 사전 확률 로드 실패 ({e}). 정보 추론 없이 진행")
    return out


def predict(species, revealed_moves=(), item=None, ability=None):
    """
    return: {"item", "item_p", "ability", "tera", "moves"} 또는 None (데이터 없는 종)
    item/ability: 이미 알려진 값이면 넘겨서 세트를 좁힘 (모르면 None)
    moves: 공개된 기술을 뺀 유력 기술 id 목록 (확률 내림차순)
    """
    sets = _sets().get(_id(species))
    if not sets:
        return None
    rev = {_id(m) for m in revealed_moves}
    ok = [s for s in sets if rev <= s[4]] or sets
    if item:
        ok = [s for s in ok if _id(item) in s[1]] or ok
    if ability:
        ok = [s for s in ok if _id(ability) in s[2]] or ok

    items, abilities, teras, moves = Counter(), Counter(), Counter(), Counter()
    for w, its, abs_, tes, mvs in ok:
        for c, xs in ((items, its), (abilities, abs_), (teras, tes)):
            for x in xs:
                c[x] += w / len(xs)
        for m in mvs:
            if m not in rev:
                moves[m] += w
    total = sum(s[0] for s in ok)
    top = lambda c: c.most_common(1)[0][0] if c else None
    return {
        "item": top(items),
        "item_p": items.most_common(1)[0][1] / total if items else 0.0,
        "ability": top(abilities),
        "tera": top(teras),
        "moves": [m for m, _ in moves.most_common()],
    }


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    p = predict("kingambit")
    assert p and p["ability"] == "supremeoverlord" and "suckerpunch" in p["moves"][:4], p
    # 공개 기술로 세트 좁히기: Low Kick 공개 → Low Kick 없는 Leftovers 전용 세트(가중치 40) 제외
    assert p["item"] == "leftovers" and predict("kingambit", ["lowkick"])["item"] == "blackglasses"
    assert "lowkick" not in predict("kingambit", ["lowkick"])["moves"]
    # OU 세트에 없고 랜덤배틀에만 있는 종: 도구 없음
    rb = next(sp for sp in _sets() if all(not s[1] for s in _sets()[sp]))
    assert predict(rb)["item"] is None and predict(rb)["moves"]
    assert predict("notapokemon") is None
    print("set_prior OK", p["item"], p["tera"], p["moves"][:4])
