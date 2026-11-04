"""입력 감사: Foul Play가 결정 시점에 보는 관측 상태(fp/state_probe.py 덤프)와 학생 인코더가 같은 결정에서 만든 텐서를 필드 단위로 대조한다.
A) 학생이 열을 가진 필드: 값이 같은지 (HP, 상태이상, 랭크, 날씨/지형/트릭룸, 사이드 조건, 효과, 기술 PP 비율, 마지막 사용 기술, 등장 후 쓴 기술, 수면/맹독 카운터, 휘발성 상태, 소원/미래예지)
B) 학생이 열이 없는 필드: 교사의 값이 기본값이 아닌 결정의 비율 (사이드 조건 중 열이 없는 것 등)
C) 교사만 아는 상대 정보: 교사는 도구/특성을 아는데 학생은 숨김인 (결정, 슬롯) 쌍 (남은 추론 규칙의 몫)
결정 단위로 센다. 인코더를 고친 뒤 이 도구를 다시 돌려 A가 맞고 B/C가 비는지 확인하는 회귀 검사로 쓴다 (워커 1개면 reveals.py의 추적 중 예외도 출력).
사용: python tools/state_audit.py --probe "data/dagger/raw/d7b/st_rec_*_h*.jsonl" [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
from src.core.tensor_encoder import BASE_NUM, EXT_START, NAME_INDEX, SIDE_EXT
from src.core.events import SLOT_DIM
from src.training.build_fp_dataset import _states, load_jsonl
from src.training.replay_dataset import _norm

V = json.load(open("data/vocab.json", encoding="utf-8"))
INV_MOVE = {i: k for k, i in V["move"].items()}
INV_ITEM = {i: k for k, i in V["item"].items()}
INV_ABIL = {i: k for k, i in V["ability"].items()}
BOOST = {"attack": 3, "defense": 4, "special-attack": 5, "special-defense": 6, "speed": 7}
WEATHER = (("sun", 0), ("desolate", 0), ("rain", 2), ("primordial", 2), ("sand", 4), ("hail", 6), ("snow", 6), ("deltastream", 8))
TERRAIN = (("electric", 12), ("grassy", 14), ("psychic", 16), ("misty", 18))
SIDE_COL = {"reflect": 0, "lightscreen": 2, "auroraveil": 4, "tailwind": 6, "stealthrock": 8, "spikes": 9, "toxicspikes": 10, "stickyweb": 11}
COVERED_VOL = {"taunt", "encore", "disable"}
EFFECT_BY_NAME = NAME_INDEX
EXT = BASE_NUM + SLOT_DIM          # 포켓몬별 추가 수치 시작 열


def collect(pattern, workers):
    by_tag = defaultdict(lambda: {"st": []})
    for r in load_jsonl(pattern):
        d = by_tag[(r["_src"], r["tag"])]
        if r["type"] == "state":
            d["st"].append(r)
        elif r["type"] in ("start", "protocol"):
            d[r["type"]] = r
    jobs, metas = [], []
    for d in by_tag.values():
        if "protocol" not in d or "start" not in d or not d["st"]:
            continue
        protocol, user = d["protocol"]["lines"], d["start"]["user"]
        side = next((s[2] for s in (l.split("|") for l in protocol) if len(s) > 4 and s[1] == "player" and s[3] == user), None)
        if side is not None:
            jobs.append((protocol, side, user, d["start"]["team"]))
            metas.append(d["st"])
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            all_states = pool.map(_states, jobs, chunksize=4)
    else:
        all_states = [_states(j) for j in jobs]
    items = []
    for recs, states in zip(metas, all_states):
        if states is None:
            continue
        queues = defaultdict(list)
        for st, turn, kind in states:
            queues[(turn, kind)].append(st)
        for r in recs:
            q = queues[(r["turn"], int(r["force_switch"]))]
            if q:
                items.append((q.pop(0)["obs"], r))
    return items


def slot_of(cat, name):
    """종 이름 -> 학생 슬롯 번호 (cat[:,10] 종 id 기준)"""
    sid = V["species"].get(_norm(name))
    if sid is None:
        return None
    hit = np.where(cat[:, 10] == sid)[0]
    return int(hit[0]) if len(hit) else None


def decode_last(row, ids):
    """학생의 마지막 행동 원핫 -> ("move", 이름) | ("switch",) | None"""
    for j in range(4):
        if row[EXT + 6 + j] > 0.5:
            return ("move", INV_MOVE.get(int(ids[j]), "?"))
    if row[EXT + 10] > 0.5:
        return ("switch",)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    items = collect(args.probe, args.workers)
    n = len(items)
    print(f"교사 상태 기록과 맞춘 결정 {n}개")
    match = defaultdict(lambda: [0, 0])
    gap = defaultdict(int)
    vol_match = defaultdict(lambda: [0, 0])
    vol_names = {"user": Counter(), "opp": Counter()}
    only_fp = {"item": Counter(), "ability": Counter()}
    both_diff = Counter()
    bad_examples = defaultdict(list)

    def chk(field, ok, ex=None):
        match[field][0] += 1
        match[field][1] += bool(ok)
        if not ok and ex is not None and len(bad_examples[field]) < 3:
            bad_examples[field].append(ex)

    unk_it, unk_ab = V["item"]["<unk>"] + 1, V["ability"]["<unk>"] + 1
    for obs, r in items:
        my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv, fv = [np.asarray(x) for x in obs]
        any_gap = defaultdict(bool)
        g = r["global"]
        w = str(g["weather"]).lower().replace("_", "")
        for key, col in WEATHER:
            if key in w:
                chk("날씨 켜짐", fv[col] > 0.5, (w, fv[col]))
                break
        else:
            chk("날씨 꺼짐", all(fv[c] < 0.5 for c in (0, 2, 4, 6, 8)), (w,))
        t = str(g["terrain"]).lower()
        for key, col in TERRAIN:
            if key in t:
                chk("지형 켜짐", fv[col] > 0.5, (t, fv[col]))
                break
        chk("트릭룸", (fv[10] > 0.5) == bool(g["trick_room"]), (g["trick_room"], fv[10]))
        for k, (side, cat, num, mv, base, eff0, ext0) in enumerate((("user", my_cat, my_num, my_mv, 20, 58, EXT_START), ("opp", opp_cat, opp_num, opp_mv, 32, 65, EXT_START + SIDE_EXT))):
            s, srole = r[side], ("me" if side == "user" else "opp")
            for key, v in s["side_conditions"].items():
                key = re.sub(r"[^a-z0-9]", "", key.lower())
                if key in SIDE_COL:
                    col = base + SIDE_COL[key]
                    want = v / 3.0 if key == "spikes" else (v / 2.0 if key == "toxicspikes" else 1.0)
                    chk(f"사이드 조건 {key}", (abs(fv[col] - want) < 0.05) if key in ("spikes", "toxicspikes") else fv[col] > 0.5, (side, v, fv[col]))
                elif key == "safeguard":
                    chk("사이드 조건 safeguard", fv[44 if base == 20 else 45] > 0, (side, v))
                elif key == "toxiccount":
                    pass                                  # 활성 포켓몬의 맹독 카운터는 포켓몬별 열(아래)에서 대조
                elif key == "protect":
                    pass                                  # 방어 연속은 포켓몬별 열(아래)에서 대조
                else:
                    any_gap["사이드 조건(열 없음): " + key] = True
            wish_on = s["wish"][0] > 0
            chk("소원 진행 여부", (fv[ext0 + SIDE_EXT - 4] > 0.5) == wish_on, (side, s["wish"], fv[ext0 + SIDE_EXT - 4]))
            fs_on = s["future_sight"][0] > 0
            chk("미래예지 진행 여부", (fv[ext0 + SIDE_EXT - 2] > 0.5) == fs_on, (side, s["future_sight"], fv[ext0 + SIDE_EXT - 2]))
            a = s["active"]
            if a is None:
                continue
            for v in a["volatiles"]:
                vol_names[side][v] += 1
                col = EFFECT_BY_NAME.get(re.sub(r"[^a-z0-9]", "", v.lower()))
                if col is None:
                    vol_match[v][0] += 1
                    any_gap["휘발성 상태: poke-env Effect에 없음"] = True
                else:
                    vol_match[v][0] += 1
                    vol_match[v][1] += fv[ext0 + col] > 0.5
            for v in COVERED_VOL:
                if v in a["volatiles"]:
                    chk(f"효과 {v}", fv[eff0 + ["taunt", "encore", "disable"].index(v)] > 0.5, (side, v))
            ai = slot_of(cat, a["name"])
            if ai is not None:
                row = num[ai]
                ids = cat[ai, 5:9]
                # 마지막 행동 (교사: 기술 이름 또는 "switch 종")
                lu = s["last_used"][0]
                got = decode_last(row, ids)
                want = ("switch",) if lu.startswith("switch ") else (("move", _norm(lu)) if lu else None)
                chk("마지막 사용 기술", got == want, (side, lu, got))
                # 등장 후 쓴 기술
                used_want = {_norm(x) for x in a["used_since_switch"]}
                used_got = {INV_MOVE.get(int(ids[j]), "?") for j in range(4) if row[EXT + 12 + j] > 0.5}
                chk("등장 후 쓴 기술", used_got == used_want, (side, sorted(used_want), sorted(used_got)))
                # 카운터
                if a["status"] == "slp" or a["sleep_turns"] > 0 or a["rest_turns"] > 0:
                    chk("수면 카운터 유무", row[EXT + 2] > 0, (side, a["sleep_turns"], a["rest_turns"], float(row[EXT + 2])))
                if a["status"] == "tox":
                    chk("맹독 카운터 유무", row[EXT + 3] > 0, (side, float(row[EXT + 3])))
            for p in [a] + [x for x in s["reserve"] if x]:
                i = slot_of(cat, p["name"])
                if i is None or p["max_hp"] in (0, None):
                    continue
                if side == "opp" and not p["revealed"]:
                    chk("HP 비율 (상대 미공개)", abs(num[i, 2] - p["hp"] / p["max_hp"]) < 0.03, (p["name"], float(num[i, 2]), p["hp"] / p["max_hp"]))
                else:
                    chk("HP 비율", abs(num[i, 2] - p["hp"] / p["max_hp"]) < 0.03, (side, p["name"], float(num[i, 2]), p["hp"] / p["max_hp"]))
                    st_id = V["status"]["fnt"] if p["hp"] <= 0 else (V["status"].get(p["status"], 0) if p["status"] else 0)
                    chk("상태이상 (기절 포함)", int(cat[i, 4]) == st_id or (not p["status"] and int(cat[i, 4]) == V["status"]["none"]), (side, p["status"], int(cat[i, 4])))
                if p is a:
                    for bk, col in BOOST.items():
                        chk("랭크", abs(num[i, col] * 6 - p["boosts"].get(bk, 0)) < 0.5, (side, bk, float(num[i, col] * 6), p["boosts"].get(bk, 0)))
                if side == "user":
                    for m in p["moves"]:
                        j = next((j for j in range(4) if int(cat[i, 5 + j]) == V["move"].get(_norm(m["name"]), -1)), None)
                        if j is not None and m.get("max_pp"):
                            chk("기술 PP 비율(내 쪽)", abs(mv[i, j, 2] - m["pp"] / m["max_pp"]) < 0.05, (p["name"], m["name"], float(mv[i, j, 2]), m["pp"], m["max_pp"]))
                else:
                    fi = p["item"] not in ("unknownitem", None, "", "None")
                    fa = p["ability"] not in (None, "")
                    for kind, known, col, unk, inv in (("item", fi, 0, unk_it, INV_ITEM), ("ability", fa, 1, unk_ab, INV_ABIL)):
                        sid = int(cat[i, col])
                        if known and sid == unk:
                            only_fp[kind][p["item"] if kind == "item" else (p["name"], p["ability"])] += 1
                            any_gap[f"{'도구' if kind == 'item' else '특성'}: 교사는 알고 학생은 숨김"] = True
                        elif known and sid != unk:
                            ok = inv.get(sid) == _norm(p[kind])
                            chk(f"{'도구' if kind == 'item' else '특성'} 값 (둘 다 앎)", ok, (p["name"], p[kind], inv.get(sid)))
                            if not ok:
                                both_diff[(kind, p[kind], inv.get(sid))] += 1
                        elif not known and sid != unk:
                            any_gap[f"{'도구' if kind == 'item' else '특성'}: 학생만 앎 (교사는 모름)"] = True
        for k in any_gap:
            gap[k] += 1

    print("\n===== A) 학생이 열을 가진 필드: 값 일치 =====")
    print(f"{'필드':26s} {'비교':>8s} {'일치율':>8s}")
    for k, (c, ok) in sorted(match.items()):
        print(f"{k:26s} {c:8d} {ok / c:8.4f}")
    for k, ex in bad_examples.items():
        if match[k][1] / match[k][0] < 0.995:
            print(f"  불일치 예 {k}: {ex}")
    print("\n휘발성 상태별: 교사에게 있을 때 학생의 멀티핫도 켜진 비율 (결정 수):")
    print("  " + ", ".join(f"{k} {ok}/{c}" for k, (c, ok) in sorted(vol_match.items(), key=lambda x: -x[1][0])[:40]))
    print("\n===== B/C) 학생이 못 보는 필드: 해당하는 결정 비율 (전체 결정 기준) =====")
    for k, c in sorted(gap.items(), key=lambda x: -x[1]):
        print(f"{k:46s} {c:6d}  {100 * c / n:5.1f}%")
    print("\n교사는 알고 학생은 숨김 - 도구:", only_fp["item"].most_common(12), "| 합계", sum(only_fp["item"].values()))
    print("교사는 알고 학생은 숨김 - 특성 (종, 특성):", only_fp["ability"].most_common(10), "| 합계", sum(only_fp["ability"].values()))
    if both_diff:
        print("둘 다 아는데 값이 다른 경우:", both_diff.most_common(8))
    if args.workers == 1:
        import src.core.events as E
        print("\nreveals.py 추적 예외:", sum(E.REV_ERRORS.values()), E.REV_ERRORS.most_common(5))


if __name__ == "__main__":
    main()
