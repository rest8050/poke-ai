"""
Foul Play(탐색 봇) 대전 기록 → 증류 학습 데이터 (train_fp_distill.py 입력).

입력
  --decisions  Foul Play가 남긴 결정 JSONL (foul-play repo의 fp/decision_log.py, 환경변수 FP_DECISION_LOG)
               start(팀 전체) / decision(전체 탐색 분포) / end(승자) / protocol(자기 시점 공개 로그) 레코드
  --protocols  상대편(poke-env)이 저장한 배틀별 공개 프로토콜 JSONL ({"tag", "won", "protocol"}). 생략 가능 —
               FoulPlay vs FoulPlay(자가대전)처럼 상대편 쪽 수집기가 없으면 각 컨테이너의 protocol 레코드를 대신 씀
               (자가대전은 양쪽 컨테이너 dec 파일에 같은 배틀 태그가 나오므로 파일 단위로 구분해서 따로 처리 — 한쪽으로 덮이지 않음)
처리
  프로토콜을 Foul Play 쪽 시점으로 재생하되, 내 팀 정보는 사후 추정이 아니라 Foul Play가 실제로 쓴 팀(start 레코드)으로 채움
  → 실시간 인코더와 같은 완전한 내 팀 정보. 결정 지점(턴 시작/기절 후 교체)마다 Foul Play의 탐색 분포를 22칸 행동 분포로 변환
출력 npz: OBS_KEYS + action_mask + target(22) + value_target, teacher_value(탐색이 본 이 쪽 승률 0~1, 결정마다), lens(배틀별 길이), outcome

사용: python src/training/build_fp_dataset.py --decisions "data/fp_gen/ms050/dec_*.jsonl" --protocols "data/fp_gen/ms050/prot_*.jsonl" --out data/fp_gen/ms050/fp_data.npz
자가대전: python src/training/build_fp_dataset.py --decisions "data/fp_selfplay/ms050/dec_*.jsonl" --out data/fp_selfplay/ms050/fp_data.npz
"""
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir != os.path.dirname(_current_dir):
    if os.path.exists(os.path.join(_current_dir, "data", "vocab.json")):
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        break
    _current_dir = os.path.dirname(_current_dir)
import argparse
import glob
import json
import multiprocessing as mp
from collections import Counter, defaultdict

import numpy as np
from poke_env.battle import Battle

from src.training.replay_dataset import OBS_KEYS, _LOGGER, _capture, _norm, _preview_mon, legal_from_request

GAMMA = 0.995


def load_jsonl(pattern):
    """pattern: 콤마로 여러 glob 패턴 지정 가능 (train_fp_distill.py --data와 동일한 규칙)"""
    paths = sorted({p for pat in pattern.split(",") for p in glob.glob(pat.strip())})
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_src"] = path  # 자가대전은 양쪽 컨테이너가 같은 배틀 태그를 쓰므로 파일로 구분
                    yield r


def _num(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]


def team_info(team):
    """Foul Play 팀 dict(자기 팀, Showdown 형식 nature/evs/ivs 포함) → _preview_mon이 쓰는 dict.
    노력치/성격까지 넣어 종족값 대신 실제 능력치를 인코더에 줌 (내 팀만; export 생략 시 EV=0/IV=31이 Showdown 기본값)"""
    return {_norm(m["species"]): {
        "moves": [_norm(x) for x in m["moves"]], "item": _norm(m["item"]) or None,
        "ability": _norm(m["ability"]) or None, "tera": _norm(m["tera_type"]) or None,
        "nature": _norm(m.get("nature")) or "serious", "level": _num(m.get("level"), 100),
        "evs": [_num((m.get("evs") or {}).get(k), 0) for k in STAT_KEYS],
        "ivs": [_num((m.get("ivs") or {}).get(k), 31) for k in STAT_KEYS],
    } for m in team}


def _choice_index(choice, moves, team):
    """'기술' | '기술-tera' | 'switch 종' → 22칸 행동 인덱스 (매칭 안 되면 None)"""
    if choice.startswith("switch "):
        sp = _norm(choice[len("switch "):])
        k = next((k for k, s in enumerate(team) if s == sp), None)
        if k is None:       # 폼이 바뀐 이름(예: 오거폰 테라 폼 'ogerpontealtera')은 접두사로 매칭 (한 팀에 같은 종은 하나)
            k = next((k for k, s in enumerate(team) if s and (s.startswith(sp) or sp.startswith(s))), None)
        return None if k is None else 8 + k
    tera = choice.endswith("-tera")
    mid = _norm(choice.removesuffix("-tera").removesuffix("-mega"))
    return next((i + (4 if tera else 0) for i, m in enumerate(moves) if m == mid), None)


CONSTRAINT_ITEMS = {"choicescarf", "choiceband", "choicespecs", "assaultvest"}       # 기술 선택을 제한하는 도구
CONSTRAINT_EFFECTS = {"TAUNT", "ENCORE", "DISABLE", "TORMENT", "IMPRISON", "HEAL_BLOCK", "THROAT_CHOP", "LOCKED_MOVE"}


def legal_from_options(r, state):
    """결정 기록의 옵션 목록(Foul Play가 서버 요청에서 얻은 실제 합법 행동, 모든 세계의 합집합) -> (22칸 합법 마스크, 매칭 실패한 옵션 이름들).
    옵션이 없으면 (None, [])"""
    names = {o[0] for w in (r.get("worlds") or []) for o in w.get("o", [])}
    if not names:
        return None, []
    moves, team = [_norm(m) for m in state["moves"]], [_norm(x) for x in state["team_species"]]
    legal, bad = np.zeros(22, bool), []
    for n in sorted(names):
        i = _choice_index(n, moves, team)
        if i is None:
            bad.append(n)
        else:
            legal[i] = True
    return legal, bad


def to_target(state, policy, keep_ratio, choice=None):
    """[(선택, 가중치)] → (22칸 분포, 매칭된 질량 비율, 실제로 둔 수의 인덱스). 선택: '기술', '기술-tera', 'switch 종'.
    keep_ratio: 최고 가중치의 이 비율 이상만 남김. 기본 0.2 = 방문이 극히 적은 꼬리만 제거, 0이면 전체 탐색 분포 그대로
    (Foul Play가 실제로 두는 규칙은 0.75지만, 그대로 쓰면 지식이 아니라 그 선택 절차까지 모방하게 됨)
    choice: 그 턴에 Foul Play가 실제로 둔 수(정책 그라디언트용 action_taken) — 없으면 None 반환"""
    best = max(w for _, w in policy)
    policy = [(c, w) for c, w in policy if w >= keep_ratio * best]
    t = np.zeros(22, np.float32)
    moves, team = [_norm(m) for m in state["moves"]], [_norm(s) for s in state["team_species"]]
    total = sum(w for _, w in policy) or 1.0
    for c, w in policy:
        idx = _choice_index(c, moves, team)
        if idx is not None and state["mask"][idx]:
            t[idx] += w
    s = float(t.sum())
    action_idx = _choice_index(choice, moves, team) if choice is not None else None
    if action_idx is not None and not state["mask"][action_idx]:
        action_idx = None
    return (t / s if s > 0 else None), s / total, action_idx


def _apply_request(st, req, kind):
    """상태의 행동 마스크를 서버 요청 기반 순수 합법 마스크로 교체 (못 만들면 규칙 마스크 유지)"""
    ids = [_norm(x) for x in st["moves"]]
    m = legal_from_request(req, ids, st["team_species"], kind)
    if m is not None and m.any():
        st["mask"], st["mask_src"] = m, "request"
        if kind == "turn":      # PP가 바닥난 기술 슬롯 (마스크 점검에서 "제약 없는 상태"의 정당한 예외)
            empty = {_norm(x.get("id") or x.get("move")) for x in req["active"][0].get("moves", []) if x.get("pp", 1) == 0}
            st["pp_empty"] = np.array([i in empty for i in ids] + [False] * (4 - len(ids)))


def replay_states(protocol, side, name, team):
    """Foul Play 시점으로 재생해 결정 지점의 상태들 [(상태, 게임 턴, 종류 0=턴 1=기절 후 교체)]"""
    lines = [l.split("|") for l in protocol if l.startswith("|") and l != "|"]
    lines = [s for s in lines if len(s) > 1]
    info = team_info(team)
    battle = Battle(f"fp-{side}", name, _LOGGER, gen=9)
    states, forced = [], False
    last_req, waiting = None, None      # 서버 요청: 턴 결정의 요청은 |turn| 줄 바로 뒤에 오고, 강제 교체의 요청은 교체 직전에 이미 와 있음
    for s in lines:
        ev = s[1]
        if ev == "request":
            try:
                req = json.loads("|".join(s[2:]))
            except ValueError:
                continue
            if waiting is not None and req.get("active"):
                _apply_request(states[waiting][0], req, "turn")
                waiting = None
            last_req = req
            continue
        own = len(s) > 2 and s[2].startswith(side) and len(s[2]) > 2 and s[2][2] in "abc"
        if own and ev == "switch" and forced:
            st = _capture(battle, "forced")
            if last_req and last_req.get("forceSwitch"):
                _apply_request(st, last_req, "forced")
            states.append((st, battle.turn, 1))
            forced = False
        elif own and ev == "faint":
            forced = True
        if ev == "poke" and len(s) > 3:
            if s[2] == side:
                battle._team[f"{side}: {s[3].split(',')[0]}"] = _preview_mon(s[3], info)
            else:
                battle._register_teampreview_pokemon(s[2], s[3])
            continue
        if ev in ("win", "tie"):
            break
        try:
            battle.parse_message(s)
        except NotImplementedError:
            pass
        except Exception:
            return None
        if ev == "turn":
            states.append((_capture(battle, "turn"), battle.turn, 0))
            waiting = len(states) - 1
    return states


def _states(task):
    return replay_states(*task)


def build(decisions, protocols, min_matched, keep_ratio, workers=1):
    decisions = list(decisions)  # 아래서 두 번(결정 집계 + 자기기록 프로토콜 추출) 훑음
    # (파일, 태그)로 묶음: 자가대전은 두 컨테이너가 같은 배틀 태그를 쓰므로 태그만으로 묶으면 서로 덮어씀
    by_tag = defaultdict(lambda: {"dec": []})
    for r in decisions:
        d = by_tag[(r["_src"], r["tag"])]
        if r["type"] == "decision":
            d["dec"].append(r)
        else:
            d[r["type"]] = r
    # 공개 프로토콜: 상대편(poke-env) 기록이 있으면 그걸, 없으면(자가대전) FoulPlay 자신이 남긴 걸 씀. 태그만으로 묶음(양쪽 다 같은 공개 로그)
    prot_by_tag = {}
    for p in protocols:
        prot_by_tag.setdefault(p["tag"], p["protocol"])
    for r in decisions:
        if r["type"] == "protocol":
            prot_by_tag.setdefault(r["tag"], r["lines"])

    stat = Counter()
    unmatched = defaultdict(lambda: [0, None])       # 22칸에 못 넣은 옵션 이름 -> [횟수, 첫 사례 (태그, 턴)]
    out, jobs = [], []
    for (_src, tag), d in by_tag.items():
        # 자기 컨테이너의 프로토콜을 우선 (자기 쪽 서버 요청 |request|가 들어 있어 순수 합법 마스크를 만들 수 있음), 없으면 상대편 수집분
        protocol = d["protocol"]["lines"] if "protocol" in d else prot_by_tag.get(tag)
        if not protocol or "start" not in d or "end" not in d or d["end"]["winner"] is None:
            stat["배틀 제외(결정/종료/프로토콜 기록 없음)"] += 1
            continue
        user = d["start"]["user"]
        side = next((s[2] for s in (l.split("|") for l in protocol) if len(s) > 4 and s[1] == "player" and s[3] == user), None)
        if side is None:
            stat["배틀 제외(플레이어 못 찾음)"] += 1
            continue
        jobs.append((d, user, (protocol, side, user, d["start"]["team"])))
    if workers > 1:  # 배틀별 재생은 서로 독립 → 프로세스 풀
        with mp.get_context("spawn").Pool(workers) as pool:
            all_states = pool.map(_states, [j[2] for j in jobs], chunksize=4)
    else:
        all_states = [_states(j[2]) for j in jobs]
    for (d, user, _), states in zip(jobs, all_states):
        if states is None:
            stat["배틀 제외(재생 실패)"] += 1
            continue
        queues = defaultdict(list)
        for st, turn, kind in states:
            queues[(turn, kind)].append(st)
        steps, tvals = [], []
        for r in d["dec"]:
            key = (r["turn"], int(r["force_switch"]))
            if not queues[key]:
                stat["결정 제외(상태 못 찾음)"] += 1
                continue
            st = queues[key].pop(0)
            from_req = st.get("mask_src") == "request"      # replay_states가 서버 요청에서 만든 순수 합법 마스크 (없으면 규칙 마스크)
            stat["마스크: 서버 요청 기반" if from_req else "마스크: 규칙 대체(요청 못 찾음)"] += 1
            opt, bad = legal_from_options(r, st)             # Foul Play가 고려한 옵션: 진단용 (마스크는 요청 기반)
            if opt is not None:
                for n in bad:
                    unmatched[n][0] += 1
                    unmatched[n][1] = unmatched[n][1] or (r["tag"], r["turn"])
                stat["옵션 중 22칸 매칭 실패가 있는 결정"] += bool(bad)
                if from_req:
                    stat["FP 옵션이 요청 합법 밖 행동을 포함한 결정"] += bool((opt & ~st["mask"]).any())
                    stat["요청상 합법인데 FP 옵션에 없는 행동이 있는 결정(FP 자체 가지치기)"] += bool((st["mask"] & ~opt).any())
            info = st["active_info"]
            if from_req and key[1] == 0 and info["first_turn"] and info["item"] not in CONSTRAINT_ITEMS and not (set(info["effects"]) & CONSTRAINT_EFFECTS) and info["n_moves"]:
                stat["제약 없는 상태(등장 직후 첫 턴, 비고정 도구, 도발·앵콜 등 없음)"] += 1
                pe = st.get("pp_empty", np.zeros(4, bool))
                if not (st["mask"][:4][:info["n_moves"]] | pe[:info["n_moves"]]).all():         # PP가 바닥난 기술은 정당하게 빠짐
                    stat["　그중 공개된 기술이 마스크에서 빠진 결정"] += 1
                    unmatched["[제약 없는 상태에서 기술 누락] " + ",".join(str(m) for m in st["moves"]) + " | 마스크 " + "".join("1" if x else "0" for x in st["mask"][:8])][0] += 1
            target, matched, action_idx = to_target(st, r["policy"], keep_ratio, r.get("choice"))
            if target is None or matched < min_matched:
                stat["결정 제외(행동 매칭 실패)"] += 1
                continue
            steps.append((st, target, -1 if action_idx is None else action_idx))
            tvals.append(float(r["value"]))
            stat["결정 사용"] += 1
        if len(steps) < 2:
            continue
        outcome = 1.0 if d["end"]["winner"] == user else -1.0
        T = len(steps)
        seq = {k: np.stack([s["obs"][i] for s, _, _ in steps]) for i, k in enumerate(OBS_KEYS)}
        seq["action_mask"] = np.stack([s["mask"] for s, _, _ in steps])
        seq["target"] = np.stack([t for _, t, _ in steps])
        seq["action_taken"] = np.array([a for _, _, a in steps], np.int64)  # 정책 그라디언트용, -1=실제 선택 매칭 실패
        seq["value_target"] = (GAMMA ** np.arange(T - 1, -1, -1, dtype=np.float32)) * outcome - np.array([s["phi"] for s, _, _ in steps], np.float32)
        seq["teacher_value"] = np.array(tvals, np.float32)
        seq["outcome"] = outcome
        out.append(seq)
        stat["배틀 사용"] += 1
    return out, stat, unmatched


def main(args):
    protocols = load_jsonl(args.protocols) if args.protocols else []
    seqs, stat, unmatched = build(load_jsonl(args.decisions), protocols, args.min_matched, args.keep_ratio, args.workers)
    if not seqs:
        raise SystemExit(f"❌ 만들어진 배틀이 없음: {dict(stat)}")
    data = {k: np.concatenate([s[k] for s in seqs]) for k in OBS_KEYS + ["action_mask", "target", "value_target", "teacher_value", "action_taken"]}
    data["lens"] = np.array([len(s["action_mask"]) for s in seqs])
    data["outcome"] = np.array([s["outcome"] for s in seqs], np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **data)
    ent = -(data["target"] * np.log(np.clip(data["target"], 1e-9, 1))).sum(-1)
    print(f"✅ {len(seqs)}판 / 결정 {int(data['lens'].sum())}개 → {args.out}")
    print(f"   Foul Play 승률(이 데이터) {np.mean(data['outcome'] > 0):.3f} | 교사 분포 평균 엔트로피 {ent.mean():.3f} | 최상위 행동 평균 확률 {data['target'].max(-1).mean():.3f}")
    print("   ", dict(stat))
    n_mask = (stat["마스크: 서버 요청 기반"] + stat["마스크: 규칙 대체(요청 못 찾음)"]) or 1
    n_free = stat["제약 없는 상태(등장 직후 첫 턴, 비고정 도구, 도발·앵콜 등 없음)"]
    n_prune = stat["요청상 합법인데 FP 옵션에 없는 행동이 있는 결정(FP 자체 가지치기)"]
    print(f"   마스크 점검: 서버 요청 기반 {stat['마스크: 서버 요청 기반']}개 ({stat['마스크: 서버 요청 기반'] / n_mask:.1%}), 규칙 대체 {stat['마스크: 규칙 대체(요청 못 찾음)']}개")
    print(f"   옵션(Foul Play) 대비: 22칸 매칭 실패가 있는 결정 {stat['옵션 중 22칸 매칭 실패가 있는 결정']}개 ({stat['옵션 중 22칸 매칭 실패가 있는 결정'] / n_mask:.2%}), "
          f"요청 합법 밖 옵션 {stat['FP 옵션이 요청 합법 밖 행동을 포함한 결정']}개, FP 자체 가지치기 {n_prune}개 ({n_prune / n_mask:.1%})")
    print(f"   제약 없는 상태 {n_free}개 중 공개된 기술이 마스크에서 빠진 결정 {stat['　그중 공개된 기술이 마스크에서 빠진 결정']}개 ({stat['　그중 공개된 기술이 마스크에서 빠진 결정'] / max(n_free, 1):.2%})")
    log = os.path.splitext(args.out)[0] + "_unmatched.txt"          # 22칸에 못 넣은 옵션과 제약 없는 상태의 기술 누락 사례 (횟수 내림차순)
    with open(log, "w", encoding="utf-8") as f:
        for n, (c, ex) in sorted(unmatched.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{c}"+chr(9)+f"{n}"+chr(9)+f"{ex}"+chr(10))
    print(f"   매칭 실패 옵션 {len(unmatched)}종 -> {log}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help="Foul Play 결정 JSONL glob")
    ap.add_argument("--protocols", default=None,
                    help="상대편(poke-env)이 남긴 프로토콜 JSONL glob. 자가대전(FoulPlay vs FoulPlay)처럼 상대편 기록이 없으면 생략 — "
                         "FoulPlay 자신이 남긴 protocol 레코드를 씀")
    ap.add_argument("--out", default="data/fp/fp_data.npz")
    ap.add_argument("--workers", type=int, default=8, help="재생 병렬 프로세스 수")
    ap.add_argument("--keep-ratio", type=float, default=0.2, help="탐색 분포에서 최고 확률의 이 비율 이상만 목표로. 낮을수록 원본 분포에 가까움(0=전체, Foul Play 자체 선택 규칙은 0.75지만 그대로 쓰면 알고리즘 절차를 모방하게 됨)")
    ap.add_argument("--min-matched", type=float, default=0.6, help="탐색 분포 중 우리 행동 칸에 매칭돼야 하는 최소 질량 비율")
    main(ap.parse_args())
