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
  (--opp-labels: 상대 의도 라벨 opp_sw/opp_tera [N], opp_mv_id/opp_mv_p [N,5], opp_tgt [N,6] — 상대 쪽 탐색 분포에서, 없으면 opp_sw=NaN)

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

from src.training.replay_dataset import OBS_KEYS, _LOGGER, _capture, _norm, _preview_mon

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
        return next((8 + k for k, s in enumerate(team) if s == sp), None)
    tera = choice.endswith("-tera")
    mid = _norm(choice.removesuffix("-tera").removesuffix("-mega"))
    return next((i + (4 if tera else 0) for i, m in enumerate(moves) if m == mid), None)


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


OPP_TOP = 5   # 상대 기술 분포로 남기는 상위 기술 수 (나머지 질량은 '기타')
_VOCAB = json.load(open("data/vocab.json", encoding="utf-8"))
_SPECIES_NAME = {i: n for n, i in _VOCAB["species"].items()}


def opp_labels(policy, keep_ratio, opp_team_cat):
    """상대 쪽 탐색 분포 → 상대 의도 라벨 (모두 전체 질량 대비 비율, 매칭 안 된 질량은 버림 — 손실이 다시 정규화).
    policy: 상대 컨테이너가 남긴 [(선택, 가중치)], opp_team_cat: 내 관측의 상대 6칸 [6, CAT] (종 ID는 10번 열, 프리뷰로 6마리 모두 채워짐)
    반환: (교체 질량, 테라 질량, 기술 ID [OPP_TOP], 기술 질량 [OPP_TOP], 교체 대상 슬롯 질량 [6])"""
    best = max(w for _, w in policy)
    policy = [(c, w) for c, w in policy if w >= keep_ratio * best]
    total = sum(w for _, w in policy) or 1.0
    slots = [_SPECIES_NAME.get(int(i), "") for i in opp_team_cat[:, 10]]
    sw, tera, tgt, mv = 0.0, 0.0, np.zeros(6, np.float32), defaultdict(float)
    for c, w in policy:
        w /= total
        if c.startswith("switch "):
            sw += w
            sp = _norm(c[len("switch "):])
            k = next((k for k, s in enumerate(slots) if s and s == sp), None)
            if k is None:   # 폼 차이(프리뷰 'sinistcha' vs 실제 'sinistchamasterpiece')는 접두사로 매칭 (tensor_encoder와 같은 규칙)
                k = next((k for k, s in enumerate(slots) if s and (s.startswith(sp) or sp.startswith(s))), None)
            if k is not None:
                tgt[k] += w
        else:
            tera += w if c.endswith("-tera") else 0.0
            mid = _VOCAB["move"].get(_norm(c.removesuffix("-tera").removesuffix("-mega")))
            if mid:
                mv[mid] += w
    top = sorted(mv.items(), key=lambda kv: -kv[1])[:OPP_TOP]
    ids = np.array([i for i, _ in top] + [0] * (OPP_TOP - len(top)), np.int32)
    ps = np.array([p for _, p in top] + [0.0] * (OPP_TOP - len(top)), np.float32)
    return np.float32(sw), np.float32(tera), ids, ps, tgt


def replay_states(protocol, side, name, team):
    """Foul Play 시점으로 재생해 결정 지점의 상태들 [(상태, 게임 턴, 종류 0=턴 1=기절 후 교체)]"""
    lines = [l.split("|") for l in protocol if l.startswith("|") and l != "|"]
    lines = [s for s in lines if len(s) > 1]
    info = team_info(team)
    battle = Battle(f"fp-{side}", name, _LOGGER, gen=9)
    states, forced = [], False
    for s in lines:
        ev = s[1]
        own = len(s) > 2 and s[2].startswith(side) and len(s[2]) > 2 and s[2][2] in "abc"
        if own and ev == "switch" and forced:
            st = _capture(battle, "forced")
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
    return states


def _states(task):
    return replay_states(*task)


def build(decisions, protocols, min_matched, keep_ratio, workers=1, opp=False):
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

    peers = defaultdict(list)   # 배틀 태그 -> 그 배틀의 기록 파일들 (자가대전은 양쪽 2개)
    for src, tag in by_tag:
        peers[tag].append(src)
    stat = Counter()
    out, jobs = [], []
    for (_src, tag), d in by_tag.items():
        protocol = prot_by_tag.get(tag)
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
        steps, tvals, olabs = [], [], []
        opp_dec = {}   # 반대편의 같은 턴 결정 (자가대전 한정): (턴, 강제교체) -> 기록
        if opp:
            others = [s for s in peers[d["start"]["tag"]] if s != d["start"]["_src"]]
            if len(others) == 1:
                for o in by_tag[(others[0], d["start"]["tag"])]["dec"]:
                    opp_dec.setdefault((o["turn"], int(o["force_switch"])), o)
        for r in d["dec"]:
            key = (r["turn"], int(r["force_switch"]))
            if not queues[key]:
                stat["결정 제외(상태 못 찾음)"] += 1
                continue
            st = queues[key].pop(0)
            target, matched, action_idx = to_target(st, r["policy"], keep_ratio, r.get("choice"))
            if target is None or matched < min_matched:
                stat["결정 제외(행동 매칭 실패)"] += 1
                continue
            steps.append((st, target, -1 if action_idx is None else action_idx))
            tvals.append(float(r["value"]))
            o = opp_dec.get(key) if key[1] == 0 else None      # 내가 강제 교체일 때는 상대 결정이 없음
            olabs.append(opp_labels(o["policy"], keep_ratio, st["obs"][3]) if o else (np.float32("nan"), np.float32(0), np.zeros(OPP_TOP, np.int32), np.zeros(OPP_TOP, np.float32), np.zeros(6, np.float32)))
            stat["상대 라벨 있음" if o else "상대 라벨 없음"] += 1
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
        if opp:
            seq["opp_sw"], seq["opp_tera"] = (np.array([l[i] for l in olabs], np.float32) for i in (0, 1))
            seq["opp_mv_id"], seq["opp_mv_p"], seq["opp_tgt"] = (np.stack([l[i] for l in olabs]) for i in (2, 3, 4))
        seq["outcome"] = outcome
        out.append(seq)
        stat["배틀 사용"] += 1
    return out, stat


def main(args):
    protocols = load_jsonl(args.protocols) if args.protocols else []
    seqs, stat = build(load_jsonl(args.decisions), protocols, args.min_matched, args.keep_ratio, args.workers, args.opp_labels)
    if not seqs:
        raise SystemExit(f"❌ 만들어진 배틀이 없음: {dict(stat)}")
    opp_keys = ["opp_sw", "opp_tera", "opp_mv_id", "opp_mv_p", "opp_tgt"] if args.opp_labels else []
    data = {k: np.concatenate([s[k] for s in seqs]) for k in OBS_KEYS + ["action_mask", "target", "value_target", "teacher_value", "action_taken"] + opp_keys}
    data["lens"] = np.array([len(s["action_mask"]) for s in seqs])
    data["outcome"] = np.array([s["outcome"] for s in seqs], np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **data)
    ent = -(data["target"] * np.log(np.clip(data["target"], 1e-9, 1))).sum(-1)
    print(f"✅ {len(seqs)}판 / 결정 {int(data['lens'].sum())}개 → {args.out}")
    print(f"   Foul Play 승률(이 데이터) {np.mean(data['outcome'] > 0):.3f} | 교사 분포 평균 엔트로피 {ent.mean():.3f} | 최상위 행동 평균 확률 {data['target'].max(-1).mean():.3f}")
    print("   ", dict(stat))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help="Foul Play 결정 JSONL glob")
    ap.add_argument("--protocols", default=None,
                    help="상대편(poke-env)이 남긴 프로토콜 JSONL glob. 자가대전(FoulPlay vs FoulPlay)처럼 상대편 기록이 없으면 생략 — "
                         "FoulPlay 자신이 남긴 protocol 레코드를 씀")
    ap.add_argument("--out", default="data/fp/fp_data.npz")
    ap.add_argument("--opp-labels", action="store_true",
                    help="자가대전 전용: 같은 배틀 반대편 기록의 탐색 분포로 상대 의도 라벨(opp_sw/opp_tera/opp_mv_*/opp_tgt)을 함께 저장 (DAgger는 쓰지 말 것 — 반대편 policy가 실제 행동 분포가 아님)")
    ap.add_argument("--workers", type=int, default=8, help="재생 병렬 프로세스 수")
    ap.add_argument("--keep-ratio", type=float, default=0.2, help="탐색 분포에서 최고 확률의 이 비율 이상만 목표로. 낮을수록 원본 분포에 가까움(0=전체, Foul Play 자체 선택 규칙은 0.75지만 그대로 쓰면 알고리즘 절차를 모방하게 됨)")
    ap.add_argument("--min-matched", type=float, default=0.6, help="탐색 분포 중 우리 행동 칸에 매칭돼야 하는 최소 질량 비율")
    main(ap.parse_args())
