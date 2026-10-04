"""
Showdown 리플레이 원문 → (실전과 같은 인코더 입력, 사람 행동, 가치 목표) 시퀀스.

한 판을 한 플레이어 시점으로 poke-env Battle에 재생하고, 결정 지점마다 encode_battle()을 그대로 사용.
- 결정 지점: 매 턴 시작(기술/교체/테라), 기절 후 교체
- 라벨: 그 턴 그 플레이어의 첫 행동. 선택이 로그에 안 보이는 턴(잠듦/풀죽음/고정 기술 등)은 버림
- 내 팀: 관전자 로그라 요청(request)이 없음 → 같은 로그 전체에서 드러난 기술/도구/특성/테라를
  팀 프리뷰 시점에 미리 채움 (사후 정보. 실전의 "내 팀은 다 안다"에 가깝게)
- 가치 목표: PPO와 같은 의미 = γ^(남은 결정 수)·승패 - Φ(현재 상태)

ponytail 근사: 유턴/볼트체인지 뒤 교체 대상, 조로아크 환영, 끝까지 안 드러난 내 도구/테라 타입은 미반영.
"""
import logging
import re
from collections import defaultdict

import numpy as np
from poke_env.battle import Battle, Move, Pokemon, PokemonType
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

from src.core.battle_eval import potential
from src.core.tensor_encoder import BattleTensorEncoder

OBS_KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
_LOGGER = logging.getLogger("replay_dataset")
_LOGGER.setLevel(logging.CRITICAL)
logging.getLogger("poke-env").setLevel(logging.ERROR)  # 리플레이의 모르는 효과(Gooey 등) 경고 숨김
_ENCODER = None


def _norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _key(ident):
    """'p1a: Nick' → 'p1: Nick'"""
    return ident[:2] + ident[3:] if len(ident) > 3 and ident[2] != ":" else ident


def _hindsight(lines, side):
    """같은 로그 전체에서 내 포켓몬별로 드러난 정보: 종 → {moves, item, ability, tera}"""
    nick2sp = {}
    for s in lines:
        if s[1] in ("switch", "drag", "replace") and s[2].startswith(side):
            nick2sp[_key(s[2])] = _norm(s[3].split(",")[0])
    info = defaultdict(lambda: {"moves": [], "item": None, "ability": None, "tera": None})
    for s in lines:
        if len(s) < 4:
            continue
        owner = next((_key(x[5:]) for x in s[3:] if x.startswith("[of] ")), _key(s[2]))
        sp = nick2sp.get(owner)
        if sp is None or not owner.startswith(side):
            continue
        d = info[sp]
        if s[1] == "move" and _key(s[2]) == owner and not any(x.startswith("[from]") for x in s[4:]):
            mid = _norm(s[3])
            if mid not in d["moves"] and len(d["moves"]) < 4:
                d["moves"].append(mid)
        elif s[1] in ("-item", "-enditem") and d["item"] is None:
            d["item"] = _norm(s[3])
        elif s[1] == "-ability" and d["ability"] is None:
            d["ability"] = _norm(s[3])
        elif s[1] == "-terastallize":
            d["tera"] = _norm(s[3])
        for x in s[3:]:
            if x.startswith("[from] item: ") and d["item"] is None:
                d["item"] = _norm(x[len("[from] item: "):])
            elif x.startswith("[from] ability: ") and d["ability"] is None:
                d["ability"] = _norm(x[len("[from] ability: "):])
    return info


def _preview_mon(details, info):
    mon = Pokemon(details=details, gen=9)
    mon._max_hp = mon._current_hp = 100
    d = info.get(_norm(details.split(",")[0]))
    if d:
        for mid in d["moves"]:
            try:
                mon._moves[mid] = Move(mid, gen=9)
            except Exception:
                pass
        if d["item"]:
            mon._item = d["item"]
        if d["ability"]:
            mon.ability = d["ability"]
        if d["tera"]:
            try:
                mon._terastallized_type = PokemonType.from_name(d["tera"])
            except Exception:
                pass
        if d.get("nature") and d.get("evs"):  # BC 리플레이는 이 정보가 없어서(관전자 로그) 여기 안 탐 → 항상 종족값 그대로
            try:
                stats = compute_raw_stats(mon.species, d["evs"], d.get("ivs") or [31] * 6, d.get("level") or 100,
                                          d["nature"], GenData.from_gen(9))
                mon._stats = dict(zip(["hp", "atk", "def", "spa", "spd", "spe"], stats))
            except Exception:
                pass
    return mon


def _capture(battle, kind):
    """현재 상태를 인코딩하고, 요청 데이터 대신 규칙으로 합법 행동 마스크를 만듦"""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = BattleTensorEncoder()
    battle._can_tera = kind == "turn" and not battle.used_tera
    tensors = _ENCODER.encode_battle(battle)
    active = battle.active_pokemon
    team = list(battle.team.values())[:6]
    mask = np.zeros(22, dtype=bool)
    moves = list(active.moves)[:4] if (kind == "turn" and active) else []
    for i in range(len(moves)):
        mask[i] = True
        mask[4 + i] = battle._can_tera
    for k, mon in enumerate(team):
        if not mon.fainted and mon is not active:
            mask[8 + k] = True
    return {
        "obs": [t[0].numpy() for t in tensors[:7]], "mask": mask, "phi": potential(battle),
        "moves": moves, "team_species": [mon.species for mon in team], "tera": False,
    }


def _label_switch(state, details):
    sp = _norm(details.split(",")[0])
    for k, team_sp in enumerate(state["team_species"]):
        if _norm(team_sp) == sp:
            return 8 + k
    return None


def replay_to_sequence(log_text, side, gamma=0.995):
    """
    return: dict(obs 7개 [T, ...], action_mask [T,22], action [T], value_target [T]) 또는 None
    side: "p1" / "p2"
    """
    lines = [l.split("|") for l in log_text.split("\n") if l.startswith("|") and l != "|"]
    lines = [s for s in lines if len(s) > 1]
    name = next((s[3] for s in lines if s[1] == "player" and len(s) > 3 and s[2] == side), None)
    winner = next((s[2] for s in lines if s[1] == "win" and len(s) > 2), None)
    if not name or winner is None:
        return None
    outcome = 1.0 if winner == name else -1.0
    info = _hindsight(lines, side)

    battle = Battle(f"replay-{side}", name, _LOGGER, gen=9)
    steps, pending, forced = [], None, False

    def resolve(state, action):
        if action is not None and state["mask"][action]:
            steps.append((state, action))

    for s in lines:
        ev = s[1]
        own = len(s) > 2 and s[2].startswith(side) and len(s[2]) > 2 and s[2][2] in "abc"

        # 1) 이 줄을 반영하기 전에 처리할 것 (라벨 결정, 기절 후 교체 시점 상태)
        if own and ev == "-terastallize" and pending:
            pending["tera"] = True
        elif own and ev == "move" and pending:
            if not any(x.startswith("[from]") for x in s[4:]):
                mid = _norm(s[3])
                if mid in pending["moves"]:
                    resolve(pending, pending["moves"].index(mid) + (4 if pending["tera"] else 0))
            pending = None
        elif own and ev == "switch":
            if forced:
                state = _capture(battle, "forced")
                resolve(state, _label_switch(state, s[3]))
                forced = False
            elif pending:
                resolve(pending, _label_switch(pending, s[3]))
                pending = None
        elif own and ev == "cant":
            pending = None
        elif own and ev == "faint":
            pending, forced = None, True

        # 2) 배틀 상태 갱신
        if ev == "poke" and len(s) > 3:
            if s[2] == side:
                mon = _preview_mon(s[3], info)
                battle._team[f"{side}: {s[3].split(',')[0]}"] = mon
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
            return None  # 재생 불가능한 로그는 통째로 버림

        # 3) 턴 시작 = 새 결정 지점
        if ev == "turn":
            pending, forced = _capture(battle, "turn"), False

    if not steps:
        return None
    T = len(steps)
    phi = np.array([st["phi"] for st, _ in steps], dtype=np.float32)
    out = {k: np.stack([st["obs"][i] for st, _ in steps]) for i, k in enumerate(OBS_KEYS)}
    out["action_mask"] = np.stack([st["mask"] for st, _ in steps])
    out["action"] = np.array([a for _, a in steps], dtype=np.int64)
    out["value_target"] = (gamma ** np.arange(T - 1, -1, -1, dtype=np.float32)) * outcome - phi
    out["outcome"] = outcome
    return out


if __name__ == "__main__":
    import os
    import sys
    import time
    from collections import Counter

    import pyarrow.parquet as pq

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rows = pq.read_table("data/metamon/raw/train-00000-of-00046.parquet", columns=["log"]).slice(0, n).to_pylist()
    t0, seqs, failed, kinds, turns = time.time(), 0, 0, Counter(), 0
    for r in rows:
        n_turns = r["log"].count("\n|turn|")
        for side in ("p1", "p2"):
            out = replay_to_sequence(r["log"], side)
            if out is None:
                failed += 1
                continue
            seqs += 1
            turns += n_turns
            T = len(out["action"])
            assert all(out[k].shape[0] == T for k in OBS_KEYS + ["action_mask", "value_target"])
            assert out["action_mask"][np.arange(T), out["action"]].all(), "라벨이 마스크 밖"
            assert out["my_move_num"].shape[1:] == (6, 4, 46) and out["my_team_cat"].shape[1:] == (6, 10)
            assert abs(out["value_target"][-1] - (out["outcome"] - 0)) < 2.0
            for a in out["action"]:
                kinds["tera" if 4 <= a < 8 else "move" if a < 4 else "switch"] += 1
    dt = time.time() - t0
    total = sum(kinds.values())
    print(f"리플레이 {n}판 × 2시점: 성공 {seqs}, 실패 {failed} | 결정 {total}개 (로그 턴 대비 {total / max(1, turns):.0%})"
          f" | 기술 {kinds['move']}, 교체 {kinds['switch']}, 테라 {kinds['tera']} | {dt / max(1, seqs):.2f}초/시점")
