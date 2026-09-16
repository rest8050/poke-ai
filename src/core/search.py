"""
추론 시 1턴 탐색: 정책 상위 k개 행동을 (내 행동 × 상대 유력 기술) 보수 행렬로 다시 평가.

score(a) = λ·log π(a) + (1-λ)·[0.5·최악 + 0.5·평균]
- 상대 기술: 공개된 공격기 + 세트 사전 확률로 추론한 공격기 (최대 4개)
- 한 칸: 스피드/우선도로 선공 결정 → 기대 데미지(난수 0.5 고정) → 선공이 쓰러뜨리면 반격 없음
- 교체: 교체해 들어오는 포켓몬이 상대 기술을 맞음

PPO 학습 rollout에는 쓰지 말 것: 탐색이 행동을 바꾸면 log_prob이 정책과 어긋남. 평가/배포 전용.
ponytail: 1턴만 봄 (장판·랭크업·교체 이득 못 봄), 테라 행동은 테라 효과 없이 기술로만 계산. 다턴은 poke-engine MCTS로 교체.
"""
import math

from src.core.rct_player import _id, cur_hp, damage, field, stat
from src.core.set_prior import predict as predict_set
from src.core.tensor_encoder import _move_obj


class _Mid:
    """난수를 0.5로 고정 → 기대 데미지"""
    def random(self):
        return 0.5


def decode_action(battle, idx):
    """22칸 행동 -> ("move", Move, z, tera) | ("switch", Pokemon) | None (실행 불가)"""
    active = battle.active_pokemon
    if idx < 8 and active:
        moves = list(active.moves.values())
        m = idx % 4
        if m < len(moves) and moves[m].id in {x.id for x in battle.available_moves}:
            gimmick = idx >= 4
            z = gimmick and battle.can_z_move and moves[m].id in {x.id for x in active.available_z_moves}
            tera = gimmick and not z and bool(getattr(battle, "can_tera", False))
            return ("move", moves[m], z, tera)
    elif 8 <= idx <= 13:
        team = list(battle.team.values())
        k = idx - 8
        if k < len(team) and team[k] in battle.available_switches:
            return ("switch", team[k])
    return None


def _speed(battle, mon):
    b = mon.boosts.get("spe", 0)
    s = stat(mon, "spe") * (max(2, 2 + b) / max(2, 2 - b))
    return s * 0.5 if _id(mon.status) == "par" else s


def _hit(battle, atk, dfn, move):
    """상대 현재 HP 대비 기대 데미지 비율 (최대 1)"""
    if not move or not move.base_power:
        return 0.0
    return min(1.0, damage(battle, _Mid(), atk, dfn, move) / max(1.0, cur_hp(battle, dfn)))


def opponent_moves(battle, opp):
    known = [m for m in opp.moves.values() if m.base_power]
    item = None if opp.item == "unknown_item" else opp.item
    prior = predict_set(opp.species, list(opp.moves), item, opp.ability)
    guessed = [m for m in (_move_obj(mid) for mid in (prior or {}).get("moves", [])) if m is not None and m.base_power]
    return (known + guessed)[:4] or [None]


def cell(battle, me, opp, action, opp_move):
    """(준 피해 비율) - (받은 피해 비율) + 기절 보너스"""
    if action[0] == "switch":
        taken = _hit(battle, opp, action[1], opp_move)
        return -taken - (0.5 if taken >= 1 else 0.0)
    my_move = action[1]
    dealt, taken = _hit(battle, me, opp, my_move), _hit(battle, opp, me, opp_move)
    my_key = (my_move.priority, _speed(battle, me))
    opp_key = (opp_move.priority if opp_move else 0, _speed(battle, opp))
    if field(battle, "trickroom"):
        my_key, opp_key = (my_key[0], -my_key[1]), (opp_key[0], -opp_key[1])
    if my_key > opp_key and dealt >= 1:
        taken = 0.0
    elif my_key < opp_key and taken >= 1:
        dealt = 0.0
    return dealt - taken + (0.5 if dealt >= 1 else 0.0) - (0.5 if taken >= 1 else 0.0)


def search_pick(battle, probs, lam=0.5, k=3):
    """probs: [22] 정책 확률 (마스킹 반영). 고를 게 없으면 None"""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    cands = [(a, decode_action(battle, a)) for a in probs.topk(k).indices.tolist() if probs[a] > 0]
    cands = [(a, d) for a, d in cands if d is not None]
    if not cands or me is None or opp is None:
        return cands[0][0] if cands else None

    opp_moves = opponent_moves(battle, opp)
    values = {}
    for a, d in cands:
        if d[0] == "move" and not d[1].base_power:
            continue  # 변화기는 1턴 데미지로 평가 불가 → 아래에서 중립값
        cells = [cell(battle, me, opp, d, om) for om in opp_moves]
        values[a] = 0.5 * min(cells) + 0.5 * sum(cells) / len(cells)
    # ponytail: 변화기는 평가된 후보들의 평균(중립)을 줘서 정책 확률로만 갈리게 함
    neutral = sum(values.values()) / len(values) if values else 0.0
    return max(cands, key=lambda c: lam * math.log(probs[c[0]].item()) + (1 - lam) * values.get(c[0], neutral))[0]
