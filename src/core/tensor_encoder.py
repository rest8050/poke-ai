import torch
import numpy as np
import json
import os
import re
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Optional

from src.core.events import SLOT_DIM, TURN_DIM


def effective_base_stats(pkmn, base_stats):
    """내 포켓몬의 종족값 대신 실제 노력치/성격이 반영된 능력치(아는 경우만). 상대는 항상 모르므로 호출 안 함.
    poke-env가 실전에선 서버 request의 진짜 수치를, 우리가 만든 재생 데이터(FoulPlay 자기 팀)는 compute_raw_stats로 채움 (전부 없으면 종족값 그대로)"""
    stats = getattr(pkmn, "stats", None) or {}
    if all(stats.get(k) is not None for k in ("hp", "atk", "def", "spa", "spd", "spe")):
        return stats
    return base_stats



def extract_enum_str(val: Any) -> str:
    """
    poke-env의 Enum 객체(PokemonType.FIRE, Status.BRN, Weather.SUNNYDAY 등),
    문자열 ("FIRE (pokemon type) object"), 또는 래퍼 객체에서 안전하게 순수 속성 명칭 추출
    """
    if val is None:
        return ""
    
    # 1. Python Enum 또는 name 속성을 가진 객체 (단, str.name 제외)
    if hasattr(val, "name") and isinstance(getattr(val, "name"), str):
        val_name = getattr(val, "name")
        if val_name and "(" not in val_name and "." not in val_name and " " not in val_name:
            return val_name.strip().lower()

    # 2. value 속성이 문자열일 때
    if hasattr(val, "value") and isinstance(getattr(val, "value"), str):
        val_val = getattr(val, "value")
        if val_val and "(" not in val_val and "." not in val_val and " " not in val_val:
            return val_val.strip().lower()

    # 3. str(val)로 변환된 긴 문자열 정제 ("FIRE (pokemon type) object", "PokemonType.FIRE", "FIRE")
    s = str(val).strip()
    if not s:
        return ""

    if "(" in s:
        s = s.split("(")[0]
    if "." in s:
        s = s.split(".")[-1]
    if ":" in s:
        s = s.split(":")[-1]
    if " " in s:
        s = s.split()[0]

    return s.strip().lower()


class VocabManager:
    """
    결정론적 vocab.json 기반 문자열 -> 고유 ID 변환 매니저
    - 0: 빈 슬롯 (Strict Padding)
    - <unk>: 사전에 등록되지 않은 처음 본 개체 (novel)
    - <unk>+1: 아직 안 드러난 값 (hidden) — 도구/특성/기술에만 사용
    """
    _instance = None

    def __init__(self, vocab_path: str = "data/vocab.json"):
        self.vocab = {}
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
            except Exception as e:
                print(f"⚠️ vocab.json 로드 실패: {e}")

    def unk_id(self, kind: str) -> int:
        kd = self.vocab.get(kind, {})
        return kd.get("<unk>", len(kd) + 1)

    def slot_id(self, kind: str, name: Any, hidden: bool = False) -> int:
        """도구/특성/기술 ID. 0 = 빈 자리, <unk> = 어휘에 없는 처음 본 개체, unk+1 = 아직 안 드러난 값(hidden)"""
        return self.unk_id(kind) + 1 if hidden else self.get_id(kind, name)

    @classmethod
    def get_instance(cls, vocab_path: str = "data/vocab.json"):
        if cls._instance is None:
            cls._instance = VocabManager(vocab_path)
        return cls._instance

    def get_id(self, kind: str, name: Any) -> int:
        """
        이름/Enum을 파싱하여 ID를 반환.
        - 파라미터가 없거나 빈 경우: 0 (오직 빈 슬롯/패딩)
        - 사전에 없는 경우: <unk> ID (미지 토큰, 결코 0이 아님!)
        """
        clean_str = extract_enum_str(name)
        if not clean_str:
            return 0 # 0은 오직 진짜 빈 슬롯/패딩에만 할당!
            
        norm_name = re.sub(r'[^a-z0-9]', '', clean_str)
        if not norm_name:
            return 0
            
        kind_dict = self.vocab.get(kind, {})
        if norm_name in kind_dict:
            return kind_dict[norm_name]
            
        # 사전에 없는 미지의 포켓몬/기술일 경우 <unk> ID 할당 (Fix 2: 0으로 지워지는 버그 방지)
        return kind_dict.get("<unk>", len(kind_dict) + 1)


def get_dynamic_turns(val: Any, effect_name: str, current_turn: int, battle: Any = None) -> Tuple[float, float]:
    """
    날씨/필드/벽의 지속 턴을 동적으로 계산:
    1. 날씨 바위 도구(Heat Rock, Damp Rock, Smooth Rock, Icy Rock), 빛의점토(Light Clay), 지형연장(Terrain Extender) 파악
    2. 무한 날씨(시작의바다, 단구의흔적, 델타스트림) 동적 1.0 처리
    3. val이 '남은 턴 수' (turns_left) 인지 '시작된 턴 번호' (start_turn) 인지 자동 판별
    """
    if val is None or val is False or val == 0:
        return 0.0, 0.0

    eff_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(effect_name))

    # 1. 무한 날씨 특수 처리 (Desolate Land, Primordial Sea, Delta Stream 등)
    if any(k in eff_str for k in ["desolateland", "primordialsea", "deltastream", "harshsunshine", "heavyrain"]):
        return 1.0, 1.0

    # 2. 보유 도구 동적 수집 (내 팀 / 상대 팀 / 활성화 포켓몬)
    held_items = set()
    if battle:
        teams = []
        if hasattr(battle, "team") and battle.team:
            teams.extend(battle.team.values())
        if hasattr(battle, "opponent_team") and battle.opponent_team:
            teams.extend(battle.opponent_team.values())
        for pkmn in teams:
            it = extract_enum_str(getattr(pkmn, "item", None))
            if it:
                held_items.add(re.sub(r'[^a-z0-9]', '', it))

    # 3. 효과별 동적 최대 지속 턴 (Max Duration) 결정
    if any(k in eff_str for k in ["sun", "sunny", "rain", "sand", "hail", "snow"]):
        if "heatrock" in held_items and "sun" in eff_str:
            max_duration = 8.0
        elif "damprock" in held_items and "rain" in eff_str:
            max_duration = 8.0
        elif "smoothrock" in held_items and "sand" in eff_str:
            max_duration = 8.0
        elif "icyrock" in held_items and ("hail" in eff_str or "snow" in eff_str):
            max_duration = 8.0
        else:
            max_duration = 5.0
    elif any(k in eff_str for k in ["reflect", "lightscreen", "auroraveil"]):
        max_duration = 8.0 if "lightclay" in held_items else 5.0
    elif "tailwind" in eff_str:
        max_duration = 4.0
    elif "trickroom" in eff_str:
        max_duration = 5.0
    elif "terrain" in eff_str:
        max_duration = 8.0 if "terrainextender" in held_items else 5.0
    elif "safeguard" in eff_str:
        max_duration = 5.0
    else:
        max_duration = 5.0

    if val is True:
        return 1.0, 1.0

    if isinstance(val, (int, float)):
        v = float(val)
        
        # 4. val 형태 동적 판별 (남은 턴 수 vs 시작 턴 번호). poke-env는 보통 시작 턴 번호를 주므로
        # Case B(시작 턴)를 먼저 본다 — 안 그러면 "시작 턴 번호가 우연히 최대 지속시간과 같은" 경계에서
        # (예: 훈열의돌로 8턴 지속인데 정확히 턴8에 시작) 남은 턴 수로 오인식하는 버그가 생김.
        # Case B: val이 시작된 턴 번호 (v <= current_turn 이고 (current_turn - v) < max_duration 인 경우)
        if 0 <= v <= current_turn and (current_turn - v) < max_duration:
            elapsed = current_turn - v
            turns_left_cnt = max(0.0, max_duration - elapsed)
        # Case A: val이 이미 남은 턴 수 (0 < v <= max_duration 이고 current_turn > max_duration 인 경우)
        elif 0 < v <= max_duration and current_turn > max_duration:
            turns_left_cnt = v
        # Case C: 기타 지정 수치
        else:
            turns_left_cnt = max(0.0, min(v, max_duration))

        turns_norm = min(1.0, max(0.0, turns_left_cnt / max_duration))
        act = 1.0 if turns_norm > 0 else 0.0
        return act, turns_norm

    return 1.0, 1.0


def layers(val: Any, max_layers: float) -> float:
    """압정/독압정: poke-env 값은 겹수"""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return min(1.0, float(val) / max_layers)
    return 1.0 / max_layers


# 기술 부가효과 벡터 (기대값 = 확률 × 효과량, 대략 -1~1)
MOVE_EFFECT_NAMES = [
    "drain", "recoil", "extra_hits", "crit_stage", "fixed_damage",                       # 0-4 데미지 보정
    "tgt_brn", "tgt_par", "tgt_psn", "tgt_slp", "tgt_frz", "tgt_flinch", "tgt_confusion",   # 5-11 상대 상태이상
    "tgt_offense", "tgt_defense", "tgt_speed",                                               # 12-14 상대 랭크
    "self_offense", "self_defense", "self_speed",                                            # 15-17 자기 랭크
    "heal", "self_switch", "force_switch",                                                   # 18-20 회복/템포
    "hazard_set", "hazard_removal", "screen", "trick_room", "weather_terrain_tailwind",      # 21-25 필드
    "protect", "disrupt", "residual_on_target", "substitute", "item_manipulation",           # 26-30 방해/유틸
    "alt_offense_stat", "ignore_target_boosts", "self_sacrifice", "charge_or_recharge", "contact",  # 31-35
    "priority_conditional",                                                                          # 36 (선공권이 상대의 이번 턴 선택에 달림 — 아래 설명)
]
MOVE_EFFECT_DIM = len(MOVE_EFFECT_NAMES)
MOVE_NUM_DIM = 9 + MOVE_EFFECT_DIM  # 기본 수치 7 + 상대 액티브 배율 1 + 부가효과 + 선공권 우열(끝에 추가)

# Showdown 데이터에 필드가 없고 코드로만 구현된 효과 (poke-engine choice_effects.rs 와 같은 방식으로 목록 보완)
_HAZARD_SET_EXTRA = {"ceaselessedge", "stoneaxe"}
_HAZARD_REMOVAL = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
_ITEM_MANIPULATION = {"knockoff", "trick", "switcheroo", "thief", "covet", "bugbite", "pluck", "incinerate", "corrosivegas"}
_DISRUPT = {"taunt", "encore", "disable", "torment", "healblock", "imprison", "throatchop"}
# 선공권이 고정이 아니라 "이번 턴 상대가 뭘 냈는지"에 달린 기술. poke-env 정적 데이터의 priority 필드는
# 명목값(예: thunderclap=1)만 있고 이 조건을 반영 안 해서, 실제로는 조건이 안 맞으면 선공이 아니게 됨
# (thunderclap: 상대가 공격기가 아니면 우선도 0으로 처리됨 / suckerpunch, upperhand: 조건 불충족 시 기술 자체가 실패).
# 명목 priority 피처(4번)는 그대로 두고, "이 값은 상대 선택에 달려 있어 불확실하다"는 플래그만 따로 추가함
# (날씨볼처럼 관측 가능한 상태에서 정답을 계산해 넣는 것과 달리, 상대의 동시 선택은 관측 불가능해 정답을 못 만듦 —
#  대신 모델이 이미 갖고 있는 상대 행동 예측(opp_action_head)과 엮어 스스로 위험도를 배우게 유도하는 용도).
_CONDITIONAL_PRIORITY = {"thunderclap", "suckerpunch", "upperhand"}
_RESIDUAL = {"leechseed", "saltcure", "partiallytrapped", "nightmare"}
_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
_SCREENS = {"reflect", "lightscreen", "auroraveil"}
_SELF_TARGETS = {"self", "adjacentallyorself", "allies", "allyside", "allyteam"}
# 회복량이 날씨/상대 능력치에 따라 달라져 데이터 필드가 비어 있는 회복기 (기본값으로 근사)
_HEAL_EXTRA = {"rest": 1.0, "moonlight": 0.5, "synthesis": 0.5, "morningsun": 0.5, "shoreup": 0.5,
               "strengthsap": 0.5, "painsplit": 0.5, "junglehealing": 0.25, "lunarblessing": 0.25}
# 코드로 구현된 자기 랭크 변화 (저주는 고스트 타입이 아닐 때 기준)
_SELF_BOOST_EXTRA = {"curse": {"atk": 1, "def": 1, "spe": -1}, "bellydrum": {"atk": 6},
                     "takeheart": {"spa": 1, "spd": 1}, "stockpile": {"def": 1, "spd": 1}}
# ponytail: 흑안개, 길동무, 치유방울, 멸망의노래, 검은눈빛/블록(교체 봉쇄), 변신, 잠꼬대는 벡터에 표현 안 됨 → ID 임베딩에 의존
_STATUS_SLOT = {"brn": 5, "par": 6, "psn": 7, "tox": 7, "slp": 8, "frz": 9}


def _norm_id(x: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", extract_enum_str(x))


def _add_boosts(vec: List[float], boosts: Optional[Dict[str, int]], chance: float, base: int) -> None:
    """랭크 변화 기대값을 공격계/방어계/스피드 3칸에 누적 (2랭크 = 1.0)"""
    for stat, stage in (boosts or {}).items():
        slot = {"atk": 0, "spa": 0, "def": 1, "spd": 1, "spe": 2}.get(stat)
        if slot is not None:
            vec[base + slot] += stage * chance / 2.0


@lru_cache(maxsize=4096)
def move_effect_features(move_id: str) -> Tuple[float, ...]:
    """기술 id → 부가효과 벡터 (MOVE_EFFECT_NAMES 순서). 모르는 기술은 0벡터"""
    vec = [0.0] * MOVE_EFFECT_DIM
    move = _move_obj(move_id)
    if move is None:
        return tuple(vec)
    mid = move.id
    self_target = _norm_id(getattr(move, "target", "")) in _SELF_TARGETS

    vec[0] = float(getattr(move, "drain", 0) or 0)
    vec[1] = float(getattr(move, "recoil", 0) or 0)
    vec[2] = min(1.0, max(0.0, float(getattr(move, "expected_hits", 1) or 1) - 1) / 4.0)  # 추가 타격 수 (단타=0, 5타=1)
    vec[3] = min(1.0, max(0, (getattr(move, "crit_ratio", 0) or 0) - 1) / 2.0)
    vec[4] = 1.0 if getattr(move, "damage", None) else 0.0

    # 기술 자체 효과 (변화기: 도깨비불, 칼춤, 차밍보이스 등)
    status = _norm_id(getattr(move, "status", None))
    if status in _STATUS_SLOT and not self_target:
        vec[_STATUS_SLOT[status]] += 1.0
    _add_boosts(vec, getattr(move, "boosts", None), 1.0, 15 if self_target else 12)
    _add_boosts(vec, getattr(move, "self_boost", None), 1.0, 15)
    _add_boosts(vec, _SELF_BOOST_EXTRA.get(mid), 1.0, 15)
    volatile = _norm_id(getattr(move, "volatile_status", None))
    if volatile == "yawn":
        vec[8] += 1.0  # ponytail: 하품은 다음 턴 수면이라 수면 확률 1로 근사
    if volatile == "confusion" and not self_target:
        vec[11] += 1.0

    # 확률 부가효과 (열탕 화상 30%, 문포스 특공 -1 30%, 고속스핀 자기 스피드 +1 등)
    secondary = getattr(move, "secondary", None) or []
    for sec in secondary:
        chance = (sec.get("chance") or 100) / 100.0
        sec_status = _norm_id(sec.get("status"))
        if sec_status in _STATUS_SLOT:
            vec[_STATUS_SLOT[sec_status]] += chance
        sec_volatile = _norm_id(sec.get("volatileStatus"))
        if sec_volatile == "flinch":
            vec[10] += chance
        elif sec_volatile == "confusion":
            vec[11] += chance
        _add_boosts(vec, sec.get("boosts"), chance, 12)
        _add_boosts(vec, (sec.get("self") or {}).get("boosts"), chance, 15)

    heal = float(getattr(move, "heal", 0) or 0)
    if _norm_id(getattr(move, "slot_condition", None)) == "wish":
        heal = 0.5
    vec[18] = _HEAL_EXTRA.get(mid, heal)
    vec[19] = 1.0 if getattr(move, "self_switch", None) else 0.0
    vec[20] = 1.0 if getattr(move, "force_switch", False) else 0.0

    side = _norm_id(getattr(move, "side_condition", None))
    vec[21] = 1.0 if side in _HAZARDS or mid in _HAZARD_SET_EXTRA else 0.0
    vec[22] = 1.0 if mid in _HAZARD_REMOVAL else 0.0
    vec[23] = 1.0 if side in _SCREENS else 0.0
    pseudo = _norm_id(getattr(move, "pseudo_weather", None))
    vec[24] = 1.0 if pseudo == "trickroom" else 0.0
    vec[25] = 1.0 if (getattr(move, "weather", None) or getattr(move, "terrain", None)
                      or side == "tailwind" or (pseudo and pseudo != "trickroom")) else 0.0

    vec[26] = 1.0 if getattr(move, "is_protect_move", False) or getattr(move, "stalling_move", False) else 0.0
    secondary_volatiles = {_norm_id(s.get("volatileStatus")) for s in secondary}
    vec[27] = 1.0 if (volatile in _DISRUPT or secondary_volatiles & _DISRUPT) and not self_target else 0.0
    vec[28] = 1.0 if (volatile in _RESIDUAL or secondary_volatiles & _RESIDUAL) and not self_target else 0.0
    vec[29] = 1.0 if volatile == "substitute" else 0.0
    vec[30] = 1.0 if mid in _ITEM_MANIPULATION else 0.0

    entry = getattr(move, "entry", {}) or {}
    vec[31] = 1.0 if entry.get("overrideOffensiveStat") or getattr(move, "use_target_offensive", False) else 0.0
    vec[32] = 1.0 if getattr(move, "ignore_defensive", False) else 0.0
    vec[33] = 1.0 if getattr(move, "self_destruct", None) else 0.0
    flags = getattr(move, "flags", None) or set()
    vec[34] = 1.0 if "charge" in flags or "recharge" in flags else 0.0
    vec[35] = 1.0 if "contact" in flags else 0.0
    vec[36] = 1.0 if mid in _CONDITIONAL_PRIORITY else 0.0

    for i in range(5, 12):
        vec[i] = min(1.0, vec[i])
    for i in range(12, 18):
        vec[i] = max(-1.0, min(1.0, vec[i]))
    return tuple(vec)


def defender_ability(mon: Any) -> Optional[str]:
    """상성 판단용 방어측 특성: 공개됐거나 후보가 1개일 때만. 미공개면 ""(타입 상성만) — 이름 기반 세트 추정은 쓰지 않음"""
    from src.core.rct_player import ability as known_ability
    return known_ability(mon) or ""


# 날씨에 따라 타입이 바뀌는 기술(웨더볼) — poke-env는 이런 걸 자동으로 재계산해주지 않아서
# move.type을 그대로 읽으면 항상 고정(노말) 취급됨. 실전 버그로 확인됨(모래바람 중 웨더볼을
# 여전히 물타입인 것처럼 판단 → 하마돈 상대 반감기를 씀). 여기서 실제 타입으로 바로잡음.
_WEATHER_MOVE_TYPE = {
    "sunnyday": "fire", "desolateland": "fire",
    "raindance": "water", "primordialsea": "water",
    "sandstorm": "rock",
    "hail": "ice", "snow": "ice", "snowscape": "ice",
}


def _current_weather(battle: Any) -> str:
    return next((re.sub(r"[^a-z0-9]", "", str(k).lower()) for k in (getattr(battle, "weather", None) or {})), "")


def effective_move_type(battle: Any, move: Any) -> str:
    """move.type을 그대로 쓰되, 날씨의존 기술(지금은 웨더볼)만 현재 날씨 기준 실제 타입으로 교체."""
    move_type = extract_enum_str(getattr(move, "type", None))
    if getattr(move, "id", None) != "weatherball":
        return move_type
    return _WEATHER_MOVE_TYPE.get(_current_weather(battle), "normal")


_RAIN_WEATHER = {"raindance", "primordialsea"}
_SUN_WEATHER = {"sunnyday", "desolateland"}
_WEATHER_ACC_BOOST_MOVES = {"hurricane", "thunder"}  # 비=필중, 쾌청=명중 50%


def effective_move_power(battle: Any, move: Any, base_power: float) -> float:
    """물/불타입 기술의 비/쾌청 보정(1.5배/0.5배)과 웨더볼의 날씨 중 위력 2배를 반영.
    poke-env는 move.base_power를 고정값으로만 주고 날씨 보정을 안 해줌."""
    if not base_power or battle is None:
        return base_power
    w = _current_weather(battle)
    if not w:
        return base_power
    if getattr(move, "id", None) == "weatherball":
        return base_power * 2.0
    mtype = extract_enum_str(getattr(move, "type", None))
    if w in _RAIN_WEATHER:
        if mtype == "water":
            return base_power * 1.5
        if mtype == "fire":
            return base_power * 0.5
    elif w in _SUN_WEATHER:
        if mtype == "fire":
            return base_power * 1.5
        if mtype == "water":
            return base_power * 0.5
    return base_power


def effective_move_accuracy(battle: Any, move: Any, accuracy: float) -> float:
    """허리케인/천둥: 비=필중(1.0), 쾌청=명중 50%(0.5). poke-env는 고정 명중률(0.7)만 줌."""
    if battle is None or getattr(move, "id", None) not in _WEATHER_ACC_BOOST_MOVES:
        return accuracy
    w = _current_weather(battle)
    if w in _RAIN_WEATHER:
        return 1.0
    if w in _SUN_WEATHER:
        return 0.5
    return accuracy


def move_effectiveness(battle: Any, move: Any, defender: Any, def_ability: Optional[str]) -> float:
    """
    기술이 방어측에게 실제로 들어가는 배율 / 4 (0=무효, 0.25=등배, 1=4배)
    타입(테라 후 타입 포함, 날씨의존 타입 포함), 특성 면역(저수/부유 등), 풍선, 중력, 불가사의부적 반영.
    변화기와 방어측 없음은 등배(0.25)로 둠 — 무효와 구분은 분류 피처가 함.
    ponytail: 공격측 틀깨기, 프리즈드라이/플라잉프레스 같은 특수 상성은 미반영
    """
    if defender is None or not (getattr(move, "base_power", 0) or 0):
        return 0.25
    from src.core.rct_player import eff_move
    move_type = effective_move_type(battle, move)
    return min(4.0, eff_move(battle, move_type, defender, def_ability)) / 4.0


def _battle_has_trickroom(battle: Any) -> bool:
    fields = getattr(battle, "fields", None) or {}
    keys = fields if isinstance(fields, dict) else ({fields} if not isinstance(fields, (list, tuple, set)) else fields)
    return any("trickroom" in re.sub(r'[^a-z0-9]', '', extract_enum_str(k)) for k in keys)


def priority_speed_edge(battle: Any, move: Any, own: Any, opp: Any) -> float:
    """
    이 기술이 상대가 "드러낸 기술 중 최고 우선도"보다 먼저 나갈지: +1 먼저, -1 나중, 0 동률/불명.
    실제 규칙 그대로: 우선도 등급이 다르면 스피드 무관하게 등급만으로 결정 (Thunderclap류처럼 상대가
    실제로 그 기술을 낼지는 예측 영역이라 여기선 다루지 않음 — "상대가 쓸 수 있는 선공기가 있다"는
    드러난 사실만 반영). 등급이 같으면 종족값 기준 스피드로 근사 비교.
    ponytail: 스탯 랭크업/마비/추진력 등 스피드 보정은 미반영(둘 다 미보정 종족값 비교), 상대 도구는
    반영 안 함(species_stats만) — 정밀 비교보다 "등급이 다르면 확정적으로 진다/이긴다"는 큰 신호가 핵심.
    """
    if move is None or opp is None:
        return 0.0
    my_pri = getattr(move, "priority", 0) or 0
    opp_moves = (getattr(opp, "moves", {}) or {}).values()
    opp_pri = max((getattr(m, "priority", 0) or 0 for m in opp_moves), default=0)
    if my_pri != opp_pri:
        return 1.0 if my_pri > opp_pri else -1.0
    my_spe = (getattr(own, "base_stats", None) or {}).get("spe", 0) if own is not None else 0
    opp_spe = (getattr(opp, "base_stats", None) or {}).get("spe", 0)
    if _battle_has_trickroom(battle):
        my_spe, opp_spe = -my_spe, -opp_spe
    if my_spe == opp_spe:
        return 0.0
    return 1.0 if my_spe > opp_spe else -1.0


def extract_move_features(move: Any, battle: Any = None) -> Tuple[float, float, float, float, float, float, float]:
    """
    기술(Move) 객체에서 7가지 명시적 수치 및 부가효과 피처 추출:
    1. base_power / 200.0 (위력)
    2. accuracy (명중률)
    3. current_pp / max_pp (PP 비율)
    4. priority / 7.0 (우선도/선공권: -7~+5 ➔ -1.0~+0.71)
    5. category (물리: 0.33, 특수: 0.67, 변화기: 1.0)
    6. secondary_effect_flag (반동, 회복/흡혈, 방어기, 상태이상/랭크변화 부가효과 0.0~1.0)
    7. move_type_norm (기술 타입 ID / 20.0 — 타입 면역/상성 전략 학습용)
    """
    if move is None:
        return 0.0, 1.0, 1.0, 0.0, 0.33, 0.0, 0.0

    base_power_raw = getattr(move, "base_power", 0) or 0
    accuracy_raw = getattr(move, "accuracy", 1.0) or 1.0
    if battle is not None:
        base_power_raw = effective_move_power(battle, move, base_power_raw)
        accuracy_raw = effective_move_accuracy(battle, move, accuracy_raw)
    bp = base_power_raw / 200.0
    acc = accuracy_raw
    
    curr_pp = getattr(move, "current_pp", 1) or 0
    max_pp = getattr(move, "max_pp", 1) or 1
    pp_ratio = curr_pp / max_pp

    # 우선도 (-7 ~ +5)
    priority = (getattr(move, "priority", 0) or 0) / 7.0

    # 기술 분류 (PHYSICAL: 0.33, SPECIAL: 0.67, STATUS: 1.0)
    cat_str = extract_enum_str(getattr(move, "category", None))
    if "special" in cat_str:
        cat_val = 0.67
    elif "status" in cat_str or "change" in cat_str:
        cat_val = 1.0
    else:
        cat_val = 0.33

    # 부가효과 (반동, 회복, 방어기, 상태이상/랭크변화/흡혈 유무)
    sec_val = 0.0
    if getattr(move, "recoil", 0):
        sec_val += 0.25
    if getattr(move, "drain", 0) or getattr(move, "heal", 0):
        sec_val += 0.25
    if getattr(move, "secondary", None) or getattr(move, "self_boost", None) or getattr(move, "boosts", None):
        sec_val += 0.25
    if getattr(move, "stalling_move", False) or getattr(move, "is_protect_move", False):
        sec_val += 0.25

    # 기술 타입 (electric=5, fire=2, ground=9, ... / 20.0 정규화)
    # TYPE_NORM_MAP: 타입 문자열 → vocab ID를 직접 사용하지 않고 런타임에서 추출
    _TYPE_MAP = {
        "normal": 1, "fire": 2, "water": 3, "grass": 4, "electric": 5,
        "ice": 6, "fighting": 7, "poison": 8, "ground": 9, "flying": 10,
        "psychic": 11, "bug": 12, "rock": 13, "ghost": 14, "dragon": 15,
        "steel": 16, "dark": 17, "fairy": 18, "stellar": 19,
    }
    move_type_raw = effective_move_type(battle, move) if battle is not None else extract_enum_str(getattr(move, "type", None))
    move_type_id = _TYPE_MAP.get(move_type_raw, 0)
    move_type_norm = move_type_id / 20.0

    return bp, acc, pp_ratio, priority, cat_val, sec_val, move_type_norm


@lru_cache(maxsize=2048)
def _move_obj(move_id: str):
    """추론된 기술 id -> poke-env Move (수치 피처용)"""
    try:
        from poke_env.battle import Move
        move = Move(move_id, gen=9)
        move.entry  # 존재하지 않는 기술이면 여기서 예외
        return move
    except Exception:
        return None


# my_num / opp_num 레이아웃 [.., 23]
#   0~11 배틀 수치, 12~18 종족값, 19 도구 추론됨, 20 특성 추론됨, 21 추론 기술 수/4, 22 팀 프리뷰로만 확인
# my_cat / opp_cat 레이아웃 [.., 10]
#   0 item, 1 ability, 2 type1, 3 type2, 4 status, 5~8 moves, 9 tera type
# my_num / opp_num 레이아웃 [.., BASE_NUM] (죽은 칸 정리 후):
#   0 필드 위, 1 기절, 2 HP비율, 3~7 공/방/특공/특방/스피드 랭크, 8 레벨, 9 기믹 활성, 10 기믹 사용가능,
#   11 종족값(또는 내 쪽은 아는 경우 실능력치) HP, 12~16 공/방/특공/특방/스피드, 17 몸무게,
#   18 아직 팀프리뷰로만 보임(상대 전용) — 뒤 SLOT_DIM칸은 상대 슬롯별 증거 (events.py)
BASE_NUM = 19
NUM_DIM, CAT_DIM = BASE_NUM + SLOT_DIM, 11  # 카테고리 [0..9]=도구/특성/타입/상태/기술/테라, [10]=종 ID (상대만 모델이 씀)
# 양쪽 활성 포켓몬의 행동 제약 효과 (field_vec 뒤쪽 열, 내 쪽 다음 상대 쪽): 기술을 막는 효과 3 + 교체를 막는 효과 1 + 남은 턴 비율 3
# (트집·기술 고정·교감은 poke-env가 효과로 안 남기거나 데이터에 거의 없어서 뺌: 난동 같은 고정은 합법 마스크가 "기술 1개만 가능"으로 알려 줌)
# 남은 턴 = (지속 턴 - poke-env 카운터) / 지속 턴. 카운터는 시작 때 0, 턴이 끝날 때마다 +1. 지속 턴은 프로토콜 실측(앵콜/도발은 시작 2턴 뒤 종료 = 결정 2번 제약)과 게임 규칙
EFFECT_FLAGS = ("TAUNT", "ENCORE", "DISABLE")
TRAP_EFFECTS = ("TRAPPED", "OCTOLOCK", "NO_RETREAT", "PARTIALLY_TRAPPED")
EFFECT_TURNS = {"TAUNT": 3, "ENCORE": 3, "DISABLE": 4}
EFFECT_SIDE = len(EFFECT_FLAGS) + 1 + len(EFFECT_TURNS)
EFFECT_DIM = 2 * EFFECT_SIDE
EFFECT_START = 48 + TURN_DIM
FIELD_DIM = EFFECT_START + EFFECT_DIM


def active_effect_vec(mon: Any) -> List[float]:
    """활성 포켓몬의 행동 제약 효과 -> [플래그 3, 교체 봉쇄 1, 남은 턴 비율 3]"""
    v = [0.0] * EFFECT_SIDE
    eff = {e.name: c for e, c in (getattr(mon, "effects", None) or {}).items()} if mon is not None else {}
    for i, n in enumerate(EFFECT_FLAGS):
        v[i] = float(n in eff)
    v[len(EFFECT_FLAGS)] = float(any(n in eff for n in TRAP_EFFECTS))
    for j, (n, dur) in enumerate(EFFECT_TURNS.items()):
        if n in eff:
            v[len(EFFECT_FLAGS) + 1 + j] = max(0.0, (dur - eff[n]) / dur)
    return v


class BattleTensorEncoder:
    """
    poke-env의 Battle 객체 또는 RCT Mod의 JSON 상태를 
    DeepPokemonBattleTransformerNet 입력 규격 텐서로 변환하는 전용 엔코더
    """
    
    def __init__(self, vocab_path: str = "data/vocab.json", device: str = "cpu"):
        self.vocab_mgr = VocabManager.get_instance(vocab_path)
        self.device = device
        self.blind_preview = False  # True면 팀 프리뷰로 알게 된 상대 팀을 안 씀 (마인크래프트처럼 프리뷰가 없는 환경 재현: 상대는 등장해야 보임)

    def encode_battle(self, battle: Any) -> Tuple[torch.Tensor, ...]:
        """
        poke-env의 Battle 객체 변환 (Enum 파싱 & 딕셔너리 weather 호환)
        """
        my_cat = np.zeros((1, 6, CAT_DIM), dtype=np.int64)
        my_num = np.zeros((1, 6, NUM_DIM), dtype=np.float32)
        my_m_num = np.zeros((1, 6, 4, MOVE_NUM_DIM), dtype=np.float32)
        
        opp_cat = np.zeros((1, 6, CAT_DIM), dtype=np.int64)
        opp_num = np.zeros((1, 6, NUM_DIM), dtype=np.float32)
        opp_m_num = np.zeros((1, 6, 4, MOVE_NUM_DIM), dtype=np.float32)
        
        field_vec = np.zeros((1, FIELD_DIM), dtype=np.float32)
        action_mask = np.zeros((1, 22), dtype=bool)
        legal_mask = np.zeros((1, 22), dtype=bool)     # 서버 요청이 알려 준 순수 합법성 (모델 입력 특징용). action_mask = 여기에 휴리스틱 가지치기를 더한 것 (로짓 마스킹 전용)

        if not hasattr(battle, "team") or not battle.team:
            return self._to_tensors(my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask, legal_mask)

        # --- A. 내 팀 6마리 인코딩 ---
        my_team_list = list(battle.team.values())
        my_team_keys = list(battle.team.keys())
        active_pkmn = getattr(battle, "active_pokemon", None)
        opp_active_mon = getattr(battle, "opponent_active_pokemon", None)
        opp_active_ability = defender_ability(opp_active_mon) if opp_active_mon else None
        my_active_ability = extract_enum_str(getattr(active_pkmn, "ability", None)) if active_pkmn else None
        # poke-env는 요청(request)의 teraType을 읽지 않아 테라 전엔 tera_type이 None → 원본 요청에서 직접 읽음
        request_tera = {m.get("ident"): m.get("teraType")
                        for m in ((getattr(battle, "last_request", None) or {}).get("side") or {}).get("pokemon", [])}

        for i in range(min(6, len(my_team_list))):
            pkmn = my_team_list[i]
            is_active = (pkmn == active_pkmn)
            
            my_num[0, i, 0] = 1.0 if is_active else 0.0
            my_num[0, i, 1] = 1.0 if getattr(pkmn, "fainted", False) else 0.0
            my_num[0, i, 2] = getattr(pkmn, "current_hp_fraction", 1.0)

            boosts = getattr(pkmn, "boosts", {}) or {}
            my_num[0, i, 3] = boosts.get("atk", 0) / 6.0
            my_num[0, i, 4] = boosts.get("def", 0) / 6.0
            my_num[0, i, 5] = boosts.get("spa", 0) / 6.0
            my_num[0, i, 6] = boosts.get("spd", 0) / 6.0
            my_num[0, i, 7] = boosts.get("spe", 0) / 6.0
            my_num[0, i, 8] = (getattr(pkmn, "level", 100) or 100) / 100.0

            # 수치 슬롯 9: 현재 포켓몬 기믹 활성화 상태 (다이맥스 남은 턴/3.0, 테라스탈 1.0, 메가진화 1.0)
            is_tera = getattr(pkmn, "is_terastallized", False)
            is_dyna = getattr(pkmn, "is_dynamaxed", False)
            dyna_turns = (getattr(battle, "dynamax_turns_left", None) or 3) if is_dyna else 0

            clean_species = re.sub(r'[^a-z0-9]', '', extract_enum_str(getattr(pkmn, "species", "")))
            is_mega = clean_species.endswith("mega") or clean_species.endswith("megax") or clean_species.endswith("megay")

            if is_dyna:
                gimmick_state = min(1.0, max(0.33, float(dyna_turns) / 3.0))
            elif is_tera or is_mega:
                gimmick_state = 1.0
            else:
                gimmick_state = 0.0

            my_num[0, i, 9] = gimmick_state

            # 수치 슬롯 10: 트레이너/포켓몬 기믹 발동 가능 여부
            can_tera = getattr(battle, "can_tera", False)
            can_dyna = getattr(battle, "can_dynamax", False)
            can_mega = getattr(battle, "can_mega_evolve", False) or getattr(battle, "can_z_move", False)
            my_num[0, i, 10] = 1.0 if (can_tera or can_dyna or can_mega) else 0.0

            # 카테고리 ID (species_id 슬롯 제거, 인덱스 1씩 시프트)
            # [0]=item, [1]=ability, [2]=type1, [3]=type2, [4]=status, [5:9]=moves
            my_item, my_abil = getattr(pkmn, "item", None), getattr(pkmn, "ability", None)
            my_cat[0, i, 0] = self.vocab_mgr.slot_id("item", my_item, my_item == "unknown_item")
            my_cat[0, i, 1] = self.vocab_mgr.slot_id("ability", my_abil, not my_abil)
            my_cat[0, i, 2] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_1", None))
            my_cat[0, i, 3] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_2", None))
            my_cat[0, i, 10] = self.vocab_mgr.get_id("species", getattr(pkmn, "species", None))
            my_cat[0, i, 4] = self.vocab_mgr.get_id("status", getattr(pkmn, "status", None))
            my_cat[0, i, 9] = self.vocab_mgr.get_id("type", getattr(pkmn, "tera_type", None) or request_tera.get(my_team_keys[i]))

            moves_dict = getattr(pkmn, "moves", {}) or {}
            for m_idx, move in enumerate(moves_dict.values()):
                if m_idx < 4:
                    move_id = getattr(move, "id", None)
                    my_cat[0, i, 5 + m_idx] = self.vocab_mgr.get_id("move", move_id)
                    my_m_num[0, i, m_idx, :7] = extract_move_features(move, battle)
                    my_m_num[0, i, m_idx, 7] = move_effectiveness(battle, move, opp_active_mon, opp_active_ability)
                    my_m_num[0, i, m_idx, 8:8 + MOVE_EFFECT_DIM] = move_effect_features(move_id)
                    my_m_num[0, i, m_idx, 8 + MOVE_EFFECT_DIM] = priority_speed_edge(battle, move, pkmn, opp_active_mon)
            for m_idx in range(min(4, len(moves_dict)), 4):  # 아직 안 드러난 기술 자리
                my_cat[0, i, 5 + m_idx] = self.vocab_mgr.slot_id("move", None, True)

            # species base_stats 7개 수치화 (my_num 뒤쪽 7 슬롯: 11~17)
            # 내 포켓몬은 노력치/성격을 알면(실전, 혹은 FoulPlay 재생) 종족값 대신 실제 능력치를 씀 → 아래 도구 배율도 그 위에 곱해짐
            base_stats = getattr(pkmn, "base_stats", None) or {}
            eff_stats = effective_base_stats(pkmn, base_stats)
            hp_val = eff_stats.get("hp", 0) or 0
            atk_val = eff_stats.get("atk", 0) or 0
            def_val = eff_stats.get("def", 0) or 0
            spa_val = eff_stats.get("spa", 0) or 0
            spd_val = eff_stats.get("spd", 0) or 0
            spe_val = eff_stats.get("spe", 0) or 0
            
            item_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(getattr(pkmn, "item", "")))
            ability_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(getattr(pkmn, "ability", "")))
            
            if item_str == "choiceband":
                atk_val *= 1.5
            elif item_str == "choicespecs":
                spa_val *= 1.5
            elif item_str == "choicescarf":
                spe_val *= 1.5
            elif item_str == "lifeorb":
                # 생명의구슬은 기술 위력 1.3배지만, 인공지능 인지를 위해 공/특공 1.3배로 근사치 적용
                atk_val *= 1.3
                spa_val *= 1.3
            elif item_str == "assaultvest":
                spd_val *= 1.5
            elif item_str == "eviolite":
                # 엄밀히는 미진화체 전용이나, 실전 데이터 특성상 장착 시 1.5배로 근사
                def_val *= 1.5
                spd_val *= 1.5
            elif item_str == "boosterenergy" and ability_str in ["protosynthesis", "quarkdrive"]:
                stats_dict = {"atk": atk_val, "def": def_val, "spa": spa_val, "spd": spd_val, "spe": spe_val}
                highest = max(stats_dict, key=stats_dict.get)
                if highest == "spe": spe_val *= 1.5
                elif highest == "atk": atk_val *= 1.3
                elif highest == "def": def_val *= 1.3
                elif highest == "spa": spa_val *= 1.3
                elif highest == "spd": spd_val *= 1.3

            my_num[0, i, 11] = hp_val / 255.0
            my_num[0, i, 12] = atk_val / 255.0
            my_num[0, i, 13] = def_val / 255.0
            my_num[0, i, 14] = spa_val / 255.0
            my_num[0, i, 15] = spd_val / 255.0
            my_num[0, i, 16] = spe_val / 255.0
            my_num[0, i, 17] = min(1.0, (getattr(pkmn, "weight", 0) or 0) / 500.0)


        # --- B. 상대 팀 6마리 및 공개된 기술 인코딩 ---
        opp_team_list = list(battle.opponent_team.values()) if hasattr(battle, "opponent_team") else []
        n_seen = len(opp_team_list)
        seen_species = [re.sub(r'[^a-z0-9]', '', str(getattr(p, "species", ""))) for p in opp_team_list]
        for p in ([] if self.blind_preview else (getattr(battle, "teampreview_opponent_team", None) or [])):
            sp = re.sub(r'[^a-z0-9]', '', str(getattr(p, "species", "")))
            # ponytail: 폼 차이(프리뷰 'urshifu' vs 실제 'urshifurapidstrike')는 접두사로만 매칭
            if sp and not any(s.startswith(sp) or sp.startswith(s) for s in seen_species):
                opp_team_list.append(p)
        opp_active = getattr(battle, "opponent_active_pokemon", None)
        ev = battle.__dict__.get("_ev")  # 턴 사건/상대 증거 (events.py가 parse_message에서 채움)

        for i in range(min(6, len(opp_team_list))):
            pkmn = opp_team_list[i]
            is_active = (pkmn == opp_active)
            
            opp_num[0, i, 0] = 1.0 if is_active else 0.0
            opp_num[0, i, 1] = 1.0 if getattr(pkmn, "fainted", False) else 0.0
            opp_num[0, i, 2] = getattr(pkmn, "current_hp_fraction", 1.0)
            
            boosts = getattr(pkmn, "boosts", {}) or {}
            opp_num[0, i, 3] = boosts.get("atk", 0) / 6.0
            opp_num[0, i, 4] = boosts.get("def", 0) / 6.0
            opp_num[0, i, 5] = boosts.get("spa", 0) / 6.0
            opp_num[0, i, 6] = boosts.get("spd", 0) / 6.0
            opp_num[0, i, 7] = boosts.get("spe", 0) / 6.0
            opp_num[0, i, 8] = (getattr(pkmn, "level", 100) or 100) / 100.0

            is_tera = getattr(pkmn, "is_terastallized", False)
            is_dyna = getattr(pkmn, "is_dynamaxed", False)
            dyna_turns = (getattr(battle, "opponent_dynamax_turns_left", None) or 3) if is_dyna else 0

            clean_species = re.sub(r'[^a-z0-9]', '', extract_enum_str(getattr(pkmn, "species", "")))
            is_mega = clean_species.endswith("mega") or clean_species.endswith("megax") or clean_species.endswith("megay")

            if is_dyna:
                gimmick_state = min(1.0, max(0.33, float(dyna_turns) / 3.0))
            elif is_tera or is_mega:
                gimmick_state = 1.0
            else:
                gimmick_state = 0.0

            opp_num[0, i, 9] = gimmick_state
            opp_num[0, i, 10] = 0.0  # 상대 기믹 가능 여부는 모를 수 있으므로 기본값 0.0

            opp_num[0, i, BASE_NUM - 1] = 0.0 if i < n_seen else 1.0  # 아직 팀프리뷰로만 보임

            # 공개 안 된 도구/특성/기술/테라 타입은 추정해서 채우지 않고 hidden 토큰으로 둠 (모델이 불확실한 채로 판단)
            moves_dict = getattr(pkmn, "moves", {}) or {}
            item_name = getattr(pkmn, "item", None)
            ability_name = getattr(pkmn, "ability", None)

            opp_cat[0, i, 0] = self.vocab_mgr.slot_id("item", item_name, item_name == "unknown_item")
            opp_cat[0, i, 1] = self.vocab_mgr.slot_id("ability", ability_name, not ability_name)
            opp_cat[0, i, 2] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_1", None))
            opp_cat[0, i, 3] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_2", None))
            opp_cat[0, i, 10] = self.vocab_mgr.get_id("species", getattr(pkmn, "species", None))
            opp_cat[0, i, 4] = self.vocab_mgr.get_id("status", getattr(pkmn, "status", None))

            for m_idx, move in enumerate(moves_dict.values()):
                if m_idx < 4:
                    move_id = getattr(move, "id", None)
                    opp_cat[0, i, 5 + m_idx] = self.vocab_mgr.get_id("move", move_id)
                    opp_m_num[0, i, m_idx, :7] = extract_move_features(move, battle)
                    opp_m_num[0, i, m_idx, 7] = move_effectiveness(battle, move, active_pkmn, my_active_ability)
                    opp_m_num[0, i, m_idx, 8:8 + MOVE_EFFECT_DIM] = move_effect_features(move_id)
                    opp_m_num[0, i, m_idx, 8 + MOVE_EFFECT_DIM] = priority_speed_edge(battle, move, pkmn, active_pkmn)
            for m_idx in range(min(4, len(moves_dict)), 4):  # 아직 안 드러난 기술 자리
                opp_cat[0, i, 5 + m_idx] = self.vocab_mgr.slot_id("move", None, True)

            # 테라 타입: 이미 테라스탈했으면 type_1이 곧 테라 타입, 아니면 미공개
            tera_name = getattr(pkmn, "type_1", None) if is_tera else None
            opp_cat[0, i, 9] = self.vocab_mgr.get_id("type", tera_name)

            # 상대 포켓방 base_stats
            base_stats = getattr(pkmn, "base_stats", None) or {}
            hp_val = base_stats.get("hp", 0)
            atk_val = base_stats.get("atk", 0)
            def_val = base_stats.get("def", 0)
            spa_val = base_stats.get("spa", 0)
            spd_val = base_stats.get("spd", 0)
            spe_val = base_stats.get("spe", 0)
            
            item_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(item_name or ""))
            ability_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(ability_name or ""))
            
            if item_str == "choiceband":
                atk_val *= 1.5
            elif item_str == "choicespecs":
                spa_val *= 1.5
            elif item_str == "choicescarf":
                spe_val *= 1.5
            elif item_str == "lifeorb":
                # 생명의구슬은 기술 위력 1.3배지만, 인공지능 인지를 위해 공/특공 1.3배로 근사치 적용
                atk_val *= 1.3
                spa_val *= 1.3
            elif item_str == "assaultvest":
                spd_val *= 1.5
            elif item_str == "eviolite":
                # 엄밀히는 미진화체 전용이나, 실전 데이터 특성상 장착 시 1.5배로 근사
                def_val *= 1.5
                spd_val *= 1.5
            elif item_str == "boosterenergy" and ability_str in ["protosynthesis", "quarkdrive"]:
                stats_dict = {"atk": atk_val, "def": def_val, "spa": spa_val, "spd": spd_val, "spe": spe_val}
                highest = max(stats_dict, key=stats_dict.get)
                if highest == "spe": spe_val *= 1.5
                elif highest == "atk": atk_val *= 1.3
                elif highest == "def": def_val *= 1.3
                elif highest == "spa": spa_val *= 1.3
                elif highest == "spd": spd_val *= 1.3

            opp_num[0, i, 11] = hp_val / 255.0
            opp_num[0, i, 12] = atk_val / 255.0
            opp_num[0, i, 13] = def_val / 255.0
            opp_num[0, i, 14] = spa_val / 255.0
            opp_num[0, i, 15] = spd_val / 255.0
            opp_num[0, i, 16] = spe_val / 255.0
            opp_num[0, i, 17] = min(1.0, (getattr(pkmn, "weight", 0) or 0) / 500.0)
            opp_num[0, i, BASE_NUM:] = ev.slot_vec(re.sub(r"[^a-z0-9]", "", str(getattr(pkmn, "species", "")).lower())) if ev else 0.0

        # --- C. 필드, 날씨, 룸 & 내/상대 사이드 조건(벽, 장판, 순풍) 48D 정밀 인코딩 ---
        if ev:  # 직전 턴 사건 10칸 (인덱스 48~57)
            field_vec[0, 48:EFFECT_START] = ev.last
        # C.1 현재 턴 수 정규화 (50턴 기준 - 인덱스 47)
        turn_num = getattr(battle, "turn", 1) or 1
        field_vec[0, 47] = min(1.0, turn_num / 50.0)

        # C.2 날씨 (Weather) 및 남은 턴 수 인코딩 (0~9)
        weathers = getattr(battle, "weather", None) or getattr(battle, "weathers", None)
        if weathers:
            weather_dict = weathers if isinstance(weathers, dict) else ({weathers: True} if not isinstance(weathers, (list, tuple, set)) else {k: True for k in weathers})
            for w_key, w_val in weather_dict.items():
                w_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(w_key))
                act, turns_norm = get_dynamic_turns(w_val, w_str, turn_num, battle)
                if "sun" in w_str or "desolateland" in w_str:
                    field_vec[0, 0], field_vec[0, 1] = act, turns_norm
                elif "rain" in w_str or "primordialsea" in w_str:
                    field_vec[0, 2], field_vec[0, 3] = act, turns_norm
                elif "sand" in w_str:
                    field_vec[0, 4], field_vec[0, 5] = act, turns_norm
                elif "hail" in w_str or "snow" in w_str:
                    field_vec[0, 6], field_vec[0, 7] = act, turns_norm
                elif "deltastream" in w_str:
                    field_vec[0, 8], field_vec[0, 9] = act, turns_norm

        # C.3 글로벌 필드, 트릭룸, 지형 (10~19)
        fields = getattr(battle, "fields", None)
        if fields:
            fields_dict = fields if isinstance(fields, dict) else ({fields: True} if not isinstance(fields, (list, tuple, set)) else {k: True for k in fields})
            for f_key, f_val in fields_dict.items():
                f_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(f_key))
                act, turns_norm = get_dynamic_turns(f_val, f_str, turn_num, battle)
                if "trickroom" in f_str:
                    field_vec[0, 10], field_vec[0, 11] = act, turns_norm
                elif "electric" in f_str:
                    field_vec[0, 12], field_vec[0, 13] = act, turns_norm
                elif "grassy" in f_str:
                    field_vec[0, 14], field_vec[0, 15] = act, turns_norm
                elif "psychic" in f_str:
                    field_vec[0, 16], field_vec[0, 17] = act, turns_norm
                elif "misty" in f_str:
                    field_vec[0, 18], field_vec[0, 19] = act, turns_norm
                elif "gravity" in f_str:
                    field_vec[0, 46] = 1.0

        # C.4 내 진영 사이드 조건 (20~31) & 상대 진영 사이드 조건 (32~43) - 벽, 순풍, 장판 1대1 대칭 인코딩
        def _encode_side(side_conds_data, base_idx):
            if not side_conds_data:
                return
            side_dict = side_conds_data if isinstance(side_conds_data, dict) else ({side_conds_data: True} if not isinstance(side_conds_data, (list, tuple, set)) else {k: True for k in side_conds_data})
            for s_key, s_val in side_dict.items():
                s_str = re.sub(r'[^a-z0-9]', '', extract_enum_str(s_key))
                act, turns_norm = get_dynamic_turns(s_val, s_str, turn_num, battle)
                if "reflect" in s_str:
                    field_vec[0, base_idx + 0], field_vec[0, base_idx + 1] = act, turns_norm
                elif "lightscreen" in s_str:
                    field_vec[0, base_idx + 2], field_vec[0, base_idx + 3] = act, turns_norm
                elif "auroraveil" in s_str:
                    field_vec[0, base_idx + 4], field_vec[0, base_idx + 5] = act, turns_norm
                elif "tailwind" in s_str:
                    field_vec[0, base_idx + 6], field_vec[0, base_idx + 7] = act, turns_norm
                elif "stealthrock" in s_str:
                    field_vec[0, base_idx + 8] = 1.0
                elif "toxicspikes" in s_str: # toxicspikes를 spikes보다 반드시 먼저 검사!
                    field_vec[0, base_idx + 10] = layers(s_val, 2.0)
                elif "spikes" in s_str:
                    field_vec[0, base_idx + 9] = layers(s_val, 3.0)
                elif "stickyweb" in s_str:
                    field_vec[0, base_idx + 11] = 1.0
                elif "safeguard" in s_str:
                    field_vec[0, 44 if base_idx == 20 else 45] = turns_norm

        _encode_side(getattr(battle, "side_conditions", None), base_idx=20)
        _encode_side(getattr(battle, "opponent_side_conditions", None), base_idx=32)

        field_vec[0, EFFECT_START:EFFECT_START + EFFECT_SIDE] = active_effect_vec(active_pkmn)
        field_vec[0, EFFECT_START + EFFECT_SIDE:] = active_effect_vec(opp_active_mon)

        # --- D. 행동 마스킹 ---
        if active_pkmn and hasattr(battle, "available_moves"):
            active_moves = list(getattr(active_pkmn, "moves", {}).values())
            avail_move_ids = [getattr(m, "id", None) for m in getattr(battle, "available_moves", [])]
            # 4~7번 = 해당 기술을 Z기술(7세대) 또는 테라스탈(9세대)과 함께 사용 (메가/다이맥스는 player가 규칙으로 자동 발동)
            can_tera = getattr(battle, "can_tera", False)
            z_ids = set()
            if getattr(battle, "can_z_move", False):
                z_ids = {getattr(m, "id", None) for m in (getattr(active_pkmn, "available_z_moves", None) or [])}

            for m_idx in range(min(4, len(active_moves))):
                mv = active_moves[m_idx]
                mv_id = getattr(mv, "id", None)
                curr_pp = getattr(mv, "current_pp", 1)

                if mv_id in avail_move_ids and curr_pp > 0:
                    legal_mask[0, m_idx] = True
                    if mv_id in z_ids or can_tera:
                        legal_mask[0, 4 + m_idx] = True
                    # 💡 [휴리스틱 마스킹] 무의미한 턴 낭비 방지 룰 (action_mask에만 적용, legal_mask에는 안 닿음)
                    is_valid = True
                    
                    # 1. 상대가 이미 상태이상인데, 또 상태이상을 거는 변화기(도깨비불, 맹독 등) 사용 금지
                    if hasattr(mv, "status") and mv.status is not None:
                        opp_pkmn = getattr(battle, "opponent_active_pokemon", None)
                        if opp_pkmn and opp_pkmn.status is not None:
                            is_valid = False
                            
                    # 2. 상대가 부유(Levitate) 특성이나 비행 타입일 때 땅 타입 공격기(지진 등) 금지
                    if hasattr(mv, "type") and extract_enum_str(mv.type) == "ground" and getattr(mv, "base_power", 0) > 0:
                        opp_pkmn = getattr(battle, "opponent_active_pokemon", None)
                        if opp_pkmn:
                            opp_type1 = extract_enum_str(getattr(opp_pkmn, "type_1", ""))
                            opp_type2 = extract_enum_str(getattr(opp_pkmn, "type_2", ""))
                            opp_ability = extract_enum_str(getattr(opp_pkmn, "ability", ""))
                            if "flying" in [opp_type1, opp_type2] or opp_ability == "levitate":
                                is_valid = False

                    # 3. 이미 깔려있는 1회용 장판기(스텔스록, 끈적끈적네트) 및 벽(리플렉터 등) 중복 사용 금지
                    if mv_id in ["stealthrock", "stickyweb"]:
                        opp_side = getattr(battle, "opponent_side_conditions", {})
                        opp_side_str = [extract_enum_str(k) for k in opp_side.keys()]
                        if mv_id in opp_side_str:
                            is_valid = False
                            
                    if mv_id in ["reflect", "lightscreen", "auroraveil", "tailwind", "safeguard"]:
                        my_side = getattr(battle, "side_conditions", {})
                        my_side_str = [extract_enum_str(k) for k in my_side.keys()]
                        if mv_id in my_side_str:
                            is_valid = False

                    if is_valid:
                        action_mask[0, m_idx] = True
                        if mv_id in z_ids or can_tera:
                            action_mask[0, 4 + m_idx] = True

        if hasattr(battle, "available_switches"):
            avail_switches = getattr(battle, "available_switches", [])
            for s_idx in range(min(6, len(my_team_list))):
                pkmn = my_team_list[s_idx]
                is_fainted = getattr(pkmn, "fainted", False)
                if pkmn in avail_switches and not is_fainted and pkmn != active_pkmn:
                    action_mask[0, 8 + s_idx] = True
                    legal_mask[0, 8 + s_idx] = True

        if not action_mask.any():
            action_mask[0, 0] = True
        if not legal_mask.any():
            legal_mask[0, 0] = True

        return self._to_tensors(my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask, legal_mask)

    def _to_tensors(self, my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask, legal_mask):
        return (
            torch.tensor(my_cat, dtype=torch.long, device=self.device),
            torch.tensor(my_num, dtype=torch.float32, device=self.device),
            torch.tensor(my_m_num, dtype=torch.float32, device=self.device),
            torch.tensor(opp_cat, dtype=torch.long, device=self.device),
            torch.tensor(opp_num, dtype=torch.float32, device=self.device),
            torch.tensor(opp_m_num, dtype=torch.float32, device=self.device),
            torch.tensor(field_vec, dtype=torch.float32, device=self.device),
            torch.tensor(action_mask, dtype=torch.bool, device=self.device),
            torch.tensor(legal_mask, dtype=torch.bool, device=self.device)
        )


# =====================================================================
# 테스트용 파이썬 Enum 정의 (실제 poke-env 모듈 객체 구조 모사)
# =====================================================================
class MockPokemonType(Enum):
    FIRE = "fire"
    WATER = "water"
    ELECTRIC = "electric"
    PSYCHIC = "psychic"

class MockStatus(Enum):
    BRN = "brn"
    PAR = "par"

class MockWeather(Enum):
    SUN = "sun"
    SUNNYDAY = "sunnyday"
    RAIN = "rain"


class DummyMove:
    def __init__(self, move_id, base_power=90, accuracy=1.0, current_pp=15, max_pp=15, move_type=None):
        self.id = move_id
        self.base_power = base_power
        self.accuracy = accuracy
        self.current_pp = current_pp
        self.max_pp = max_pp
        self.type = move_type  # 기술 타입 (Enum 또는 문자열)


class DummyPokemon:
    def __init__(self, species, item="leftovers", ability="blaze", type1=MockPokemonType.FIRE, type2=None, status=MockStatus.BRN, moves=None, hp=1.0):
        self.species = species
        self.item = item
        self.ability = ability
        self.type_1 = type1  # Enum 객체 전달!
        self.type_2 = type2
        self.status = status # Enum 객체 전달!
        self.current_hp_fraction = hp
        self.fainted = (hp <= 0)
        self.level = 100
        self.boosts = {"atk": 1, "spe": 2}
        self.is_terastallized = False
        self.moves = {m.id: m for m in (moves or [DummyMove("thunderbolt"), DummyMove("flamethrower")])}


class DummyBattle:
    def __init__(self):
        # Enum 객체 및 "FIRE (pokemon type) object" 문자열 모사 포켓몬 생성
        p1 = DummyPokemon("pikachu", item="lightball", type1=MockPokemonType.ELECTRIC, moves=[DummyMove("thunderbolt", 90, 1.0, 15, move_type=MockPokemonType.ELECTRIC), DummyMove("quickattack", 40, 1.0, 30, move_type="normal")])
        p2 = DummyPokemon("charizard", item="heatrock", type1="FIRE (pokemon type) object", status="BRN (status) object", moves=[DummyMove("flamethrower", 90, 1.0, 15, move_type=MockPokemonType.FIRE)])
        p3 = DummyPokemon("unknown_custom_mon", type1=MockPokemonType.WATER, hp=0.0) # 사전에 없는 포켓몬!
        
        self.active_pokemon = p1
        self.team = {"pikachu": p1, "charizard": p2, "unknown_custom_mon": p3}
        
        opp1 = DummyPokemon("mewtwo", type1=MockPokemonType.PSYCHIC, moves=[DummyMove("psystrike", 100, 1.0, 10, move_type=MockPokemonType.PSYCHIC)])
        self.opponent_active_pokemon = opp1
        self.opponent_team = {"mewtwo": opp1}
        
        self.available_moves = [p1.moves["thunderbolt"]]
        self.available_switches = [p2]
        self.can_terastallize = True
        
        self.turn = 12
        # poke-env 규약: 값 = 시작 턴. 쾌청 8턴 시작 → 8-(12-8)=4턴 남음 (0.5), 트릭룸 8턴 시작 → 5-4=1턴 (0.2)
        self.weather = {MockWeather.SUNNYDAY: 8}
        self.fields = {"trickroom": 8}
        self.side_conditions = {"tailwind": 10, "reflect": 9}


def main():
    print("🧪 [Fix 검증] Enum 객체, Weather/TrickRoom 남은 턴 수, <unk> 미지 포켓몬 테스트 시작...")
    
    encoder = BattleTensorEncoder(vocab_path="data/vocab.json")
    mock_battle = DummyBattle()
    
    tensors = encoder.encode_battle(mock_battle)
    my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask, legal_mask = tensors
    
    print("\n✅ 턴 카운터 및 장기 전략 필드 텐서 검증:")
    print(f"  1. 쾌청 활성화 (field_vec[0, 0]): {field_vec[0, 0].item() == 1.0}")
    print(f"  2. 쾌청 남은 턴 수 (4/8 = 0.5) (field_vec[0, 1]): {field_vec[0, 1].item():.4f}")
    print(f"  3. 트릭룸 활성화 (field_vec[0, 10]): {field_vec[0, 10].item() == 1.0}")
    print(f"  4. 트릭룸 남은 턴 수 (1/5 = 0.2) (field_vec[0, 11]): {field_vec[0, 11].item():.4f}")
    print(f"  5. 순풍 활성화 & 남은 턴 수 (2/4 = 0.5) (field_vec[0, 26, 27]): {field_vec[0, 26].item() == 1.0}, {field_vec[0, 27].item():.4f}")
    print(f"  6. 현재 턴 수 정규화 (12/50 = 0.24) (field_vec[0, 47]): {field_vec[0, 47].item():.4f}")
    print(f"  7. 사전에 없는 unknown_custom_mon <unk> ID: {my_cat[0, 2, 0].item()} (0(빈슬롯)이 아니라 <unk>로 인식!)")
    
    assert field_vec[0, 10].item() == 1.0, "Trick Room should be active!"
    assert abs(field_vec[0, 11].item() - 0.2) < 1e-4, f"Trick Room turns left ratio expected 0.2, got {field_vec[0, 11].item()}"
    assert abs(field_vec[0, 1].item() - 0.5) < 1e-4, f"Sun turns left ratio expected 0.5, got {field_vec[0, 1].item()}"
    
    print("\n🎉 트릭룸/날씨/순풍/벽 남은 턴 수 텐서 반영 검증 완료!")


if __name__ == "__main__":
    main()
