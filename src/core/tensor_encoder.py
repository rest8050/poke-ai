import torch
import numpy as np
import json
import os
import re
from enum import Enum
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Optional

from src.core.set_prior import predict as predict_set


WEATHER_MAP = {
    "sun": 1, "sunnyday": 1, "desolateland": 1, "harshsunshine": 1, "sunny": 1,
    "rain": 2, "raindance": 2, "primordialsea": 2, "heavyrain": 2,
    "sand": 3, "sandstorm": 3,
    "hail": 4, "snow": 4, "snowscape": 4,
    "deltastream": 5,
}


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
    - 0: 빈/미공개 패딩 슬롯 (Strict Padding)
    - <unk>: 사전에 등록되지 않은 아이템/포켓몬 (Unknown Token)
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
        
        # 4. val 형태 동적 판별 (남은 턴 수 vs 시작 턴 번호)
        # Case A: val이 이미 남은 턴 수 (0 < v <= max_duration 이고 current_turn > max_duration 인 경우)
        if 0 < v <= max_duration and current_turn > max_duration:
            turns_left_cnt = v
        # Case B: val이 시작된 턴 번호 (v <= current_turn 이고 (current_turn - v) < max_duration 인 경우)
        elif 0 <= v <= current_turn and (current_turn - v) < max_duration:
            elapsed = current_turn - v
            turns_left_cnt = max(0.0, max_duration - elapsed)
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


def extract_move_features(move: Any) -> Tuple[float, float, float, float, float, float, float]:
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

    bp = (getattr(move, "base_power", 0) or 0) / 200.0
    acc = (getattr(move, "accuracy", 1.0) or 1.0)
    
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
    move_type_raw = extract_enum_str(getattr(move, "type", None))
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
NUM_DIM, CAT_DIM = 23, 10


class BattleTensorEncoder:
    """
    poke-env의 Battle 객체 또는 RCT Mod의 JSON 상태를 
    DeepPokemonBattleTransformerNet 입력 규격 텐서로 변환하는 전용 엔코더
    """
    
    def __init__(self, vocab_path: str = "data/vocab.json", device: str = "cpu"):
        self.vocab_mgr = VocabManager.get_instance(vocab_path)
        self.device = device

    def encode_battle(self, battle: Any) -> Tuple[torch.Tensor, ...]:
        """
        poke-env의 Battle 객체 변환 (Enum 파싱 & 딕셔너리 weather 호환)
        """
        my_cat = np.zeros((1, 6, CAT_DIM), dtype=np.int64)
        my_num = np.zeros((1, 6, NUM_DIM), dtype=np.float32)
        my_m_num = np.zeros((1, 6, 4, 7), dtype=np.float32)
        
        opp_cat = np.zeros((1, 6, CAT_DIM), dtype=np.int64)
        opp_num = np.zeros((1, 6, NUM_DIM), dtype=np.float32)
        opp_m_num = np.zeros((1, 6, 4, 7), dtype=np.float32)
        
        field_vec = np.zeros((1, 48), dtype=np.float32)
        action_mask = np.zeros((1, 22), dtype=bool)

        if not hasattr(battle, "team") or not battle.team:
            return self._to_tensors(my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask)

        # --- A. 내 팀 6마리 인코딩 ---
        my_team_list = list(battle.team.values())
        my_team_keys = list(battle.team.keys())
        active_pkmn = getattr(battle, "active_pokemon", None)
        # poke-env는 요청(request)의 teraType을 읽지 않아 테라 전엔 tera_type이 None → 원본 요청에서 직접 읽음
        request_tera = {m.get("ident"): m.get("teraType")
                        for m in ((getattr(battle, "last_request", None) or {}).get("side") or {}).get("pokemon", [])}

        for i in range(min(6, len(my_team_list))):
            pkmn = my_team_list[i]
            is_active = (pkmn == active_pkmn)
            
            my_num[0, i, 0] = 1.0 if is_active else 0.0
            my_num[0, i, 1] = 1.0 if getattr(pkmn, "fainted", False) else 0.0
            my_num[0, i, 2] = i / 5.0
            my_num[0, i, 3] = getattr(pkmn, "current_hp_fraction", 1.0)
            
            boosts = getattr(pkmn, "boosts", {}) or {}
            my_num[0, i, 4] = boosts.get("atk", 0) / 6.0
            my_num[0, i, 5] = boosts.get("def", 0) / 6.0
            my_num[0, i, 6] = boosts.get("spa", 0) / 6.0
            my_num[0, i, 7] = boosts.get("spd", 0) / 6.0
            my_num[0, i, 8] = boosts.get("spe", 0) / 6.0
            my_num[0, i, 9] = (getattr(pkmn, "level", 100) or 100) / 100.0
            
            # 수치 슬롯 10: 현재 포켓몬 기믹 활성화 상태 (다이맥스 남은 턴/3.0, 테라스탈 1.0, 메가진화 1.0)
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

            my_num[0, i, 10] = gimmick_state

            # 수치 슬롯 11: 트레이너/포켓몬 기믹 발동 가능 여부
            can_tera = getattr(battle, "can_tera", False)
            can_dyna = getattr(battle, "can_dynamax", False)
            can_mega = getattr(battle, "can_mega_evolve", False) or getattr(battle, "can_z_move", False)
            my_num[0, i, 11] = 1.0 if (can_tera or can_dyna or can_mega) else 0.0

            # 카테고리 ID (species_id 슬롯 제거, 인덱스 1씩 시프트)
            # [0]=item, [1]=ability, [2]=type1, [3]=type2, [4]=status, [5:9]=moves
            my_cat[0, i, 0] = self.vocab_mgr.get_id("item", getattr(pkmn, "item", None))
            my_cat[0, i, 1] = self.vocab_mgr.get_id("ability", getattr(pkmn, "ability", None))
            my_cat[0, i, 2] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_1", None))
            my_cat[0, i, 3] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_2", None))
            my_cat[0, i, 4] = self.vocab_mgr.get_id("status", getattr(pkmn, "status", None))
            my_cat[0, i, 9] = self.vocab_mgr.get_id("type", getattr(pkmn, "tera_type", None) or request_tera.get(my_team_keys[i]))

            moves_dict = getattr(pkmn, "moves", {}) or {}
            for m_idx, move in enumerate(moves_dict.values()):
                if m_idx < 4:
                    move_id = getattr(move, "id", None)
                    my_cat[0, i, 5 + m_idx] = self.vocab_mgr.get_id("move", move_id)
                    my_m_num[0, i, m_idx, :] = extract_move_features(move)

            # species base_stats 7개 수치화 (my_num 뒤쪽 7 슬롯: 12~18)
            base_stats = getattr(pkmn, "base_stats", None) or {}
            hp_val = base_stats.get("hp", 0)
            atk_val = base_stats.get("atk", 0)
            def_val = base_stats.get("def", 0)
            spa_val = base_stats.get("spa", 0)
            spd_val = base_stats.get("spd", 0)
            spe_val = base_stats.get("spe", 0)
            
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

            my_num[0, i, 12] = hp_val / 255.0
            my_num[0, i, 13] = atk_val / 255.0
            my_num[0, i, 14] = def_val / 255.0
            my_num[0, i, 15] = spa_val / 255.0
            my_num[0, i, 16] = spd_val / 255.0
            my_num[0, i, 17] = spe_val / 255.0
            my_num[0, i, 18] = min(1.0, (getattr(pkmn, "weight", 0) or 0) / 500.0)


        # --- B. 상대 팀 6마리 및 공개된 기술 인코딩 ---
        opp_team_list = list(battle.opponent_team.values()) if hasattr(battle, "opponent_team") else []
        n_seen = len(opp_team_list)
        seen_species = [re.sub(r'[^a-z0-9]', '', str(getattr(p, "species", ""))) for p in opp_team_list]
        for p in getattr(battle, "teampreview_opponent_team", None) or []:
            sp = re.sub(r'[^a-z0-9]', '', str(getattr(p, "species", "")))
            # ponytail: 폼 차이(프리뷰 'urshifu' vs 실제 'urshifurapidstrike')는 접두사로만 매칭
            if sp and not any(s.startswith(sp) or sp.startswith(s) for s in seen_species):
                opp_team_list.append(p)
        opp_active = getattr(battle, "opponent_active_pokemon", None)

        for i in range(min(6, len(opp_team_list))):
            pkmn = opp_team_list[i]
            is_active = (pkmn == opp_active)
            
            opp_num[0, i, 0] = 1.0 if is_active else 0.0
            opp_num[0, i, 1] = 1.0 if getattr(pkmn, "fainted", False) else 0.0
            opp_num[0, i, 2] = i / 5.0
            opp_num[0, i, 3] = getattr(pkmn, "current_hp_fraction", 1.0)
            
            boosts = getattr(pkmn, "boosts", {}) or {}
            opp_num[0, i, 4] = boosts.get("atk", 0) / 6.0
            opp_num[0, i, 5] = boosts.get("def", 0) / 6.0
            opp_num[0, i, 6] = boosts.get("spa", 0) / 6.0
            opp_num[0, i, 7] = boosts.get("spd", 0) / 6.0
            opp_num[0, i, 8] = boosts.get("spe", 0) / 6.0
            opp_num[0, i, 9] = (getattr(pkmn, "level", 100) or 100) / 100.0

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

            opp_num[0, i, 10] = gimmick_state
            opp_num[0, i, 11] = 0.0  # 상대 기믹 가능 여부는 모를 수 있으므로 기본값 0.0

            opp_num[0, i, 22] = 0.0 if i < n_seen else 1.0

            # 정보 추론: 공개 안 된 도구/특성/기술/테라 타입을 세트 사전 확률로 채움
            moves_dict = getattr(pkmn, "moves", {}) or {}
            item_name = getattr(pkmn, "item", None)
            ability_name = getattr(pkmn, "ability", None)
            item_unknown = item_name == "unknown_item"
            prior = predict_set(getattr(pkmn, "species", ""), list(moves_dict),
                                None if item_unknown else item_name, ability_name)
            if prior:
                if item_unknown and prior["item"]:
                    item_name = prior["item"]
                    opp_num[0, i, 19] = 1.0
                if not ability_name and prior["ability"]:
                    ability_name = prior["ability"]
                    opp_num[0, i, 20] = 1.0

            opp_cat[0, i, 0] = self.vocab_mgr.get_id("item", item_name)
            opp_cat[0, i, 1] = self.vocab_mgr.get_id("ability", ability_name)
            opp_cat[0, i, 2] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_1", None))
            opp_cat[0, i, 3] = self.vocab_mgr.get_id("type", getattr(pkmn, "type_2", None))
            opp_cat[0, i, 4] = self.vocab_mgr.get_id("status", getattr(pkmn, "status", None))

            for m_idx, move in enumerate(moves_dict.values()):
                if m_idx < 4:
                    move_id = getattr(move, "id", None)
                    opp_cat[0, i, 5 + m_idx] = self.vocab_mgr.get_id("move", move_id)
                    opp_m_num[0, i, m_idx, :] = extract_move_features(move)

            if prior:
                n_rev = min(4, len(moves_dict))
                guesses = [m for m in (_move_obj(mid) for mid in prior["moves"]) if m is not None][:4 - n_rev]
                for j, move in enumerate(guesses):
                    opp_cat[0, i, 5 + n_rev + j] = self.vocab_mgr.get_id("move", move.id)
                    opp_m_num[0, i, n_rev + j, :] = extract_move_features(move)
                opp_num[0, i, 21] = len(guesses) / 4.0

            # 테라 타입: 이미 테라스탈했으면 type_1이 곧 테라 타입, 아니면 추론값
            tera_name = getattr(pkmn, "type_1", None) if is_tera else (prior or {}).get("tera")
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

            opp_num[0, i, 12] = hp_val / 255.0
            opp_num[0, i, 13] = atk_val / 255.0
            opp_num[0, i, 14] = def_val / 255.0
            opp_num[0, i, 15] = spa_val / 255.0
            opp_num[0, i, 16] = spd_val / 255.0
            opp_num[0, i, 17] = spe_val / 255.0
            opp_num[0, i, 18] = min(1.0, (getattr(pkmn, "weight", 0) or 0) / 500.0)

        # --- C. 필드, 날씨, 룸 & 내/상대 사이드 조건(벽, 장판, 순풍) 48D 정밀 인코딩 ---
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
                    # 💡 [휴리스틱 마스킹] 무의미한 턴 낭비 방지 룰
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

        if not action_mask.any():
            action_mask[0, 0] = True

        return self._to_tensors(my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask)

    def _to_tensors(self, my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask):
        return (
            torch.tensor(my_cat, dtype=torch.long, device=self.device),
            torch.tensor(my_num, dtype=torch.float32, device=self.device),
            torch.tensor(my_m_num, dtype=torch.float32, device=self.device),
            torch.tensor(opp_cat, dtype=torch.long, device=self.device),
            torch.tensor(opp_num, dtype=torch.float32, device=self.device),
            torch.tensor(opp_m_num, dtype=torch.float32, device=self.device),
            torch.tensor(field_vec, dtype=torch.float32, device=self.device),
            torch.tensor(action_mask, dtype=torch.bool, device=self.device)
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
        p2 = DummyPokemon("charizard", type1="FIRE (pokemon type) object", status="BRN (status) object", moves=[DummyMove("flamethrower", 90, 1.0, 15, move_type=MockPokemonType.FIRE)])
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
    my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask = tensors
    
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
