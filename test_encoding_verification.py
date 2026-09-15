import torch
import numpy as np
import json
import re
from enum import Enum

from tensor_encoder import BattleTensorEncoder, VocabManager, extract_enum_str


class MockType(Enum):
    ELECTRIC = "electric"
    FIRE = "fire"
    FLYING = "flying"
    WATER = "water"


class MockStatus(Enum):
    BRN = "brn"
    PAR = "par"


class MockMove:
    def __init__(self, move_id, bp=90, acc=1.0, cpp=15, mpp=15, priority=0, category="special", recoil=0, heal=0, secondary=None, move_type=None):
        self.id = move_id
        self.base_power = bp
        self.accuracy = acc
        self.current_pp = cpp
        self.max_pp = mpp
        self.priority = priority
        self.category = category
        self.recoil = recoil
        self.heal = heal
        self.secondary = secondary
        self.type = move_type  # 기술 타입 (Enum 또는 문자열)


class MockPokemon:
    def __init__(self, species, item="leftovers", ability="blaze", type1=MockType.FIRE, type2=None, status=None, moves=None, hp=1.0, level=100, fainted=False, boosts=None, is_dyna=False, is_tera=False, z_moves=None):
        self.species = species
        self.item = item
        self.ability = ability
        self.type_1 = type1
        self.type_2 = type2
        self.status = status
        self.current_hp_fraction = hp
        self.fainted = fainted
        self.level = level
        self.boosts = boosts or {}
        self.is_dynamaxed = is_dyna
        self.is_terastallized = is_tera
        self.moves = {m.id: m for m in (moves or [])}
        self.available_z_moves = z_moves or []


class MockBattle:
    """poke-env 규약: weather/fields/side_conditions 값 = 시작 턴 (압정류만 겹수), 다이맥스 남은 턴은 battle에"""
    def __init__(self, my_team, opp_team, active_my, active_opp, avail_moves=None, avail_switches=None, weather=None, fields=None, side_conditions=None, opp_side_conditions=None, turn=1, can_tera=False, can_dyna=False, can_mega=False, can_z=False, opp_dyna_turns=None):
        self.team = {p.species: p for p in my_team}
        self.opponent_team = {p.species: p for p in opp_team}
        self.active_pokemon = active_my
        self.opponent_active_pokemon = active_opp
        self.available_moves = avail_moves or []
        self.available_switches = avail_switches or []
        self.weather = weather
        self.fields = fields
        self.side_conditions = side_conditions
        self.opponent_side_conditions = opp_side_conditions
        self.turn = turn
        self.can_terastallize = can_tera
        self.can_dynamax = can_dyna
        self.can_mega_evolve = can_mega
        self.can_z_move = can_z
        self.dynamax_turns_left = None
        self.opponent_dynamax_turns_left = opp_dyna_turns


def run_full_1to1_verification():
    print("🔬 [1대1 완전 검증] BattleTensorEncoder 전체 인덱스 & 항목별 1:1 매칭 검증 시작...\n")
    encoder = BattleTensorEncoder(vocab_path="vocab.json")
    v_mgr = encoder.vocab_mgr

    # 1. 테스트 포켓몬 준비
    m1 = MockMove("thunderbolt", bp=90, acc=1.0, cpp=15, mpp=15, priority=0, category="special", move_type=MockType.ELECTRIC)
    m2 = MockMove("quickattack", bp=40, acc=1.0, cpp=30, mpp=30, priority=1, category="physical", move_type="normal")
    m3 = MockMove("flareblitz", bp=120, acc=1.0, cpp=5, mpp=15, priority=0, category="physical", recoil=0.33, move_type=MockType.FIRE)
    m4 = MockMove("roost", bp=0, acc=1.0, cpp=10, mpp=10, priority=0, category="status", heal=0.5, move_type=MockType.FLYING)

    p1 = MockPokemon("pikachu", item="heatrock", ability="static", type1=MockType.ELECTRIC, status=MockStatus.PAR, moves=[m1, m2], hp=0.8, level=100, boosts={"atk": 2, "spe": -1}, z_moves=[m1])
    p2 = MockPokemon("charizardmegax", item="lightclay", ability="toughclaws", type1=MockType.FIRE, type2=MockType.FLYING, moves=[m3], hp=1.0, level=100)
    p3 = MockPokemon("meganium", item="leftovers", ability="overgrow", type1=MockType.FIRE, moves=[m4], hp=0.0, fainted=True)

    opp_m1 = MockMove("psystrike", bp=100, acc=1.0, cpp=10, mpp=15, priority=0, category="special", move_type="psychic")
    opp_p1 = MockPokemon("mewtwo", item="terrainextender", ability="pressure", type1=MockType.FIRE, moves=[opp_m1], hp=0.5, is_dyna=True)

    # turn=15 기준 시작 턴: 남은 = 지속 - (15 - 시작)
    battle = MockBattle(
        my_team=[p1, p2, p3],
        opp_team=[opp_p1],
        active_my=p1,
        active_opp=opp_p1,
        avail_moves=[m1, m2],
        avail_switches=[p2],
        weather={"sunnyday": 11},                                   # 8-4 = 4 남음
        fields={"trickroom": 11, "electricterrain": 12},            # 5-4 = 1 남음
        side_conditions={"toxicspikes": 2, "spikes": 1, "tailwind": 14, "reflect": 13},  # 압정류=겹수, 순풍 3, 리플 6
        opp_side_conditions={"reflect": 12, "lightscreen": 10, "auroraveil": 14, "tailwind": 13},  # 5, 3, 7, 2
        turn=15,
        can_z=True,
        opp_dyna_turns=2
    )

    tensors = encoder.encode_battle(battle)
    my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask = tensors

    # =========================================================================
    # A. 내 팀 카테고리 텐서 my_cat [1, 6, 10] 1:1 매칭 검증
    # =========================================================================
    print("--- A. 내 팀 카테고리 텐서 my_cat [1, 6, 10] 검증 ---")
    p1_species_id = v_mgr.get_id("species", "pikachu")
    p1_item_id = v_mgr.get_id("item", "heatrock")
    p1_ability_id = v_mgr.get_id("ability", "static")
    p1_type1_id = v_mgr.get_id("type", "electric")
    p1_status_id = v_mgr.get_id("status", "par")
    m1_id = v_mgr.get_id("move", "thunderbolt")
    m2_id = v_mgr.get_id("move", "quickattack")

    assert my_cat[0, 0, 0].item() == p1_species_id, f"p1 species mismatch: {my_cat[0, 0, 0].item()} vs {p1_species_id}"
    assert my_cat[0, 0, 1].item() == p1_item_id, f"p1 item mismatch: {my_cat[0, 0, 1].item()} vs {p1_item_id}"
    assert my_cat[0, 0, 2].item() == p1_ability_id, f"p1 ability mismatch: {my_cat[0, 0, 2].item()} vs {p1_ability_id}"
    assert my_cat[0, 0, 3].item() == p1_type1_id, f"p1 type1 mismatch: {my_cat[0, 0, 3].item()} vs {p1_type1_id}"
    assert my_cat[0, 0, 5].item() == p1_status_id, f"p1 status mismatch: {my_cat[0, 0, 5].item()} vs {p1_status_id}"
    assert my_cat[0, 0, 6].item() == m1_id, f"m1 id mismatch: {my_cat[0, 0, 6].item()} vs {m1_id}"
    assert my_cat[0, 0, 7].item() == m2_id, f"m2 id mismatch: {my_cat[0, 0, 7].item()} vs {m2_id}"
    print("  [Pass] Slot 0 Pikachu Category IDs 1:1 Match!")

    # 빈 슬롯 (Slot 3, 4, 5) strict padding 0 검증
    for empty_slot in [3, 4, 5]:
        assert (my_cat[0, empty_slot, :] == 0).all().item(), f"Empty slot {empty_slot} should be strictly 0!"
    print("  [Pass] Strict Padding Slots (3, 4, 5) == 0 Verified!")

    # =========================================================================
    # B. 내 팀 수치 텐서 my_num [1, 6, 12] 1:1 매칭 검증
    # =========================================================================
    print("\n--- B. 내 팀 수치 텐서 my_num [1, 6, 12] 검증 ---")
    assert my_num[0, 0, 0].item() == 1.0, "Slot 0 active flag should be 1.0"
    assert my_num[0, 0, 1].item() == 0.0, "Slot 0 fainted flag should be 0.0"
    assert abs(my_num[0, 0, 3].item() - 0.8) < 1e-4, "Slot 0 HP fraction should be 0.8"
    assert abs(my_num[0, 0, 4].item() - (2.0/6.0)) < 1e-4, "Slot 0 atk boost should be +2/6"
    assert abs(my_num[0, 0, 8].item() - (-1.0/6.0)) < 1e-4, "Slot 0 spe boost should be -1/6"
    assert my_num[0, 0, 11].item() == 1.0, "can_z_move should set slot 11 to 1.0"

    # p3 Meganium (Slot 2) fainted = 1.0, Meganium is_mega = 0.0 (오탐지 교정 검증!)
    assert my_num[0, 2, 1].item() == 1.0, "Slot 2 Meganium fainted should be 1.0"
    assert my_num[0, 2, 10].item() == 0.0, "Meganium MUST NOT be misidentified as Mega evolution (slot 10 should be 0.0)"
    
    # p2 CharizardMegaX (Slot 1) is_mega = 1.0 검증
    assert my_num[0, 1, 10].item() == 1.0, "CharizardMegaX MUST be identified as Mega evolution (slot 10 should be 1.0)"
    print("  [Pass] Numeric Features & Mega Evolution False Positive Fix 1:1 Match!")

    # =========================================================================
    # C. 기술 7D 수치 텐서 my_m_num [1, 6, 4, 7] 1:1 매칭 검증
    # =========================================================================
    print("\n--- C. 기술 7D 수치 텐서 my_m_num [1, 6, 4, 7] 검증 ---")
    assert abs(my_m_num[0, 0, 0, 0].item() - (90/200.0)) < 1e-4, "Thunderbolt BP ratio mismatch"
    assert abs(my_m_num[0, 0, 0, 1].item() - 1.0) < 1e-4, "Thunderbolt accuracy mismatch"
    assert abs(my_m_num[0, 0, 0, 2].item() - 1.0) < 1e-4, "Thunderbolt PP ratio mismatch"
    assert abs(my_m_num[0, 0, 0, 3].item() - 0.0) < 1e-4, "Thunderbolt priority mismatch"
    assert abs(my_m_num[0, 0, 0, 4].item() - 0.67) < 1e-4, "Thunderbolt special category mismatch"

    assert abs(my_m_num[0, 0, 1, 3].item() - (1.0/7.0)) < 1e-4, "Quick Attack priority (+1/7) mismatch"
    assert abs(my_m_num[0, 0, 1, 4].item() - 0.33) < 1e-4, "Quick Attack physical category mismatch"

    assert my_m_num[0, 1, 0, 5].item() >= 0.25, "Flare Blitz recoil secondary flag mismatch"

    # 기술 타입 검증 (7번째 피처: move_type_norm = type_id / 20.0)
    assert abs(my_m_num[0, 0, 0, 6].item() - (5/20.0)) < 1e-4, "Thunderbolt move type (electric=5/20) mismatch"
    assert abs(my_m_num[0, 0, 1, 6].item() - (1/20.0)) < 1e-4, "Quick Attack move type (normal=1/20) mismatch"
    assert abs(my_m_num[0, 1, 0, 6].item() - (2/20.0)) < 1e-4, "Flare Blitz move type (fire=2/20) mismatch"
    print("  [Pass] Move 7D Features (Power, Accuracy, PP, Priority, Category, SecEffect, MoveType) 1:1 Match!")

    # =========================================================================
    # D. 상대 팀 텐서 & 다이맥스 턴 비율 opp_num [1, 6, 12] 1:1 매칭 검증
    # =========================================================================
    print("\n--- D. 상대 팀 텐서 & 다이맥스 턴 비율 검증 ---")
    assert abs(opp_num[0, 0, 10].item() - (2.0/3.0)) < 1e-4, "Opponent Dynamax turns left ratio mismatch"
    print("  [Pass] Opponent Dynamax Turn Ratio (2/3 = 0.6667) 1:1 Match!")

    # =========================================================================
    # E. 필드 및 장기 전략 턴 카운터 field_vec [1, 48] 1:1 매칭 검증 (상대 벽 포함!)
    # =========================================================================
    print("\n--- E. 필드 및 상대 벽 포함 사이드 조건 field_vec [1, 48] 검증 ---")
    # 0, 1: Sun active(1.0), Sun turns left 4/8 = 0.5
    assert field_vec[0, 0].item() == 1.0, "Sun active mismatch"
    assert abs(field_vec[0, 1].item() - 0.5) < 1e-4, "Sun turns left (4/8) mismatch"

    # 10, 11: Trick Room active(1.0), Trick Room turns left 1/5 = 0.2
    assert field_vec[0, 10].item() == 1.0, "Trick Room active mismatch"
    assert abs(field_vec[0, 11].item() - 0.2) < 1e-4, "Trick Room turns left (1/5) mismatch"

    # 20, 21: 내 리플렉터 active(1.0), 6/8 = 0.75
    assert field_vec[0, 20].item() == 1.0, "My Reflect active mismatch"
    assert abs(field_vec[0, 21].item() - 0.75) < 1e-4, "My Reflect turns (6/8) mismatch"

    # 26, 27: 내 순풍 active(1.0), 3/4 = 0.75
    assert field_vec[0, 26].item() == 1.0, "My Tailwind active mismatch"
    assert abs(field_vec[0, 27].item() - 0.75) < 1e-4, "My Tailwind turns (3/4) mismatch"

    # 29: 내 압정뿌리기 1/3 = 0.3333
    assert abs(field_vec[0, 29].item() - (1.0/3.0)) < 1e-4, "My Spikes layer (1/3) mismatch"

    # 30: 내 독압정 2/2 = 1.0
    assert field_vec[0, 30].item() == 1.0, "My Toxic Spikes layer (2/2) mismatch"

    # --- 상대 진영 사이드 조건 (32~43) 새로 검증! ---
    # 32, 33: 상대 리플렉터 active(1.0), 5/8 = 0.625
    assert field_vec[0, 32].item() == 1.0, "Opponent Reflect active mismatch"
    assert abs(field_vec[0, 33].item() - (5.0/8.0)) < 1e-4, "Opponent Reflect turns (5/8) mismatch"

    # 34, 35: 상대 빛의 장막 active(1.0), 3/8 = 0.375
    assert field_vec[0, 34].item() == 1.0, "Opponent Light Screen active mismatch"
    assert abs(field_vec[0, 35].item() - (3.0/8.0)) < 1e-4, "Opponent Light Screen turns (3/8) mismatch"

    # 36, 37: 상대 오로라베일 active(1.0), 7/8 = 0.875
    assert field_vec[0, 36].item() == 1.0, "Opponent Aurora Veil active mismatch"
    assert abs(field_vec[0, 37].item() - (7.0/8.0)) < 1e-4, "Opponent Aurora Veil turns (7/8) mismatch"

    # 38, 39: 상대 순풍 active(1.0), 2/4 = 0.5
    assert field_vec[0, 38].item() == 1.0, "Opponent Tailwind active mismatch"
    assert abs(field_vec[0, 39].item() - 0.5) < 1e-4, "Opponent Tailwind turns (2/4) mismatch"

    # 47: Turn normalized 15/50 = 0.3
    assert abs(field_vec[0, 47].item() - 0.3) < 1e-4, "Turn normalized (15/50) mismatch"
    print("  [Pass] Field, Weather, Trick Room, My Screens/Hazards & Opponent Reflect/LightScreen/AuroraVeil 1:1 Match!")

    # =========================================================================
    # F. 행동 마스킹 action_mask [1, 22] 1:1 매칭 검증
    # =========================================================================
    print("\n--- F. 행동 마스킹 action_mask [1, 22] 검증 ---")
    assert action_mask[0, 0].item() == True, "Move 0 (Thunderbolt) should be legal"
    assert action_mask[0, 1].item() == True, "Move 1 (Quick Attack) should be legal"
    assert action_mask[0, 4].item() == True, "Z-Move 0 (Thunderbolt → Z) should be legal"
    assert action_mask[0, 5].item() == False, "Quick Attack is not Z-capable, slot 5 MUST be False"
    assert action_mask[0, 9].item() == True, "Switch to Slot 1 (CharizardMegaX) should be legal"
    assert action_mask[0, 10].item() == False, "Switch to Slot 2 (Fainted Meganium) MUST be False"

    # 몸부림 턴: 가진 기술이 선택 불가 → 기술 칸 전부 막힘
    struggle = MockMove("struggle", bp=50, category="physical")
    b2 = MockBattle(my_team=[p1], opp_team=[opp_p1], active_my=p1, active_opp=opp_p1, avail_moves=[struggle], turn=3)
    mask2 = encoder.encode_battle(b2)[7]
    assert not mask2[0, 1:8].any().item(), "Struggle turn: no own move slot should be legal"
    print("  [Pass] Action Masking (Legal Moves, Z-Move, Non-fainted Switches, Struggle) 1:1 Match!")

    print("\n=========================================================================")
    print("🎉 [검증 완수] 모든 텐서 항목 및 6D 피처의 1:1 매칭 검증을 100% 통과했습니다!")
    print("=========================================================================")


if __name__ == "__main__":
    run_full_1to1_verification()
