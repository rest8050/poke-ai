"""build_fp_dataset.to_target: 정책 그라디언트용 action_taken(실제로 둔 수의 22칸 인덱스) 추출이 정확한지 확인."""
import numpy as np

from src.training.build_fp_dataset import OPP_TOP, _VOCAB, opp_labels, to_target


def _state(moves, team, mask_all=True):
    return {"moves": moves, "team_species": team, "mask": np.ones(22, dtype=bool) if mask_all else np.zeros(22, dtype=bool)}


def test_move_choice_gives_matching_index():
    st = _state(["thunderbolt", "surf", "icebeam", "toxic"], ["pikachu", "gyarados"])
    policy = [("thunderbolt", 0.6), ("surf", 0.4)]
    target, matched, action_idx = to_target(st, policy, 0.2, choice="thunderbolt")
    assert action_idx == 0  # 1번째 기술 슬롯
    assert target is not None and matched > 0.9


def test_tera_move_choice_offset_by_4():
    st = _state(["thunderbolt", "surf", "icebeam", "toxic"], ["pikachu"])
    policy = [("thunderbolt", 1.0)]
    _, _, action_idx = to_target(st, policy, 0.2, choice="thunderbolt-tera")
    assert action_idx == 4  # 테라 슬롯은 +4


def test_switch_choice_index():
    st = _state(["thunderbolt"], ["pikachu", "gyarados", "charizard"])
    policy = [("thunderbolt", 1.0)]
    _, _, action_idx = to_target(st, policy, 0.2, choice="switch charizard")
    assert action_idx == 8 + 2  # 세 번째 팀원 = 교체 슬롯 10


def test_unmatched_choice_gives_none():
    st = _state(["thunderbolt"], ["pikachu"])
    policy = [("thunderbolt", 1.0)]
    _, _, action_idx = to_target(st, policy, 0.2, choice="earthquake")  # 이 포켓몬 기술 아님
    assert action_idx is None


def test_choice_illegal_in_mask_gives_none():
    st = _state(["thunderbolt", "surf"], ["pikachu"], mask_all=False)  # 전부 불법
    policy = [("thunderbolt", 1.0)]
    _, _, action_idx = to_target(st, policy, 0.2, choice="thunderbolt")
    assert action_idx is None


def test_no_choice_arg_returns_none():
    st = _state(["thunderbolt"], ["pikachu"])
    _, _, action_idx = to_target(st, [("thunderbolt", 1.0)], 0.2)
    assert action_idx is None


def _opp_cat(species):
    cat = np.zeros((6, 11), np.int64)
    for k, sp in enumerate(species):
        cat[k, 10] = _VOCAB["species"][sp]
    return cat


def test_opp_labels_masses_and_slots():
    cat = _opp_cat(["gliscor", "blissey", "clefable", "toxapex", "kingambit", "dondozo"])
    policy = [("protect", 0.5), ("roost-tera", 0.2), ("switch dondozo", 0.2), ("switch kingambit", 0.1), ("spite", 0.01)]   # spite는 최고의 20% 미만이라 버려짐
    sw, tera, ids, ps, tgt = opp_labels(policy, 0.2, cat)
    assert abs(sw - 0.3 / 1.0) < 1e-6 and abs(tera - 0.2) < 1e-6              # 가중치 합 1.0
    assert tgt[5] > tgt[4] > 0 and abs(tgt.sum() - sw) < 1e-6                 # 대상 질량 = 교체 질량, 슬롯 6 = dondozo
    assert ids.shape == (OPP_TOP,) and abs(ps.sum() - 0.7) < 1e-6             # 기술 질량 = 1 - 교체
    assert ids[0] == _VOCAB["move"]["protect"] and ids[1] == _VOCAB["move"]["roost"]     # 테라 접미사는 같은 기술로 합침


def test_opp_labels_form_prefix_match():
    cat = _opp_cat(["sinistchamasterpiece", "blissey", "clefable", "toxapex", "kingambit", "dondozo"])
    _, _, _, _, tgt = opp_labels([("switch sinistcha", 1.0)], 0.2, cat)
    assert tgt[0] == 1.0


if __name__ == "__main__":
    test_move_choice_gives_matching_index()
    test_tera_move_choice_offset_by_4()
    test_switch_choice_index()
    test_unmatched_choice_gives_none()
    test_choice_illegal_in_mask_gives_none()
    test_no_choice_arg_returns_none()
    test_opp_labels_masses_and_slots()
    test_opp_labels_form_prefix_match()
    print("ok")
