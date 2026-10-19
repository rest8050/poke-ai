"""build_fp_dataset.to_target: 정책 그라디언트용 action_taken(실제로 둔 수의 22칸 인덱스) 추출이 정확한지 확인."""
import numpy as np

from src.training.build_fp_dataset import load_jsonl, to_target


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


def test_load_jsonl_reads_plain_and_gzip(tmp_path=None):
    import gzip, json, tempfile, os
    d = tempfile.mkdtemp()
    recs = [{"type": "start", "tag": "a"}, {"type": "end", "tag": "a"}]
    with open(os.path.join(d, "dec_x_00.jsonl"), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + chr(10) for r in recs))
    with gzip.open(os.path.join(d, "dec_x_01.jsonl.gz"), "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + chr(10) for r in recs))
    got = list(load_jsonl(os.path.join(d, "dec_x_*.jsonl")))             # 같은 패턴으로 평문과 .gz를 둘 다 읽음
    assert len(got) == 4 and {r["_src"].split(os.sep)[-1] for r in got} == {"dec_x_00.jsonl", "dec_x_01.jsonl.gz"}


if __name__ == "__main__":
    test_load_jsonl_reads_plain_and_gzip()
    test_move_choice_gives_matching_index()
    test_tera_move_choice_offset_by_4()
    test_switch_choice_index()
    test_unmatched_choice_gives_none()
    test_choice_illegal_in_mask_gives_none()
    test_no_choice_arg_returns_none()
    print("ok")
