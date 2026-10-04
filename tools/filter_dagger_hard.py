"""record_games.py가 남긴 jsonl에서 "불리했던 판"만 추려냄: 학생 자신의 value가 threshold 이하로
min_consecutive턴 이상 연속인 구간이 있는 판만 통과. 통과한 판은 replay_label.py에 그대로 넣을 수 있음
(같은 스키마: tag/user/export/frames, values는 제거).
사용: python tools/filter_dagger_hard.py <in.jsonl> <out.jsonl> [threshold=-0.2] [min_consecutive=3]
"""
import json
import sys


def has_hard_stretch(values, threshold, min_consecutive):
    run = 0
    for v in values:
        run = run + 1 if v <= threshold else 0
        if run >= min_consecutive:
            return True
    return False


def main(in_path, out_path, threshold, min_consecutive):
    total = kept = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            row = json.loads(line)
            if has_hard_stretch(row.get("values", []), threshold, min_consecutive):
                row.pop("values", None)
                fout.write(json.dumps(row) + "\n")
                kept += 1
    print(f"{in_path}: {total}판 중 {kept}판 통과(threshold={threshold}, 연속 {min_consecutive}턴) -> {out_path}")


def _test():
    assert has_hard_stretch([0.1, -0.3, -0.3, -0.3, 0.2], -0.2, 3)          # 정확히 3연속 -> 통과
    assert not has_hard_stretch([0.1, -0.3, -0.3, 0.2, -0.3], -0.2, 3)      # 연속 끊김 -> 탈락
    assert not has_hard_stretch([-0.3, -0.3], -0.2, 3)                     # 2턴뿐이라 부족 -> 탈락
    assert has_hard_stretch([-0.2, -0.2, -0.2], -0.2, 3)                   # 경계값(<=) -> 통과
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _test()
    else:
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else -0.2
        min_consecutive = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        main(sys.argv[1], sys.argv[2], threshold, min_consecutive)
