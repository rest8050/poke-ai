"""체크포인트 이름 일괄 변경: 파일(+cfg/.last), 결과 장부의 모델 이름, 스크립트/문서/메모리의 참조를 한 번에.
사용: python tools/rename_models.py            # 계획만 출력 (아무것도 안 바꿈)
      python tools/rename_models.py --apply    # 적용 (장부는 results/ledger.jsonl.bak_before_rename에 백업)
이름 규칙: <구조>_<학습 규모>_<특징>  (구조 v1/ent/v3, 규모 pilot/full). 접두사 supervised_v2_fp_ 는 평가 도구가 파일을 찾는 규칙이라 유지"""
import glob
import json
import os
import re
import shutil
import sys

MAP = {"v10boot": "v1_full_boot", "v12r3": "v1_full_r3", "pilA": "v1_pilot", "pilB": "ent_pilot", "v3pil": "v3_pilot",
       "v3u": "v3_pilot_unrev", "v3lse": "v3_pilot_lse", "v3flat": "v3_pilot_flat", "v3full": "v3_full"}
PRE = "supervised_v2_fp_"
MEMORY = os.path.expanduser("~/.claude/projects/C--Users-lsh-Desktop-coble/memory")
TEXT = (glob.glob("logs/*.sh") + glob.glob("tools/*.py") + glob.glob("tools/*.sh") + ["tools/README.md", "README.md"] + glob.glob("tests/*.py")
        + glob.glob("deploy/*.sh") + glob.glob(os.path.join(MEMORY, "*.md")))
TEXT = [p for p in TEXT if os.path.isfile(p) and os.path.basename(p) != "rename_models.py"]
apply = "--apply" in sys.argv


def sub_text(s):
    n = 0
    for old, new in MAP.items():
        s, k = re.subn(rf"{PRE}{old}(?![A-Za-z0-9_])", PRE + new, s); n += k          # supervised_v2_fp_<old>.pt 형태
        s, k = re.subn(rf"(?<![A-Za-z0-9_]){old}(?![A-Za-z0-9_])", new, s); n += k   # 단독으로 쓰인 이름 (h2h_summary_v12r3_run1 같은 파일명 속 이름은 제외)
    return s, n


# 1) 체크포인트 파일
for old, new in MAP.items():
    for src in sorted(glob.glob(f"checkpoints/{PRE}{old}.pt*")):
        dst = src.replace(f"{PRE}{old}.pt", f"{PRE}{new}.pt", 1)
        print(f"파일  {os.path.basename(src)} -> {os.path.basename(dst)}")
        if apply:
            assert not os.path.exists(dst), dst
            os.rename(src, dst)
# 포인터 파일(로컬)
if os.path.exists("checkpoints/CURRENT"):
    t = open("checkpoints/CURRENT").read(); t2, n = sub_text(t)
    if n:
        print(f"포인터 checkpoints/CURRENT: {t.strip()} -> {t2.strip()}")
        if apply: open("checkpoints/CURRENT", "w").write(t2)
# 2) 장부
if os.path.exists("results/ledger.jsonl"):
    rows, changed = [], 0
    for line in open("results/ledger.jsonl", encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            rows.append(line.rstrip("\n")); continue
        for k in ("a", "b"):
            if r.get(k) in MAP:
                r[k] = MAP[r[k]]; changed += 1
        rows.append(json.dumps(r, ensure_ascii=False))
    print(f"장부  이름 필드 {changed}개 변경")
    if apply:
        shutil.copy("results/ledger.jsonl", "results/ledger.jsonl.bak_before_rename")
        open("results/ledger.jsonl", "w", encoding="utf-8").write("\n".join(rows) + "\n")
# 3) 스크립트/문서/메모리
for p in TEXT:
    s = open(p, encoding="utf-8").read(); s2, n = sub_text(s)
    if n:
        print(f"텍스트 {p}: {n}곳")
        if apply: open(p, "w", encoding="utf-8", newline="").write(s2)
print("적용 완료" if apply else "(계획만 출력 — --apply로 적용)")
