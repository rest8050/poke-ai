"""팀 풀 JSON → Foul Play 팀 폴더(foul-play/fp/teams/teams/gen9/<이름>/team_XXXX). Foul Play가 자기 팀 폼 변화를 못 찾는 종은 제외.
사용: python tools/export_fp_teams.py data/team_pool_train_a.json train_a"""
import json
import os
import re
import shutil
import sys

EXCLUDE = {"keldeo"}  # 전투 중 폼이 바뀌어 "Could not find X in team_dict"로 Foul Play가 죽는 종 (실제로 관측된 것만)
pool, name = sys.argv[1], sys.argv[2]
root = os.path.join("..", "foul-play", "fp", "teams", "teams", "gen9", name)
teams = json.load(open(pool, encoding="utf-8-sig"))["teams"]
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root)
kept = 0
for t in teams:
    species = {re.sub(r"[^a-z0-9]", "", b.split("\n")[0].split(" @ ")[0].lower()) for b in t["export"].strip().split("\n\n")}
    if any(sp.startswith(e) for sp in species for e in EXCLUDE):
        continue
    with open(os.path.join(root, f"team_{kept:04d}"), "w", encoding="utf-8", newline="\n") as f:
        f.write(t["export"].strip() + "\n")
    kept += 1
print(f"{name}: {kept}/{len(teams)}팀 → {os.path.abspath(root)}")
