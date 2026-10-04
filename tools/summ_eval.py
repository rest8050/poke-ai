import json, glob, os, sys
label, pool = sys.argv[1], sys.argv[2]
d = sorted(glob.glob("logs/analysis_*"), key=os.path.getmtime)[-1]
s = json.load(open(d + "/summary.json", encoding="utf-8"))
print(f"{label} | {pool} | {s['wins']}/{s['battles']} = {s['win_rate']} {s['win_rate_95ci']} | {os.path.basename(d.replace(chr(92), '/'))}")
