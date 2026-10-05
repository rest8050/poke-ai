"""타워 층별 트레이너 JSON 생성: team_screen 결과에서 '폼 없음/지역폼만' 팀 중 승률 상위를 골라 층 순서(아래=약, 위=강)로 배정.
사용: python tools/make_tower_trainers.py <출력 디렉터리> [층 수=8] [최대 겹침=2]
- 지역폼은 RCT JSON의 "aspects": ["hisuian"|"galarian"|"alolan"|"paldean"]로 지정 (서버 팩 트레이너가 같은 방식을 씀)
- 그 외 폼(가면 오거폰·령수폼 등)은 서버에서 재현이 검증되지 않아 제외
- 트레이너 id는 타워에 이미 서 있는 개체의 TrainerId를 그대로 써서(개체는 id만 저장, 팀은 데이터팩에서 읽음) 개체를 새로 소환하지 않아도 됨"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_trainers_from_pool import convert_pokemon, parse_export   # noqa: E402

FLOOR_IDS = ["main_1", "main_3", "main_5", "rare_1", "rare_3", "rare_5", "rain", "sun"]   # y=76,86,...,146 (아래→위)
REGIONAL = {"Alola": "alolan", "Galar": "galarian", "Hisui": "hisuian", "Paldea": "paldean"}
toid = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def convert_with_aspects(p):
    out = convert_pokemon(p, [])
    if p.get("forme") in REGIONAL:
        out["aspects"] = [REGIONAL[p["forme"]]]
    return out


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    floors = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    cands = {t["name"]: t for t in json.load(open("results/team_screen/cands2.json", encoding="utf-8"))["teams"]}
    last = {r["team"]: r for r in map(json.loads, open("results/team_screen/stage2.jsonl", encoding="utf-8"))}
    ranked = sorted(((last[n]["w"] / (last[n]["w"] + last[n]["l"]), n) for n in cands if n in last), reverse=True)
    chosen = []
    for rate, n in ranked:
        if rate <= 0.5:
            break
        mons = parse_export(cands[n]["export"])
        if any(p.get("forme") and p["forme"] not in REGIONAL for p in mons):
            continue
        sp = {toid(p["species"]) for p in mons}
        if any(len(sp & c[3]) > max_overlap for c in chosen):
            continue
        chosen.append((rate, n, mons, sp))
        if len(chosen) == floors:
            break
    chosen.sort(key=lambda c: c[0])   # 아래 층 = 약한 쪽
    for fid, (rate, n, mons, _) in zip(FLOOR_IDS, chosen):
        trainer = {"name": f"strong {n}", "ai": {"type": "pokeai", "data": {"verbose": True}}, "battleRules": {"maxItemUses": 0},
                   "bag": [], "battleFormat": "GEN_9_SINGLES", "team": [convert_with_aspects(p) for p in mons]}
        (out_dir / f"{fid}.json").write_text(json.dumps(trainer, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{fid:7s} <- {n} ({rate:.0%}) {[p['species'] for p in mons]}")


if __name__ == "__main__":
    main()
