"""팀 풀(export 텍스트)에서 무작위로 팀을 뽑아 RCT 트레이너 json으로 변환.
파싱은 로컬 pokemon-showdown 패키지의 Teams.import()에 맡김(젠더/자연/성격/테라타입 등 자체 파싱 안 함).
사용: python tools/make_trainers_from_pool.py <pool.json> <n> <out_dir> <name_prefix> [seed]
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE_PARSE = r"""
const { Teams } = require(process.argv[1]);
const { Dex } = require(process.argv[2]);
const txt = require("fs").readFileSync(0, "utf-8");
const mons = Teams.import(txt);
for (const p of mons) {
    const s = Dex.species.get(p.species);
    p.baseSpecies = s.baseSpecies;   // 폼이면 기본종, 아니면(Ting-Lu 등 원래 이름에 -가 있는 경우) 자기 자신
    p.forme = s.forme;
}
console.log(JSON.stringify(mons));
"""


def parse_export(export_text: str):
    p = subprocess.run(
        ["node", "-e", NODE_PARSE,
         str(ROOT / "pokemon-showdown/dist/sim/teams.js"), str(ROOT / "pokemon-showdown/dist/sim/dex.js")],
        input=export_text, capture_output=True, text=True, cwd=ROOT,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout)


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


MEGA_SHOWDOWN_ITEMS = {"booster_energy", "wellspring_mask", "hearthflame_mask", "cornerstone_mask"}  # 서버의 MegaShowdown 모드가 제공(cobblemon 네임스페이스에는 없음)


def to_item_id(s: str) -> str:
    i = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return f"mega_showdown:{i}" if i in MEGA_SHOWDOWN_ITEMS else i


def convert_pokemon(p: dict, warnings: list):
    species_raw = p["species"]
    species = p["baseSpecies"]  # 폼이면 기본형(Dex.species.get().baseSpecies), Ting-Lu처럼 원래 이름에 -가 있으면 그대로
    if p.get("forme"):
        warnings.append(f"{species_raw} -> {species} (폼 차이 무시, 기본형으로)")
    out = {
        "moveset": [to_id(m) for m in p["moves"]],
        "ivs": {k: v for k, v in p["ivs"].items()},  # 31도 생략하지 않고 전부 명시 (RCT의 빈 ivs 기본값이 불명확해서)
        "species": to_id(species),
        "level": p.get("level", 100),
        "nature": p["nature"].lower(),
        "ability": to_id(p["ability"]),
    }
    evs = {k: v for k, v in p["evs"].items() if v}
    if evs:
        out["evs"] = evs
    if p.get("item"):
        out["heldItem"] = [to_item_id(p["item"])]
    gender = p.get("gender", "")
    if gender == "M":
        out["gender"] = "MALE"
    elif gender == "F":
        out["gender"] = "FEMALE"
    return out


def main():
    pool_path, n, out_dir, prefix = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
    rng = random.Random(seed)

    pool = json.load(open(pool_path, encoding="utf-8"))
    teams = pool["teams"]
    picked = rng.sample(teams, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    for i, t in enumerate(picked):
        mons = parse_export(t["export"])
        trainer = {
            "name": f"{prefix} {t['name']}",
            "ai": {"type": "pokeai", "data": {"verbose": True}},
            "battleRules": {"maxItemUses": 0},
            "bag": [],
            "battleFormat": "GEN_9_SINGLES",
            "team": [convert_pokemon(p, warnings) for p in mons],
        }
        out_path = out_dir / f"{prefix}_{i}.json"
        out_path.write_text(json.dumps(trainer, indent=2, ensure_ascii=False), encoding="utf-8")
        tera = [p.get("teraType") for p in mons]
        print(f"{out_path.name}: {t['name']} <- {[p['species'] for p in mons]}  (참고용 테라타입: {tera})")

    if warnings:
        print("\n주의(폼 단순화):")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
