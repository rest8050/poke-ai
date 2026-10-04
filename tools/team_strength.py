"""서버 로스터 팀 하나를 고정해 같은 모델끼리 대전: A=고정 팀, B=기준 풀 무작위 → A 승률이 그 팀의 상대적 강도.
사용: team_strength.py <ckpt> <로스터 JSON> <팀 인덱스> <판 수> <태그(영숫자 5자 이하)> [기준 풀 JSON]  (Showdown :8000 필요)"""
import asyncio, json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from tools.h2h import make


async def main(ckpt, roster, idx, n, tag, ref_path):
    team = json.load(open(roster, encoding="utf-8-sig"))["teams"][idx]
    ref = [t["export"] for t in json.load(open(ref_path, encoding="utf-8-sig"))["teams"]]
    pa, pb = make(ckpt, f"s{tag}a", [team["export"]]), make(ckpt, f"s{tag}b", ref)
    await pa.battle_against(pb, n_battles=n)
    wa, wb = pa.n_won_battles, pb.n_won_battles
    print(f"RESULT {team['name']} {wa} {wb} {pa.n_finished_battles - wa - wb}", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
                     sys.argv[6] if len(sys.argv) > 6 else os.path.join(ROOT, "data", "team_pool_metamon_holdout.json")))
