"""AI가 잘 쓰는 강한 팀 선별: 후보 팀마다 '그 팀 하나로 고정한 AI' 대 '래더 팀 풀의 무작위 AI'를 붙여 승률(=팀+운용 실력)을 잰 뒤,
단계마다 판 수를 늘리며 상위만 남기는 successive halving. 같은 모델이 양쪽을 맡음.
사용: python tools/team_screen.py run [--model v1_full_r3] [--n0 400] [--k 24,80,160] [--keep 72,24] [--port 8000] [--workers 5]
  단계별 결과: results/team_screen/stage<i>.jsonl (한 팀 끝날 때마다 한 줄 → 중간에 끊어도 같은 명령으로 이어서 진행)
  최종: data/team_pool_strong.json (마지막 단계 승률 순, 종이 겹치는 팀은 제외) + results/team_screen/final.txt
선별 점수 = 승수에 사전(평균 45%, 가중 20판)을 더한 사후평균(적은 판 수로 우연히 높은 팀이 올라오는 걸 줄임)
최종 승률은 마지막 단계의 새 판만으로 보고(선별에 안 쓰인 판이라 과대평가가 없음)"""
import argparse
import asyncio
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from tools.arena import port_open, resolve, watchdog

D = os.environ.get("TEAM_SCREEN_DIR", "results/team_screen")
OPP_POOL = "data/team_pool_metamon_holdout.json"
toid = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def species_of(exp):
    out = set()
    for blk in exp.strip().split("\n\n"):
        first = re.sub(r"\s*\((M|F)\)\s*$", "", blk.strip().split("\n")[0].split(" @")[0].strip())
        m = re.search(r"\(([^()]+)\)$", first)
        out.add(toid(m.group(1) if m else first))
    return out


def read_stage(i):
    path = f"{D}/stage{i}.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def score(w, n):
    return (w + 9) / (n + 20)   # 사전 평균 45%, 가중 20판


async def worker(a):
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.concurrency import POKE_LOOP
    from src.core.model import model_from_ckpt
    from src.evaluation.analyze_model import LoggingPlayer
    from src.core.team_pool import RandomPoolTeambuilder

    teams = json.load(open(a.cands, encoding="utf-8"))["teams"][a.idx::a.workers]
    opp = [t["export"] for t in json.load(open(OPP_POOL, encoding="utf-8-sig"))["teams"]]
    model = model_from_ckpt(a.ckpt).eval()
    server = ServerConfiguration(f"ws://localhost:{a.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
    out = f"{D}/stage{a.stage}.jsonl"
    done = {r["team"] for r in read_stage(a.stage)}
    voided = set()

    class Player(LoggingPlayer):
        res = None

        def _battle_finished_callback(self, battle):
            tag = battle.battle_tag
            self.turn_logs.pop(tag, None)
            self.history.pop(tag, None)
            if self.res is not None:
                self.res.append("void" if tag in voided else "draw" if battle.won is None else "w" if battle.won else "l")

    for n, t in enumerate(teams):
        if t["name"] in done:
            continue

        def make(suffix, exports):
            p = Player(mode="argmax", model=model, search_lambda=None, battle_format="gen9ou", server_configuration=server,
                       account_configuration=AccountConfiguration(f"s{a.stage}{a.idx}x{n % 1000}{suffix}", None), max_concurrent_battles=a.concurrent)
            p._team = RandomPoolTeambuilder(exports)
            return p
        pa, pb = make("a", [t["export"]]), make("b", opp)
        pa.res = []
        watch = asyncio.run_coroutine_threadsafe(watchdog(pa, a.max_battle_sec, voided), POKE_LOOP)
        await pa.battle_against(pb, n_battles=a.k)
        watch.cancel()
        r = pa.res
        line = json.dumps({"stage": a.stage, "team": t["name"], "w": r.count("w"), "l": r.count("l"), "void": r.count("void"), "t": time.time()}) + "\n"
        fd = os.open(out, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
        os.write(fd, line.encode()); os.close(fd)
        for p in (pa, pb):   # 팀마다 접속이 쌓이지 않게 정리
            try:
                asyncio.run_coroutine_threadsafe(p.ps_client.stop_listening(), POKE_LOOP).result(5)
            except Exception:
                pass


def run_stage(a, stage, cands_path, k):
    todo = len(json.load(open(cands_path, encoding="utf-8"))["teams"])
    base = [sys.executable, os.path.abspath(__file__), "--worker", "--stage", str(stage), "--cands", cands_path, "--k", str(k), "--ckpt", a.ckpt,
            "--port", str(a.port), "--workers", str(a.workers), "--concurrent", str(a.concurrent), "--max-battle-sec", str(a.max_battle_sec)]
    procs = [subprocess.Popen(base + ["--idx", str(i)], stdout=open(f"logs/team_screen_s{stage}_{i}.log", "w"), stderr=subprocess.STDOUT)
             for i in range(a.workers)]
    t0, last = time.time(), 0
    while any(p.poll() is None for p in procs):
        time.sleep(10)
        if time.time() - last > 120:
            last = time.time()
            print(f"  [단계 {stage}] {len(read_stage(stage))}/{todo}팀 완료 ({(time.time() - t0) / 60:.0f}분)", flush=True)
    if len(read_stage(stage)) < todo:
        print(f"  경고: 단계 {stage}에서 {todo - len(read_stage(stage))}팀이 끝나지 않음 (logs/team_screen_s{stage}_*.log). 같은 명령으로 이어서 진행 가능", flush=True)


def controller(a):
    a.ckpt = resolve(a.model)
    ks, keeps = [int(x) for x in a.k.split(",")], [int(x) for x in a.keep.split(",")]
    assert len(ks) == len(keeps) + 1
    os.makedirs(D, exist_ok=True)
    sd = None
    if not port_open(a.port):
        sd = subprocess.Popen([shutil.which("node"), "pokemon-showdown", "start", "--no-security", str(a.port)], cwd="pokemon-showdown",
                              stdout=open("logs/showdown_team_screen.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if port_open(a.port):
                break
            time.sleep(1)
    try:
        c0 = f"{D}/cands0.json"
        if not os.path.exists(c0):   # 후보: 래더 팀 풀에서 무작위(시드 고정). 평가용 상대 풀(holdout)과 겹치는 팀은 제외
            pool = json.load(open("data/team_pool_metamon.json", encoding="utf-8-sig"))["teams"]
            random.Random(0).shuffle(pool)
            json.dump({"teams": pool[:a.n0]}, open(c0, "w", encoding="utf-8"), ensure_ascii=False)
        cands = c0
        for s, k in enumerate(ks):
            print(f"== 단계 {s}: 후보 {len(json.load(open(cands, encoding='utf-8'))['teams'])}팀 x {k}판 ({a.model})", flush=True)
            run_stage(a, s, cands, k)
            if s < len(keeps):
                allr = {}
                for st in range(s + 1):
                    for r in read_stage(st):
                        w, n = allr.get(r["team"], (0, 0))
                        allr[r["team"]] = (w + r["w"], n + r["w"] + r["l"])
                cur = {t["name"]: t for t in json.load(open(cands, encoding="utf-8"))["teams"]}
                ranked = sorted((n for n in cur if n in allr), key=lambda n: -score(*allr[n]))[:keeps[s]]
                cands = f"{D}/cands{s + 1}.json"
                json.dump({"teams": [cur[n] for n in ranked]}, open(cands, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"   상위 {len(ranked)}팀 통과 (사후평균 컷 {score(*allr[ranked[-1]]):.1%})", flush=True)
        final(a, cands)
    finally:
        if sd:
            sd.terminate()


def final(a, cands):
    last = {r["team"]: r for r in read_stage(len(a.k.split(",")) - 1)}
    cur = {t["name"]: t for t in json.load(open(cands, encoding="utf-8"))["teams"]}
    allw = {}
    for st in range(len(a.k.split(","))):
        for r in read_stage(st):
            w, n = allw.get(r["team"], (0, 0)); allw[r["team"]] = (w + r["w"], n + r["w"] + r["l"])
    rows = sorted((n for n in cur if n in last), key=lambda n: -last[n]["w"] / max(1, last[n]["w"] + last[n]["l"]))
    chosen, lines = [], []
    for n in rows:
        sp = species_of(cur[n]["export"])
        dup = any(len(sp & species_of(c["export"])) > 3 for c in chosen)
        r = last[n]; games = r["w"] + r["l"]
        lines.append(f"{'제외(종 겹침)' if dup else '선정':10s} {n:14s} 마지막 단계 {r['w']}/{games} = {r['w'] / max(1, games):5.1%} | 누적 {allw[n][0]}/{allw[n][1]} = {allw[n][0] / max(1, allw[n][1]):5.1%} | {', '.join(sorted(sp))}")
        if not dup:
            chosen.append(cur[n])
    json.dump({"source": "tools/team_screen.py", "format": "gen9ou", "model": a.model, "teams": chosen}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    open(f"{D}/final.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"\n최종 {len(chosen)}팀 -> {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--model", default="v1_full_r3"); ap.add_argument("--n0", type=int, default=400)
    ap.add_argument("--k", default="24,80,160"); ap.add_argument("--keep", default="72,24")
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--concurrent", type=int, default=12); ap.add_argument("--max-battle-sec", type=int, default=900)
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--stage", type=int); ap.add_argument("--cands")
    ap.add_argument("--idx", type=int); ap.add_argument("--ckpt"); ap.add_argument("--out", default="data/team_pool_strong.json")
    a = ap.parse_args()
    if a.worker:
        a.k = int(a.k)
        asyncio.run(worker(a))
    else:
        controller(a)
