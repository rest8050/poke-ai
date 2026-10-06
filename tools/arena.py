"""후보 대 챔피언 순차 검정(SPRT) 대전: 판이 끝날 때마다 results/ledger.jsonl에 한 줄씩 쓰고, 결론이 나는 즉시 멈춤.
사용: python tools/arena.py <후보> <챔피언> [--delta 0.03] [--alpha 0.05] [--beta 0.05] [--cap 10000] [--workers 6] [--port 8000]
  후보/챔피언 = 체크포인트 경로 또는 이름(v1_full_r3 -> checkpoints/supervised_v2_fp_v1_full_r3.pt)
  풀(holdout/rare/randomset)을 작업자에 균등 배정하고, 판정은 풀별 판 수를 맞춘 데이터로 함. Showdown이 안 떠 있으면 직접 띄우고 끝나면 내림
  판당 제한 시간(--max-battle-sec) 넘기면 강제 종료하고 'void'로 기록(집계 제외) — 멈춘 판 하나가 전체를 붙잡지 않게
종료 코드: 0=낫다 1=나쁘다 2=차이 없음 3=상한 도달/비정상 종료"""
import argparse
import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from tools.ledger import LEDGER, append, balanced, model_id, read
from tools.sprt import NAMES, decide, wilson

POOLS = "holdout=data/team_pool_metamon_holdout.json,rare=data/team_pool_rare.json,randomset=data/team_pool_randomset.json"
EXIT = {"better": 0, "worse": 1, "equal": 2, "continue": 3}


def resolve(x):
    return x if os.path.exists(x) else f"checkpoints/supervised_v2_fp_{x}.pt"


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def watchdog(player, limit, voided):
    first = {}
    while True:
        await asyncio.sleep(min(15, max(1, limit / 3)))
        now = time.time()
        for tag, b in list(player.battles.items()):
            if b.finished:
                continue
            if now - first.setdefault(tag, now) > limit and tag not in voided:
                voided.add(tag)
                await player.ps_client.send_message("/forfeit", tag)


async def worker(a):
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.concurrency import POKE_LOOP
    import json
    from src.core.model import model_from_ckpt
    from src.evaluation.analyze_model import LoggingPlayer
    from src.core.team_pool import RandomPoolTeambuilder

    pool_name, pool_path = a.pool.split("=", 1)
    exports = [t["export"] for t in json.load(open(pool_path, encoding="utf-8-sig"))["teams"]]
    ma, mb = model_from_ckpt(a.a_path).eval(), model_from_ckpt(a.b_path).eval()
    (na, ha), (nb, hb) = model_id(a.a_path), model_id(a.b_path)
    server = ServerConfiguration(f"ws://localhost:{a.port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")
    voided = set()

    class Player(LoggingPlayer):
        recorder = False

        def _battle_finished_callback(self, battle):
            tag = battle.battle_tag
            self.turn_logs.pop(tag, None)   # 판마다 쌓이는 메모리 정리
            self.history.pop(tag, None)
            if self.recorder:
                winner = "void" if tag in voided else "draw" if battle.won is None else "a" if battle.won else "b"
                append({"kind": "game", "run": a.run, "t": time.time(), "a": na, "a_hash": ha, "b": nb, "b_hash": hb,
                        "pool": pool_name, "winner": winner, "turns": battle.turn, "tag": tag}, a.ledger)

    chunk = 0
    while True:   # 일정 판 수마다 플레이어를 새로 만들어 메모리를 제한 (poke_env가 끝난 판을 계속 들고 있음)
        chunk += 1

        def make(model, suffix):
            p = Player(mode="argmax", model=model, search_lambda=None, battle_format="gen9ou", server_configuration=server,
                       account_configuration=AccountConfiguration(f"w{a.run}{a.idx}{chunk % 100}{suffix}", None),
                       max_concurrent_battles=a.concurrent)
            p._team = RandomPoolTeambuilder(exports)
            return p
        pa, pb = make(ma, "a"), make(mb, "b")
        pa.recorder = True
        watch = asyncio.run_coroutine_threadsafe(watchdog(pa, a.max_battle_sec, voided), POKE_LOOP)
        await pa.battle_against(pb, n_battles=a.chunk)
        watch.cancel()


def controller(a):
    a.a_path, a.b_path = resolve(a.a), resolve(a.b)
    (na, ha), (nb, hb) = model_id(a.a_path), model_id(a.b_path)
    run = uuid.uuid4().hex[:4]
    runs = {run, a.resume} if a.resume else {run}   # --resume: 중단된 같은 검정의 기록을 이어서 셈 (이전 실행에서 결론이 안 난 경우)
    pools = POOLS.split(",")
    assert a.workers % len(pools) == 0, f"작업자 수는 풀 수({len(pools)})의 배수여야 풀이 균등해짐"
    sd = None
    if not port_open(a.port):
        sd = subprocess.Popen([shutil.which("node"), "pokemon-showdown", "start", "--no-security", str(a.port)], cwd="pokemon-showdown",
                              stdout=open(f"logs/showdown_arena_{run}.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if port_open(a.port):
                break
            time.sleep(1)
        else:
            sd.kill(); sys.exit("Showdown 시작 실패")
    print(f"[arena {run}] 후보 {na}({ha}) 대 챔피언 {nb}({hb}) | δ={a.delta} α={a.alpha} β={a.beta} 상한 {a.cap}판 | 작업자 {a.workers} x 동시 {a.concurrent}", flush=True)
    base = [sys.executable, os.path.abspath(__file__), "--worker", "--run", run, "--a-path", a.a_path, "--b-path", a.b_path, "--port", str(a.port),
            "--concurrent", str(a.concurrent), "--chunk", str(a.chunk), "--max-battle-sec", str(a.max_battle_sec), "--ledger", a.ledger]
    procs = [subprocess.Popen(base + ["--idx", str(i), "--pool", pools[i % len(pools)]], stdout=open(f"logs/arena_{run}_{i}.log", "w"),
                              stderr=subprocess.STDOUT) for i in range(a.workers)]
    t0, last_print, s, per = time.time(), 0, None, {}
    try:
        while True:
            time.sleep(5)
            rows = [r for r in read(a.ledger) if r.get("run") in runs and r["kind"] == "game"]
            w, l, per = balanced(rows)
            s = decide(w, l, a.delta, a.alpha, a.beta)
            played = sum(r["winner"] != "void" for r in rows)
            if time.time() - last_print > 30:
                last_print = time.time()
                print(f"[{(time.time() - t0) / 60:5.1f}분] 총 {played}판 (균형 {w + l}판: 후보 {w}승 {l}패, 무효 {len(rows) - played}) "
                      f"LLR +δ {s['up']:+.2f} / -δ {s['dn']:+.2f}  경계 {s['A']:.2f}/{s['B']:.2f}", flush=True)
            if s["decision"] != "continue" or played >= a.cap:
                break
            if len(rows) >= int(1.5 * a.cap) + 20:
                print("무효 판이 너무 많아 중단 (--max-battle-sec 확인)", flush=True)
                break
            if all(p.poll() is not None for p in procs):
                print("작업자가 모두 종료됨 (logs/arena_*.log 확인)", flush=True)
                break
    finally:
        for p in procs:
            p.terminate()
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()
        if sd:
            sd.terminate()
    lo, hi = wilson(s["wins"], s["n"]) if s else (0, 1)
    verdict = s["decision"] if s else "continue"
    print(f"\n== 결과: {NAMES[verdict] if verdict != 'continue' else '결론 없음(상한 도달/중단)'} | 후보 {na} {s['wins']}승 {s['losses']}패 "
          f"(균형 {s['n']}판) 승률 {s['wins'] / max(1, s['n']):.1%} [95% {lo:.1%}~{hi:.1%}]" if s else "결과 없음", flush=True)
    print("   풀별:", {p: f"{x}/{n}" for p, (x, n) in per.items()}, flush=True)
    append({"kind": "decision", "run": run, "t": time.time(), "a": na, "b": nb, "delta": a.delta, "alpha": a.alpha, "beta": a.beta,
            "decision": verdict, "wins": s["wins"] if s else 0, "losses": s["losses"] if s else 0}, a.ledger)
    sys.exit(EXIT[verdict])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?"); ap.add_argument("b", nargs="?")
    ap.add_argument("--delta", type=float, default=0.03); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05); ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--concurrent", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=60); ap.add_argument("--max-battle-sec", type=int, default=900)
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--resume", default=None); ap.add_argument("--worker", action="store_true"); ap.add_argument("--run"); ap.add_argument("--idx", type=int)
    ap.add_argument("--pool"); ap.add_argument("--a-path"); ap.add_argument("--b-path")
    args = ap.parse_args()
    if args.worker:
        asyncio.run(worker(args))
    else:
        controller(args)
