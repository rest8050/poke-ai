"""
모델 전체 로그 수집 + 패배 분석.

체크포인트로 N판을 돌리며 매 턴 결정(합법 행동별 확률, 가치, 상성, 테라)과
배틀 전체 프로토콜 로그, 상대의 실제 팀(세트 추론 정확도 검증용)을 저장하고 요약한다.

사용 예:
  python src/evaluation/analyze_model.py --battles 200 --opponent rct
  python src/evaluation/analyze_model.py --mode argmax --search 0.5 --replays
  python src/evaluation/analyze_model.py --pool data/team_pool_rare.json   # 기본은 Metamon 평가 전용 500팀(holdout), 낯선 파티는 rare/randomset

출력: logs/analysis_<시각>/
  turns.jsonl    턴별 결정 로그
  battles.jsonl  배틀별 결과/팀/선두/세트 추론 정확도/원본 프로토콜 로그
  summary.json   집계 (콘솔에도 출력)
  replays/       --replays 시 Showdown HTML 리플레이
"""
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir != os.path.dirname(_current_dir):
    if os.path.exists(os.path.join(_current_dir, "data", "vocab.json")):
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        break
    _current_dir = os.path.dirname(_current_dir)
import argparse
import asyncio
import json
import math
import multiprocessing as mp
import re
import time
import warnings
from collections import Counter, defaultdict

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder

from src.core.battle_eval import potential
from src.core.model import DeepPokemonBattleTransformerNet, load_compatible, zero_unseen_id_rows
from src.core.player import SmartPokemonPlayer
from src.core.rct_player import RCTBattleAIPlayer, _id, eff_move
from src.core.search import decode_action, search_pick
from src.core.set_prior import predict as predict_set
from src.core.team_pool import RandomPoolTeambuilder, load_team_pool

warnings.filterwarnings("ignore")
OPPONENTS = {"rct": RCTBattleAIPlayer, "heuristic": SimpleHeuristicsPlayer, "random": RandomPlayer}


def action_name(d):
    if d is None:
        return None
    if d[0] == "switch":
        return f"switch:{d[1].species}"
    _, move, z, tera = d
    return f"{move.id}{'+tera' if tera else ''}{'+z' if z else ''}"


def mon_state(mon, own):
    if mon is None:
        return None
    return {
        "species": mon.species, "hp": round(mon.current_hp_fraction, 3), "status": _id(mon.status) or None,
        "boosts": {k: v for k, v in mon.boosts.items() if v}, "types": [_id(t) for t in mon.types if t],
        "tera": _id(mon.tera_type) if mon.is_terastallized else None,
        "item": mon.item if own or mon.item != "unknown_item" else None,
        "ability": mon.ability,
    }


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def team_species(names):
    return frozenset(norm(n) for n in names)


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


class LoggingPlayer(SmartPokemonPlayer):
    def __init__(self, mode="sample", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.turn_logs = defaultdict(list)
        self.leads = {}

    def teampreview(self, battle):
        order = super().teampreview(battle)
        self.leads[battle.battle_tag] = list(battle.team.values())[int(order.split()[1][0]) - 1].species
        return order

    def choose_move(self, battle):
        tag = battle.battle_tag
        tensors = self.encoder.encode_battle(battle)
        mask = tensors[7][0]
        opp_num = tensors[4][0]
        policy_idx, probs, _, value, self.history[tag] = self.model.get_action(*tensors[:7], self.history.get(tag), tensors[7])
        probs = probs[0]
        if self.mode == "sample":
            policy_idx = int(torch.multinomial(probs, 1).item())
        chosen = policy_idx
        if self.search_lambda is not None:
            chosen = search_pick(battle, probs, self.search_lambda) or policy_idx

        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        legal = []
        for i in torch.nonzero(mask).flatten().tolist():
            d = decode_action(battle, i)
            entry = {"idx": i, "name": action_name(d), "p": round(probs[i].item(), 4)}
            if d and d[0] == "move" and d[1].base_power and opp is not None:
                entry["eff"] = eff_move(battle, _id(d[1].type), opp)
            legal.append(entry)
        by_idx = {e["idx"]: e for e in legal}
        damaging = [e for e in legal if "eff" in e]
        best_eff = max((e["eff"] for e in damaging), default=None)
        d = decode_action(battle, chosen)
        entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum().item()

        self.turn_logs[tag].append({
            "battle": tag, "turn": battle.turn, "me": mon_state(me, True), "opp": mon_state(opp, False),
            "forced_switch": not bool(mask[:8].any()),
            "legal": sorted(legal, key=lambda e: -e["p"]),
            "chosen": {"idx": chosen, "name": action_name(d), "p": by_idx.get(chosen, {}).get("p"),
                       "eff": by_idx.get(chosen, {}).get("eff")},
            "policy_choice": policy_idx, "search_changed": chosen != policy_idx,
            "value": round(value, 4), "phi": round(potential(battle), 4), "entropy": round(entropy, 4), "best_eff": best_eff,
            "opp_item_guessed": bool(opp_num[:, 19].any()), "fallback": d is None,
        })
        return self.order_for(battle, chosen) or self.choose_random_move(battle)


def prior_accuracy(opp_true_team, tera_by_species):
    """팀 프리뷰 시점(공개 정보 없음) 추론 vs 상대 실제 세트"""
    item_hit = item_n = move_hit = move_n = tera_hit = tera_n = 0
    for mon in opp_true_team:
        p = predict_set(mon.species)
        if not p:
            continue
        if p["item"] and mon.item:
            item_n += 1
            item_hit += p["item"] == _id(mon.item)
        true_moves = {_id(m) for m in mon.moves}
        if true_moves:
            move_n += len(true_moves)
            move_hit += len(true_moves & set(p["moves"][:4]))
        true_tera = tera_by_species.get(norm(mon.species))
        if p["tera"] and true_tera:
            tera_n += 1
            tera_hit += p["tera"] == norm(true_tera)
    return {"item": [item_hit, item_n], "moves": [move_hit, move_n], "tera": [tera_hit, tera_n]}


def auc(pairs):
    """(점수, 승리여부) → 이긴 판 점수가 진 판 점수보다 높을 확률 (0.5 = 무작위)"""
    pos = sorted(v for v, w in pairs if w)
    neg = sorted(v for v, w in pairs if not w)
    if not pos or not neg:
        return None
    import bisect
    s = sum(bisect.bisect_left(neg, v) + 0.5 * (bisect.bisect_right(neg, v) - bisect.bisect_left(neg, v)) for v in pos)
    return round(s / (len(pos) * len(neg)), 3)


def value_auc(battles, turns):
    """가치 헤드는 (결과 − Φ) 잔차를 학습 → 승률 추정은 V + Φ. 국면 구간별 AUC"""
    won = {b["battle"]: b["won"] for b in battles}
    phases = {"t1-5": (1, 5), "t6-10": (6, 10), "t11-20": (11, 20), "t21+": (21, 10 ** 9)}
    out = {}
    for name, (lo, hi) in phases.items():
        ts = [t for t in turns if lo <= t["turn"] <= hi and "phi" in t]
        out[name] = {k: auc([(f(t), won[t["battle"]]) for t in ts]) for k, f in
                     (("V+phi", lambda t: t["value"] + t["phi"]), ("phi", lambda t: t["phi"]), ("V", lambda t: t["value"]))}
    return out


def summarize(battles, turns):
    n = len(battles)
    wins = sum(b["won"] for b in battles)
    decisions = [t for t in turns if not t["forced_switch"] and not t["fallback"]]
    chosen_switch = [t for t in decisions if t["chosen"]["name"] and t["chosen"]["name"].startswith("switch:")]
    attacks = [t for t in decisions if t["chosen"]["eff"] is not None]
    se_available = [t for t in attacks if t["best_eff"] and t["best_eff"] >= 2]

    def rate(sub, total):
        return round(len(sub) / max(1, len(total)), 3)

    by_lead, by_matchup = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for b in battles:
        by_lead[b["lead"]][0] += b["won"]; by_lead[b["lead"]][1] += 1
        key = f'{b["my_team"]} vs {b["opp_team"]}'
        by_matchup[key][0] += b["won"]; by_matchup[key][1] += 1

    acc = {k: [sum(b["prior_accuracy"][k][0] for b in battles), sum(b["prior_accuracy"][k][1] for b in battles)]
           for k in ("item", "moves", "tera")}
    first_values = {b["battle"]: b["first_value"] for b in battles if b["first_value"] is not None}
    won_v = [v for b in battles if b["won"] for v in [first_values.get(b["battle"])] if v is not None]
    lost_v = [v for b in battles if not b["won"] for v in [first_values.get(b["battle"])] if v is not None]
    team_wr = defaultdict(lambda: [0, 0])
    for b in battles:
        team_wr[b["my_team"]][0] += b["won"]; team_wr[b["my_team"]][1] += 1

    return {
        "battles": n, "wins": wins, "win_rate": round(wins / max(1, n), 3), "win_rate_95ci": wilson(wins, n),
        "avg_turns": round(sum(b["turns"] for b in battles) / max(1, n), 1),
        "avg_turns_won_lost": [round(sum(b["turns"] for b in battles if b["won"]) / max(1, wins), 1),
                               round(sum(b["turns"] for b in battles if not b["won"]) / max(1, n - wins), 1)],
        "avg_mons_left_won_lost": [round(sum(b["my_left"] for b in battles if b["won"]) / max(1, wins), 2),
                                   round(sum(b["opp_left"] for b in battles if not b["won"]) / max(1, n - wins), 2)],
        "decisions": len(decisions),
        "switch_rate": rate(chosen_switch, decisions),
        "status_move_rate": rate([t for t in decisions if t["chosen"]["eff"] is None
                                  and not (t["chosen"]["name"] or "").startswith("switch:")], decisions),
        "missed_super_effective": rate([t for t in se_available if t["chosen"]["eff"] < 2], se_available),
        "chose_immune_move": rate([t for t in attacks if t["chosen"]["eff"] == 0], attacks),
        "chose_resisted_when_neutral_available": rate(
            [t for t in attacks if t["chosen"]["eff"] < 1 and t["best_eff"] >= 1], attacks),
        "search_changed_rate": rate([t for t in decisions if t["search_changed"]], decisions),
        "fallback_rate": rate([t for t in turns if t["fallback"]], turns),
        "avg_entropy": round(sum(t["entropy"] for t in decisions) / max(1, len(decisions)), 3),
        "tera_used_battles": rate([b for b in battles if b["tera_turn"] is not None], battles),
        "tera_turn_hist": dict(sorted(Counter(b["tera_turn"] for b in battles if b["tera_turn"] is not None).items())),
        "win_rate_with_without_tera": [round(sum(b["won"] for b in g) / max(1, len(g)), 3) for g in (
            [b for b in battles if b["tera_turn"] is not None], [b for b in battles if b["tera_turn"] is None])],
        "value_turn1_mean_won_lost": [round(sum(won_v) / max(1, len(won_v)), 3), round(sum(lost_v) / max(1, len(lost_v)), 3)],
        "value_auc_by_phase": value_auc(battles, turns),
        "prior_accuracy": {k: round(h / max(1, t), 3) for k, (h, t) in acc.items()},
        "team_win_rate": {k: [w, t] for k, (w, t) in sorted(team_wr.items(), key=lambda kv: kv[1][0] / kv[1][1])},
        "worst_leads": sorted(([k, w, t] for k, (w, t) in by_lead.items() if t >= 3), key=lambda x: x[1] / x[2])[:5],
    }


async def play(args, n_battles, wid, out_dir):
    """워커 하나: 자기 플레이어 쌍으로 n판을 돌리고 (배틀 기록, 턴 기록)을 돌려줌"""
    model = DeepPokemonBattleTransformerNet(pokemon_embed_dim=args.embed_dim, history_dim=args.history_dim,
                                            latent_dim=args.latent_dim, num_layers=args.layers)
    expanded, skipped = load_compatible(model, torch.load(args.ckpt, map_location="cpu"))
    if args.zero_unseen_rows:
        zero_unseen_id_rows(model)
    assert not skipped, f"체크포인트와 모델 구조가 안 맞음 (버려진 키 {len(skipped)}개): --embed-dim 등 확인"
    model.eval()
    if wid == 0:
        print(f"✅ {args.ckpt} 로드 (확장 {expanded}, 버림 {skipped})")

    random_fmt = "random" in args.format  # 랜덤배틀은 서버가 팀을 배정 → 팀 풀 사용 안 함
    if random_fmt:
        pool, pool_names = None, {}
    else:
        pool = load_team_pool(args.pool) if wid == 0 else _quiet_load(args.pool)
        pool_data = json.load(open(args.pool, encoding="utf-8-sig"))["teams"]
        pool_names = {team_species(m.species or m.nickname for m in Teambuilder.parse_showdown_team(t["export"])): t["name"]
                      for t in pool_data}

    tag = f"{wid}x{int(time.time()) % 100000}"  # 워커마다 다른 계정 (Showdown 이름 18자 제한)
    common = dict(battle_format=args.format, server_configuration=LocalhostServerConfiguration,
                  max_concurrent_battles=args.concurrent)
    me = LoggingPlayer(model=model, mode=args.mode, search_lambda=args.search,
                       save_replays=os.path.join(out_dir, "replays") if args.replays else False,
                       account_configuration=AccountConfiguration(f"ana{tag}", None), **common)
    me.encoder.blind_preview = bool(args.blind)
    opp = OPPONENTS[args.opponent](account_configuration=AccountConfiguration(f"{args.opponent[:3]}a{tag}", None), **common)
    if pool:
        opp_pool = _quiet_load(args.opp_pool) if args.opp_pool else pool
        me._team = RandomPoolTeambuilder(pool)
        opp._team = RandomPoolTeambuilder(opp_pool)
    try:
        await asyncio.wait_for(me.battle_against(opp, n_battles=n_battles), timeout=args.battle_timeout or None)
    except asyncio.TimeoutError:
        print(f"⚠️ 워커 {wid}: {args.battle_timeout:.0f}초 초과 → 끝난 {sum(b.finished for b in me.battles.values())}/{n_battles}판만 집계")

    battles, turns = [], []
    for btag, b in me.battles.items():
        if not b.finished:
            continue
        ob = opp.battles.get(btag)
        opp_true = list(ob.team.values()) if ob else []
        # poke-env는 자기 팀 teraType을 요청에서 안 읽음 → 원본 요청에서 직접
        opp_tera = {norm(m.get("details", "").split(",")[0]): m.get("teraType")
                    for m in ((ob.last_request if ob else {}).get("side") or {}).get("pokemon", [])}
        logs = me.turn_logs.get(btag, [])
        tera_turn = next((t["turn"] for t in logs if t["chosen"]["name"] and "+tera" in t["chosen"]["name"]), None)
        battles.append({
            "battle": btag, "won": bool(b.won), "turns": b.turn,
            "my_team": pool_names.get(team_species(m.species for m in b.team.values()), "?"),
            "opp_team": pool_names.get(team_species(m.species for m in opp_true), "?") if opp_true else "?",
            "lead": me.leads.get(btag), "tera_turn": tera_turn,
            "my_left": sum(not m.fainted for m in b.team.values()),
            "opp_left": 6 - sum(m.fainted for m in b.opponent_team.values()),
            "first_value": logs[0]["value"] if logs else None,
            "prior_accuracy": prior_accuracy(opp_true, opp_tera),
            "opp_true_team": [{"species": m.species, "item": m.item, "ability": m.ability,
                               "tera": opp_tera.get(norm(m.species)), "moves": list(m.moves)} for m in opp_true],
            "protocol": ["|".join(msg) for msg in b._replay_data],
        })
        turns.extend(logs)
    return battles, turns


def _quiet_load(path):
    with open(path, encoding="utf-8-sig") as f:
        return [t["export"] for t in json.load(f)["teams"]]


def worker(args, n_battles, wid, out_dir):
    torch.set_num_threads(1)  # 워커끼리 CPU 스레드 경쟁 방지
    return asyncio.run(play(args, n_battles, wid, out_dir))


def main(args):
    out_dir = os.path.join(args.out, f"analysis_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    n_workers = max(1, min(args.workers, args.battles))
    counts = [args.battles // n_workers + (1 if i < args.battles % n_workers else 0) for i in range(n_workers)]

    print(f"체크포인트 {args.ckpt} | 팀 풀 {args.pool} | 상대 {args.opponent} | {args.battles}판 (워커 {n_workers}개)")
    t0 = time.time()
    if n_workers == 1:
        results = [worker(args, counts[0], 0, out_dir)]
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            results = pool.starmap(worker, [(args, c, w, out_dir) for w, c in enumerate(counts)])
    battles = [b for bs, _ in results for b in bs]
    turns = [t for _, ts in results for t in ts]
    print(f"⏱️ {len(battles)}판 {time.time() - t0:.0f}초 (워커 {n_workers}개)")

    with open(os.path.join(out_dir, "turns.jsonl"), "w", encoding="utf-8") as ft, \
         open(os.path.join(out_dir, "battles.jsonl"), "w", encoding="utf-8") as fb:
        for rec in battles:
            fb.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for t in turns:
            ft.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary = summarize(battles, turns)
    summary.update(opponent=args.opponent, mode=args.mode, search=args.search, ckpt=args.ckpt, pool=args.pool)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"📁 {out_dir}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/supervised_v2.pt")
    ap.add_argument("--blind", action="store_true", help="팀 프리뷰 정보를 모델에 안 줌 (마인크래프트 환경 재현: 상대 포켓몬은 등장해야 보임)")
    ap.add_argument("--zero-unseen-rows", action="store_true", help="미등장 도구/특성/기술의 ID 임베딩 행을 0으로 (특징 표 효과 없이 이것만 시험)")
    ap.add_argument("--battle-timeout", type=float, default=0, help="워커당 제한 시간(초). 넘으면 끝난 판만 집계 (0이면 무제한)")
    ap.add_argument("--opp-pool", default=None, help="상대 팀 풀 JSON (생략하면 --pool과 동일)")
    ap.add_argument("--format", default="gen9ou", help="대전 형식 (예: gen9randombattle → 팀 풀 대신 서버가 랜덤 팀 배정)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--history-dim", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--opponent", default="rct", choices=list(OPPONENTS))
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--mode", default="sample", choices=["sample", "argmax"], help="sample=학습과 같은 확률 샘플링")
    ap.add_argument("--search", type=float, default=None, help="1턴 탐색 혼합 λ (생략하면 정책만)")
    ap.add_argument("--concurrent", type=int, default=8, help="워커 하나당 동시 배틀 수")
    ap.add_argument("--workers", type=int, default=4, help="배틀을 나눠 돌릴 프로세스 수")
    ap.add_argument("--replays", action="store_true", help="Showdown HTML 리플레이도 저장")
    ap.add_argument("--out", default="logs")
    ap.add_argument("--pool", default="data/team_pool_metamon_holdout.json",
                    help="양쪽 팀을 뽑을 팀 풀 JSON (예: data/team_pool_rare.json, data/team_pool_randomset.json)")
    main(ap.parse_args())
