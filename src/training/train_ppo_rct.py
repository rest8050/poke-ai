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
import multiprocessing as mp
import queue
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder
from src.core.rct_player import RCTBattleAIPlayer
from src.core.player import SmartPokemonPlayer
from src.core.model import DeepPokemonBattleTransformerNet, load_compatible
from src.core.search import search_pick
from src.core.battle_eval import potential

# 수집(워커)은 CPU, 학습(메인)은 GPU
ROLLOUT_DEVICE = torch.device("cpu")
LEARN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATTLE_FORMAT = "gen9ou"
TEAM_POOL_PATH = "data/team_pool_metamon.json"  # Metamon 학습 풀 (평가 전용 _holdout.json은 학습에 쓰지 말 것)
TARGET_KL = 0.015
# 신용 할당 거리: GAE 실질 지평 ≈ 1/(1-γλ) → 0.99/0.95 약 17턴, 0.995/0.97 약 29턴
GAMMA, GAE_LAMBDA = 0.995, 0.97
LR_MIN, LR_MAX = 1e-6, 3e-4


def load_team_pool(path: str = TEAM_POOL_PATH):
    """team_pool.json에서 Showdown export 문자열 목록 로드"""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        teams = [t["export"] for t in data["teams"]]
        print(f"✅ 팀 풀 로드 완료: {len(teams)}개 팀")
        return teams
    except Exception as e:
        print(f"⚠️ 팀 풀 로드 실패 ({e}). gen9randombattle로 폴백")
        return None

class RandomPoolTeambuilder(Teambuilder):
    def __init__(self, exports):
        self.packed = [self.join_team(self.parse_showdown_team(t)) for t in exports]
        
    def yield_team(self):
        return random.choice(self.packed)

OPPONENTS = {"rct": RCTBattleAIPlayer, "heuristic": SimpleHeuristicsPlayer, "random": RandomPlayer}

OBS_KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
TENSOR_KEYS = OBS_KEYS + ["action_mask", "log_prob", "value", "advantage", "return"]


class PPOCollectingPlayer(SmartPokemonPlayer):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.trajectories = {}
        self.last_state_scores = {}
        self.is_eval = False

    def clear_memory(self):
        self.trajectories.clear()
        self.last_state_scores.clear()
        self.history.clear()
        self.battles.clear()

    def choose_move(self, battle):
        tag = battle.battle_tag
        current_score = potential(battle)  # Foul Play식 국면 평가 Φ(s)

        if not self.is_eval:
            if tag not in self.trajectories:
                self.trajectories[tag] = []
            else:
                # 잠재 함수 보상: γ·Φ(s') - Φ(s) (최적 정책 불변, 신용 할당만 가속)
                last_score = self.last_state_scores.get(tag, 0.0)
                step_reward = GAMMA * current_score - last_score
                if len(self.trajectories[tag]) > 0:
                    self.trajectories[tag][-1]["reward"] = self.trajectories[tag][-1].get("reward", 0.0) + step_reward

            self.last_state_scores[tag] = current_score

        tensors = self.encoder.encode_battle(battle)
        my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, action_mask = [t.to(ROLLOUT_DEVICE) for t in tensors]

        history = self.history.get(tag)
        if history is not None:
            history = history.to(ROLLOUT_DEVICE)

        if self.is_eval and self.search_lambda is not None:
            # 평가 전용 1턴 탐색 (학습 rollout에는 절대 쓰지 않음: log_prob이 정책과 어긋남)
            action_idx, probs, _, val, new_history = self.model.get_action(
                my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, history, action_mask
            )
            action_idx = search_pick(battle, probs[0], self.search_lambda) or action_idx
            log_prob = None
        else:
            action_idx, log_prob, val, new_history = self.model.get_action_rl(
                my_cat, my_num, my_m_num, opp_cat, opp_num, opp_m_num, field_vec, history, action_mask, deterministic=self.is_eval
            )
        self.history[tag] = new_history

        # 상대의 마지막 사용 기술 인덱스 수집 (보조 헤드 학습용)
        opp_move_label = -1  # 기본값: 정보 없음
        opp_active = battle.opponent_active_pokemon
        if opp_active and hasattr(opp_active, 'moves') and opp_active.moves:
            opp_moves_list = list(opp_active.moves.values())
            # 가장 최근에 사용된 기술을 찾아 인덱스로 변환
            for mi, mv in enumerate(opp_moves_list):
                if hasattr(mv, 'current_pp') and mv.current_pp is not None and mv.max_pp is not None:
                    if mv.current_pp < mv.max_pp and mi < 4:
                        opp_move_label = mi  # 마지막으로 PP가 줄어든 기술

        if not self.is_eval:
            self.trajectories[tag].append({
                "my_team_cat": my_cat.squeeze(0),
                "my_team_num": my_num.squeeze(0),
                "my_move_num": my_m_num.squeeze(0),
                "opp_team_cat": opp_cat.squeeze(0),
                "opp_team_num": opp_num.squeeze(0),
                "opp_move_num": opp_m_num.squeeze(0),
                "field_vec": field_vec.squeeze(0),
                "action_mask": action_mask.squeeze(0),
                "action": action_idx,
                "log_prob": log_prob.squeeze(0),
                "value": val.squeeze(0),
                "opp_move_label": opp_move_label
            })

        return self.order_for(battle, action_idx) or self.choose_random_move(battle)

def compute_gae(trajectory, final_reward, gamma=GAMMA, lam=GAE_LAMBDA):
    values = [step["value"] for step in trajectory]
    values.append(torch.tensor([0.0], device=ROLLOUT_DEVICE))

    advantages = []
    gae = 0
    for t in reversed(range(len(trajectory))):
        reward = trajectory[t].get("reward", 0.0)
        if t == len(trajectory) - 1:
            reward += final_reward

        delta = reward + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return advantages


# =====================================================================
# 워커 프로세스: 자기 플레이어 쌍으로 배틀을 돌리고 궤적을 numpy 배열로 돌려줌
# =====================================================================

def _pack_steps(steps):
    """step dict 리스트 -> 키별로 쌓은 numpy 배열 (프로세스 간 전송용)"""
    if not steps:
        return None
    out = {k: torch.stack([s[k] for s in steps]).numpy() for k in TENSOR_KEYS}
    out["action"] = np.array([s["action"] for s in steps], dtype=np.int64)
    out["opp_move_label"] = np.array([s["opp_move_label"] for s in steps], dtype=np.int64)
    # 히스토리 트랜스포머 학습용: 어느 배틀의 몇 번째 턴인지 (배틀별로 연속, 턴 순서대로 쌓임)
    out["seq_id"] = np.array([s["seq_id"] for s in steps], dtype=np.int64)
    out["seq_pos"] = np.array([s["seq_pos"] for s in steps], dtype=np.int64)
    return out


async def _worker_loop(wid, kind, cmd_q, out_q, max_concurrent, team_pool, eval_search=None):
    model = DeepPokemonBattleTransformerNet()
    model.eval()
    tag = f"{wid}p{os.getpid() % 100000}"
    common = dict(battle_format=BATTLE_FORMAT, server_configuration=LocalhostServerConfiguration,
                  max_concurrent_battles=max_concurrent)
    ai_player = PPOCollectingPlayer(model=model, account_configuration=AccountConfiguration(f"ppo{tag}", None),
                                    search_lambda=eval_search, **common)
    train_opp = OPPONENTS[kind](account_configuration=AccountConfiguration(f"{kind[:3]}{tag}", None), **common)
    eval_opp = train_opp if kind == "rct" else \
        RCTBattleAIPlayer(account_configuration=AccountConfiguration(f"rct{tag}", None), **common)
        
    if team_pool:
        ai_player._team = RandomPoolTeambuilder(team_pool)
        train_opp._team = RandomPoolTeambuilder(team_pool)
        if eval_opp is not train_opp:
            eval_opp._team = RandomPoolTeambuilder(team_pool)

    loop = asyncio.get_running_loop()

    while True:
        cmd = await loop.run_in_executor(None, cmd_q.get)
        if cmd is None:
            return
        state_dict, n_battles, is_eval = cmd
        model.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
        opp_player = eval_opp if is_eval else train_opp

        ai_player.clear_memory()
        opp_player.battles.clear()
        ai_player.is_eval = is_eval
        await ai_player.battle_against(opp_player, n_battles=n_battles)

        wins = sum(1 for b in ai_player.battles.values() if b.won)
        finished = sum(1 for b in ai_player.battles.values() if b.finished)
        steps = []
        if not is_eval:
            for seq_id, (battle_tag, trajectory) in enumerate(ai_player.trajectories.items()):
                battle_obj = ai_player.battles.get(battle_tag)
                if not trajectory or not battle_obj or battle_obj.won is None:
                    continue
                # 종료 상태 Φ=0 → 마지막 스텝에 -Φ(마지막 관측 상태)를 더해 잠재 함수 형태를 닫음
                final_reward = (1.0 if battle_obj.won else -1.0) - ai_player.last_state_scores.get(battle_tag, 0.0)
                advantages = compute_gae(trajectory, final_reward)
                for t, step in enumerate(trajectory):
                    step["advantage"] = advantages[t].detach()
                    step["return"] = (advantages[t] + step["value"]).detach()
                    step["seq_id"], step["seq_pos"] = seq_id, t
                    steps.append(step)
        out_q.put((wid, wins, finished, _pack_steps(steps), "rct" if is_eval else kind))


def worker_main(wid, kind, cmd_q, out_q, max_concurrent, team_pool, eval_search=None):
    torch.set_num_threads(1)  # 워커끼리 CPU 스레드 경쟁 방지
    try:
        asyncio.run(_worker_loop(wid, kind, cmd_q, out_q, max_concurrent, team_pool, eval_search))
    except Exception as e:
        out_q.put((wid, "error", repr(e), None, None))
        raise


# =====================================================================
# 메인 프로세스: 워커에 가중치 배포 -> 궤적 수집 -> GPU에서 PPO 업데이트
# =====================================================================

def run_workers(model, procs, cmd_qs, out_q, total_battles, is_eval):
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    n_workers = len(cmd_qs)
    counts = [total_battles // n_workers + (1 if i < total_battles % n_workers else 0) for i in range(n_workers)]
    active = [i for i, c in enumerate(counts) if c > 0]
    for i in active:
        cmd_qs[i].put((state, counts[i], is_eval))

    results, t0 = [], time.time()
    while len(results) < len(active):
        try:
            res = out_q.get(timeout=30)
        except queue.Empty:
            dead = [i for i in active if not procs[i].is_alive()]
            if dead:
                raise RuntimeError(f"워커 {dead} 프로세스가 종료됨")
            continue
        if res[1] == "error":
            raise RuntimeError(f"워커 {res[0]} 오류: {res[2]}")
        results.append(res)
        done = sum(r[2] for r in results)
        print(f"⏳ 워커 {res[0]} 완료 ({len(results)}/{len(active)}) | 누적 {done}판 | {time.time() - t0:.0f}초")
    return results


def wins_by_opponent(results):
    by = {}
    for r in results:
        w, f = by.get(r[4], (0, 0))
        by[r[4]] = (w + r[1], f + r[2])
    return " | ".join(f"{k} {w}/{f} ({w / max(f, 1) * 100:.1f}%)" for k, (w, f) in sorted(by.items()))


def parse_opponents(spec):
    """'rct:5,heuristic:3' -> ['rct']*5 + ['heuristic']*3"""
    kinds = []
    for part in spec.split(","):
        name, _, count = part.strip().partition(":")
        if name not in OPPONENTS:
            raise ValueError(f"알 수 없는 상대 '{name}' (가능: {', '.join(OPPONENTS)})")
        kinds += [name] * int(count or 1)
    return kinds


def merge_buffers(results):
    parts = [r[3] for r in results if r[3] is not None]
    if not parts:
        return None
    offset = 0
    for part in parts:  # 워커마다 0부터 매긴 배틀 번호를 전역으로 겹치지 않게
        part["seq_id"] = part["seq_id"] + offset
        offset = int(part["seq_id"].max()) + 1
    return {k: torch.from_numpy(np.concatenate([p[k] for p in parts])).to(LEARN_DEVICE) for k in parts[0]}


def ppo_update(model, optimizer, buf, batch_size, ppo_epochs):
    clip_epsilon = 0.2
    value_coef = 0.25
    # [충격 요법] 휴리스틱 봇과의 무한 교체(Deadlock)를 깨기 위해 호기심을 대폭 상승시킴 (0.005 -> 0.05)
    entropy_coef = 0.008
    aux_coef = 0.0
    n = buf["action"].shape[0]
    # 배틀 단위 구간: 한 배틀의 턴은 버퍼에서 연속, 턴 순서대로 (히스토리 트랜스포머가 배틀 전체를 봐야 함)
    seq = buf["seq_id"]
    starts = torch.nonzero(torch.cat([seq.new_ones(1, dtype=torch.bool), seq[1:] != seq[:-1]])).flatten().tolist()
    ends = starts[1:] + [n]
    n_seq = len(starts)

    for epoch in range(ppo_epochs):
        model.eval()
        order = torch.randperm(n_seq).tolist()
        epoch_policy_loss = epoch_value_loss = epoch_entropy = 0.0
        epoch_approx_kl = epoch_clip_frac = 0.0
        batch_count = 0

        i = 0
        while i < n_seq:
            # 턴 수가 batch_size에 찰 때까지 배틀을 통째로 묶음
            chosen, count = [], 0
            while i < n_seq and count < batch_size:
                chosen.append(order[i])
                count += ends[order[i]] - starts[order[i]]
                i += 1
            idx = torch.cat([torch.arange(starts[c], ends[c], device=LEARN_DEVICE) for c in chosen])
            local_seq = torch.cat([torch.full((ends[c] - starts[c],), j, device=LEARN_DEVICE) for j, c in enumerate(chosen)])
            if len(idx) < 4:
                continue
            b = {k: v[idx] for k, v in buf.items()}

            advantages = b["advantage"].view(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = b["return"].view(-1, 1)

            outputs = model.forward_sequences([b[k] for k in OBS_KEYS], local_seq, b["seq_pos"],
                                              len(chosen), action_mask=b["action_mask"])

            dist = torch.distributions.Categorical(logits=outputs["policy_logits"])
            new_log_probs = dist.log_prob(b["action"])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - b["log_prob"])
            
            with torch.no_grad():
                log_ratio = new_log_probs - b["log_prob"]
                approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean().item()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(outputs["value"], returns)

            # 보조 헤드 손실: 상대 행동 예측 (유효한 라벨만 사용)
            valid_opp = b["opp_move_label"] >= 0
            if valid_opp.any():
                aux_loss = F.cross_entropy(outputs["opp_action_logits"][valid_opp], b["opp_move_label"][valid_opp])
            else:
                aux_loss = torch.tensor(0.0, device=LEARN_DEVICE)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + aux_coef * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_entropy += entropy.item()
            epoch_approx_kl += approx_kl
            epoch_clip_frac += clip_frac
            batch_count += 1

        if batch_count:
            kl = epoch_approx_kl / batch_count
            print(f"🔄 Epoch [{epoch+1}/{ppo_epochs}] P-Loss: {epoch_policy_loss/batch_count:.4f} | "
                  f"V-Loss: {epoch_value_loss/batch_count:.4f} | Ent: {epoch_entropy/batch_count:.4f} | "
                  f"KL: {kl:.4f} | Clip: {epoch_clip_frac/batch_count:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # KL 기반 동적 학습률: KL을 target 근처로 유지 (lr은 옵티마이저에 남아 다음 iteration까지 유지)
            for g in optimizer.param_groups:
                if kl > 2.0 * TARGET_KL:
                    g["lr"] = max(g["lr"] / 1.5, LR_MIN)
                elif kl < 0.5 * TARGET_KL:
                    g["lr"] = min(g["lr"] * 1.5, LR_MAX)
            if kl > 1.5 * TARGET_KL:
                print(f"⛔ KL {kl:.4f} > {1.5 * TARGET_KL:.4f} → 남은 epoch 스킵")
                break


def main(n_iterations=100, battles_per_iter=128, opponents="rct:8", concurrent_per_worker=4,
         batch_size=1024, ppo_epochs=4, eval_every=10, eval_battles=96,
         ckpt_path="checkpoints/vs_bot_RL.pt", eval_search=None):
    kinds = parse_opponents(opponents)
    n_workers = len(kinds)
    print(f"🚀 [PPO] 병렬 수집 시작 (워커 {n_workers}개 × 동시 {concurrent_per_worker}판, 상대 {opponents}, "
          f"학습: {LEARN_DEVICE})")

    model = DeepPokemonBattleTransformerNet()
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            expanded, skipped = load_compatible(model, ckpt)
            fresh = [k for k in model.state_dict() if k not in ckpt]
            print(f"✅ 기존 가중치 로드 완료!")
            if expanded:
                print(f"   입력 확장(기존 열 보존, 새 열 0): {expanded}")
            if skipped or fresh:
                print(f"   새로 초기화: {sorted({k.split('.')[0] for k in skipped + fresh})}")
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패 ({e}). 무작위 가중치로 시작합니다.")
    else:
        print(f"⚠️ {ckpt_path} 가 없어 무작위 가중치로 시작합니다.")
    model.to(LEARN_DEVICE)

    # 처음부터 전체 가중치 학습 (단일 Phase)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    team_pool = load_team_pool()

    ctx = mp.get_context("spawn")
    cmd_qs = [ctx.Queue() for _ in range(n_workers)]
    out_q = ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(i, kinds[i], cmd_qs[i], out_q, concurrent_per_worker, team_pool, eval_search), daemon=True)
             for i in range(n_workers)]
    for p in procs:
        p.start()

    try:
        for iteration in range(1, n_iterations + 1):
            print(f"\n==================================================")
            print(f"🔥 Iteration [{iteration}/{n_iterations}] - {battles_per_iter}판 데이터 수집 시작")
            print(f"==================================================")
            t0 = time.time()
            results = run_workers(model, procs, cmd_qs, out_q, battles_per_iter, is_eval=False)
            buf = merge_buffers(results)
            wins = sum(r[1] for r in results)
            turns = 0 if buf is None else buf["action"].shape[0]
            print(f"✅ 수집 완료! {wins}승 / {battles_per_iter}판 | 수집된 총 턴 수: {turns} | {time.time() - t0:.0f}초")
            print(f"   상대별: {wins_by_opponent(results)}")
            if buf is None:
                continue



            print("🧠 PPO 가중치 업데이트 시작...")
            ppo_update(model, optimizer, buf, batch_size, ppo_epochs)
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print("💾 가중치 저장 완료!")

            if iteration % eval_every == 0:
                print(f"\n🏆 [평가 모드] 혼합 봇(RCT/Heuristic) 상대 실력 검증 ({eval_battles}판 승부)")
                results = run_workers(model, procs, cmd_qs, out_q, eval_battles, is_eval=True)
                eval_wins = sum(r[1] for r in results)
                print(f"📊 평가 결과 승률: {eval_wins}승 / {eval_battles}판 ({eval_wins/eval_battles*100:.1f}%)")
                print(f"   상대별: {wins_by_opponent(results)}")
    finally:
        for q in cmd_qs:
            q.put(None)
        for p in procs:
            p.join(timeout=10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--battles", type=int, default=260, help="반복당 총 배틀 수 (워커들이 나눠 맡음)")
    ap.add_argument("--workers", type=int, default=8, help="--opponents를 안 주면 전부 RCT 상대로 이 수만큼")
    ap.add_argument("--opponents", default=None,
                    help="워커별 학습 상대, 예: rct:5,heuristic:3 (가능: rct, heuristic, random)")
    ap.add_argument("--concurrent", type=int, default=4, help="워커 하나당 동시 배틀 수")
    ap.add_argument("--ckpt", default="checkpoints/vs_bot_RL.pt",
                    help="시작 체크포인트이자 매 iteration 저장 위치 (덮어씀). BC 체크포인트와 겹치지 않는 이름으로. "
                         "처음 시작할 땐 이 경로로 BC 체크포인트를 복사해 두고 실행 (없으면 무작위 초기화로 시작)")
    ap.add_argument("--eval-search", type=float, default=None,
                    help="평가 때만 1턴 탐색 혼합 비율 λ (예: 0.5). 생략하면 정책만")
    args = ap.parse_args()
    main(n_iterations=args.iterations, battles_per_iter=args.battles,
         opponents=args.opponents or f"rct:{args.workers}",
         concurrent_per_worker=args.concurrent, ckpt_path=args.ckpt, eval_search=args.eval_search)
