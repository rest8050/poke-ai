"""
강화학습(PPO) 미세조정: 증류된 정책(v8_full 등)에서 시작해 자가대전 승패로 정책을 개선.
한 반복 = 롤아웃(tools/rl_rollout.py 작업자들, 현재 정책 샘플링) → PPO 갱신 → 체크포인트 저장 → (주기적으로) 시작 모델과 SPRT 대전.

- advantage (--adv, 같은 반복 전체에서 정규화):
  outcome: 보상은 판 끝의 승패 ±1. 결정별 반환 G_t = γ^(남은 결정 수)·승패 (train_fp_distill.make_batches와 같은 정의), advantage = G_t − V(s_t).
    V(s_t)는 그 행동 전의 상태 값이라 편향 없는 기준값 (가치 헤드는 정책과 같이 계속 학습).
  phi: 국면 포텐셜 Φ(src/core/battle_eval.py)를 비평가로 쓴 짧은 지평 TD. δ_t = γΦ_{t+1} − Φ_t, 마지막 결정은 승패 − Φ, GAE(γ, --lam).
    lam이 작을수록 가까운 몇 턴의 국면 변화만 봄 (오프라인 검증: FP의 행동 이득과 상관이 k=1~3턴에서 가장 큼, 학습된 V나 승패 기반은 0 이하).
    Φ는 진짜 가치가 아닌 근사라서 편향이 있음 (1~3턴 뒤 평가를 올리는 쪽으로 움직임)
- 정책 손실: PPO 클립 + β·KL(현재 정책 ‖ 시작 정책) (KL 앵커: 증류로 얻은 지식에서 멀어지는 속도를 제한, --kl 0이면 앵커 없음).
- 비율의 분모(옛 정책)는 롤아웃 때가 아니라 반복 시작에 저장된 입력으로 다시 계산 (롤아웃은 fp32 입력, 저장은 fp16이라 어긋나는 것을 없앰). 갱신과 롤아웃 모두 드롭아웃 끔(eval).
- 세트 추정(belief)은 얼림. 합법 마스크(legal_mask)를 입력 특징으로 쓰는 모델(v6/v8)만 지원.
- 상대: 작업자 일부(--snap-frac)는 과거 반복 체크포인트와, 나머지는 현재 정책과의 자가대전 (양쪽 시점 모두 학습). 팀은 학습용 풀에서만 뽑고 평가(arena)는 평가 풀.

사용: python src/training/train_rl.py --run rl1 [--start checkpoints/supervised_v2_fp_v8_full.pt] [--iters 20 --games 2000 --kl 0.1 --lr 3e-5]
출력: checkpoints/rl/<run>/it000.pt(시작 복사), it001.pt, ... + w*.log(작업자 로그). Showdown(:8000)은 없으면 직접 띄우고 끝나면 내림
"""
import glob
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np

from src.core.model import model_from_ckpt, save_ckpt
from src.training import train_fp_distill as T
from src.training.replay_dataset import OBS_KEYS
from tools.arena import port_open

POOLS = ["data/team_pool_train_a2.json", "data/team_pool_randomset_train3.json", "data/team_pool_rare_train3.json"]   # 학습용 풀 (DAgger와 같음). 평가는 arena의 holdout/rare/randomset
BASE_KEYS = OBS_KEYS + ["action_mask", "legal_mask", "action_taken"]
EXTRA_KEYS = ["old_logp", "adv", "ref_logp"]   # 반복 시작에 계산해 data에 붙임 (make_batches가 시퀀스 단위로 같이 잘라 줌)
CLIP, VALUE_COEF = 0.2, 0.5
DEVICE = T.DEVICE


def policy_terms(lp_all, a, mask, old_logp, adv, ref_logp, clip, kl_coef):
    """PPO 클립 + KL 앵커. lp_all [N,22] 현재 정책 log확률, a [N] 고른 행동(-1=제외), old_logp [N] 옛 정책이 고른 행동에 준 log확률, ref_logp [N,22] 시작 정책.
    선택지가 둘 이상인 결정만 셈(하나뿐이면 기울기가 없음). 반환: (손실, 통계)"""
    valid = (a >= 0) & (mask.sum(-1) > 1)
    n = valid.sum().clamp_min(1)
    lp = torch.where(valid, lp_all.gather(-1, a.clamp(min=0)[:, None]).squeeze(-1), torch.zeros_like(lp_all[:, 0]))   # 제외된 결정은 마스크 밖 칸(-1e9)을 집을 수 있어 exp가 넘치지 않게 0으로
    old_logp = torch.where(valid, old_logp, torch.zeros_like(old_logp))
    ratio = (lp - old_logp).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    p = lp_all.exp()
    kl_ref = torch.where(mask, p * (lp_all - ref_logp), torch.zeros_like(p)).sum(-1)
    ent = -torch.where(mask, p * lp_all, torch.zeros_like(p)).sum(-1)
    loss = ((pg + kl_coef * kl_ref) * valid).sum() / n
    with torch.no_grad():
        st = {"clipfrac": (((ratio - 1).abs() > clip) & valid).sum() / n, "kl_old": ((old_logp - lp) * valid).sum() / n,
              "kl_ref": (kl_ref * valid).sum() / n, "ent": (ent * valid).sum() / n, "n": n.float()}
    return loss, st


def to_dev(b):
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}


def forward(model, b):
    out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"], legal_mask=b["legal_mask"])
    return F.log_softmax(out["policy_logits"].float(), -1), out["value"].float().view(-1)


def phi_advantage(phi, outcome, lam, gamma=T.GAMMA):
    """한 판의 Φ 열 [T]과 승패 → 결정별 GAE advantage. 비평가 = Φ (종료 상태는 0): δ_t = γΦ_{t+1} − Φ_t, 마지막은 outcome − Φ"""
    delta = np.append(gamma * phi[1:].astype(np.float64), outcome) - phi
    out, run = np.empty_like(delta), 0.0
    for t in range(len(delta) - 1, -1, -1):
        run = delta[t] + gamma * lam * run
        out[t] = run
    return out


@torch.no_grad()
def prepare(model, ref, data, offsets, n_seq, batch, adv_kind, lam):
    """반복 시작 정책으로: 고른 행동의 log확률, 시작 정책의 전체 log확률, advantage를 data에 붙임"""
    lp_a, ref_lp, g, v = [], [], [], []
    for b in T.make_batches(data, offsets, range(n_seq), batch, False):
        b = to_dev(b)
        lp_all, val = forward(model, b)
        lp_a.append(lp_all.gather(-1, b["action_taken"].clamp(min=0)[:, None]).squeeze(-1).cpu())
        ref_lp.append(forward(ref, b)[0].cpu())
        g.append(b["value_target"].cpu()); v.append(val.cpu())
    lp_a, ref_lp, g, v = (torch.cat(x).numpy() for x in (lp_a, ref_lp, g, v))
    valid = (data["action_taken"] >= 0) & (data["action_mask"].sum(1) > 1)
    raw = g - v if adv_kind == "outcome" else np.concatenate([phi_advantage(data["phi"][offsets[j]:offsets[j + 1]], float(data["outcome"][j]), lam) for j in range(n_seq)])
    adv = np.where(valid, (raw - raw[valid].mean()) / max(raw[valid].std(), 1e-6), 0.0).astype(np.float32)
    data.update(old_logp=lp_a.astype(np.float32), adv=adv, ref_logp=ref_lp.astype(np.float32))
    return {"adv_raw_std": float(np.std(raw[valid])), "v_mse": float(np.mean((g - v) ** 2)), "valid": int(valid.sum())}


def update(model, opt, data, offsets, n_seq, args):
    params = [p for p in model.parameters() if p.requires_grad]
    for ep in range(args.epochs):
        tot, nb = defaultdict(float), 0
        torch.cuda.reset_peak_memory_stats()     # 에폭별 최댓값 (누적이면 긴 판이 한 번 걸린 뒤로 내려가지 않아 늘어나는 것처럼 보임)
        for b in T.make_batches(data, offsets, range(n_seq), args.batch, True):
            b = to_dev(b)
            lp_all, v = forward(model, b)
            pg, st = policy_terms(lp_all, b["action_taken"], b["action_mask"], b["old_logp"], b["adv"], b["ref_logp"], CLIP, args.kl)
            v_loss = F.mse_loss(v, b["value_target"])
            opt.zero_grad()
            (pg + VALUE_COEF * v_loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for k, x in st.items():
                tot[k] += float(x)
            tot["v"] += float(v_loss); nb += 1
        print(f"  [PPO 에폭 {ep + 1}/{args.epochs}] " + " | ".join(f"{k} {tot[k] / nb:.4f}" for k in ("clipfrac", "kl_old", "kl_ref", "ent", "v")) + f" | VRAM 최대 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)


def rollouts(it, cur, snaps, args, run_dir):
    """작업자 --workers개를 띄워 --games판을 모음 (일부는 과거 체크포인트 상대). 반환: 모은 shard 경로"""
    per, n_snap = math.ceil(args.games / args.workers), round(args.workers * args.snap_frac)
    procs, t0 = [], time.time()
    for i in range(args.workers):
        opp = random.choice(snaps) if i < n_snap else "self"
        cmd = [sys.executable, "tools/rl_rollout.py", cur, opp, str(per), POOLS[i % len(POOLS)], f"{run_dir}/shard_{i}.npz", f"{it % 100:02d}{i:02d}",
               "--port", str(args.port), "--concurrent", str(args.concurrent)]
        procs.append(subprocess.Popen(cmd, stdout=open(f"{run_dir}/w{i}.log", "w"), stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in procs) and time.time() - t0 < args.timeout_min * 60:
        time.sleep(5)
    for p in procs:
        if p.poll() is None:
            p.terminate()
    time.sleep(2)
    for p in procs:
        if p.poll() is None:
            p.kill()
    print(f"  롤아웃 {(time.time() - t0) / 60:.1f}분 (작업자 종료 코드 {[p.returncode for p in procs]}, 과거 체크포인트 상대 {n_snap}명)", flush=True)
    return sorted(glob.glob(f"{run_dir}/shard_*.npz"))


def main(args):
    run_dir = f"checkpoints/rl/{args.run}"
    os.makedirs(run_dir, exist_ok=True)
    start0 = f"{run_dir}/it000.pt"
    model = model_from_ckpt(args.start)
    assert getattr(model, "uses_legal", False), "합법 마스크를 입력으로 쓰는 모델(v6/v8)만 지원"
    assert not getattr(model, "prev_act", False), "직전 기술 입력(prev_act) 모델은 미지원: 롤아웃 기록과 PPO 갱신이 prev_move를 담지 않음"
    ref = model_from_ckpt(args.base or args.start)      # KL 앵커이자 arena 비교 기준 (이어 학습할 땐 --base로 원래 시작 모델을 유지)
    save_ckpt(ref, start0)
    ref = ref.to(DEVICE).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model.to(DEVICE)
    for n, p in model.named_parameters():
        if n.startswith("belief."):
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0)
    sd = None
    if not port_open(args.port):
        sd = subprocess.Popen([shutil.which("node"), "pokemon-showdown", "start", "--no-security", str(args.port)], cwd="pokemon-showdown",
                              stdout=open(f"logs/showdown_rl_{args.run}.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if port_open(args.port):
                break
            time.sleep(1)
        else:
            sd.kill(); sys.exit("Showdown 시작 실패")
    print(f"[RL {args.run}] 시작 {args.start} | 반복 {args.iters} x {args.games}판, 작업자 {args.workers}, β(KL) {args.kl}, lr {args.lr}", flush=True)
    snaps = [start0]
    try:
        for it in range(1, args.iters + 1):
            t0 = time.time()
            cur = f"{run_dir}/cur.pt"
            save_ckpt(model, cur)
            for f in glob.glob(f"{run_dir}/shard_*.npz"):
                os.remove(f)
            shards = rollouts(it, cur, snaps, args, run_dir)
            if not shards:
                sys.exit("❌ 롤아웃 shard 없음: " + run_dir + "/w*.log 확인")
            keys = BASE_KEYS + (["phi"] if args.adv == "phi" else [])
            T.KEYS = keys
            data, offsets, lens, _ = T.load(",".join(shards))
            for f in shards:
                os.remove(f)
            n_seq = len(lens)
            wins = int((data["outcome"] > 0).sum())
            model.eval()
            info = prepare(model, ref, data, offsets, n_seq, args.batch, args.adv, args.lam)
            T.KEYS = keys + EXTRA_KEYS
            print(f"[반복 {it}/{args.iters}] 시퀀스 {n_seq} (이긴 시퀀스 {wins}) | 결정 {int(lens.sum())} (선택지 2개 이상 {info['valid']}, 최장 판 {int(lens.max())}결정) | advantage 표준편차(정규화 전, {args.adv}) {info['adv_raw_std']:.3f} V-MSE {info['v_mse']:.3f}", flush=True)
            update(model, opt, data, offsets, n_seq, args)
            del data
            snap = f"{run_dir}/it{it:03d}.pt"
            save_ckpt(model, snap)
            snaps.append(snap)
            print(f"  💾 {snap} ({(time.time() - t0) / 60:.1f}분)", flush=True)
            if args.eval_every and it % args.eval_every == 0:
                r = subprocess.run([sys.executable, "tools/arena.py", snap, start0, "--cap", str(args.eval_cap), "--workers", "6", "--port", str(args.port)], capture_output=True, text=True)
                print("  [시작 모델과 SPRT 대전]\n   " + "\n   ".join(r.stdout.strip().splitlines()[-2:]), flush=True)
    finally:
        if sd:
            sd.terminate()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="실행 이름 (checkpoints/rl/<run>/)")
    ap.add_argument("--start", default="checkpoints/supervised_v2_fp_v8_full.pt")
    ap.add_argument("--base", default=None, help="KL 앵커/arena 비교 기준 체크포인트 (기본: --start). 이어 학습: --start <이전 it*.pt> --base <원래 시작 모델>")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=2000, help="반복당 롤아웃 판 수")
    ap.add_argument("--adv", choices=["outcome", "phi"], default="outcome", help="advantage: outcome = 승패 − V(s), phi = 국면 포텐셜 Φ를 비평가로 쓴 짧은 지평 TD")
    ap.add_argument("--lam", type=float, default=0.5, help="--adv phi의 GAE λ (0.5 = 유효 지평 약 2턴, 0 = 1턴)")
    ap.add_argument("--kl", type=float, default=0.1, help="시작 정책에 대한 KL 앵커 계수 β (0=끔)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2, help="반복당 PPO 에폭")
    ap.add_argument("--batch", type=int, default=16, help="미니배치 = 배틀 수 (VRAM에 맞춰)")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--concurrent", type=int, default=10)
    ap.add_argument("--snap-frac", type=float, default=0.33, help="과거 체크포인트와 두는 작업자 비율 (나머지는 자가대전)")
    ap.add_argument("--timeout-min", type=int, default=40, help="반복당 롤아웃 제한 시간 (넘으면 지금까지 모인 판만 씀)")
    ap.add_argument("--eval-every", type=int, default=5, help="N반복마다 시작 모델과 arena SPRT 대전 (0=안 함)")
    ap.add_argument("--eval-cap", type=int, default=1000)
    ap.add_argument("--port", type=int, default=8000)
    main(ap.parse_args())
