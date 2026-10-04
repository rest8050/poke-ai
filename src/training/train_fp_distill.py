"""
Foul Play(엔진 탐색 교사) 증류 학습 — RL 없이 교사의 탐색 분포(혼합 전략)를 정책이 그대로 배움.

- 데이터: build_fp_dataset.py 출력 npz (실시간 인코더 입력 + Foul Play의 전체 탐색 분포 + 가치 목표), 여러 파일 가능
- 손실: 소프트 CE(교사 분포) + 가치 MSE + 시작 정책과의 KL (정책 유지). 검증 배틀에서 교사 선택 일치율과
  "교사가 시작 정책과 다르게 고른 결정을 얼마나 따라 하는가"를 측정

사용: python src/training/train_fp_distill.py --data "data/fp_selfplay/ms100/fp_data*.npz" --out checkpoints/supervised_v2_fp_next.pt
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
import glob
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.core.model import DeepPokemonBattleTransformerNet, load_compatible, model_from_ckpt, save_ckpt
from src.core.tensor_encoder import MOVE_NUM_DIM
from src.training.replay_dataset import OBS_KEYS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = OBS_KEYS + ["action_mask", "target", "value_target", "action_taken"]


def make_batches(data, offsets, idx, size, shuffle):
    idx = list(idx)
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        sel = idx[i:i + size]
        b = {k: torch.from_numpy(np.concatenate([data[k][offsets[j]:offsets[j + 1]] for j in sel])) for k in KEYS}
        b = {k: v.float() if v.dtype == torch.float16 else v for k, v in b.items()}
        lens = [offsets[j + 1] - offsets[j] for j in sel]
        b["seq_index"] = torch.cat([torch.full((n,), s) for s, n in enumerate(lens)])
        b["time_index"] = torch.cat([torch.arange(n) for n in lens])
        b["n_seq"] = len(sel)
        yield b


def pg_loss(logp, b, value_target):
    """정책 그라디언트 보조 손실: 실제 승패에서 온 advantage(value_target, 이미 베이스라인 뺌)로
    Foul Play가 실제 그 판에서 둔 수의 확률을 가중. 배치 내 정규화로 판당 표본 1개짜리 큰 분산을 줄임.
    action_taken == -1(실제 선택이 우리 22칸에 안 매칭된 결정)은 제외."""
    valid = b["action_taken"] >= 0
    if not valid.any():
        return logp.new_zeros(())
    a_idx = b["action_taken"].clamp(min=0)
    logp_a = logp.gather(-1, a_idx.unsqueeze(-1)).squeeze(-1)
    adv = value_target
    if valid.sum() > 1:
        m, s = adv[valid].mean(), adv[valid].std().clamp_min(1e-6)
        adv = (adv - m) / s
    return -(logp_a * adv.detach() * valid.float()).sum() / valid.float().sum().clamp_min(1)


def boot_value_target(b, ref_value, args):
    """value_target을 손으로 짠 잠재함수(Φ) 대신 기준 모델(ref)의 학습된 가치 헤드로 부트스트랩.
    target(t) = γ^H·V_ref(s_{t+H}) − V_ref(s_t) (같은 판 안에서 H턴 뒤가 있을 때만).
    배틀 끝 H턴 이내라 미래 시점이 없으면 npz에 저장된 기존 Φ 기반 value_target으로 대체
    (근사 — 그 구간만 옛 방식 편향이 남음. --boot-horizon 0이면 이 함수 자체를 안 씀 = 기존 동작 그대로).
    ref_value: ref.forward_sequences(...)["value"]를 [N]으로 편 것 (no_grad로 이미 계산됨)."""
    H = args.boot_horizon
    n = ref_value.shape[0]
    idx = torch.arange(n, device=ref_value.device)
    fut = (idx + H).clamp(max=n - 1)
    same_battle = (idx + H < n) & (b["seq_index"][fut] == b["seq_index"])
    boot = (args.gamma ** H) * ref_value[fut] - ref_value
    return torch.where(same_battle, boot, b["value_target"])


def run(model, ref, data, offsets, idx, args, train, opt=None):
    model.train(bool(train and not args.no_dropout))  # 학습 중에는 드롭아웃(트랜스포머/융합층 0.1)을 켬 → 오래된 데이터를 반복해서 외우는 것을 늦춤. 검증은 항상 끔
    tot = dict(ce=0.0, v=0.0, kl=0.0, pg=0.0, agree=0, n=0, chg=0, chg_agree=0)
    for b in make_batches(data, offsets, idx, args.batch_battles, train):
        b = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.set_grad_enabled(train):
            out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
            with torch.no_grad():
                ref_out = ref.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
            vt = boot_value_target(b, ref_out["value"].view(-1), args) if args.boot_horizon > 0 else b["value_target"]
            logp = F.log_softmax(out["policy_logits"], -1)
            ce = -(b["target"] * logp).sum(-1).mean()
            v = F.mse_loss(out["value"].view(-1), vt)
            kl = F.kl_div(logp, F.softmax(ref_out["policy_logits"], -1), reduction="batchmean")
            pg = pg_loss(logp, b, vt)
            loss = ce + args.value_coef * v + args.kl_coef * getattr(args, '_kl_mult', 1.0) * kl + args.pg_coef * pg
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if getattr(args, "_sched", None) is not None:
                args._sched.step()  # 학습률 스케줄은 배치마다 갱신
        with torch.no_grad():
            want, got, ref_a = b["target"].argmax(-1), logp.argmax(-1), ref_out["policy_logits"].argmax(-1)
            chg = want != ref_a  # 탐색이 원래 정책의 선택을 바꾼 결정
        n = len(want)
        tot["ce"] += ce.item() * n; tot["v"] += v.item() * n; tot["kl"] += kl.item() * n; tot["pg"] += pg.item() * n; tot["n"] += n
        tot["agree"] += (got == want).sum().item(); tot["chg"] += chg.sum().item(); tot["chg_agree"] += ((got == want) & chg).sum().item()
    n = max(1, tot["n"])
    return {"ce": tot["ce"] / n, "v": tot["v"] / n, "kl": tot["kl"] / n, "pg": tot["pg"] / n, "agree": tot["agree"] / n,
            "chg_agree": tot["chg_agree"] / max(1, tot["chg"]), "chg_rate": tot["chg"] / n, "n": tot["n"]}


def fmt(r):
    return (f"소프트CE {r['ce']:.4f} | 탐색 선택과 일치 {r['agree']:.3f} | 탐색이 바꾼 결정({r['chg_rate']:.3f})을 따라 함 {r['chg_agree']:.3f} "
            f"| V {r['v']:.3f} | KL {r['kl']:.4f} | PG {r['pg']:.4f}")


def load(pattern, fp16=False):
    files = sorted({f for pat in pattern.split(",") for f in glob.glob(pat.strip())})  # 쉼표로 여러 패턴/파일 지정 가능
    if not files:
        raise SystemExit(f"❌ 데이터 파일 없음: {pattern}")
    parts = [np.load(f) for f in files]
    data = {}
    total = int(sum(int(p["lens"].sum()) for p in parts))
    for k in KEYS:
        # 미리 전체 크기의 배열을 잡고 파일 하나씩 채움 → 순간 최대 메모리 = 결과 + 파일 하나 (np.concatenate는 결과와 조각 전체가 동시에 필요해 데이터가 크면 RAM 초과)
        first = parts[0][k]
        # ponytail: 기술 수치 칸이 늘어난(44→45 등) 옛 npz와 새 npz를 섞어 쓸 때, 짧은 쪽은 뒤를 0으로 채워 맞춤
        # (전체 재빌드 대신 임시방편 — 옛 데이터는 새 칸이 항상 0=미지, 언젠가 데이터 재빌드하면 이 분기는 필요 없어짐)
        # my_move_num/opp_move_num만 해당: 마지막 축이 "기술당 특징 칸 수"라 파일마다 달라질 수 있음.
        # 다른 키(예: value_target)는 마지막 축이 그냥 "결정 개수"라 파일마다 다른 게 정상 — 여긴 손대면 안 됨.
        pad_key = k in ("my_move_num", "opp_move_num")
        # 목표 칸 수는 로딩된 파일들 중 최댓값이 아니라 "지금 모델이 기대하는 칸 수"(MOVE_NUM_DIM) 기준 —
        # 로딩된 파일이 전부 옛 칸 수로 통일돼 있으면 파일간 불일치가 없어서 이 분기 자체가 안 도는 버그가 있었음
        tail_dim = max([MOVE_NUM_DIM] + [p[k].shape[-1] for p in parts]) if pad_key else first.shape[-1]
        shape_tail = first.shape[1:-1] + (tail_dim,) if pad_key else first.shape[1:]
        half = fp16 and k in OBS_KEYS and first.dtype == np.float32  # fp16: 관측 실수 배열을 반정밀도로 보관 (배치를 만들 때 다시 float32로), 값 정밀도 약 3자리
        out = np.zeros((total,) + shape_tail, dtype=np.float16 if half else first.dtype)
        pos = 0
        for i, p in enumerate(parts):
            a = first if i == 0 else p[k]
            if pad_key and a.shape[-1] != tail_dim:
                print(f"  ⚠️ {files[i]}의 {k} 칸 수 {a.shape[-1]} != {tail_dim} → 뒤를 0으로 채움 (재빌드 전 임시방편)")
                out[pos:pos + len(a), ..., :a.shape[-1]] = a
            else:
                out[pos:pos + len(a)] = a
            pos += len(a)
            del a
        assert pos == total, (k, pos, total)
        data[k] = out
        del first
    lens = np.concatenate([p["lens"] for p in parts])
    return data, np.concatenate([[0], np.cumsum(lens)]), lens, files


def main(args):
    data, offsets, lens, files = load(args.data, args.fp16_store)
    ids = list(range(len(lens)))
    random.Random(0).shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val, train = ids[:n_val], ids[n_val:]
    print(f"데이터 파일 {len(files)}개 | {len(lens)}판 / 결정 {int(lens.sum())}개 → 학습 {len(train)}판, 검증 {len(val)}판")

    arch = json.loads(args.arch) if args.arch else {}
    model = DeepPokemonBattleTransformerNet(**arch)
    if args.init != "none":
        expanded, skipped = load_compatible(model, torch.load(args.init, map_location="cpu"))
        # load_compatible이 체크포인트에 깊은 헤드 키가 있으면 model.enable_deep_heads()를 자동 호출해 파라미터를 늘릴 수 있음
        # → 총 파라미터 수는 반드시 이 호출 다음에 재야 함 (먼저 재면 늘어나기 전 값이라 재사용 비율이 틀어짐)
        tot = sum(p.numel() for p in model.parameters())
        own = model.state_dict()
        exact = [k for k in own if k not in expanded and k not in skipped]
        exact_n = sum(own[k].numel() for k in exact)
        expand_n = sum(own[k].numel() for k in expanded)
        reuse_frac = (exact_n + expand_n) / tot
        print(f"  워밍스타트: 완전일치 {len(exact)}개({exact_n/1e6:.2f}M) + 확장 {len(expanded)}개({expand_n/1e6:.2f}M) "
              f"= 재사용 {reuse_frac:.1%} | 버려짐(무작위 유지) {len(skipped)}개")
        if reuse_frac < 0.05:
            print("  ⚠️ 재사용 비율이 5% 미만 — 의도한 워밍스타트가 안 됐을 수 있음 (레이어 이름/구조 확인)")
    else:
        tot = sum(p.numel() for p in model.parameters())
    # KL 기준점(ref): 기본은 시작 체크포인트. 구조가 다르거나 처음부터 학습할 때는 --aux-teacher로 다른 체크포인트(예: all11)를 지정
    ref_path = args.aux_teacher or args.init
    if ref_path == "none":
        raise SystemExit("❌ --init none이면 --aux-teacher가 필요함 (KL 기준 모델)")
    ref = model_from_ckpt(ref_path)
    print(f"모델 {tot/1e6:.2f}M 구조 {arch or '기본'} | 시작 {args.init} | KL 기준 {ref_path}")
    if args.deep_heads:
        model.enable_deep_heads()  # 시작 출력은 그대로(0 초기화). ref는 시작 정책 그대로 두어 KL 기준점이 됨
    model.to(DEVICE)
    ref.to(DEVICE).eval()
    for p in ref.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.sched == "cosine":
        # 웜업(0 → lr) 후 코사인 감쇠(lr → lr × lr_min_ratio). 높은 학습률에서 첫 에폭에 정책이 무너지는 충격을 웜업이, 마지막 지점의 흔들림을 감쇠가 줄임
        import math
        total = max(1, args.epochs * math.ceil(len(train) / args.batch_battles))
        warm = int(total * args.warmup_frac)

        def mult(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            t = (step - warm) / max(1, total - warm)
            return args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        args._sched = torch.optim.lr_scheduler.LambdaLR(opt, mult)
    print("시작 전 검증:", fmt(run(model, ref, data, offsets, val, args, train=False)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_ce, best_ep, ce_hist = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        # KL 계수 배율: 첫 에폭 1.0 → 마지막 에폭 kl_final (선형). 처음부터 학습할 때 초반엔 보조 교사를 강하게, 후반엔 약하게
        args._kl_mult = 1.0 + (args.kl_final - 1.0) * (ep - 1) / max(1, args.epochs - 1)
        tr = run(model, ref, data, offsets, train, args, train=True, opt=opt)
        va = run(model, ref, data, offsets, val, args, train=False)
        print(f"[에폭 {ep}/{args.epochs}] 학습 {fmt(tr)}\n            검증 {fmt(va)}", flush=True)
        # 최근 최대 3개 평균으로 완만화해서 최저점 판단 (검증셋이 작아 한 에폭 값만으로는 잡음이 큼)
        ce_hist.append(va["ce"])
        smoothed = sum(ce_hist[-3:]) / len(ce_hist[-3:])
        if smoothed < best_ce:
            best_ce, best_ep = smoothed, ep
            save_ckpt(model, args.out)
            print(f"  💾 최저점 갱신(완만화 CE {smoothed:.4f}) → {args.out}", flush=True)
    save_ckpt(model, args.out + ".last.pt")
    print(f"💾 최종 저장: {args.out}.last.pt | 최저점: {args.out} (에폭 {best_ep}, 완만화 CE {best_ce:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fp_selfplay/ms100/fp_data*.npz", help="build_fp_dataset.py 출력 (glob)")
    ap.add_argument("--init", default="checkpoints/supervised_v2_fp_all8a.pt", help="시작 체크포인트. 앞으로는 BC가 아니라 직전 최신 모델에서 이어서 학습")
    ap.add_argument("--out", default="checkpoints/supervised_v2_fp_next.pt")
    ap.add_argument("--arch", default="", help="모델 구조 JSON. 예: {\"pokemon_embed_dim\":192,\"history_dim\":384,\"latent_dim\":384,\"num_layers\":3,\"history_mode\":\"typed\"}. 비우면 기본 구조")
    ap.add_argument("--aux-teacher", default="", help="KL 기준 모델 체크포인트 (기본: --init). --init none(처음부터 학습)일 때 필수")
    ap.add_argument("--kl-final", type=float, default=1.0, help="마지막 에폭의 KL 계수 배율 (첫 에폭 1.0에서 선형 변화). 예: 0.2")
    ap.add_argument("--no-dropout", action="store_true", help="학습 중 드롭아웃을 끔 (2026-09-27 이전 동작). 기본은 켬")
    ap.add_argument("--fp16-store", action="store_true", help="관측 배열을 float16으로 메모리에 보관 (데이터가 커서 RAM이 모자랄 때). 값 정밀도는 약 3자리")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3, help="피크 학습률. 스윕(코사인, 학생 상태 홀드아웃 CE): 1e-3 1.5005 / 2e-3 1.4906 / 3e-3 1.4958 / 4e-3 1.5034")
    ap.add_argument("--sched", choices=["const", "cosine"], default="cosine", help="학습률 스케줄. cosine=웜업 후 코사인 감쇠(기본), const=고정 (예전 동작; 고정 lr은 높이면 붕괴)")
    ap.add_argument("--warmup-frac", type=float, default=0.05, help="cosine일 때 전체 스텝 중 웜업 비율")
    ap.add_argument("--lr-min-ratio", type=float, default=0.05, help="cosine일 때 마지막 학습률 = lr × 이 값")
    ap.add_argument("--batch-battles", type=int, default=32)
    ap.add_argument("--value-coef", type=float, default=0.25)
    ap.add_argument("--kl-coef", type=float, default=0.05, help="시작 정책과의 KL (클수록 시작 정책 유지). 최적해가 (교사 + 계수*시작정책)/(1+계수) 혼합이라 크면 시작 정책 쪽으로 끌려감")
    ap.add_argument("--pg-coef", type=float, default=0.0, help="실제 승패(value_target=advantage) 기반 정책 그라디언트 보조 손실 가중치. "
                    "0=끔(기본, 기존 동작 그대로). 장기 자산(장판/트릭룸/날씨) 신용 할당을 정책에도 흘려보내려는 실험적 항 — 작게(0.03~0.1) 시작할 것")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--deep-heads", action="store_true", help="기술/교체 점수 헤드에 깊은 잔차 가지를 추가 (model.ResHead). 이미 켜진 체크포인트는 자동 감지")
    ap.add_argument("--boot-horizon", type=int, default=0, help="value_target을 손짠 잠재함수(Φ) 대신 ref의 학습된 가치 헤드로 H턴 부트스트랩"
                    "(target=γ^H·V_ref(s+H)-V_ref(s), 배틀 끝 H턴 이내면 기존 Φ 기반 값으로 대체). 0=끔(기존 동작 그대로)")
    ap.add_argument("--gamma", type=float, default=0.995, help="--boot-horizon 할인율 (replay_dataset.py 기본값과 동일)")
    main(ap.parse_args())
