"""
사람 대전 데이터(data/human/gen9ou_s0_r1600_*.npz, tools/parse_human.py 출력)로 모델을 처음부터 행동 복제(BC) 학습.
(마지막 기술 입력 prev_act 비교 실험은 이득이 없어 모델에서 제거됨. 지금은 같은 구조의 사람 데이터 BC 기준선만 학습)

- 정책 손실: 사람이 고른 행동(22칸)의 NLL, 라벨 없는 결정(action=-1)은 상태만 히스토리에 쓰고 손실에서 뺌
- 보조 손실(두 모델 공통): 상대의 이번 턴 행동 종류(기술/교체/테라)와 상대가 쓴 기술 id를 latent에서 예측 (+ 가치 MSE)
- 입력으로 쓰는 마지막 기술 = 이전 턴 시작 결정들의 own_move/opp_move (현재 결정의 값은 목표라서 쓰지 않음)
- 분할: 판 단위 (game*7919%10: 0 테스트, 1 검증, 나머지 학습; 같은 판의 두 시점은 같은 쪽). 최저 검증 NLL 에폭을 저장하고, 끝에 테스트 + 입력/히스토리 제거 실험
- 세트 추정(belief)은 팀 데이터로 사전학습한 가중치를 불러와 얼림 (정책 경로에서 어차피 detach돼서 기울기가 안 가고, 숨김정보 정답은 사람 데이터에 없음)

사용: PYTHONPATH=. python src/training/train_human.py --tag h1 --arch '{"model":"v8","my_pos":false}'
"""
import argparse
import glob
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import MAX_TURNS, build_model, load_compatible, model_from_ckpt, save_ckpt
from src.training.replay_dataset import OBS_KEYS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR, BATCH_SEQ, VALUE_COEF, AUX_COEF = 1e-3, 32, 0.25, 0.2
WARMUP_FRAC, LR_MIN_RATIO, GAMMA = 0.05, 0.05, 0.995      # 학습률 스케줄과 가치 할인율은 train_fp_distill.py와 같음
CLS = np.array([0] * 4 + [2] * 4 + [1] * 14)              # 22칸 → 종류 (0 기술, 1 교체, 2 테라 기술)
PER_DECISION = ("action_mask", "action", "turn", "opp_action", "opp_move", "own_move")


GAME_STRIDE = 100000   # 판 번호는 샤드마다 0부터라서, 샤드 번호를 곱해 더해 전체에서 유일하게 (샤드 0은 그대로)


def load_files(files):
    parts = [np.load(f) for f in files]
    lens = np.concatenate([p["lens"] for p in parts])
    n = int(lens.sum())
    d = {}
    for k in OBS_KEYS + list(PER_DECISION):
        first = parts[0][k]
        out = np.zeros((n,) + first.shape[1:], dtype=np.float16 if k in OBS_KEYS and first.dtype == np.float32 else np.int16 if k.endswith("_cat") else first.dtype)
        pos = 0
        for p in parts:
            a = p[k]
            out[pos:pos + len(a)] = a
            pos += len(a)
        d[k] = out
    for k in ("outcome", "rating"):
        d[k] = np.concatenate([p[k] for p in parts])
    d["game"] = np.concatenate([p["game"] + GAME_STRIDE * int(re.search(r"_s(\d+)_", f).group(1)) for p, f in zip(parts, files)])
    return d, lens


def subset(d, lens, ids):
    """시퀀스 번호 ids만 복사해 새 (d, lens)로"""
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    pos = np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in ids])
    return {k: v[pos] if k in OBS_KEYS or k in PER_DECISION else v[ids] for k, v in d.items()}, lens[ids]


def merge(parts):
    return {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}, np.concatenate([p[1] for p in parts])


def split_ids(d):
    """판 단위 분할 (game*7919%10: 0 테스트, 1 검증, 나머지 학습). 같은 판의 두 시점은 같은 쪽"""
    s = d["game"] * 7919 % 10
    return np.where(s >= 2)[0], np.where(s == 1)[0], np.where(s == 0)[0]


def last_moves(d, lens):
    """결정마다 [직전까지 마지막으로 쓴 내 기술 id, 상대 기술 id] (0 = 아직 없음). 턴 시작 결정에서만 갱신 (기절 후 교체 지점은 같은 턴), 현재 결정의 값은 넣지 않음"""
    turn, own, opp = d["turn"].astype(np.int64), d["own_move"].astype(np.int64), d["opp_move"].astype(np.int64)
    lm = np.zeros((len(turn), 2), np.int64)
    forced = np.zeros(len(turn), bool)
    pos = 0
    for n in lens:
        last_m = last_o = 0
        for j in range(pos, pos + n):
            lm[j] = (last_m, last_o)
            if j == pos or turn[j] != turn[j - 1]:
                last_m, last_o = own[j] or last_m, opp[j] or last_o
            else:
                forced[j] = True
        pos += n
    return lm, forced


class Aux(nn.Module):
    def __init__(self, latent, n_mv):
        super().__init__()
        self.cls, self.mv = nn.Linear(latent, 3), nn.Linear(latent, n_mv)


def batches(d, lm, starts, lens, seq_ids, size, shuffle):
    seq_ids = list(seq_ids)
    if shuffle:
        random.shuffle(seq_ids)
    for i in range(0, len(seq_ids), size):
        sel = seq_ids[i:i + size]
        pos = np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in sel])
        b = {k: torch.from_numpy(d[k][pos]) for k in OBS_KEYS + list(PER_DECISION)}
        b = {k: v.float() if v.dtype == torch.float16 else v.long() if k.endswith("_cat") or k in ("action", "opp_action", "opp_move") else v for k, v in b.items()}
        b["prev_move"] = torch.from_numpy(lm[pos])
        b["pos"] = torch.from_numpy(pos)
        b["seq_index"] = torch.cat([torch.full((int(lens[s]),), j) for j, s in enumerate(sel)])
        b["time_index"] = torch.cat([torch.arange(int(lens[s])) for s in sel])
        b["value_target"] = torch.cat([GAMMA ** torch.arange(int(lens[s]) - 1, -1, -1, dtype=torch.float32) * float(d["outcome"][s]) for s in sel])
        b["n_seq"] = len(sel)
        yield b


def step(model, aux, b):
    """한 배치의 손실 묶음"""
    b = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
    out = {k: v.float() for k, v in out.items()}
    logp = F.log_softmax(out["policy_logits"], -1)
    lab = b["action"] >= 0
    nll = torch.where(lab, -logp.gather(-1, b["action"].clamp(min=0)[:, None]).squeeze(-1), torch.zeros_like(logp[:, 0]))
    v = F.mse_loss(out["value"].view(-1), b["value_target"])
    cls_lp, mv_lp = F.log_softmax(aux.cls(out["latent"]), -1), F.log_softmax(aux.mv(out["latent"]), -1)
    oc, om = b["opp_action"] >= 0, b["opp_move"] > 0
    cls_nll = -cls_lp.gather(-1, b["opp_action"].clamp(min=0)[:, None]).squeeze(-1)
    mv_nll = -mv_lp.gather(-1, b["opp_move"].clamp(min=0)[:, None]).squeeze(-1)
    loss = nll.sum() / lab.sum().clamp_min(1) + VALUE_COEF * v + AUX_COEF * ((cls_nll * oc).sum() / oc.sum().clamp_min(1) + (mv_nll * om).sum() / om.sum().clamp_min(1))
    return loss, {"nll": nll.detach(), "lab": lab, "logp": logp.detach(), "v": v.detach(), "cls": cls_nll.detach() * oc, "oc": oc, "mv": mv_nll.detach() * om, "om": om}


def evaluate(model, aux, d, lm, starts, lens, seq_ids, nll_out=None):
    """검증/테스트: 라벨 있는 결정의 사람 행동 NLL·일치율, 상대 행동 종류/기술 보조 NLL, 가치 MSE. nll_out(전체 결정 길이 배열)에 결정별 NLL 기록"""
    model.eval()
    t = dict(nll=0.0, agree=0.0, n=0, cls=0.0, nc=0, mv=0.0, nm=0, v=0.0, nd=0)
    with torch.no_grad():
        for b in batches(d, lm, starts, lens, seq_ids, BATCH_SEQ, False):
            _, r = step(model, aux, b)
            lab = r["lab"].cpu()
            t["nll"] += r["nll"].sum().item(); t["n"] += lab.sum().item()
            t["agree"] += ((r["logp"].argmax(-1) == b["action"].to(DEVICE)) & r["lab"]).sum().item()
            t["cls"] += r["cls"].sum().item(); t["nc"] += r["oc"].sum().item(); t["mv"] += r["mv"].sum().item(); t["nm"] += r["om"].sum().item()
            t["v"] += r["v"].item() * len(lab); t["nd"] += len(lab)
            if nll_out is not None:
                nll_out[b["pos"].numpy()] = np.where(lab.numpy(), r["nll"].cpu().numpy(), np.nan)
    return {"nll": t["nll"] / max(1, t["n"]), "agree": t["agree"] / max(1, t["n"]), "cls": t["cls"] / max(1, t["nc"]), "mv": t["mv"] / max(1, t["nm"]), "v": t["v"] / max(1, t["nd"])}


def fmt(r):
    return f"사람 행동 NLL {r['nll']:.4f} | 일치율 {r['agree']:.3f} | 상대 종류 {r['cls']:.4f} | 상대 기술 {r['mv']:.3f} | V {r['v']:.3f}"


def prepare(d, lens):
    lm, forced = last_moves(d, lens)
    return lm, forced, np.concatenate([[0], np.cumsum(lens)[:-1]])


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    files = sorted(glob.glob(args.data))
    n_chunk = args.chunk_files or len(files)
    chunks = lambda fs: [fs[i:i + n_chunk] for i in range(0, len(fs), n_chunk)]
    vparts, tparts, n_tr, n_seq, n_dec, n_lab, cache = [], [], 0, 0, 0, 0, {}
    for c in chunks(files):      # 검증/테스트 판은 한 번 메모리에 모아 두고, 학습 판은 에폭마다 조각 단위로 다시 읽음 (조각이 하나면 그대로 보관)
        d, lens = load_files(c)
        lab = d["action"] >= 0
        assert d["action_mask"][np.where(lab)[0], d["action"][lab]].all(), "라벨이 합법 마스크 밖"
        tr, va, te = split_ids(d)
        vparts.append(subset(d, lens, va)); tparts.append(subset(d, lens, te))
        n_tr += len(tr); n_seq += len(lens); n_dec += len(lab); n_lab += int(lab.sum())
        if len(files) == len(c):
            cache["all"] = (d, lens)
    vd, vlens = merge(vparts); td, tlens = merge(tparts)
    vlm, _, vstarts = prepare(vd, vlens)
    tlm, tforced, tstarts = prepare(td, tlens)
    va, te = np.arange(len(vlens)), np.arange(len(tlens))
    print(f"조각(파일) {len(files)}개, 한 번에 {n_chunk}개 | 판(시점) {n_seq}, 결정 {n_dec} (라벨 {n_lab}) → 학습 {n_tr} / 검증 {len(va)} / 테스트 {len(te)} 시퀀스, 시드 {args.seed}", flush=True)

    def train_batches():
        for c in chunks(random.sample(files, len(files))):
            d, lens = cache["all"] if "all" in cache else load_files(c)
            lm, _, starts = prepare(d, lens)
            yield from batches(d, lm, starts, lens, split_ids(d)[0], BATCH_SEQ // args.accum, True)

    model = build_model(json.loads(args.arch))
    assert not getattr(model, "token_head", False), "token_head 모델은 가치 헤드/잠재벡터가 없어 이 학습기 미지원 (train_fp_distill.py만)"
    if args.init:      # 다른 체크포인트(예: FP 증류 v8_full)에서 시작: 구조가 같은 가중치만 가져오고 새 모듈은 무작위로 남김
        load_compatible(model, torch.load(args.init, map_location="cpu"))
    model.belief.load_state_dict(torch.load(f"checkpoints/belief_{model.belief.num_dim}.pt", map_location="cpu")["state"])
    for p in model.belief.parameters():
        p.requires_grad_(False)
    model.to(DEVICE)
    aux = Aux(model.latent_dim, model.n_mv).to(DEVICE)
    params = [p for p in list(model.parameters()) + list(aux.parameters()) if p.requires_grad]
    print(f"모델 {sum(p.numel() for p in params) / 1e6:.2f}M 학습 가능, 구조 {model.cfg}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total = args.epochs * (math.ceil(n_tr / BATCH_SEQ) + len(chunks(files)))   # 조각 끝의 덜 찬 묶음 때문에 에폭당 최대 조각 수만큼 더 많음 (스케줄은 끝에서 최솟값으로 고정)
    warm = int(total * WARMUP_FRAC)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else LR_MIN_RATIO + (1 - LR_MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    ckpt = f"checkpoints/human_{args.tag}.pt"
    best = (1e9, 0)
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = [0.0, 0]; opt.zero_grad()
        for i, b in enumerate(train_batches()):
            loss, r = step(model, aux, b)
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); sched.step()
                if (i + 1) // args.accum % 500 == 0:
                    print(f"  에폭 {ep} 스텝 {(i + 1) // args.accum} ({time.time() - t0:.0f}s) 학습 NLL {run[0] / run[1]:.4f}", flush=True)
            run[0] += r["nll"].sum().item(); run[1] += r["lab"].sum().item()
        v = evaluate(model, aux, vd, vlm, vstarts, vlens, va)
        print(f"[에폭 {ep}/{args.epochs}] 학습 NLL {run[0] / run[1]:.4f} | 검증 {fmt(v)} ({time.time() - t0:.0f}s)", flush=True)
        if v["nll"] < best[0]:
            best = (v["nll"], ep)
            save_ckpt(model, ckpt)
            torch.save(aux.state_dict(), ckpt + ".aux")
            print(f"  💾 최저 검증 NLL {v['nll']:.4f} → {ckpt}", flush=True)
    model = model_from_ckpt(ckpt).to(DEVICE)
    aux.load_state_dict(torch.load(ckpt + ".aux", map_location=DEVICE))
    final(model, aux, td, tlm, tforced, tstarts, tlens, te, args, best)


def final(model, aux, d, lm, forced, starts, lens, te, args, best):
    """최저 검증 에폭 모델의 테스트 지표 + 제거 실험 (마지막 기술 입력 0, 히스토리 0, 히스토리 최근 1턴)"""
    nll = np.full(len(d["action"]), np.nan, np.float32)
    lines = [f"[{args.tag}] 구조 {args.arch} 시드 {args.seed}, 최저 검증 NLL {best[0]:.4f} (에폭 {best[1]})"]
    full = evaluate(model, aux, d, lm, starts, lens, te, nll_out=nll)
    lines.append(f"테스트 전체: {fmt(full)}")
    pos = np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in te])
    act, own, ok = d["action"][pos], d["own_move"][pos].astype(np.int64), ~np.isnan(nll[pos])
    rep = ok & (own > 0) & (own == lm[pos, 0]) & ~forced[pos]
    mv = ok & (own > 0) & (own != lm[pos, 0]) & (lm[pos, 0] > 0) & ~forced[pos]
    kind = CLS[np.maximum(act, 0)]
    for name, m in (("교체 선택", ok & (kind == 1)), ("기술 선택(테라 포함)", ok & (kind != 1)), ("같은 기술 반복(마지막 기술 = 이번 기술)", rep), ("다른 기술", mv), ("강제 교체 지점 제외", ok & ~forced[pos])):
        lines.append(f"  {name:28s} n={m.sum():6d} NLL {nll[pos][m].mean():.4f}")
    rating = np.repeat(d["rating"][te], lens[te])
    for lo, hi in ((0, 1600), (1600, 1700), (1700, 1800), (1800, 9999)):
        m = ok & (rating >= lo) & (rating < hi)
        if m.sum():
            lines.append(f"  레이팅 {lo}-{hi}: n={m.sum():6d} NLL {nll[pos][m].mean():.4f}")
    np.savez_compressed(f"logs/human_{args.tag}_test.npz", nll=nll[pos], pos=pos, game=np.repeat(d["game"][te], lens[te]))
    orig = model._history
    zero = lambda ts: torch.zeros(ts.size(0), ts.size(1), model.history_out.out_features, device=ts.device, dtype=ts.dtype)

    def window1(ts):
        T = ts.size(1)
        x = model.turn_proj(ts) + model.turn_pos(torch.arange(T, device=ts.device).clamp(max=MAX_TURNS - 1))
        mask = torch.full((T, T), float("-inf"), device=ts.device).fill_diagonal_(0)
        return model.history_out(model.history_seq(x, mask=mask))

    for name, fn in (("히스토리 출력 0", zero), ("히스토리 최근 1턴", window1)):
        model._history = fn
        r = evaluate(model, aux, d, lm, starts, lens, te)
        lines.append(f"테스트 {name}: {fmt(r)}  (ΔNLL {r['nll'] - full['nll']:+.4f})")
    model._history = orig
    open(f"logs/human_{args.tag}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/human/gen9ou_s0_r1600_*.npz")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default='{"model":"v8","my_pos":false}')
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=LR, help="피크 학습률 (미세조정은 낮게)")
    ap.add_argument("--init", default="", help="시작 가중치 체크포인트 (기본: 처음부터)")
    ap.add_argument("--accum", type=int, default=1, help="기울기 누적 횟수: 판 32개를 이 수로 나눠 처리해 VRAM을 줄임 (8GB GPU에서 윈도우 화면 프로세스와 합쳐 넘치면 시스템 RAM으로 넘쳐 5배 이상 느려짐). 32의 약수")
    ap.add_argument("--chunk-files", type=int, default=0, help="학습 판을 이 수의 파일씩 읽어 RAM을 아낌 (0 = 전부 한 번에). 파일 1개 = 1000판, RAM 약 0.4GB")
    main(ap.parse_args())
