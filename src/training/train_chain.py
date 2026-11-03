"""사슬(v9) 증류 학습 (docs/chain_design.md): train_fp_distill의 손실에 사슬 보조 손실을 더함.
  L = 기존(정책 CE + 가치 + PG + 숨김정보) + λ_o·응수 CE + λ_t·테라 BCE + λ_r·결과 Δ(실제 둔 수·실제 응수 쌍) + λ_q·Q 회귀(FP 탐색 점수, 신뢰 행동만)
데이터는 build_fp_dataset 출력(opp_action/opp_move/kind가 든 data/dyn 형식). 변형은 --arch의 chain_variant: aux(정책 헤드 유지) / forced(정책 = softmax(Q/τ)).
사용: python src/training/train_chain.py --data "data/dyn/ms100_sp0*.npz" --out checkpoints/supervised_v2_fp_chainA.pt --arch '{"model":"v9","my_pos":false,"chain_variant":"aux"}' --accum 4
진단: 에폭마다 검증에서 응수 NLL(클래스 사전 분포 대비), 결과 손실과 a/o 섞었을 때의 악화(M2), Q 헤드 argmax regret(M3)을 같이 찍음"""
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
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.core.model import build_model, model_from_ckpt, save_ckpt
from src.training import train_fp_distill as T
from src.training.chain_targets import N_ACT, N_CLS, build, move_class_table
from src.training.hidden_labels import hidden_loss
from src.training.replay_dataset import OBS_KEYS

EXTRA_RAW = ["opp_action", "opp_move", "kind"]
CHAIN_KEYS = ["opp_cls", "opp_tera", "r_valid", "dhp", "faint", "act_my", "act_opp", "boost"]
DEVICE = T.DEVICE


def attach_targets(data, files, lens):
    """파일들에서 응수/종류 원본을 읽어 이어 붙이고(train_fp_distill.load와 같은 순서) 사슬 정답을 data에 추가"""
    parts = [np.load(f) for f in files]
    raw = {k: np.concatenate([p[k] for p in parts]) for k in EXTRA_RAW}
    raw["action_taken"] = data["action_taken"]
    raw["my_team_num"], raw["opp_team_num"], raw["opp_team_cat"] = data["my_team_num"], data["opp_team_num"], data["opp_team_cat"]
    t = build(raw, lens, move_class_table())
    data.update(t)
    valid = t["opp_cls"] >= 0
    prior = np.bincount(t["opp_cls"][valid], minlength=N_CLS).astype(np.float64) + 1
    data["_prior_logp"] = np.log(prior / prior.sum()).astype(np.float32)
    print(f"사슬 정답: 응수 클래스 있는 행 {valid.mean():.1%}, 결과(r) 학습 행 {t['r_valid'].mean():.1%} | 클래스 사전 {np.round(prior / prior.sum(), 3).tolist()}")


def _batches(data, offsets, idx, size, shuffle, gammas):
    """train_fp_distill.make_batches의 배틀 묶음 하나하나에 사슬 정답 열을 같이 잘라 붙임 -> (배치, 선택한 배틀 번호)"""
    idx = list(idx)
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        sel = idx[i:i + size]
        b = next(T.make_batches(data, offsets, sel, len(sel), False, gammas))
        for k in CHAIN_KEYS:
            b[k] = torch.from_numpy(np.concatenate([data[k][offsets[j]:offsets[j + 1]] for j in sel]))
        yield b, sel


def chain_losses(model, out, b, args):
    """사슬 손실 + 진단. 반환: (손실 합, 진단 dict)"""
    z = out["q_pi"].new_zeros(())
    diag = {}
    cls = b["opp_cls"].long()
    vo = cls >= 0
    lo = z
    if vo.any():
        logits = out["opp_logits"][vo]
        lo = F.cross_entropy(logits, cls[vo])
        diag["opp_nll"] = lo.detach()
        diag["opp_prior_nll"] = -b["_prior_logp"][cls[vo]].mean()
        diag["opp_acc"] = (logits.argmax(-1) == cls[vo]).float().mean()
    tera = b["opp_tera"].long()
    vt = tera >= 0
    lt = F.binary_cross_entropy_with_logits(out["opp_tera"][vt], tera[vt].float()) if vt.any() else z
    lr = z
    rv = b["r_valid"].bool()
    if rv.any():
        a = b["action_taken"][rv]
        o = cls[rv]
        h = out["chain_h"][rv]
        n = h.size(0)
        ar = torch.arange(n, device=h.device)

        def r_loss(a_, o_):
            r = model.outcome(h[ar, a_, o_])
            l = F.smooth_l1_loss(r["dhp"], b["dhp"][rv], beta=0.1) * 3 + F.binary_cross_entropy_with_logits(r["faint"], b["faint"][rv]) + \
                F.smooth_l1_loss(r["boost"], b["boost"][rv], beta=0.1)
            for key, tk in (("act_my", "act_my"), ("act_opp", "act_opp")):
                t = b[tk][rv].long()
                ok = t >= 0
                if ok.any():
                    l = l + F.cross_entropy(r[key][ok], t[ok])
            return l

        lr = r_loss(a, o)
        diag["r_loss"] = lr.detach()
        if not model.training:   # M2: 같은 배치에서 수 a / 응수 o를 섞었을 때 결과 손실이 얼마나 나빠지는가 (정답이 바뀌면 그 정보를 쓰고 있다는 뜻)
            perm = torch.randperm(n, device=h.device)
            diag["r_shuf_a"] = r_loss(a[perm], o).detach()
            diag["r_shuf_o"] = r_loss(a, o[perm]).detach()
    lq = z
    q, m = b["q_mean"][:, :N_ACT], b["action_mask"][:, :N_ACT]
    rel = (b["q_vis"][:, :N_ACT] >= T.Q_MIN_VIS) & ~torch.isnan(q) & m
    if rel.any():
        lq = F.mse_loss(out["q_pi"][:, :N_ACT][rel], q[rel]) * 10          # q 차이는 0.01~0.1 규모라 키움
        diag["q_mse"] = lq.detach() / 10
    return args.lam_o * lo + args.lam_t * lt + args.lam_r * lr + args.lam_q * lq, diag


def run(model, data, offsets, idx, args, train, opt=None):
    model.train(bool(train))
    tot = dict(ce=0.0, v=0.0, pg=0.0, hid=0.0, hid_top4=0.0, hid_nb=0, agree=0, n=0, reg=0.0, reg_n=0, reg_q=0.0, reg_base=0.0)
    dg = {}
    prior = torch.from_numpy(data["_prior_logp"]).to(DEVICE)
    for bi, (b, _) in enumerate(_batches(data, offsets, idx, T.BATCH_BATTLES // args.accum, train, getattr(model, "value_gammas", ()))):
        b = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}
        b["_prior_logp"] = prior
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
            out = {k: v.float() for k, v in out.items()}
            logp = F.log_softmax(out["policy_logits"], -1)
            ce_i = -(b["target"] * logp).sum(-1)
            ce = ce_i.mean()
            w = torch.ones_like(ce_i)
            valid, gap = T.q_terms(b)
            if train and args.hard_weight != 1:
                w = torch.where(valid & (gap.max(-1).values >= T.HARD_STAKES), torch.full_like(w, args.hard_weight), w)
            reg_arg = gap.gather(-1, logp.argmax(-1, keepdim=True)).squeeze(-1)
            reg_q = gap.gather(-1, out["q_logits"].argmax(-1, keepdim=True)).squeeze(-1)
            v = F.mse_loss(out["value"].view(-1), b["value_target"])
            pg = T.pg_loss(logp, b, b["value_target"] - out["value"].view(-1).detach())
            hid, hs = hidden_loss({"move": out["hid_move"], "item": out["hid_item"], "ability": out["hid_ability"]}, b, T.UNK)
            tot["hid_top4"] += hs.get("move_top4", 0.0) * hs["move_n"]; tot["hid_nb"] += hs["move_n"]
            cl, d = chain_losses(model, out, b, args)
            loss = (w * ce_i).sum() / w.sum() + T.VALUE_COEF * v + args.pg_coef * pg + T.HIDDEN_COEF * hid + cl
        if train:
            if bi % args.accum == 0:
                opt.zero_grad()
            (loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                for grp in args._clip_groups:
                    torch.nn.utils.clip_grad_norm_(grp, 1.0)
                opt.step()
                args._sched.step()
        with torch.no_grad():
            want, got = b["target"].argmax(-1), logp.argmax(-1)
        n = len(want)
        for k, x in d.items():
            dg[k] = dg.get(k, 0.0) + float(x) * n
        tot["ce"] += ce.item() * n; tot["v"] += v.item() * n; tot["pg"] += pg.item() * n; tot["hid"] += hid.item() * n; tot["n"] += n
        tot["agree"] += (got == want).sum().item()
        tot["reg"] += (reg_arg * valid).sum().item(); tot["reg_q"] += (reg_q * valid).sum().item(); tot["reg_n"] += valid.sum().item()
    n = max(1, tot["n"])
    r = {"reg": tot["reg"] / tot["reg_n"] if tot["reg_n"] else None, "reg_q": tot["reg_q"] / tot["reg_n"] if tot["reg_n"] else None,
         "ce": tot["ce"] / n, "v": tot["v"] / n, "pg": tot["pg"] / n, "hid": tot["hid"] / n, "hid_top4": tot["hid_top4"] / max(1, tot["hid_nb"]),
         "hidh": 0.0, "hidh_top4": 0.0, "agree": tot["agree"] / n, "n": tot["n"]}
    r.update({k: x / n for k, x in dg.items()})
    return r


def fmt(r):
    s = T.fmt(r)
    if r.get("reg_q") is not None:
        s += f" | Q-argmax regret {r['reg_q']:.4f}"
    if "opp_nll" in r:
        s += f" | 응수 NLL {r['opp_nll']:.3f} (사전 {r['opp_prior_nll']:.3f}, 정확도 {r['opp_acc']:.3f})"
    if "r_loss" in r:
        s += f" | 결과 {r['r_loss']:.3f}" + (f" (a 섞음 {r['r_shuf_a']:.3f}, o 섞음 {r['r_shuf_o']:.3f})" if "r_shuf_a" in r else "")
    if "q_mse" in r:
        s += f" | Q mse {r['q_mse']:.5f}"
    return s


def main(args):
    files = sorted({f for pat in args.data.split(",") for f in glob.glob(pat.strip())})
    data, offsets, lens, files = T.load(args.data)
    attach_targets(data, files, lens)
    if "q_mean" not in data:
        raise SystemExit("❌ q_mean/q_vis가 든 데이터가 필요")
    val, train = T.split_ids(data, offsets, T.VAL_FRAC)
    print(f"데이터 파일 {len(files)}개 | {len(lens)}판 / 결정 {int(lens.sum())}개 → 학습 {len(train)}판, 검증 {len(val)}판")
    arch = json.loads(args.arch)
    assert arch.get("model") == "v9", "--arch의 model은 v9여야 함"
    model = model_from_ckpt(args.init) if args.init else build_model(arch)
    if args.train_only:                                                   # 지정한 모듈 접두사만 학습 (나머지 동결): Q 읽기 재학습 같은 탐침용
        keys = args.train_only.split(",")
        for n, p_ in model.named_parameters():
            p_.requires_grad = n.split(".")[0] in keys
    print(f"모델 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 구조 {arch}")
    belief_params = [p for n, p in model.named_parameters() if n.startswith("belief.")]
    args._clip_groups = [belief_params, [p for n, p in model.named_parameters() if not n.startswith("belief.")]]
    pre = f"checkpoints/belief_{model.belief.num_dim}.pt"
    if not args.init:
        model.belief.load_state_dict(torch.load(pre, map_location="cpu")["state"])
    model.to(DEVICE)
    other = [p for n, p in model.named_parameters() if not n.startswith("belief.")]
    opt = torch.optim.AdamW([{"params": other, "lr": args.lr}, {"params": belief_params, "lr": args.lr * T.BELIEF_FT_LR_MULT}], weight_decay=1e-4)
    total = max(1, args.epochs * math.ceil(len(train) / T.BATCH_BATTLES))
    warm = int(total * T.WARMUP_FRAC)

    def mult(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return T.LR_MIN_RATIO + (1 - T.LR_MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    args._sched = torch.optim.lr_scheduler.LambdaLR(opt, mult)
    print("시작 전 검증:", fmt(run(model, data, offsets, val, args, train=False)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best, best_ep, hist = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        tr = run(model, data, offsets, train, args, train=True, opt=opt)
        va = run(model, data, offsets, val, args, train=False)
        print(f"[에폭 {ep}/{args.epochs}] 학습 {fmt(tr)}\n            검증 {fmt(va)}", flush=True)
        hist.append(va["reg"])
        smoothed = sum(hist[-3:]) / len(hist[-3:])
        if smoothed < best:
            best, best_ep = smoothed, ep
            save_ckpt(model, args.out)
            print(f"  💾 최저점 갱신(완만화 regret {smoothed:.4f}) → {args.out}", flush=True)
    print(f"💾 최저점: {args.out} (에폭 {best_ep}, 완만화 regret {best:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="build_fp_dataset 출력(opp_action 등 포함, data/dyn) glob, 쉼표로 여러 개")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default='{"model":"v9","my_pos":false,"chain_variant":"aux"}')
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=T.LR)
    ap.add_argument("--hard-weight", type=float, default=1.0)
    ap.add_argument("--pg-coef", type=float, default=0.05)
    ap.add_argument("--lam-o", type=float, default=0.3, help="응수 클래스 CE 가중")
    ap.add_argument("--lam-t", type=float, default=0.1, help="상대 테라 여부 BCE 가중")
    ap.add_argument("--lam-r", type=float, default=1.0, help="결과 Δ 손실 가중")
    ap.add_argument("--init", default=None, help="체크포인트 경로에서 시작 (없으면 처음부터)")
    ap.add_argument("--train-only", default=None, help="쉼표로 구분한 모듈 접두사(예: q_out)만 학습하고 나머지는 동결")
    ap.add_argument("--lam-q", type=float, default=1.0, help="Q 회귀 가중 (손실 안에서 x10 스케일)")
    main(ap.parse_args())
