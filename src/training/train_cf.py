"""M5 후속 (docs/chain_m5.md): 시뮬레이터 반사실 라벨(안 둔 수 포함)로 결과 헤드 r(s,a,o)를 학습.
라벨 = tools/cf_gen.js의 (위치, 행동 a, 응수 클래스 o)별 한 턴 뒤 HP 변화/새 기절의 반복 평균 (깨끗한 칸만). 위치 텐서는 cf_run.sh가 만든 npz.
내 쪽은 행동 상대형 스칼라 r_my(필드의 내 포켓몬 HP 변화/기절)로 학습하고, 체크포인트 이름에 _rt를 넣어 cf_eval이 배정 평가를 쓰게 함.
--mode head: 몸통 동결, 결과 경로(act/o 임베딩, w_*, chain2, r_out)만 학습  -> 표현이 이미 갖고 있나
--mode all : 몸통도 작은 학습률로 같이 학습 (정책 헤드 보존 확인은 regret_eval로 따로)
사용: python src/training/train_cf.py --init chain_p1 --cf data/cf/train.jsonl --npz data/cf/train.npz --out chain_p1_cf --mode head
평가: python tools/cf_eval.py eval data/cf/main.jsonl data/cf/pos.npz chain_p1_cf   (보류한 홀드아웃 팀 풀 위치)"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import cf_eval as E
from src.core.model import model_from_ckpt, save_ckpt
from src.training import chain_targets as CT
from src.training.replay_dataset import OBS_KEYS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEAD = ("act_mv", "act_sw", "tera_emb", "o_emb", "w_lat", "w_act", "w_opp", "chain2", "r_out", "r_my")


def labels(cf, npz):
    """시퀀스 k -> (a [C], o [C], dhp [C,12] (대응 안 되는 슬롯 nan), faint [C,12]). 깨끗하지 않은 칸은 뺌"""
    P, z, ids, off = E.load(cf, npz, cells=False)
    want = {pid: k for k, pid in enumerate(ids)}
    out = {}
    for line in open(cf, encoding="utf-8"):
        p = json.loads(line)
        k = want.get(p["id"])
        if k is None:
            continue
        r = off[k + 1] - 1
        mm, om = E.slot_map(z["my_team_cat"][r], p["pre"]["p1"]), E.slot_map(z["opp_team_cat"][r], p["pre"]["p2"])
        if None in mm:
            continue
        inv = {nm: j for j, nm in enumerate(mm)}
        A = [E.act_index(a, inv) for a in p["acts"]]
        rows = []
        for c in p["cells"]:
            if A[c["a"]] is None or A[c["a"]] >= CT.N_ACT or any(x["req"] != "move" for x in c["reps"]):
                continue
            ds = [E.sim_delta(p["pre"], x, mm, om) for x in c["reps"]]
            a_ = A[c["a"]]
            tgt = a_ - 8 if a_ >= 8 else int(np.argmax(z["my_team_num"][r][:, 0]))        # 피해를 받는 내 슬롯 (행동에서 정해짐)
            rows.append((a_, p["opps"][c["o"]]["cls"], np.mean([d[0] for d in ds], 0), np.mean([d[1] for d in ds], 0),
                         np.mean([d[0][tgt] for d in ds]), np.mean([d[1][tgt] for d in ds])))
        if rows:
            out[k] = tuple(np.array(x) for x in zip(*rows))
    return z, off, out


def batch(z, off, ks):
    rows = np.concatenate([np.arange(off[k], off[k + 1]) for k in ks])
    seq = np.concatenate([np.full(off[k + 1] - off[k], i) for i, k in enumerate(ks)])
    t = np.concatenate([np.arange(off[k + 1] - off[k]) for k in ks])
    last = np.cumsum([off[k + 1] - off[k] for k in ks]) - 1
    obs = [torch.tensor(z[key][rows]).to(DEVICE) for key in OBS_KEYS]
    return obs, torch.tensor(seq).to(DEVICE), torch.tensor(t).to(DEVICE), torch.tensor(z["action_mask"][rows]).to(DEVICE), last


def step_loss(model, z, off, lab, ks):
    obs, seq, t, mask, last = batch(z, off, ks)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        out = model.forward_sequences(obs, seq, t, len(ks), action_mask=mask)
    h = out["chain_h"].float()
    hs, tg_h, tg_f, tg_m, tg_mf = [], [], [], [], []
    for i, k in enumerate(ks):
        a, o, dh, fn, mh, mf = lab[k]
        tg_m.append(torch.tensor(mh, dtype=torch.float32)); tg_mf.append(torch.tensor(mf, dtype=torch.float32))
        hs.append(h[last[i]][torch.tensor(a), torch.tensor(o)])
        tg_h.append(torch.tensor(dh, dtype=torch.float32)); tg_f.append(torch.tensor(fn, dtype=torch.float32))
    o_ = model.outcome(torch.cat(hs))
    th, tf = torch.cat(tg_h).to(DEVICE), torch.cat(tg_f).to(DEVICE)
    tm, tmf = torch.cat(tg_m).to(DEVICE), torch.cat(tg_mf).to(DEVICE)
    v = ~torch.isnan(th)
    v[:, :6] = False                                                                   # 내 쪽은 슬롯 벡터 대신 행동 상대형 스칼라(r_my)로 학습
    mse = F.mse_loss(o_["dhp"][v], th[v])
    bce = F.binary_cross_entropy_with_logits(o_["faint"][v], tf[v])
    mse_m = F.mse_loss(o_["my_hp"], tm)
    bce_m = F.binary_cross_entropy_with_logits(o_["my_faint"], tmf)
    zero = (th[v] ** 2).mean()
    return 10 * (mse + mse_m) + bce + bce_m, mse.item(), zero.item(), bce.item(), mse_m.item(), (tm ** 2).mean().item()


def run(model, z, off, lab, ks, train, opt=None, sched=None, bs=32):
    import random
    ks = list(ks)
    if train:
        random.shuffle(ks)
    tot = np.zeros(5); n = 0
    for i in range(0, len(ks), bs):
        with torch.set_grad_enabled(train):
            loss, mse, zero, bce, mm_, mz = step_loss(model, z, off, lab, ks[i:i + bs])
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
        tot += [mse, zero, bce, mm_, mz]; n += 1
    mse, zero, bce, mm_, mz = tot / n
    return dict(mse=mse, zero=zero, red=1 - mse / zero, bce=bce, my_red=1 - mm_ / mz)             # red = 상대 슬롯, my_red = 내 쪽(배정된 슬롯) 오차 감소율


def main(a):
    z, off, lab = labels(a.cf, a.npz)
    ks = sorted(lab)
    print(f"학습 가능 위치 {len(ks)} / 칸 {sum(len(lab[k][0]) for k in ks)}")
    val = ks[-a.val:]; tr = ks[:-a.val]
    model = model_from_ckpt(f"checkpoints/supervised_v2_fp_{a.init}.pt").to(DEVICE)
    for n, p in model.named_parameters():
        p.requires_grad = n.split(".")[0] in HEAD or a.mode == "all"
    head = [p for n, p in model.named_parameters() if n.split(".")[0] in HEAD]
    rest = [p for n, p in model.named_parameters() if n.split(".")[0] not in HEAD and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head, "lr": a.lr}] + ([{"params": rest, "lr": a.trunk_lr}] if rest else []), weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(tr) / 32)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    model.eval()                                                                       # 드롭아웃 등 끔 (몸통 분포를 흔들지 않게)
    print("시작 전 검증:", run(model, z, off, lab, val, False))
    best = -1
    for ep in range(1, a.epochs + 1):
        r = run(model, z, off, lab, tr, True, opt, sched)
        v = run(model, z, off, lab, val, False)
        print(f"[에폭 {ep}/{a.epochs}] 학습 {r}\n            검증 {v}", flush=True)
        if v["red"] + v["my_red"] > best:
            best = v["red"] + v["my_red"]; save_ckpt(model, f"checkpoints/supervised_v2_fp_{a.out}.pt"); print("  💾 저장", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--cf", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["head", "all"], default="head"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--trunk-lr", type=float, default=3e-5); ap.add_argument("--val", type=int, default=150)
    main(ap.parse_args())
