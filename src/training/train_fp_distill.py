"""
Foul Play(엔진 탐색 교사) 증류 학습 — RL 없이 교사의 탐색 분포(혼합 전략)를 정책이 그대로 배움.

- 데이터: build_fp_dataset.py 출력 npz (실시간 인코더 입력 + Foul Play의 전체 탐색 분포 + 가치 목표), 여러 파일 가능
- 손실: 소프트 CE(교사 분포) + 가치 MSE + 정책 그라디언트 보조 + 숨김정보. 항상 처음부터 학습(코사인 학습률, 피크 1e-3). 검증 배틀에서 교사 선택 일치율과 regret(Q가 있을 때)을 측정
- 세트 추정 모듈(belief)은 팀 데이터로 사전학습한 checkpoints/belief_<수치 폭>.pt를 불러와 낮은 학습률로 미세조정 (없으면 pretrain_belief.py를 먼저 돌릴 것)

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

from src.core.model import build_model, load_compatible, save_ckpt
from src.core.tensor_encoder import MOVE_NUM_DIM
from src.training.hidden_labels import hidden_loss, vocab_sizes
from src.training.replay_dataset import OBS_KEYS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNK = vocab_sizes()
BELIEF_FT_LR_MULT = 0.1   # 사전학습된 세트 추정의 미세조정 학습률 배율 (배틀 상황 입력만 새로 배우면 됨)
LR = 1e-3                 # 피크 학습률. 스윕(코사인, 학생 상태 홀드아웃 CE): 1e-3 1.5005 / 2e-3 1.4906 / 3e-3 1.4958 / 4e-3 1.5034 (표준 레시피는 1e-3)
BATCH_BATTLES, VALUE_COEF, VAL_FRAC = 32, 0.25, 0.1
WARMUP_FRAC, LR_MIN_RATIO = 0.05, 0.05   # 학습률: 웜업(0 → lr) 후 코사인 감쇠(lr → lr x LR_MIN_RATIO). 높은 학습률에서 첫 에폭에 정책이 무너지는 충격을 웜업이, 마지막 지점의 흔들림을 감쇠가 줄임
HIDDEN_COEF = 1.0   # 숨김정보 손실 가중치: 세트 추정 모듈 파라미터에만 닿고(입출력 detach) 기울기 클리핑도 따로라서 정책 학습과 간섭하지 않음
KEYS = OBS_KEYS + ["action_mask", "target", "action_taken"]
CAT_KEYS = ("my_team_cat", "opp_team_cat")   # 어휘 번호 열: 메모리에는 int16로 두고 배치를 만들 때 long으로
Q_KEYS = ["q_mean", "q_vis"]   # 행동별 탐색 가치/방문 수 (build_fp_dataset 새 형식, 모든 파일에 있을 때만 로드): regret 계산(체크포인트 선택 기준)에 씀
HARD_STAKES = 0.3             # --hard-weight 대상: 결정의 중요도(Q* - 최저Q)가 이 이상 (tools/regret_breakdown.py의 '중요도 >=0.3' 구간과 같음)
Q_MIN_VIS = 100                # 이 방문 수 이상인 행동의 Q만 믿음 (tools/regret_eval.py의 기본값과 같음)
GAMMA = 0.995  # replay_dataset.py / build_fp_dataset.py와 동일


def make_batches(data, offsets, idx, size, shuffle):
    idx = list(idx)
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        sel = idx[i:i + size]
        b = {k: torch.from_numpy(np.concatenate([data[k][offsets[j]:offsets[j + 1]] for j in sel])) for k in KEYS + [k for k in Q_KEYS if k in data]}
        b = {k: v.float() if v.dtype == torch.float16 else v.long() if k in CAT_KEYS else v for k, v in b.items()}
        lens = [offsets[j + 1] - offsets[j] for j in sel]
        # 가치 타겟 = 할인된 최종 결과 γ^(남은 턴)·(±1) → 가치 헤드가 승패 기대값(승률)을 직접 배움 (예전 저장값 value_target은 Φ를 뺀 차이값이라 안 씀)
        b["value_target"] = torch.cat([GAMMA ** torch.arange(n - 1, -1, -1, dtype=torch.float32) * float(data["outcome"][j]) for j, n in zip(sel, lens)])
        if "teacher_value" in data:      # 교사(탐색)가 결정마다 본 승률 0~1 → ±1 척도 (build_fp_dataset로 만든 새 데이터에만 있음)
            b["teacher_value"] = torch.from_numpy(np.concatenate([data["teacher_value"][offsets[j]:offsets[j + 1]] for j in sel])).float() * 2 - 1
        b["seq_index"] = torch.cat([torch.full((n,), s) for s, n in enumerate(lens)])
        b["time_index"] = torch.cat([torch.arange(n) for n in lens])
        b["n_seq"] = len(sel)
        yield b


def pg_loss(logp, b, adv):
    """정책 그라디언트 보조 손실: 실제 승패에서 온 advantage(adv = 할인된 결과 − 현재 가치 예측)로
    Foul Play가 실제 그 판에서 둔 수의 확률을 가중. 배치 내 정규화로 판당 표본 1개짜리 큰 분산을 줄임.
    action_taken == -1(실제 선택이 우리 22칸에 안 매칭된 결정)은 제외."""
    valid = b["action_taken"] >= 0
    if not valid.any():
        return logp.new_zeros(())
    a_idx = b["action_taken"].clamp(min=0)
    logp_a = logp.gather(-1, a_idx.unsqueeze(-1)).squeeze(-1)
    if valid.sum() > 1:
        m, s = adv[valid].mean(), adv[valid].std().clamp_min(1e-6)
        adv = (adv - m) / s
    return -(logp_a * adv.detach() * valid.float()).sum() / valid.float().sum().clamp_min(1)


def q_terms(b):
    """행동별 탐색 가치(q_mean/q_vis)에서 (유효 결정 [N] = 믿을 행동 2개 이상, 손해 gap [N,22]).
    gap = Q* - Q_a (믿을 수 있는 행동 중 최대 Q 기준), 믿을 수 없는 합법 행동은 그 결정의 최대 손해로 침 (tools/regret_eval.py와 같은 정의)"""
    q, m = b["q_mean"], b["action_mask"]
    rel = (b["q_vis"] >= Q_MIN_VIS) & ~torch.isnan(q) & m
    valid = rel.sum(-1) >= 2
    qstar = torch.where(rel, q, torch.full_like(q, float("-inf"))).max(-1).values
    qmin = torch.where(rel, q, torch.full_like(q, float("inf"))).min(-1).values
    gap = torch.where(rel, qstar[:, None] - q, (qstar - qmin)[:, None])
    return valid, torch.where(valid[:, None], gap, torch.zeros_like(gap))


def run(model, data, offsets, idx, args, train, opt=None):
    model.train(bool(train))  # 학습 중에는 드롭아웃을 켬(융합층 0.1, 직전 기술 임베딩 0.3, 세트 추정 종 0.3. 트렁크는 기본 0.0). 검증은 항상 끔
    tot = dict(ce=0.0, v=0.0, pg=0.0, hid=0.0, hid_top4=0.0, hid_nb=0, agree=0, n=0, reg=0.0, reg_n=0)
    for bi, b in enumerate(make_batches(data, offsets, idx, BATCH_BATTLES // args.accum, train)):
        b = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):   # 손실 계산은 아래에서 float32로 (큰 모델의 메모리/시간 절약)
                out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
            out = {k: v.float() for k, v in out.items()}
            logp = F.log_softmax(out["policy_logits"], -1)
            ce_i = -(b["target"] * logp).sum(-1)
            ce = ce_i.mean()                               # 기록용 정책 손실: 교사 방문 비율에 대한 소프트 CE (가중 없음)
            w = torch.ones_like(ce_i)
            if "q_mean" in b:
                valid, gap = q_terms(b)
                if train and args.hard_weight != 1:        # 중요한 결정(Q* - 최저Q >= HARD_STAKES)의 학습 손실 가중. 평균 가중 1로 정규화해 학습률 효과는 그대로
                    w = torch.where(valid & (gap.max(-1).values >= HARD_STAKES), torch.full_like(w, args.hard_weight), w)
                reg_arg = gap.gather(-1, logp.argmax(-1, keepdim=True)).squeeze(-1)              # argmax 선택의 손해 (평가/체크포인트 선택용)
            v = F.mse_loss(out["value"].view(-1), b["value_target"])           # 기록/비교용: 항상 할인된 결과 기준
            if args.teacher_value_mix == 0:
                v_loss = v
            else:
                tv = b["teacher_value"]
                v_loss = F.mse_loss(out["value"].view(-1), torch.where(torch.isnan(tv), b["value_target"], args.teacher_value_mix * tv + (1 - args.teacher_value_mix) * b["value_target"]))
            pg = pg_loss(logp, b, b["value_target"] - out["value"].view(-1).detach())
            hid = ce.new_zeros(())
            if "hid_move" in out:      # 상대의 숨겨진 기술/도구/특성 예측 보조 손실 (정답: 같은 배틀에서 나중에 드러난 것)
                hid, hs = hidden_loss({"move": out["hid_move"], "item": out["hid_item"], "ability": out["hid_ability"]}, b, UNK)
                tot["hid_top4"] += hs.get("move_top4", 0.0) * hs["move_n"]; tot["hid_nb"] += hs["move_n"]
            loss = (w * ce_i).sum() / w.sum() + args.value_coef * v_loss + args.pg_coef * pg + HIDDEN_COEF * hid
        if train:
            # ponytail: --accum k면 배틀 32개를 k개로 나눠 기울기를 누적(VRAM 절약). 미세 배치를 같은 가중으로 평균(결정 수 차이는 무시), 에폭 끝의 덜 찬 묶음은 버림
            if bi % args.accum == 0:
                opt.zero_grad()
            (loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                for grp in args._clip_groups:   # 세트 추정 모듈은 자기 손실로만 학습하므로 기울기 클리핑도 따로 (큰 숨김정보 손실이 정책 기울기를 누르지 않게)
                    torch.nn.utils.clip_grad_norm_(grp, 1.0)
                opt.step()
                args._sched.step()  # 학습률 스케줄은 최적화 스텝마다 갱신
        with torch.no_grad():
            want, got = b["target"].argmax(-1), logp.argmax(-1)
        n = len(want)
        tot["ce"] += ce.item() * n; tot["v"] += v.item() * n; tot["pg"] += pg.item() * n; tot["hid"] += hid.item() * n; tot["n"] += n
        tot["agree"] += (got == want).sum().item()
        if "q_mean" in b:
            tot["reg"] += (reg_arg * valid).sum().item(); tot["reg_n"] += valid.sum().item()
    n = max(1, tot["n"])
    return {"reg": tot["reg"] / tot["reg_n"] if tot["reg_n"] else None, "ce": tot["ce"] / n, "v": tot["v"] / n, "pg": tot["pg"] / n, "hid": tot["hid"] / n, "hid_top4": tot["hid_top4"] / max(1, tot["hid_nb"]), "agree": tot["agree"] / n, "n": tot["n"]}


def fmt(r):
    return (f"소프트CE {r['ce']:.4f} | 탐색 선택과 일치 {r['agree']:.3f} | V {r['v']:.3f} | PG {r['pg']:.4f}" + (f" | regret {r['reg']:.4f}" if r["reg"] is not None else "") + (f" | 숨김정보 {r['hid']:.3f}(기술 상위4 적중 {r['hid_top4']:.2f})" if r["hid"] else ""))


def load(pattern):
    files = sorted({f for pat in pattern.split(",") for f in glob.glob(pat.strip())})  # 쉼표로 여러 패턴/파일 지정 가능
    if not files:
        raise SystemExit(f"❌ 데이터 파일 없음: {pattern}")
    parts = [np.load(f) for f in files]
    data = {}
    total = int(sum(int(p["lens"].sum()) for p in parts))
    for k in KEYS + (Q_KEYS if all(k in p.files for p in parts for k in Q_KEYS) else []):
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
        half = k in OBS_KEYS and first.dtype == np.float32  # 관측 실수 배열은 반정밀도로 보관 (배치를 만들 때 다시 float32로, RAM 절약), 값 정밀도 약 3자리
        out = np.zeros((total,) + shape_tail, dtype=np.float16 if half else np.int16 if k in CAT_KEYS else first.dtype)
        pos = 0
        for i, p in enumerate(parts):
            a = first if i == 0 else p[k]
            if k in CAT_KEYS:       # 어휘 번호 int64 -> int16 (RAM 약 2.5GB 절약): 넘치면 조용히 깨지므로 먼저 확인
                assert int(a.max()) <= np.iinfo(np.int16).max and int(a.min()) >= 0, (files[i], k)
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
    data["outcome"] = np.concatenate([p["outcome"] for p in parts])
    if any("teacher_value" in p.files for p in parts):      # 교사 가치가 없는 파일(옛 형식)은 NaN → 그 결정은 할인된 결과만으로 학습
        data["teacher_value"] = np.concatenate([p["teacher_value"] if "teacher_value" in p.files else np.full(int(p["lens"].sum()), np.nan, np.float32) for p in parts])
    lens = np.concatenate([p["lens"] for p in parts])
    return data, np.concatenate([[0], np.cumsum(lens)]), lens, files


def split_ids(data, offsets, val_frac, seed=0):
    """학습/검증 분할 (시퀀스 단위). 셀프플레이는 한 배틀이 양쪽 시점으로 두 번 기록돼 있어서(내 시점/상대 시점, 결과 반대) 시퀀스 단위로
    무작위 분할하면 검증 시퀀스의 거울 짝이 학습에 들어가 검증 지표가 낙관적이 됨 → 같은 두 팀 조합을 가진 시퀀스는 항상 같은 쪽으로 보냄.
    팀 조합 = 첫 턴의 내 팀/상대 팀 종 ID(정렬)를 짝으로 묶은 것 (같은 조합의 다른 배틀도 같이 묶임)"""
    first = np.asarray(offsets[:-1])
    mine = np.sort(np.asarray(data["my_team_cat"][first, :, 10]), axis=1)
    opp = np.sort(np.asarray(data["opp_team_cat"][first, :, 10]), axis=1)
    groups = {}
    for i in range(len(first)):
        groups.setdefault(tuple(sorted((mine[i].tobytes(), opp[i].tobytes()))), []).append(i)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_val, val = max(1, int(len(first) * val_frac)), []
    for k in keys:
        if len(val) >= n_val:
            break
        val += groups[k]
    in_val = set(val)
    return val, [i for i in range(len(first)) if i not in in_val]


def main(args):
    data, offsets, lens, files = load(args.data)
    if args.teacher_value_mix > 0 and "teacher_value" not in data:
        raise SystemExit("❌ --teacher-value-mix에는 교사 가치(teacher_value)가 든 npz가 하나는 있어야 함 (build_fp_dataset.py로 만든 새 형식)")
    val, train = split_ids(data, offsets, VAL_FRAC)
    print(f"데이터 파일 {len(files)}개 | {len(lens)}판 / 결정 {int(lens.sum())}개 → 학습 {len(train)}판, 검증 {len(val)}판 (같은 배틀의 양쪽 시점은 같은 쪽)")

    arch = json.loads(args.arch)
    model = build_model(arch)
    need = getattr(model, "field_proj", None) and model.field_proj[0].in_features
    if getattr(model, "uses_legal", False) and data["field_vec"].shape[1] < need:        # v6는 효과 열이 든 새 데이터가 필요 (옛 데이터는 조용히 0으로 채워져 효과 입력이 비게 됨)
        raise SystemExit(f"❌ 데이터의 field_vec이 {data['field_vec'].shape[1]}칸인데 {arch['model']}는 {need}칸이 필요: src/training/build_fp_dataset.py로 데이터를 다시 만들 것")
    print(f"모델 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 구조 {arch}")
    belief_params = [p for n, p in model.named_parameters() if n.startswith("belief.")]
    args._clip_groups = [belief_params, [p for n, p in model.named_parameters() if not n.startswith("belief.")]] if belief_params else [list(model.parameters())]
    if belief_params:      # 팀 데이터로 사전학습한 세트 추정 (src/training/pretrain_belief.py)을 낮은 학습률로 미세조정. 파일 이름은 입력 수치 폭(NUM_DIM)으로 정해져 폭이 다른 파일은 못 씀
        pre = f"checkpoints/belief_{model.belief.num_dim}.pt"
        if not os.path.exists(pre):
            raise SystemExit(f"❌ {pre} 없음: 이 모델의 세트 추정 입력 폭({model.belief.num_dim})에 맞춰 python src/training/pretrain_belief.py 로 먼저 사전학습할 것")
        model.belief.load_state_dict(torch.load(pre, map_location="cpu")["state"])
        print(f"  세트 추정 사전학습 가중치 로드: {pre} (학습률 x{BELIEF_FT_LR_MULT}로 미세조정)")
    if args.init:      # 이미 학습한 체크포인트(예: v8_full)에서 이어서 학습 (세트 추정 포함 모든 가중치를 덮어씀). 구조(--arch)는 체크포인트와 같아야 함
        load_compatible(model, torch.load(args.init, map_location="cpu"))
        print(f"  시작 가중치: {args.init}")
    model.to(DEVICE)
    other_params = [p for n, p in model.named_parameters() if not n.startswith("belief.")]
    groups = [{"params": other_params, "lr": args.lr}] + ([{"params": belief_params, "lr": args.lr * BELIEF_FT_LR_MULT}] if belief_params else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    import math
    total = max(1, args.epochs * math.ceil(len(train) / BATCH_BATTLES))
    warm = int(total * WARMUP_FRAC)

    def mult(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return LR_MIN_RATIO + (1 - LR_MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    args._sched = torch.optim.lr_scheduler.LambdaLR(opt, mult)
    print("시작 전 검증:", fmt(run(model, data, offsets, val, args, train=False)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_ce, best_ep, ce_hist = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        tr = run(model, data, offsets, train, args, train=True, opt=opt)
        va = run(model, data, offsets, val, args, train=False)
        print(f"[에폭 {ep}/{args.epochs}] 학습 {fmt(tr)}\n            검증 {fmt(va)}", flush=True)
        # 최근 최대 3개 평균으로 완만화해서 최저점 판단 (검증셋이 작아 한 에폭 값만으로는 잡음이 큼). 기준은 Q가 있으면 검증 argmax regret, 없으면 소프트 CE
        # (손실 종류와 무관하게 같은 기준으로 비교하기 위해 regret을 씀)
        metric = "regret" if va["reg"] is not None else "CE"
        ce_hist.append(va["reg"] if va["reg"] is not None else va["ce"])
        smoothed = sum(ce_hist[-3:]) / len(ce_hist[-3:])
        if smoothed < best_ce:
            best_ce, best_ep = smoothed, ep
            save_ckpt(model, args.out)
            print(f"  💾 최저점 갱신(완만화 {metric} {smoothed:.4f}) → {args.out}", flush=True)
    print(f"💾 최저점: {args.out} (에폭 {best_ep}, 완만화 {metric} {best_ce:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fp_selfplay/ms100/fp_data*.npz", help="build_fp_dataset.py 출력 (쉼표로 여러 glob)")
    ap.add_argument("--out", default="checkpoints/supervised_v2_fp_next.pt")
    ap.add_argument("--arch", default='{"model":"v8","my_pos":false}', help="모델 구조 JSON (기본: v8). 옛 구조는 {\"model\":\"v6\",\"my_pos\":false}")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=1, help="기울기 누적 횟수: 배틀 32개를 이 수로 나눠 처리해 VRAM을 줄임 (큰 모델용, 32의 약수)")
    ap.add_argument("--init", default="", help="시작 가중치 체크포인트 (기본: 처음부터)")
    ap.add_argument("--lr", type=float, default=LR, help="피크 학습률 (이어서 학습할 때는 낮게)")
    ap.add_argument("--hard-weight", type=float, default=1.0, help="중요도(Q* - 최저Q) >= 0.3인 결정의 정책 손실 가중 (1=끔). Q가 든 데이터만")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (기본 1e-4)")
    ap.add_argument("--value-coef", type=float, default=VALUE_COEF, help="가치 손실 가중치 (기본 0.25). 0이면 가치 헤드는 학습 안 되니 --pg-coef 0도 같이 (PG 기준선이 가치 예측)")
    ap.add_argument("--pg-coef", type=float, default=0.05, help="실제 승패(할인된 결과 − 가치 예측 = advantage) 기반 정책 그라디언트 보조 손실 가중치 (0=끔)")
    ap.add_argument("--teacher-value-mix", type=float, default=0.0, help="가치 타깃 = 이 비율 x 교사 가치(탐색이 결정마다 본 승률) + 나머지 x 할인된 결과. 0=결과만 (교사 가치가 있는 새 데이터 필요)")
    main(ap.parse_args())
