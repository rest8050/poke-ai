"""합법성 범주별 평가: 서버 요청 기반 순수 합법 마스크가 든 npz(build_fp_dataset 출력)에서 모델별로 범주마다 교사/학생의 교체 확률, 종류 KL, 총 KL, 교체 판별 AUC, 일치율을 재고
(기준 모델 대비) 차이의 판(배틀) 단위 부트스트랩 95% 구간을 냄. 범주: 기술 4개 가능 / 2~3개 가능 / 1개만 가능(구애 고정·앵콜 등) / 교체 봉쇄(교체 가능한 후보가 있는데 불가) — 범주별 결정 수 함께 출력.
사용: python tools/legal_eval.py --models 기준,모델B,... --sets 이름=경로[,...] [--base 기준]  (모델 이름 = checkpoints/supervised_v2_fp_<이름>.pt)"""
import argparse
import sys

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, ".")
from src.core.model import model_from_ckpt
from src.core.tensor_encoder import EFFECT_FLAGS, EFFECT_START

OBS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
CATS = ["기술 4개 가능", "기술 2~3개 가능", "기술 1개만 가능", "교체 봉쇄"]
EPS = 1e-12


def load_set(path):
    z = np.load(path)
    d = {k: z[k] for k in OBS + ["action_mask", "target", "lens"]}
    for k in ("my_move_num", "opp_move_num"):
        if d[k].shape[-1] < 46:
            d[k] = np.concatenate([d[k], np.zeros(d[k].shape[:-1] + (46 - d[k].shape[-1],), d[k].dtype)], -1)
    return d


def categorize(d):
    """결정마다 범주 번호(0..3, 자발이 아니면 -1)와 배틀 번호"""
    m = d["action_mask"]
    n_mv, n_sw = m[:, :4].sum(1), m[:, 8:14].sum(1)
    mn = d["my_team_num"]
    bench = ((mn[..., 11] > 0) & (mn[..., 1] < 0.5)).sum(1) - 1               # 살아 있는 벤치 수 (활성 제외)
    cat = np.full(len(m), -1)
    vol = (n_mv > 0) & (n_sw > 0)
    cat[vol & (n_mv == 4)] = 0
    cat[vol & ((n_mv == 2) | (n_mv == 3))] = 1
    cat[vol & (n_mv == 1)] = 2
    cat[(n_sw == 0) & (n_mv > 0) & (bench > 0)] = 3                             # 벤치가 살아 있는데 교체 불가 = 봉쇄
    battle = np.repeat(np.arange(len(d["lens"])), d["lens"])
    return cat, battle


def groups(d, cat):
    """(이름, 결정 마스크) 목록: 합법성 범주 4개 + 행동 제약 효과 범주(데이터에 효과 열이 있을 때만) + 전체. 효과 = 내 활성의 도발/앵콜/사슬묶기"""
    out = [(c, cat == i) for i, c in enumerate(CATS)]
    fv = d["field_vec"]
    if fv.shape[1] >= EFFECT_START + len(EFFECT_FLAGS):
        eff = (fv[:, EFFECT_START:EFFECT_START + len(EFFECT_FLAGS)] > 0).any(1)
        out += [("제약 효과 중 (내 활성)", eff & (cat >= 0) & (cat <= 2)), ("기술 2~3개 & 제약 효과 중", eff & (cat == 1)), ("기술 2~3개 & 제약 효과 없음", ~eff & (cat == 1)),
                ("기술 1개만 & 제약 효과 중", eff & (cat == 2)), ("기술 1개만 & 제약 효과 없음", ~eff & (cat == 2))]
    return out + [("전체(자발)", cat >= 0)]


def kl(t, s):
    t, s = np.clip(t, EPS, 1), np.clip(s, EPS, 1)
    return (np.where(t > EPS, t * (np.log(t) - np.log(s)), 0.0)).sum(-1)


def infer(model, d, dev):
    mv_in = model.embeddings.move_encoder.fc[0].in_features - 48
    out, lens = [], d["lens"]
    off = np.concatenate([[0], np.cumsum(lens)])
    with torch.no_grad():
        for i in range(0, len(lens), 32):
            sl = slice(off[i], off[min(i + 32, len(lens))])
            obs = [torch.tensor(d[k][sl][..., :mv_in] if k in ("my_move_num", "opp_move_num") else d[k][sl]).to(dev) for k in OBS]
            ch = lens[i:i + 32]
            si = torch.cat([torch.full((n,), j) for j, n in enumerate(ch)]).to(dev); ti = torch.cat([torch.arange(n) for n in ch]).to(dev)
            lg = model.forward_sequences(obs, si, ti, len(ch), action_mask=torch.tensor(d["action_mask"][sl]).to(dev))["policy_logits"]
            out.append(torch.softmax(lg.double(), -1).cpu().numpy())
    return np.concatenate(out)


def auc(score, pos):
    if pos.sum() == 0 or pos.all():
        return float("nan")
    r = rankdata(score)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def per_decision(P, T):
    ts, ss = T[:, 8:14].sum(1), P[:, 8:14].sum(1)
    kind = kl(np.stack([1 - ts, ts], -1), np.stack([1 - ss, ss], -1))
    return {"kind": kind, "total": kl(T, P), "agree": (P.argmax(-1) == T.argmax(-1)).astype(float), "ps": ss, "ts": ts}


def boot(a, b, mask, battle, rng, n_boot=2000):
    idx = np.where(mask)[0]
    u, inv = np.unique(battle[idx], return_inverse=True)
    sa, sb, n = (np.bincount(inv, weights=x[idx]) for x in (a, b, np.ones(len(a))))
    pt = (sa.sum() - sb.sum()) / n.sum()
    ds = []
    for _ in range(n_boot):
        p = rng.randint(0, len(u), len(u))
        ds.append((sa[p].sum() - sb[p].sum()) / n[p].sum())
    return pt, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--base", default=None, help="차이를 낼 기준 모델 (기본: --models의 첫 번째)")
    ap.add_argument("--max-battles", type=int, default=0, help="디버그용: 앞쪽 배틀 N개만")
    args = ap.parse_args()
    models = args.models.split(",")
    base = args.base or models[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(0)
    nets = {n: model_from_ckpt(f"checkpoints/supervised_v2_fp_{n}.pt").to(dev).eval() for n in models}
    for spec in args.sets.split(","):
        name, path = spec.split("=", 1)
        d = load_set(path)
        if args.max_battles:
            k = int(np.cumsum(d["lens"])[args.max_battles - 1])
            d = {key: (val[:args.max_battles] if key == "lens" else val[:k]) for key, val in d.items()}
        cat, battle = categorize(d)
        T = d["target"].astype(np.float64)
        R = {n: per_decision(infer(net, d, dev), T) for n, net in nets.items()}
        print(f"\n===== {name} ({path}): 배틀 {len(d['lens'])}, 결정 {len(cat)}, 자발·봉쇄 범주 결정 {int((cat >= 0).sum())} =====")
        gr = groups(d, cat)
        print("범주별 결정 수: " + " | ".join(f"{c} {int(m.sum())}" for c, m in gr[:-1]))
        print(f"{'범주':16s} {'모델':10s} {'교사 P(교체)':>10s} {'학생 P(교체)':>10s} {'종류 KL':>8s} {'총 KL':>7s} {'일치율':>7s} {'AUC':>6s}")
        for cn, msk in gr:
            if msk.sum() == 0:
                continue
            for n in models:
                r = R[n]
                print(f"{cn:16s} {n:10s} {r['ts'][msk].mean():10.3f} {r['ps'][msk].mean():10.3f} {r['kind'][msk].mean():8.3f} {r['total'][msk].mean():7.3f} {r['agree'][msk].mean():7.3f} "
                      f"{auc(r['ps'][msk], r['ts'][msk] > 0.5):6.3f}")
        print(f"\n[{name}] 기준 {base} 대비 차이 (판 단위 부트스트랩 95% 구간; KL은 음수가 개선, 일치율은 양수가 개선)")
        for n in models:
            if n == base:
                continue
            print(f"  {n} - {base}")
            for cn, msk in gr:
                if msk.sum() < 30:
                    continue
                parts = []
                for key, lab in (("kind", "종류 KL"), ("total", "총 KL"), ("agree", "일치율")):
                    pt, lo, hi = boot(R[n][key], R[base][key], msk, battle, rng)
                    parts.append(f"{lab} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                print(f"    {cn:16s} (n={int(msk.sum()):6d}) " + " | ".join(parts))


if __name__ == "__main__":
    main()
