import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.model import model_from_ckpt

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
BINS = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]


def run(ckpt, paths):
    model = model_from_ckpt(ckpt); model.eval()
    move_in = model.embeddings.move_encoder.fc[0].in_features - 48
    R = {k: [] for k in ("s_arg", "t_arg", "s_sw", "t_sw", "vol", "hp", "ohp")}
    for path in paths:
        z0 = np.load(path)
        z = {k: z0[k] for k in KEYS + ["action_mask", "target", "lens"]}
        for k in ("my_move_num", "opp_move_num"):
            if z[k].shape[-1] < move_in:
                z[k] = np.concatenate([z[k], np.zeros(z[k].shape[:-1] + (move_in - z[k].shape[-1],), dtype=z[k].dtype)], -1)
        off = np.concatenate([[0], np.cumsum(z["lens"])])
        with torch.no_grad():
            for b in range(len(z["lens"])):
                s, e = off[b], off[b + 1]; T = e - s
                out = model.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(T, dtype=torch.long),
                                              torch.arange(T), 1, torch.tensor(z["action_mask"][s:e]))
                p = torch.softmax(out["policy_logits"], -1).numpy()
                t = z["target"][s:e]
                am = z["action_mask"][s:e].astype(bool)
                mn, on = z["my_team_num"][s:e], z["opp_team_num"][s:e]
                R["s_arg"].append(p.argmax(-1)); R["t_arg"].append(t.argmax(-1))
                R["s_sw"].append(p[:, 8:14].sum(-1)); R["t_sw"].append(t[:, 8:14].sum(-1))
                R["vol"].append(am[:, :8].any(-1) & am[:, 8:14].any(-1))
                R["hp"].append((mn[:, :, 0] * mn[:, :, 2]).sum(-1)); R["ohp"].append((on[:, :, 0] * on[:, :, 2]).sum(-1))
    return {k: np.concatenate(v) for k, v in R.items()}


paths = sys.argv[2:]
for ckpt in sys.argv[1].split(","):
    R = run(ckpt, paths)
    v = R["vol"]
    s_isw, t_isw = (R["s_arg"] >= 8) & (R["s_arg"] < 14), (R["t_arg"] >= 8) & (R["t_arg"] < 14)
    print(f"\n=== {ckpt}  결정 {len(v)}, 교체 가능한 자발 결정 {v.sum()} (강제교체/기술만 가능 상태 제외)")
    print(f"자발 결정에서 교체 비율  교사 argmax {t_isw[v].mean():.3f} / 학생 argmax {s_isw[v].mean():.3f} | 평균 교체 확률질량 교사 {R['t_sw'][v].mean():.3f} / 학생 {R['s_sw'][v].mean():.3f}")
    over = v & s_isw & ~t_isw; under = v & ~s_isw & t_isw
    print(f"학생만 교체(과잉) {over.sum()}건 ({over.sum() / v.sum():.1%}), 교사만 교체(과소) {under.sum()}건 ({under.sum() / v.sum():.1%})")
    print("내 활성 HP 구간별 자발 결정의 교체 비율 (교사 / 학생 / 건수)")
    for lo, hi in BINS:
        m = v & (R["hp"] >= lo) & (R["hp"] < hi)
        if m.sum(): print(f"  HP {lo:.2f}-{min(hi, 1):.2f}: {t_isw[m].mean():.3f} / {s_isw[m].mean():.3f} / {m.sum()}")
    print(f"과잉교체 상태의 평균 내 활성 HP {R['hp'][over].mean():.2f}, 상대 활성 HP {R['ohp'][over].mean():.2f}")
