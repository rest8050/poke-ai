"""세트 정보 오라클 시험(약점 2: 숨은 정보가 얼마나 병목인가): 학생 상태 홀드아웃(d7_ho,d6_ho,dg5)의 각 결정에서, 상대의 아직 안 드러난 기술/도구/특성 중
그 배틀이 끝나기 전에 드러난 것(마지막 턴 관측)을 미리 채워 넣고 학습 안 한 채 그대로 추론해 regret이 얼마나 변하는지 잼 (미래 정보 누설 = 상한 근사, 끝내 안 드러난 것은 그대로 숨김).
사용: python tools/oracle_fill.py <모델 이름|경로> ..."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "tools")
import legal_eval as L
from regret_eval import regrets
from src.core.model import model_from_ckpt
from src.training.hidden_labels import vocab_sizes

UNK = vocab_sizes()
paths = "data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz".split(",")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zs = [np.load(p) for p in paths]
q = np.concatenate([z["q_mean"] for z in zs]); vis = np.concatenate([z["q_vis"] for z in zs])
M = np.concatenate([L.load_set(p)["action_mask"] for p in paths]); T = np.concatenate([z["target"] for z in zs]).astype(np.float64)
rel = (vis >= 100) & ~np.isnan(q) & M
valid = (M.sum(1) >= 2) & (rel.sum(1) >= 2)
qstar = np.where(rel, q, -np.inf).max(1)


def fill(s):
    """s의 사본에 마지막 턴에서 드러난 기술/도구/특성을 채움. 반환: (채운 세트, 채운 칸 수 통계)"""
    s = {k: v.copy() for k, v in s.items()}
    cat, mv = s["opp_team_cat"], s["opp_move_num"]
    off = np.concatenate([[0], np.cumsum(s["lens"])]); n_item = n_ab = n_mv = 0
    for b in range(len(s["lens"])):
        a, e = off[b], off[b + 1] - 1
        lc, lm = cat[e], mv[e]
        for t in range(a, e + 1):
            for m in range(6):
                if cat[t, m, 10] <= 0 or lc[m, 10] != cat[t, m, 10]:
                    continue
                for col, key in ((0, "item"), (1, "ability")):
                    if cat[t, m, col] == UNK[key] + 1 and 1 <= lc[m, col] < UNK[key]:
                        cat[t, m, col] = lc[m, col]; n_item += col == 0; n_ab += col == 1
                cur = cat[t, m, 5:9]; fut = lc[m, 5:9]
                hid = [i for i in range(4) if not (1 <= cur[i] < UNK["move"])]
                new = [i for i in range(4) if 1 <= fut[i] < UNK["move"] and fut[i] not in cur]
                for slot, j in zip(hid, new):
                    cat[t, m, 5 + slot] = fut[j]; mv[t, m, slot] = lm[m, j] if lm.ndim == 3 else mv[t, m, slot]; n_mv += 1
    return s, (n_item, n_ab, n_mv)


for name in sys.argv[1:]:
    net = model_from_ckpt(name if os.path.exists(name) else f"checkpoints/supervised_v2_fp_{name}.pt").to(dev).eval()
    res = {}
    for tag in ("원본", "오라클 채움"):
        Ps, stat = [], np.zeros(3, int)
        for p in paths:
            s = L.load_set(p)
            if tag != "원본":
                s, st = fill(s); stat += st
            Ps.append(L.infer(net, s, dev))
        P = np.concatenate(Ps)
        rs, *_ = regrets(P, q, rel, qstar, T)
        ok = valid & ~np.isnan(rs)
        res[tag] = np.where(ok, rs, np.nan)
        extra = f" | 채운 칸: 도구 {stat[0]} 특성 {stat[1]} 기술 {stat[2]}" if tag != "원본" else ""
        print(f"{os.path.basename(name):28s} {tag:8s} regret {np.nanmean(res[tag]):.4f} (결정 {int(ok.sum())}){extra}")
    both = ~np.isnan(res["원본"]) & ~np.isnan(res["오라클 채움"])
    d = res["오라클 채움"][both] - res["원본"][both]
    print(f"  차이(채움-원본) {d.mean():+.4f} (표준오차 {d.std() / np.sqrt(len(d)):.4f}, 음수 = 숨은 정보를 알면 좋아짐)")
