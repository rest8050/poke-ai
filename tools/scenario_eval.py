"""장면 세트 평가: 장면 npz(build_fp_dataset 출력, 장면 하나 = 시퀀스 하나)에서 마지막 결정(= 테스트 장면)의 교사 분포와 모델 분포를 비교.
사용: scenario_eval.py <npz> <ckpt> [<ckpt> ...]   (장면 id는 npz와 같은 이름의 .ids.json: 시퀀스 순서의 id 목록, 없으면 번호)
행동 이름: 0-3 기술, 4-7 테라+기술, 8-13 교체(팀 슬롯 순서) -> my_team_cat 슬롯 종/기술 id를 어휘에서 이름으로 풀어 보여줌"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.model import model_from_ckpt

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
V = json.load(open("data/vocab.json", encoding="utf-8"))
INV = {k: {i: n for n, i in V[k].items()} for k in ("species", "move")}


def names(cat, num):
    """마지막 결정의 행동 번호 -> 이름 (활성 포켓몬 = my_team_num[...,0] 표시가 1인 슬롯)"""
    act = int(np.argmax(num[:, 0]))
    mv = [INV["move"].get(int(x), "?") for x in cat[act, 5:9]]
    out = {i: mv[i % 4] + ("+tera" if i >= 4 else "") for i in range(8)}
    for k in range(6):
        out[8 + k] = "→" + INV["species"].get(int(cat[k, 10]), "?")
    return out


def main(npz, ckpts):
    z = np.load(npz)
    z = {k: z[k] for k in KEYS + ["action_mask", "target", "lens"]}
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    ids_path = os.path.splitext(npz)[0] + ".ids.json"
    ids = json.load(open(ids_path)) if os.path.exists(ids_path) else list(range(len(z["lens"])))
    models = {}
    for c in ckpts:
        m = model_from_ckpt(f"checkpoints/supervised_v2_fp_{c}.pt")
        m.eval()
        models[c] = m
    rows = []
    for b in range(len(z["lens"])):
        s, e = off[b], off[b + 1]
        T = e - s
        nm = names(z["my_team_cat"][e - 1], z["my_team_num"][e - 1])
        t = z["target"][e - 1]
        top = np.argsort(-t)[:3]
        row = {"id": ids[b] if b < len(ids) else b, "teacher": [(nm[int(i)], round(float(t[i]), 3)) for i in top], "models": {}}
        for c, m in models.items():
            with torch.no_grad():
                out = m.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(T, dtype=torch.long), torch.arange(T), 1, torch.tensor(z["action_mask"][s:e]))
            p = torch.softmax(out["policy_logits"][-1], -1).numpy()
            st = np.argsort(-p)[:3]
            row["models"][c] = {"top": [(nm[int(i)], round(float(p[i]), 3)) for i in st], "p_teacher_top": round(float(p[int(top[0])]), 3), "agree": bool(st[0] == top[0])}
        rows.append(row)
        print(row["id"], "교사", row["teacher"])
        for c, r in row["models"].items():
            print("   ", c, r["top"], "교사 최상위에 둔 확률", r["p_teacher_top"], "일치" if r["agree"] else "불일치")
    json.dump(rows, open(os.path.splitext(npz)[0] + ".eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
