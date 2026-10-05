"""포켓몬 토큰(pkmn_fc 출력 128차원, 팀 크로스어텐션 뒤 128차원)에서 '핵심 기술/도구를 들고 있는가'를 선형 분류기로 되읽어 정보 손실을 점검.
AUC가 1에 가까우면 압축 후에도 정보가 남아 있다는 뜻 (남아 있다 != 판단에 쓰인다). 사용: python tools/probe_tokens.py [ckpt] [npz]"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.core.model import model_from_ckpt

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
V = json.load(open("data/vocab.json", encoding="utf-8"))
GROUPS = {
    "설치기(락/압정/독압정/네트)": ("move", {"stealthrock", "spikes", "toxicspikes", "stickyweb"}),
    "벽(리플렉터/빛의장막/오로라베일)": ("move", {"reflect", "lightscreen", "auroraveil"}),
    "회복기": ("move", {"roost", "recover", "softboiled", "slackoff", "morningsun", "synthesis", "moonlight", "shoreup", "strengthsap", "rest", "wish"}),
    "피벗(유턴/볼트체인지/플립턴/잔재주)": ("move", {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "shedtail"}),
    "강화기(칼춤/명상/용춤/나쁜음모/나비춤)": ("move", {"swordsdance", "calmmind", "dragondance", "nastyplot", "quiverdance", "bulkup", "shellsmash", "tailglow"}),
    "선공기(신속/기습/물대포류)": ("move", {"extremespeed", "suckerpunch", "aquajet", "iceshard", "machpunch", "bulletpunch", "shadowsneak", "fakeout"}),
    "도구 구애스카프": ("item", {"choicescarf"}), "도구 구애안경/머리띠": ("item", {"choicespecs", "choiceband"}),
    "도구 먹다남은음식": ("item", {"leftovers"}), "도구 튼튼한부츠": ("item", {"heavydutyboots"}), "도구 기합의띠": ("item", {"focussash"}),
    "도구 생명의구슬": ("item", {"lifeorb"}), "도구 돌격조끼": ("item", {"assaultvest"}),
}
COLS = {"item": [0], "move": [5, 6, 7, 8]}


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/supervised_v2_fp_v12r3.pt"
    path = sys.argv[2] if len(sys.argv) > 2 else "data/dagger/dg2_ho.npz"
    model = model_from_ckpt(ck).eval()
    z0 = np.load(path)
    z = {k: z0[k] for k in KEYS + ["action_mask", "lens"]}
    move_in = model.embeddings.move_encoder.fc[0].in_features - 48
    for k in ("my_move_num", "opp_move_num"):   # 옛 인코더로 만든 npz는 기술 벡터가 짧아 뒤를 0으로 채움
        if z[k].shape[-1] < move_in:
            z[k] = np.concatenate([z[k], np.zeros(z[k].shape[:-1] + (move_in - z[k].shape[-1],), dtype=z[k].dtype)], -1)
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    store, att = [], []
    h1 = model.pkmn_fc.register_forward_hook(lambda m, i, o: store.append(o.detach().numpy()))
    h2 = model.team_cross[-1].register_forward_hook(lambda m, i, o: att.append(o[0].detach().numpy()))
    pre, post, cats, bids = [], [], [], []
    with torch.no_grad():
        for b in range(len(z["lens"])):
            s, e = off[b], off[b + 1]
            store.clear(); att.clear()
            model.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(e - s, dtype=torch.long), torch.arange(e - s), 1,
                                    torch.tensor(z["action_mask"][s:e]))
            pre.append(np.stack(store[:6], 1)); post.append(att[0]); cats.append(z["my_team_cat"][s:e]); bids += [b] * (e - s)
    h1.remove(); h2.remove()
    pre, post, cats = np.concatenate(pre), np.concatenate(post), np.concatenate(cats)
    bids = np.array(bids)
    n = len(bids)
    sel = np.random.default_rng(0).choice(n, min(n, 6000), replace=False)   # 상태 6000개 x 6마리
    train_b = sel[bids[sel] % 10 < 7]; test_b = sel[bids[sel] % 10 >= 7]
    print(f"상태 {n}개 중 {len(sel)}개 표본(배틀 단위 70/30 분할), 토큰 = 내 쪽 6마리 x 128차원")
    print(f"{'정보':34s} {'양성비율':>7s} {'pkmn_fc 직후 AUC':>16s} {'크로스어텐션 뒤 AUC':>18s}")
    res = {}
    for name, (kind, names) in GROUPS.items():
        ids = {V[kind][x] for x in names if x in V[kind]}
        y = np.isin(cats[..., COLS[kind]], list(ids)).any(-1)    # [N,6]
        row = []
        for tok in (pre, post):
            Xtr, Xte = tok[train_b].reshape(-1, 128), tok[test_b].reshape(-1, 128)
            ytr, yte = y[train_b].reshape(-1), y[test_b].reshape(-1)
            if yte.sum() < 20 or ytr.sum() < 20:
                row.append(float("nan")); continue
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xtr), ytr)
            row.append(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
        res[name] = row
        print(f"{name:34s} {y[test_b].mean():7.3f} {row[0]:16.4f} {row[1]:18.4f}")
    a = np.array(list(res.values()))
    print(f"\n평균 AUC: pkmn_fc 직후 {np.nanmean(a[:, 0]):.4f} / 크로스어텐션 뒤 {np.nanmean(a[:, 1]):.4f}")


if __name__ == "__main__":
    main()
