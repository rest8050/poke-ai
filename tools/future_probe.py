"""학생 몸통이 여러 턴 뒤의 자원 결과를 이미 담고 있는가 (동결 프로브). 데이터: 학생이 FP300과 둔 판의 학생 시점 상태 (data/winrate_v6_fp300/lab_stu.npz, 484판 18.9k 결정, teacher_value = FP가 본 학생 쪽 승률).
목표(결정 t의 같은 판 t+k 결정에서): 내 생존 수, 상대 생존 수, HP 우위(내 HP 합 - 상대 HP 합) (k = 1, 3, 6; 판이 그 전에 끝나면 제외). + 판의 승패.
예측기(전부 "현재 값에서의 변화"를 맞춤; 1차 실행은 수준(level)을 맞추게 해서 지속 예측이 모든 학습 예측기를 이겨 기준선이 불공정했음 -> logs/future_probe_v1_level_target.txt, 판정 임계값은 그대로 두고 목표만 변화량으로 바꿈):
지속(변화 0) / HGB(현재 자원 요약 + 턴) / HGB(현재 원시 특징 전부) / 릿지(학생 잠재 256 + 현재 요약) / HGB(잠재 + 현재 요약).
사전 등록한 판정 (k=3,6 평균, 판 단위 부트스트랩, MSE 비 = 릿지(잠재) / HGB(원시)): >= 1.05 → 몸통이 여러 턴 뒤 결과를 선형으로 못 꺼냄 = 보조 감독이 채울 여지, <= 0.95 → 몸통이 이미 앞섬, 그 사이 → 불분명.
승패: AUC를 FP 가치 V / HGB(원시) / 로지스틱(잠재) / 로지스틱(V + 잠재)로 비교 (V에 잠재를 더해 AUC가 안 오르면 승패에 V 밖의 정보가 몸통에도 없음).
사용: python tools/future_probe.py --model v8_full"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from legal_eval import OBS                                           # noqa: E402
from regret_eval import boot_ci                                      # noqa: E402
from src.core.model import model_from_ckpt                           # noqa: E402

KS = (1, 3, 6)


def latents(net, d, dev):
    mv_in = net.embeddings.move_encoder.fc[0].in_features - 48
    lens, off, out = d["lens"], np.concatenate([[0], np.cumsum(d["lens"])]), []
    with torch.no_grad():
        for i in range(0, len(lens), 32):
            sl = slice(off[i], off[min(i + 32, len(lens))])
            obs = [torch.tensor(d[k][sl][..., :mv_in] if k in ("my_move_num", "opp_move_num") else d[k][sl]).to(dev) for k in OBS]
            ch = lens[i:i + 32]
            si = torch.cat([torch.full((n,), j) for j, n in enumerate(ch)]).to(dev); ti = torch.cat([torch.arange(n) for n in ch]).to(dev)
            out.append(net.forward_sequences(obs, si, ti, len(ch), action_mask=torch.tensor(d["action_mask"][sl]).to(dev))["latent"].float().cpu().numpy())
    return np.concatenate(out)


def state_stats(d):
    """결정마다 (내 생존, 상대 생존, HP 우위)"""
    my, op = d["my_team_num"], d["opp_team_num"]
    my_alive = ((my[..., 11] > 0) & (my[..., 1] < 0.5)).sum(1)
    op_alive = (op[..., 1] < 0.5).sum(1)
    hp = (my[..., 2] * (my[..., 1] < 0.5)).sum(1) - (op[..., 2] * (op[..., 1] < 0.5)).sum(1)
    return np.stack([my_alive, op_alive, hp], 1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v8_full")
    ap.add_argument("--data", default="data/winrate_v6_fp300/lab_stu.npz")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = dict(np.load(args.data))
    net = model_from_ckpt(args.model if os.path.exists(args.model) else f"checkpoints/supervised_v2_fp_{args.model}.pt").to(dev).eval()
    Z = latents(net, d, dev)
    lens = d["lens"]; n = len(Z)
    game = np.repeat(np.arange(len(lens)), lens); t = np.concatenate([np.arange(l) for l in lens])
    S = state_stats(d)
    raw = np.concatenate([d["my_team_num"].reshape(n, -1), d["opp_team_num"].reshape(n, -1), d["field_vec"]], 1)
    test = (game % 5) < 2
    rng = np.random.RandomState(0)
    print(f"{n}결정 {len(lens)}판 (시험 {len(np.unique(game[test]))}판) | 모델 {args.model} | 잠재 {Z.shape[1]}차원, 원시 {raw.shape[1]}차원")
    names = ("내 생존", "상대 생존", "HP 우위")
    ratios = {k: [] for k in KS}
    print(f"{'목표':10s} {'k':>2s} {'n(시험)':>7s} | {'지속 MSE':>9s} {'HGB현재요약':>10s} {'HGB원시':>8s} {'릿지잠재':>8s} {'HGB잠재':>8s} | 릿지잠재/HGB원시 [95%]")
    sc = StandardScaler().fit(Z[~test])
    cur = np.concatenate([S, t[:, None]], 1)                               # 현재 자원 요약 + 턴
    Zc = np.concatenate([sc.transform(Z), (cur - cur[~test].mean(0)) / cur[~test].std(0)], 1)   # 잠재 + 현재 요약
    for k in KS:
        ok = (t + k) < np.repeat(lens, lens)
        idx = np.arange(n)
        for j, nm in enumerate(names):
            dy = np.where(ok, S[np.minimum(idx + k, n - 1), j] - S[:, j], np.nan)     # 모든 예측기는 "현재 값에서의 변화"를 맞춤 (지속 = 0 예측이 기준)
            tr, te = ok & ~test, ok & test
            pers = dy[te] ** 2
            r_cur = (HGBR(max_iter=200).fit(cur[tr], dy[tr]).predict(cur[te]) - dy[te]) ** 2
            r_hgb = (HGBR(max_iter=200).fit(raw[tr], dy[tr]).predict(raw[te]) - dy[te]) ** 2
            r_rid = (Ridge(alpha=10.0).fit(Zc[tr], dy[tr]).predict(Zc[te]) - dy[te]) ** 2
            r_hz = (HGBR(max_iter=200).fit(np.concatenate([Z, cur], 1)[tr], dy[tr]).predict(np.concatenate([Z, cur], 1)[te]) - dy[te]) ** 2
            g = game[te]; ug = np.unique(g)
            by = lambda r: np.array([r[g == u].sum() for u in ug])
            a_, b_ = by(r_rid), by(r_hgb)
            boots = [a_[i].sum() / b_[i].sum() for i in (rng.choice(len(ug), len(ug)) for _ in range(300))]
            if k in (3, 6):
                ratios[k].append(np.array(boots))
            print(f"{nm:10s} {k:2d} {int(te.sum()):7d} | {pers.mean():9.4f} {r_cur.mean():10.4f} {r_hgb.mean():8.4f} {r_rid.mean():8.4f} {r_hz.mean():8.4f} | {r_rid.mean() / r_hgb.mean():.3f} [{np.percentile(boots, 2.5):.3f},{np.percentile(boots, 97.5):.3f}]")
    allr = np.mean([b for k in (3, 6) for b in ratios[k]], 0)
    print(f"\n[사전 등록 판정] k=3,6 평균 MSE 비(릿지 잠재 / HGB 원시) = {allr.mean():.3f} [{np.percentile(allr, 2.5):.3f},{np.percentile(allr, 97.5):.3f}] -> "
          + ("보조 감독 여지 있음 (>=1.05)" if np.percentile(allr, 2.5) >= 1.05 or allr.mean() >= 1.05 else "몸통이 이미 앞섬 (<=0.95)" if allr.mean() <= 0.95 else "불분명"))
    # 승패
    win = (d["outcome"][game] > 0).astype(int); v = d["teacher_value"].astype(np.float64)
    tr, te = ~test, test
    f = lambda s: roc_auc_score(win[te], s)
    hgb = HGBC(max_iter=200).fit(raw[tr], win[tr]).predict_proba(raw[te])[:, 1]
    lg = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Z[tr]), win[tr]).predict_proba(sc.transform(Z[te]))[:, 1]
    X = np.concatenate([sc.transform(Z), ((v - v[tr].mean()) / v[tr].std())[:, None] * 3], 1)
    lv = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr], win[tr]).predict_proba(X[te])[:, 1]
    print(f"\n[승패 AUC, 학생 시점 (학생 승률 {win.mean():.3f})] FP 가치 V {f(v[te]):.3f} | HGB(원시) {f(hgb):.3f} | 로지스틱(잠재) {f(lg):.3f} | 로지스틱(V + 잠재) {f(lv):.3f}")
    for nm, m in (("초반 t<8", t[te] < 8), ("중반 8~20", (t[te] >= 8) & (t[te] < 20)), ("후반 >=20", t[te] >= 20)):
        if m.sum() > 100 and 0 < win[te][m].mean() < 1:
            print(f"  {nm:10s} n={int(m.sum()):5d} | V {roc_auc_score(win[te][m], v[te][m]):.3f} | 잠재 {roc_auc_score(win[te][m], lg[m]):.3f} | V+잠재 {roc_auc_score(win[te][m], lv[m]):.3f} | 평균 V {v[te][m].mean():.3f} 실제 승률 {win[te][m].mean():.3f}")


if __name__ == "__main__":
    main()
