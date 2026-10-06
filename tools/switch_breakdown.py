"""교체 판단 분해 진단: python tools/switch_breakdown.py <모델이름> ...  (checkpoints/supervised_v2_fp_<이름>.pt, dg2_ho 기준)
교사가 교체한 결정에서 학생이 교체하는 비율/후보 정확도, 교사가 기술을 고른 결정에서 교체로 새는 비율/기술 정확도. 두 모델을 주면 정확 일치 증가분을 구간별로 분해, 상대 활성의 공개 기술 수별 표도 출력"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from src.core.model import model_from_ckpt
from src.training.hidden_labels import vocab_sizes
UNK_MOVE = vocab_sizes()["move"]
KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
z0 = np.load("data/dagger/dg2_ho.npz"); z = {k: z0[k] for k in KEYS + ["action_mask", "target", "lens"]}
off = np.concatenate([[0], np.cumsum(z["lens"])])
res = {}
for name in sys.argv[1:]:
    m = model_from_ckpt(f"checkpoints/supervised_v2_fp_{name}.pt").eval()
    mv_in = m.embeddings.move_encoder.fc[0].in_features - 48 if hasattr(m.embeddings, "move_encoder") else 46
    P, T, V, R = [], [], [], []
    with torch.no_grad():
        for b in range(len(z["lens"])):
            s, e = off[b], off[b + 1]; n = e - s
            obs = []
            for k in KEYS:
                x = z[k][s:e]
                if k in ("my_move_num", "opp_move_num") and x.shape[-1] < mv_in:
                    x = np.concatenate([x, np.zeros(x.shape[:-1] + (mv_in - x.shape[-1],), x.dtype)], -1)
                obs.append(torch.tensor(x))
            out = m.forward_sequences(obs, torch.zeros(n, dtype=torch.long), torch.arange(n), 1, torch.tensor(z["action_mask"][s:e]))
            P.append(out["policy_logits"].exp().numpy()); T.append(z["target"][s:e])
            act = z["opp_team_num"][s:e][:, :, 0].argmax(1); mvs = z["opp_team_cat"][s:e][np.arange(n), act, 5:9]
            R.append(((mvs >= 1) & (mvs < UNK_MOVE)).sum(-1))          # 상대 활성이 지금까지 공개한 기술 수
            am = z["action_mask"][s:e].astype(bool); V.append(am[:, :8].any(-1) & am[:, 8:14].any(-1))
    P, T, V, R = np.concatenate(P), np.concatenate(T), np.concatenate(V), np.concatenate(R); P, T, R = P[V], T[V], R[V]
    ta, sa = T.argmax(-1), P.argmax(-1)
    t_sw = (ta >= 8) & (ta < 14); s_sw = (sa >= 8) & (sa < 14)
    r = dict(n=len(P), n_tsw=int(t_sw.sum()),
             sw_recall=s_sw[t_sw].mean(), sw_exact=(sa == ta)[t_sw].mean(), sw_which=(sa == ta)[t_sw & s_sw].mean(),
             mv_false_sw=s_sw[~t_sw].mean(), mv_exact=(sa == ta)[~t_sw].mean(), mv_which=(sa == ta)[~t_sw & ~s_sw].mean(),
             sw_prec=t_sw[s_sw].mean(), share_sw=t_sw.mean())
    res[name] = r
    print(f"== {name}: 자발 결정 {r['n']} | 교사 교체 {r['n_tsw']}건({r['share_sw']:.3f})")
    print(f"  교사=교체: 학생도 교체 {r['sw_recall']:.3f} | 정확한 교체 후보까지 일치 {r['sw_exact']:.3f} | (교체 선택한 경우 후보 정확도 {r['sw_which']:.3f})")
    print(f"  교사=기술: 학생이 교체로 샘 {r['mv_false_sw']:.3f} | 정확한 기술 일치 {r['mv_exact']:.3f} | (기술 선택한 경우 기술 정확도 {r['mv_which']:.3f})")
    print(f"  학생이 교체를 고른 것 중 교사도 교체 {r['sw_prec']:.3f}")
    print("  상대 활성의 공개 기술 수별: 결정 수 | 교사 교체 비율 | 학생 교체 비율 | 학생 교체 재현율 | 교체로 새는 비율 | 교체 정밀도(학생이 교체했을 때 교사도) | 정확한 행동 일치")
    for k in range(5):
        sel = R == k
        if sel.sum() < 30:
            continue
        prec = t_sw[sel & s_sw].mean() if (sel & s_sw).any() else float('nan')
        print(f"    {k}개: {int(sel.sum()):5d} | {t_sw[sel].mean():.3f} | {s_sw[sel].mean():.3f} | {s_sw[sel & t_sw].mean() if (sel & t_sw).any() else float('nan'):.3f} | "
              f"{s_sw[sel & ~t_sw].mean() if (sel & ~t_sw).any() else float('nan'):.3f} | {prec:.3f} | {(sa == ta)[sel].mean():.3f}")
if len(res) == 2:   # 두 모델을 주면 정확 일치 증가분을 구간별로 분해 (뒤 모델 - 앞 모델)
    (na, a), (nb, b) = res.items(); s = a["share_sw"]
    print(f"정확 일치 증가분 분해({nb} - {na}): 교사=교체 구간 {s*(b['sw_exact']-a['sw_exact'])*100:+.2f}%p, 교사=기술 구간 {(1-s)*(b['mv_exact']-a['mv_exact'])*100:+.2f}%p")
