"""층별 선형 프로브: 트렁크가 임계값 양(2타 KO, 타수 경쟁)과 교체 판단을 몇 층에서 갖는가 (판 단위로 나눈 홀드아웃에서 로지스틱 프로브의 AUC와 경계 근처 정확도).
사용: python tools/probe_layers.py <체크포인트 이름 또는 경로>   (기본 v9. 2026-11-12 v9 결과: 2타 KO 경계 근처 0.65~0.67로 모든 층에서 안 읽힘, 날 입력 MLP는 0.87~0.89) 특징 = 층 l 출력의 [CLS, 내 활성, 상대 활성, 내 활성 기술 4, 상대 활성 기술 4, 내 포켓몬 6개 최대 풀링] (12 x 256).
정답은 모델이 입력으로 받는 수치에서 계산: f = 데미지/최대 체력 (mv_extra 채널 1 x 1.5), h = 맞는 쪽의 현재 체력 비율, 스피드 = mon_extra[..., 6]."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
import legal_eval as L
from src.core.model import model_from_ckpt

SETS = ["data/dagger/d7_ho.npz", "data/dagger/d6_ho.npz", "data/dagger/dg5.npz"]
dev = torch.device("cuda")
torch.manual_seed(0)
CK = sys.argv[1] if len(sys.argv) > 1 else "v9"
net = model_from_ckpt(CK if os.path.exists(CK) else f"checkpoints/supervised_v2_fp_{CK}.pt").to(dev).eval()

cap = {}
net.layers[0].register_forward_pre_hook(lambda m, a: cap.__setitem__(0, a[0]))
for i, layer in enumerate(net.layers):
    layer.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(i + 1, o))
_orig = net._matchup_feats
def _wrap(*a, **k):
    r = _orig(*a, **k)
    cap["mon_extra"], cap["mv_extra"], cap["O"] = r[3], r[4], r[1]
    return r
net._matchup_feats = _wrap

feats = [[] for _ in range(len(net.layers) + 1)]
raw, pol = [], []
PK = ("벤치 대 벤치", "벤치 대 상대 활성")
pfeats, praw = {k: [[] for _ in feats] for k in PK}, {k: [] for k in PK}
with torch.no_grad():
    for p in SETS:
        d = L.load_set(p)
        lens = d["lens"]; off = np.concatenate([[0], np.cumsum(lens)])
        for i in range(0, len(lens), 32):
            sl = slice(off[i], off[min(i + 32, len(lens))])
            obs = [torch.tensor(d[k][sl]).to(dev) for k in L.OBS]
            ch = lens[i:i + 32]
            si = torch.cat([torch.full((n,), j) for j, n in enumerate(ch)]).to(dev)
            ti = torch.from_numpy(d["pos"][sl]).long().to(dev)
            lg = net.forward_sequences(obs, si, ti, len(ch), action_mask=torch.tensor(d["action_mask"][sl]).to(dev))["policy_logits"]
            pol.append(torch.softmax(lg.float(), -1).cpu().numpy())
            B = lg.size(0); ar = torch.arange(B, device=dev)
            am, ao = obs[1][:, :, 0].argmax(1), obs[4][:, :, 0].argmax(1)
            # 쌍 과제: 내 벤치 i 하나와 상대 o 하나(벤치 / 활성)를 결정마다 무작위로 뽑음. 정답 = i의 기술 4칸이 o에게 주는 최대 데미지/최대 체력 (O[b,o,i,기술,1] x 1.5)
            alive = lambda n: (n[:, :, 11] > 0) & (n[:, :, 2] > 0)
            ok_i, ok_o = alive(obs[1]), alive(obs[4])
            ok_i[ar, am], ok_o[ar, ao] = False, False
            pi, po = (torch.rand(B, 6, device=dev) * ok_i).argmax(1), (torch.rand(B, 6, device=dev) * ok_o).argmax(1)
            OS = cap["O"].float().view(B, 6, 6, 4, 4)
            for k, o, okk in ((PK[0], po, ok_i.any(1) & ok_o.any(1)), (PK[1], ao, ok_i.any(1))):
                pr = OS[ar, o, pi]
                praw[k].append(torch.stack([pr[:, :, 1].amax(1) * 1.5, (okk & (pr[:, :, 3].amax(1) > 0)).float()], -1).cpu().numpy())
            for l in range(len(feats)):
                x = cap[l].float()
                mv = x[:, 12:60].view(B, 12, 4, -1)
                for k, o in ((PK[0], po), (PK[1], ao)):
                    pfeats[k][l].append(torch.cat([x[ar, pi], x[ar, 6 + o], mv[ar, pi].flatten(1)], -1).half().cpu())
                f = torch.cat([x[:, 63], x[ar, am], x[ar, 6 + ao], mv[ar, am].flatten(1), mv[ar, 6 + ao].flatten(1), x[:, :6].amax(1)], -1)
                feats[l].append(f.half().cpu())
            me, mx = cap["mv_extra"].float(), cap["mon_extra"].float()
            raw.append(torch.cat([me[ar, am, :, 1] * 1.5, me[ar, am, :, 2], me[ar, 6 + ao, :, 1] * 1.5,          # 0:4 내 기술 f, 4:8 내 기술 KO 플래그, 8:12 상대 기술 f
                                  obs[1][ar, am, 2:3], obs[4][ar, ao, 2:3], mx[ar, am, 6:7]], -1).cpu().numpy())  # 12 내 체력, 13 상대 체력, 14 스피드(log2 비율/2)
    T = np.concatenate([np.load(p)["target"] for p in SETS]); M = np.concatenate([np.load(p)["action_mask"] for p in SETS])
    lens = np.concatenate([np.load(p)["lens"] for p in SETS])
feats = [torch.cat(f) for f in feats]
raw, pol = np.concatenate(raw), np.concatenate(pol)
battle = np.repeat(np.arange(len(lens)), lens)
test = np.isin(battle, np.random.RandomState(0).permutation(len(lens))[: len(lens) // 4])

f_my, ko_flag, f_op, h_my, h_op, spd = raw[:, :4].max(1), raw[:, 4:8].max(1), raw[:, 8:12].max(1), raw[:, 12], raw[:, 13], raw[:, 14]
vol = (M[:, :4].sum(1) > 0)
n_my, n_op = np.ceil(h_op / np.maximum(f_my, 1e-6)), np.ceil(h_my / np.maximum(f_op, 1e-6))
t_sw = T[:, 8:14].sum(1)
# (이름, 정답, 쓰는 행, 경계 근처 행)
tasks = [
    ("[대조] 한 방 KO 플래그(입력에 있음)", ko_flag > 0.5, vol & (f_my > 0), None),
    ("[대조] 내가 더 빠름(입력에 있음)", spd > 0, vol & (spd != 0), np.abs(spd) < 0.1),
    ("내 활성이 2타 KO (2f >= h, 한 방 KO 제외)", 2 * f_my >= h_op, vol & (f_my > 0) & (f_my < h_op), np.abs(2 * f_my / np.maximum(h_op, 1e-6) - 1) < 0.2),
    ("상대 활성이 2타 KO (공개 기술)", 2 * f_op >= h_my, vol & (f_op > 0) & (f_op < h_my), np.abs(2 * f_op / np.maximum(h_my, 1e-6) - 1) < 0.2),
    ("타수 경쟁에서 이김 (타수 + 선공)", (n_my < n_op) | ((n_my == n_op) & (spd > 0)), vol & (f_my > 0) & (f_op > 0), n_my == n_op),
    ("교사가 교체를 고름 (자발 결정. 경계 근처 = 교사 교체 질량 >= 0.8)", T.argmax(1) >= 8, vol & (M[:, 8:14].sum(1) > 0), t_sw >= 0.8),
]


def fit(X, y, tr, te, hidden=0):
    Xt = X[tr].float().to(dev); mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-4)
    Xt, Xe = (Xt - mu) / sd, (X[te].float().to(dev) - mu) / sd
    yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    m = torch.nn.Linear(X.size(1), 1) if not hidden else torch.nn.Sequential(torch.nn.Linear(X.size(1), hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(600):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(m(Xt).squeeze(-1), yt).backward()
        opt.step()
    with torch.no_grad():
        return m(Xe).squeeze(-1).cpu().numpy()


print(f"결정 {len(T)} (프로브 학습 {int((~test).sum())} / 시험 {int(test.sum())}, 판 단위 분할)")
rawX = torch.tensor(raw).half()
for name, y, rows, band in tasks:
    tr, te = rows & ~test, rows & test
    yt = y[te]
    print(f"\n== {name} | 시험 행 {int(te.sum())}, 양성 비율 {yt.mean():.3f}" + (f", 경계 근처 {int((band & te).sum())}" if band is not None else ""))
    def show(label, s):
        acc = ((s > 0) == yt).mean()
        b = f" | 경계 근처 정확도 {((s > 0) == yt)[band[te]].mean():.3f}" if band is not None and (band & te).sum() > 30 else ""
        print(f"  {label:26s} AUC {L.auc(s, yt):.3f} 정확도 {acc:.3f}{b}")
    show("날 입력 수치 15칸 선형", fit(rawX, y, tr, te))
    show("날 입력 수치 15칸 MLP", fit(rawX, y, tr, te, 256))
    for l in range(len(feats)):
        show(f"층 {l} 선형" + (" (입력 토큰)" if l == 0 else ""), fit(feats[l], y, tr, te))
    if name.startswith("교사가 교체"):
        show("이 모델 정책의 교체 질량", np.log(np.clip(pol[te][:, 8:14].sum(1), 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - pol[te][:, 8:14].sum(1), 1e-9, 1)))

# 쌍 과제: 특징 = 층 l의 [내 벤치 i 토큰, 상대 o 토큰, i의 기술 토큰 4개] (6 x 256). 상대 활성 쌍은 데미지가 기술 토큰 입력(mv_extra)에 있는 대조군, 벤치 대 벤치는 어텐션 편향으로만 들어감
for k in PK:
    r = np.concatenate(praw[k]); f, rows = r[:, 0], r[:, 1] > 0
    y, band = f >= 0.5, np.abs(f / 0.5 - 1) < 0.2
    tr, te = rows & ~test, rows & test
    yt = y[te]
    print(f"\n== 내 벤치의 최대 데미지 >= 최대 체력의 50% ({k}) | 시험 행 {int(te.sum())}, 양성 비율 {yt.mean():.3f}, 경계 근처 {int((band & te).sum())}")
    for l in range(len(feats)):
        s = fit(torch.cat(pfeats[k][l]), y, tr, te)
        print(f"  층 {l} 선형{' (입력 토큰)' if l == 0 else '':10s} AUC {L.auc(s, yt):.3f} 정확도 {((s > 0) == yt).mean():.3f} | 경계 근처 정확도 {((s > 0) == yt)[band[te]].mean():.3f}")
