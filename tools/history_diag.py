"""사람 대전 데이터에서 '과거를 보면 이번 선택이 더 잘 맞는가' 진단 (히스토리가 쓸모 있는 신호가 있는지).
    PYTHONPATH=. python tools/history_diag.py [--seeds 3]
v8_full의 턴 벡터(CLS, 내/상대 활성 토큰, 필드; 얼린 특징)를 결정마다 뽑아 캐시하고, 같은 크기의 작은 인과 트랜스포머를 4가지 입력으로 학습:
  A 현재 턴만 | B 과거 턴 상태까지 | C 과거 턴 상태 + 과거 턴마다의 직전 행동(내 행동, 상대 행동) | D 현재 턴만 + 바로 직전 턴 행동
목표 두 가지: 상대의 이번 턴 선택 (기술/교체/테라, opp_action)과 사람 자신의 선택 (기술/교체/테라). 판 단위로 train/val/test 분할(test 10%).
판정: B/C가 A보다 나은지 (테스트 NLL 차이, 판 단위 부트스트랩 95% 구간)"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import model_from_ckpt
from src.training.replay_dataset import OBS_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--variants", default="ABCD")
ap.add_argument("--targets", default="opp,own")
args = ap.parse_args()
FEAT = "data/human/diag_feat.npy"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

files = sorted(glob.glob("data/human/gen9ou_s0_r1600_*.npz"))
parts = [np.load(f) for f in files]
cat = lambda k: np.concatenate([p[k] for p in parts])
lens, game = cat("lens"), cat("game")
action, opp, turn = cat("action"), cat("opp_action").astype(np.int64), cat("turn").astype(np.int64)
N, S = len(action), len(lens)
starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
print(f"조각 {len(files)}개, 시퀀스 {S}, 결정 {N}", flush=True)

if not os.path.exists(FEAT):
    net = model_from_ckpt("checkpoints/supervised_v2_fp_v8_full.pt").to(dev).eval()
    feats, t0 = [], time.time()
    with torch.no_grad():
        for p in parts:
            obs, mask = [p[k] for k in OBS_KEYS], p["action_mask"]
            for i in range(0, len(mask), 256):
                x = [torch.from_numpy(o[i:i + 256]).to(dev) for o in obs]
                feats.append(net._turn_vec(net._encode_turn(*x, legal=torch.from_numpy(mask[i:i + 256]).to(dev))).half().cpu().numpy())
            print(f"  특징 추출 {sum(len(f) for f in feats)}/{N} ({time.time() - t0:.0f}s)", flush=True)
    np.save(FEAT, np.concatenate(feats))
    del net
feat = torch.from_numpy(np.load(FEAT)).to(dev)
assert len(feat) == N

# 목표와 입력용 행동 (행동 종류: 0 기술, 1 교체, 2 테라)
own_cls = np.where(action < 0, -1, np.where(action < 4, 0, np.where(action < 8, 2, 1)))
forced = np.zeros(N, bool)                       # 기절 후 교체 지점 = 앞 결정과 턴 번호가 같음
prev_own, prev_opp = np.full(N, 22), np.full(N, 3)    # 직전 턴(턴 시작 결정)에서 내가 고른 행동 id(22 = 모름)와 상대 행동 종류(3 = 모름)
for s in range(S):
    a, n = starts[s], lens[s]
    last_first = {}                              # 턴 번호 -> 그 턴의 첫 결정 위치
    for j in range(a, a + n):
        if j > a and turn[j] == turn[j - 1]:
            forced[j] = True
        else:
            last_first[turn[j]] = j
        earlier = [t for t in last_first if t < turn[j]]
        if earlier:
            q = last_first[max(earlier)]
            prev_own[j] = action[q] if action[q] >= 0 else 22
            prev_opp[j] = opp[q] if opp[q] >= 0 else 3
seq_of = np.repeat(np.arange(S), lens)
# 누적 습관 특징 (과거 턴 시작 결정들만으로 계산): 상대 교체/테라/기술 비율, 내 교체 비율, 최근 3턴 상대 교체 수, 진행 턴 수
HAB = 8
habit = np.zeros((N, HAB), np.float32)
for s in range(S):
    a, n = starts[s], lens[s]
    o_cls, m_cls = [], []
    for j in range(a, a + n):
        if j > a and turn[j] == turn[j - 1]:
            continue
        no, nm = max(len(o_cls), 1), max(len(m_cls), 1)
        habit[j] = [min(len(o_cls), 30) / 30, o_cls.count(1) / no, float(2 in o_cls), o_cls.count(0) / no,
                    m_cls.count(1) / nm, float(2 in m_cls), o_cls[-3:].count(1) / 3, np.log1p(len(o_cls)) / 3.5]
        if opp[j] >= 0:
            o_cls.append(opp[j])
        if own_cls[j] >= 0:
            m_cls.append(own_cls[j])
    for j in range(a, a + n):                       # 기절 후 교체 지점은 같은 턴 시작 결정의 특징을 따름
        if j > a and turn[j] == turn[j - 1]:
            habit[j] = habit[j - 1]
sseq = game * 7919 % 10                         # 시퀀스별: 0 test, 1 val, 나머지 train (같은 판의 두 시점은 같은 쪽)
split = sseq[seq_of]
win_dec = np.repeat(cat("win").astype(np.int64), lens)
TARGETS = {"opp": (opp, opp >= 0), "own": (own_cls, (own_cls >= 0) & ~forced), "win": (win_dec, np.ones(N, bool))}
lm_opp, lm_own = np.zeros(N, np.int64), np.zeros(N, np.int64)     # 직전까지 마지막으로 쓴 기술 id (상대/나, 0 = 아직 없음)
if "opp_move" in parts[0].files:
    opp_mv, own_mv = cat("opp_move").astype(np.int64), cat("own_move").astype(np.int64)
    for s in range(S):
        last_o = last_m = 0
        for j in range(starts[s], starts[s] + lens[s]):
            lm_opp[j], lm_own[j] = last_o, last_m
            if j == starts[s] or turn[j] != turn[j - 1]:        # 턴 시작 결정에서만 갱신 (기절 후 교체 지점은 같은 턴)
                last_o, last_m = opp_mv[j] or last_o, own_mv[j] or last_m
    ok_o, ok_m = (opp_mv > 0) & (lm_opp > 0), (own_mv > 0) & (lm_own > 0) & ~forced
    TARGETS["opp_rep"] = ((opp_mv == lm_opp).astype(np.int64), ok_o)        # 상대가 이번에 쓴 기술이 지난번 쓴 기술과 같은가
    TARGETS["own_rep"] = ((own_mv == lm_own).astype(np.int64), ok_m)
t_pown, t_popp, t_turn, t_hab, t_lmo, t_lmm = [torch.from_numpy(x).to(dev) for x in (prev_own, prev_opp, np.minimum(turn, 255), habit, lm_opp, lm_own)]


class Net(nn.Module):
    def __init__(self, window1, use_act, habit_feat, use_mv, small_mv=False, d=256):
        super().__init__()
        self.window1, self.use_act, self.habit_feat, self.use_mv, self.small_mv = window1, use_act, habit_feat, use_mv, small_mv
        self.hab = nn.Linear(HAB, d)
        self.e_omv, self.e_mmv = nn.Embedding(1000, d), nn.Embedding(1000, d)
        self.e_mv16, self.mv_proj, self.mv_drop = nn.Embedding(2000, 16), nn.Linear(32, d), nn.Dropout(0.3)   # H: 저차원 임베딩 + 드롭아웃으로 규제한 마지막 기술 id
        self.norm, self.proj, self.pos = nn.LayerNorm(feat.size(1)), nn.Linear(feat.size(1), d), nn.Embedding(256, d)
        self.e_own, self.e_opp = nn.Embedding(23, d), nn.Embedding(4, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 512, 0.1, "gelu", batch_first=True, norm_first=True), 2)
        self.out = nn.Linear(d, 3)

    def forward(self, ix, valid):
        x = self.proj(self.norm(feat[ix].float())) + self.pos(t_turn[ix])
        if self.use_act:
            x = x + self.e_own(t_pown[ix]) + self.e_opp(t_popp[ix])
        if self.habit_feat:
            x = x + self.hab(t_hab[ix])
        if self.use_mv:
            x = x + self.e_omv(t_lmo[ix]) + self.e_mmv(t_lmm[ix])
        if self.small_mv:
            x = x + self.mv_proj(self.mv_drop(torch.cat([self.e_mv16(t_lmo[ix]), self.e_mv16(t_lmm[ix] + 1000)], -1)))
        T = ix.size(1)
        mask = torch.full((T, T), float("-inf"), device=ix.device)
        mask = mask.fill_diagonal_(0) if self.window1 else torch.triu(mask, 1)
        return self.out(self.enc(x, mask=mask))


def batches(seq_ids, size, shuffle, rng):
    seq_ids = rng.permutation(seq_ids) if shuffle else seq_ids
    for i in range(0, len(seq_ids), size):
        ids = seq_ids[i:i + size]
        L = int(lens[ids].max())
        ix = np.zeros((len(ids), L), np.int64)
        valid = np.zeros((len(ids), L), bool)
        for b, s in enumerate(ids):
            ix[b, :lens[s]] = starts[s] + np.arange(lens[s])
            valid[b, :lens[s]] = True
        yield torch.from_numpy(ix).to(dev), torch.from_numpy(valid).to(dev)


def evaluate(net, seq_ids, y, ok):
    """결정별 NLL과 확률 (ok인 결정만; 나머지는 NaN)"""
    nll, prob = np.full(N, np.nan, np.float32), np.zeros((N, 3), np.float32)
    net.eval()
    with torch.no_grad():
        for ix, valid in batches(seq_ids, 64, False, None):
            lp = F.log_softmax(net(ix, valid).float(), -1)
            flat, v = ix[valid].cpu().numpy(), lp[valid]
            prob[flat] = v.exp().cpu().numpy()
            tgt = torch.from_numpy(np.maximum(y[flat], 0)).to(dev)
            nll[flat] = np.where(ok[flat], (-v.gather(1, tgt[:, None]).squeeze(1)).cpu().numpy(), np.nan)
    return nll, prob


def run(variant, target, seed):
    y, ok = TARGETS[target]
    torch.manual_seed(seed)
    net = Net(window1=variant in "ADEFH", use_act=variant in "CDE", habit_feat=variant == "E", use_mv=variant in "FG", small_mv=variant == "H").to(dev)
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.01)
    tr, va, te = np.where(sseq >= 2)[0], np.where(sseq == 1)[0], np.where(sseq == 0)[0]
    y_t, ok_t = torch.from_numpy(np.maximum(y, 0)).to(dev), torch.from_numpy(ok).to(dev)
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        net.train()
        for ix, valid in batches(tr, 32, True, rng):
            m = valid & ok_t[ix]
            loss = F.cross_entropy(net(ix, valid)[m].float(), y_t[ix][m])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        v = np.nanmean(evaluate(net, va, y, ok)[0][np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in va])])
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return evaluate(net, te, y, ok), best


VARIANTS = {k: v for k, v in {"A": "현재 턴만", "B": "+과거 턴 상태", "C": "+과거 상태와 행동", "D": "현재 턴 + 직전 턴 행동",
                              "E": "현재 턴 + 직전 행동 + 누적 습관", "F": "현재 턴 + 마지막 기술 id", "G": "+과거 턴 상태 + 마지막 기술 id", "H": "현재 턴 + 마지막 기술 id(규제)"}.items() if k in args.variants}
te_ids = np.where(sseq == 0)[0]
te_pos = np.concatenate([np.arange(starts[s], starts[s] + lens[s]) for s in te_ids])
te_game = game[seq_of[te_pos]]
lines = []
NAMES = {"opp": "상대의 이번 턴 선택", "own": "사람 자신의 선택", "win": "그 시점 플레이어의 승패",
         "opp_rep": "상대가 마지막으로 쓴 기술을 또 쓰는가", "own_rep": "사람이 마지막으로 쓴 기술을 또 쓰는가"}
for target in args.targets.split(","):
    y, ok = TARGETS[target]
    sel = te_pos[ok[te_pos]]
    prior = np.bincount(y[ok & (split >= 2)], minlength=3) / (ok & (split >= 2)).sum()
    base_nll = -np.log(prior[y[sel]]).mean()
    lines.append(f"== 목표: {target} ({NAMES[target]}), 테스트 결정 {len(sel)}, 판 {len(np.unique(game[seq_of[sel]]))}, 클래스 비율 {np.round(prior, 3).tolist()}, 클래스 비율만 쓴 NLL {base_nll:.4f}")
    res = {}
    for v in VARIANTS:
        runs = []
        for seed in range(args.seeds):
            t0 = time.time()
            (nll, prob), best = run(v, target, seed)
            runs.append((nll, prob))
            print(f"[{target} {v} seed{seed}] val NLL {best:.4f}, test NLL {np.nanmean(nll[sel]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        res[v] = (np.mean([r[0] for r in runs], 0), [np.nanmean(r[0][sel]) for r in runs])
        lines.append(f"  {v} {VARIANTS[v]:18s} 테스트 NLL {np.mean(res[v][1]):.4f} (시드별 {', '.join(f'{x:.4f}' for x in res[v][1])})")
    ug, inv = np.unique(game[seq_of[sel]], return_inverse=True)
    rng = np.random.RandomState(0)
    for late, tag in ((False, "전체"), (True, "턴 8 이후")):
        m = np.ones(len(sel), bool) if not late else turn[sel] >= 8
        for a, b in [p for p in (("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("D", "E"), ("D", "C"), ("B", "C"), ("A", "F"), ("F", "G"), ("B", "G"), ("A", "G"), ("A", "H"), ("F", "H")) if set(p) <= set(VARIANTS)]:
            d = (res[a][0][sel] - res[b][0][sel])[m]       # 양수 = b가 a보다 NLL이 낮음(좋음)
            gs, gn = np.bincount(inv[m], d, len(ug)), np.bincount(inv[m], None, len(ug))
            boot = [gs[i].sum() / gn[i].sum() for i in (rng.randint(0, len(ug), len(ug)) for _ in range(1000))]
            lines.append(f"  [{tag}] NLL({a}) - NLL({b}) = {d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  (양수 = {b}가 더 좋음)")
open("logs/history_diag.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
