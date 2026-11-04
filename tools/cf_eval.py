"""M5 반사실 평가 (docs/chain_m5.md). 사용:
  cf_eval.py equiv <cf.jsonl> <equiv.npz>                 시뮬레이터 정답 변환이 실제 인코더 Δ와 같은지 (정답 신뢰도 검증)
  cf_eval.py eval  <cf.jsonl> <pos.npz> <ckpt> [<ckpt>..]   결과 헤드 r(s,a,o)의 반사실 Δ 예측을 시뮬레이터 정답과 비교 (이름에 _rt가 든 체크포인트는 내 쪽을 행동 상대형 r_my로 배정해 평가)
Δ 정의는 chain_targets.py와 같다: 12슬롯 HP 비율 변화, 새 기절, 다음 활성 슬롯. (부스트는 이번엔 제외)
슬롯 대응: 내 슬롯 = 팀 텐서 순서(종으로 시뮬레이터 포켓몬과 맞춤), 상대 슬롯 = 공개된 것만 (미공개 슬롯은 비교에서 뺌)"""
import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.core.model import model_from_ckpt
from src.training import chain_targets as CT

KEYS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
V = json.load(open("data/vocab.json", encoding="utf-8"))
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
INV_SP = {i: norm(n) for n, i in V["species"].items()}


def fuzzy(x, names):
    """텐서 쪽 종 이름 -> 시뮬레이터 이름 (폼 표기 차이: 텐서 'indeedee' = 시뮬레이터 'indeedeef'). 같거나 한쪽이 다른 쪽의 접두면 같은 종으로 봄"""
    hit = [n for n in names if x is not None and (n == x or (min(len(n), len(x)) >= 4 and (n.startswith(x) or x.startswith(n))))]
    return hit[0] if len(hit) == 1 else None


def slot_map(cat, mons):
    """텐서 슬롯 j -> 시뮬레이터 종 이름(정규화). 시뮬레이터 팀 목록은 교체하면 순서가 바뀌므로 번호가 아니라 이름으로 맞춘다. 못 맞추면 None"""
    names = [norm(m["sp"]) for m in mons]
    return [fuzzy(INV_SP.get(int(cat[j, 10])), names) for j in range(6)]


def sim_delta(pre, post, mm, om):
    """시뮬레이터 상태쌍 -> (dhp[12], faint[12], act_my, act_opp). 대응 안 되는 슬롯은 dhp/faint가 nan, 활성이 대응 안 되면 -2"""
    dhp, fnt = np.full(12, np.nan, np.float32), np.full(12, np.nan, np.float32)
    act = []
    for side, mp in (("p1", mm), ("p2", om)):
        a_, b_ = ({norm(m["sp"]): m for m in st[side]} for st in (pre, post))
        for j, nm in enumerate(mp):
            if nm is None:
                continue
            a, b = a_[nm], b_[nm]
            dhp[j + 6 * (side == "p2")] = b["hp"] / b["mhp"] - a["hp"] / a["mhp"]
            fnt[j + 6 * (side == "p2")] = float(b["f"] and not a["f"])
        cur = [j for j, nm in enumerate(mp) if nm is not None and b_[nm]["act"]]
        act.append(cur[0] if cur else (-2 if any(nm is None for nm in mp) else -1))
    return dhp, fnt, act[0], act[1]


def load(cf, npz, cells=True):
    """npz 시퀀스에 맞는 장면 id 목록을 돌려줌. 빌더가 일부 배틀(턴 1 위치 등)과 결정을 버리므로 ids.json은 시퀀스와 1:1이 아님
    -> 순서를 지키는 탐욕 맞춤: 시퀀스의 마지막 행 턴과 내 팀 종 집합이 장면의 기대값과 같은 첫 id"""
    P = {}
    for line in open(cf, encoding="utf-8"):
        p = json.loads(line)
        if not cells:
            p.pop("cells")                                          # 큰 학습 세트는 위치 정보만 (칸은 필요할 때 다시 읽음)
        P[p["id"]] = p
    z = np.load(npz)
    z = {k: z[k] for k in z.files}
    cand = json.load(open(npz.replace(".npz", ".ids.json")))
    off = np.concatenate([[0], np.cumsum(z["lens"])])

    def expect(sid):
        pid = sid.split("|")[0]
        if "|" not in sid:
            return P[pid]["pre"]["turn"]
        post = next(c for c in P[pid]["cells"] if (c["a"], c["o"]) == tuple(map(int, sid.split("|")[1:])))["reps"][0]
        return post["turn"] if post["req"] == "move" else P[pid]["pre"]["turn"]

    ids, i = [], 0
    for k in range(len(z["lens"])):
        last = off[k + 1] - 1
        ok = lambda c: expect(c) == z["turn"][last] and None not in slot_map(z["my_team_cat"][last], P[c.split("|")[0]]["pre"]["p1"])
        j = next((j for j in range(i, min(i + 80, len(cand))) if ok(cand[j])), None)      # 위치 결정 행이 빌더에서 빠진 시퀀스는 맞는 id가 없음 -> None (포인터는 그대로)
        ids.append(None if j is None else cand[j])
        if j is not None:
            i = j + 1
    return P, z, ids, off


def act_index(a, mm_inv):
    if a["kind"] == "move":
        return int(a["choice"].split()[1]) - 1 + (4 if a["tera"] else 0)
    k = mm_inv.get(norm(a["sp"]))
    return None if k is None else 8 + k


def enc_delta(z, r):
    """인코더 행 r -> r+1의 Δ를 내 슬롯 기준으로 (상대 슬롯은 종 번호로 r 쪽 슬롯에 맞춤: 상대 공개 순서가 바뀌어 슬롯이 섞이는 행이 있음)"""
    n0, n1 = np.concatenate([z["my_team_num"][r], z["opp_team_num"][r]]), np.concatenate([z["my_team_num"][r + 1], z["opp_team_num"][r + 1]])
    sp0, sp1 = z["opp_team_cat"][r][:, 10], z["opp_team_cat"][r + 1][:, 10]
    perm = list(range(6)) + [6 + (list(sp1).index(x) if x in sp1 else j) for j, x in enumerate(sp0)]
    n1 = n1[perm]
    act = lambda sl: int(np.argmax(sl[:, 0])) if sl[:, 0].max() > 0.5 else -1
    return n1[:, 2] - n0[:, 2], np.clip(n1[:, 1] - n0[:, 1], 0, 1), act(n1[:6]), act(n1[6:])


def equiv(cf, npz):
    """시뮬레이터 정답 변환이 실제 인코더와 같은지: 한 턴 뒤 요청이 일반 결정(move)인 장면만 비교 (강제 교체 요청에서는 FP가 결정 행을 안 남겨 다음 행이 없음)"""
    P, z, ids, off = load(cf, npz)
    n = dict(mv=0, mv_ok=0, hp=0, hp_ok=0, fnt=0, fnt_ok=0, am=0, am_ok=0, ao=0, ao_ok=0, skip=0, bad=[])
    for b, sid in enumerate(ids):
        if sid is None:
            continue
        pid, ai, oi = sid.split("|")
        p, cell = P[pid], next(c for c in P[pid]["cells"] if c["a"] == int(ai) and c["o"] == int(oi))
        post = cell["reps"][0]
        e, r = off[b + 1], off[b + 1] - 2                          # 위치 행 = 끝에서 두 번째, 한 턴 뒤 행 = 마지막
        if post["req"] != "move" or e - off[b] < 2:
            n["skip"] += 1
            continue
        mm, om = slot_map(z["my_team_cat"][r], p["pre"]["p1"]), slot_map(z["opp_team_cat"][r], p["pre"]["p2"])
        if None in mm:
            n["skip"] += 1
            continue
        act = int(np.argmax(z["my_team_num"][r][:, 0]))
        mv = [norm({i: k for k, i in V["move"].items()}.get(int(x), "")) for x in z["my_team_cat"][r][act, 5:9]]
        n["mv"] += 1                                               # 요청의 기술 순서 = 텐서의 기술 칸 순서인가
        n["mv_ok"] += int(all(norm(a["move"]) == mv[int(a["choice"].split()[1]) - 1] for a in p["acts"] if a["kind"] == "move"))
        dhp, fnt, am, ao = sim_delta(p["pre"], post, mm, om)
        hp_e, f_e, am_e, ao_e = enc_delta(z, r)
        v = ~np.isnan(dhp)
        err = np.abs(dhp[v] - hp_e[v])
        n["hp"] += v.sum(); n["hp_ok"] += int((err < 0.025).sum())
        n["fnt"] += v.sum(); n["fnt_ok"] += int((fnt[v] == f_e[v]).sum())
        n["am"] += 1; n["am_ok"] += int(am == am_e)
        if ao != -2:
            n["ao"] += 1; n["ao_ok"] += int(ao == ao_e)
        if (err >= 0.025).any() and len(n["bad"]) < 6:
            n["bad"].append((sid, np.round(dhp, 3).tolist(), np.round(hp_e, 3).tolist()))
    print(f"비교 {n['mv']}장면, 제외 {n['skip']}")
    for k in ("mv", "hp", "fnt", "am", "ao"):
        print(f"{k}: {n[k + '_ok']}/{n[k]} = {n[k + '_ok'] / max(n[k], 1):.3f}")
    for x in n["bad"]:
        print("불일치 예", x)


def rank_corr(a, b):
    """스피어만 상관 (동점은 평균 순위)"""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    ra = np.argsort(np.argsort(a + 1e-9 * np.arange(len(a)))); rb = np.argsort(np.argsort(b + 1e-9 * np.arange(len(b))))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(cf, npz, ckpts):
    P, z, ids, off = load(cf, npz)
    models = {c: model_from_ckpt(f"checkpoints/supervised_v2_fp_{c}.pt").eval() for c in ckpts}
    res = {c: dict(se=[], se0=[], seb=[], floor=[], fse=[], fse0=[], fseb=[], ffloor=[], rho=[], top1=[], top1w=[], rho_q=[], n_pos=0, n_cell=0, grp={}) for c in ckpts}
    skipped = excl = 0
    for b, pid in enumerate(ids):
        if pid is None:
            continue
        p = P[pid]
        s, e = off[b], off[b + 1]
        r = e - 1
        mm, om = slot_map(z["my_team_cat"][r], p["pre"]["p1"]), slot_map(z["opp_team_cat"][r], p["pre"]["p2"])
        if None in mm:
            skipped += 1
            continue
        inv = {nm: j for j, nm in enumerate(mm)}
        A = [act_index(a, inv) for a in p["acts"]]
        ok = [i for i, a in enumerate(A) if a is not None and a < CT.N_ACT]
        O = [o["cls"] for o in p["opps"]]
        # 정답: 칸 (a,o)마다 반복 평균 + 반복 분산
        dh, fn, vr, fvr, nr = {}, {}, {}, {}, {}
        for c in p["cells"]:
            if c["a"] not in ok:
                continue
            if any(x["req"] != "move" for x in c["reps"]):             # 판이 끝나거나 강제 교체/피벗 요청으로 끝난 칸은 제외 (인코더 정의와 같은 깨끗한 칸만)
                excl += 1
                continue
            ds = [sim_delta(p["pre"], x, mm, om) for x in c["reps"]]
            H, F = np.stack([d[0] for d in ds]), np.stack([d[1] for d in ds])
            dh[c["a"], c["o"]], fn[c["a"], c["o"]] = H.mean(0), F.mean(0)
            vr[c["a"], c["o"]], fvr[c["a"], c["o"]] = H.var(0, ddof=1), F.var(0, ddof=1)
            nr[c["a"], c["o"]] = len(ds)
        keys = sorted(dh)
        if not keys:
            continue
        slotv = ~np.isnan(np.stack([dh[k] for k in keys])).any(0)
        if not slotv.any():
            continue
        Th = np.stack([dh[k][slotv] for k in keys]); Tf = np.stack([fn[k][slotv] for k in keys])
        Vh = np.stack([vr[k][slotv] for k in keys]); Vf = np.stack([fvr[k][slotv] for k in keys])
        Rn = np.array([nr[k] for k in keys])[:, None]                # 칸마다 평균에 쓴 표본 수 (반복 x 클래스 기술 수)
        for c, m in models.items():
            with torch.no_grad():
                out = m.forward_sequences([torch.tensor(z[k][s:e]) for k in KEYS], torch.zeros(e - s, dtype=torch.long), torch.arange(e - s), 1, torch.tensor(z["action_mask"][s:e]))
                h = out["chain_h"][-1]                              # [14,K,R]
                o = m.outcome(h.reshape(-1, h.size(-1)))
                dhp = o["dhp"].reshape(14, CT.N_CLS, 12).numpy()
                fp = torch.sigmoid(o["faint"]).reshape(14, CT.N_CLS, 12).numpy()
                if "_rt" in c:                                      # 행동 상대형 내 쪽 출력: 피해받는 슬롯(비교체 = 현재 활성, 교체 = 들어오는 슬롯)에만 r_my 값을 놓고 나머지 내 슬롯은 0
                    mh = o["my_hp"].reshape(14, CT.N_CLS).numpy(); mf = torch.sigmoid(o["my_faint"]).reshape(14, CT.N_CLS).numpy()
                    dhp[:, :, :6] = 0; fp[:, :, :6] = 0
                    act = int(np.argmax(z["my_team_num"][r][:, 0]))
                    for a_ in range(CT.N_ACT):
                        t_ = a_ - 8 if a_ >= 8 else act
                        dhp[a_, :, t_] = mh[a_]; fp[a_, :, t_] = mf[a_]
                po = torch.softmax(out["opp_logits"][-1], -1).numpy()
                qh = out["q_hat"][-1].numpy()
            Ph = np.stack([dhp[A[k[0]], p["opps"][k[1]]["cls"]][slotv] for k in keys])
            Pf = np.stack([fp[A[k[0]], p["opps"][k[1]]["cls"]][slotv] for k in keys])
            mean_h, mean_f = Th.mean(0, keepdims=True), Tf.mean(0, keepdims=True)
            x = res[c]
            x["n_pos"] += 1; x["n_cell"] += len(keys)
            x["se"].append(((Ph - Th) ** 2).mean()); x["se0"].append((Th ** 2).mean()); x["seb"].append(((mean_h - Th) ** 2).mean()); x["floor"].append((Vh / Rn).mean())
            x["fse"].append(((Pf - Tf) ** 2).mean()); x["fse0"].append((Tf ** 2).mean()); x["fseb"].append(((mean_f - Tf) ** 2).mean()); x["ffloor"].append((Vf / Rn).mean())
            isopp = np.arange(12)[slotv] >= 6
            kinds = np.array([("교체" if p["acts"][k[0]]["kind"] == "switch" else "테라 기술" if p["acts"][k[0]]["tera"] else "기술") for k in keys])
            for g, m_ in [("내 슬롯", np.tile(~isopp, (len(keys), 1))), ("상대 슬롯", np.tile(isopp, (len(keys), 1)))] + [(f"행동 {kd}", np.tile(np.ones(len(isopp), bool), (len(keys), 1)) & (kinds == kd)[:, None]) for kd in ("기술", "테라 기술", "교체")]:
                a_ = x["grp"].setdefault(g, np.zeros(4))
                a_ += [((Ph - Th) ** 2)[m_].sum(), (Th ** 2)[m_].sum(), ((mean_h - Th) ** 2)[m_].sum(), m_.sum()]
            # 수 순위: 이득 = 내 HP 변화 합 - 상대 HP 변화 합 (대응되는 슬롯만)
            side = (np.arange(12)[slotv] >= 6) * -2 + 1.0
            adv_t, adv_p = (Th * side).sum(1), (Ph * side).sum(1)
            acts = sorted({k[0] for k in keys}); opps = sorted({k[1] for k in keys})
            T = np.full((len(acts), len(opps)), np.nan); Pm = T.copy()
            for i, k in enumerate(keys):
                T[acts.index(k[0]), opps.index(k[1])] = adv_t[i]; Pm[acts.index(k[0]), opps.index(k[1])] = adv_p[i]
            for j in range(len(opps)):
                rc = rank_corr(Pm[:, j], T[:, j])
                if not np.isnan(rc):
                    x["rho"].append(rc)
            Tm, PmM = np.nanmean(T, 1), np.nanmean(Pm, 1)
            x["top1"].append(float(np.argmax(PmM) == np.argmax(Tm)))
            w = po[[p["opps"][o]["cls"] for o in opps]]; w = w / w.sum()
            x["top1w"].append(float(np.argmax(np.nansum(Pm * w, 1)) == np.argmax(Tm)))
            qa = np.array([qh[A[a]] @ po for a in acts])
            rq = rank_corr(qa, Tm)
            if not np.isnan(rq):
                x["rho_q"].append(rq)
    print(f"시퀀스 {len(ids)} 중 id 못 맞춘 것 {sum(i is None for i in ids)}, 슬롯 대응 실패로 제외 {skipped}, 깨끗하지 않은 칸 제외 {excl}")
    for c, x in res.items():
        mean = lambda k: float(np.mean(x[k]))
        ex = lambda a, b, fl: (mean(a) - mean(b)) / max(mean(a) - mean(fl), 1e-12)
        print(f"===== {c}: 위치 {x['n_pos']} 칸 {x['n_cell']}")
        print(f"HP Δ MSE: 모델 {mean('se'):.5f} | 0 예측 {mean('se0'):.5f} | 위치 평균(a,o 무시, 정답 평균 사용=낙관적 기준) {mean('seb'):.5f} | 난수 하한 {mean('floor'):.5f}")
        print(f"  설명 비율 (0 예측 -> 하한): {ex('se0', 'se', 'floor'):.3f} | 위치 평균 기준 대비 설명 비율: {(mean('seb') - mean('se')) / max(mean('seb') - mean('floor'), 1e-12):.3f}")
        for g, a_ in x["grp"].items():
            print(f"  [{g}] HP Δ MSE: 모델 {a_[0] / a_[3]:.5f} | 0 예측 {a_[1] / a_[3]:.5f} | 위치 평균 {a_[2] / a_[3]:.5f} | 모델이 0 예측 대비 줄인 비율 {1 - a_[0] / a_[1]:.3f}")
        print(f"새 기절 MSE: 모델 {mean('fse'):.5f} | 0 예측 {mean('fse0'):.5f} | 위치 평균 {mean('fseb'):.5f} | 난수 하한 {mean('ffloor'):.5f} | 설명 비율 {ex('fse0', 'fse', 'ffloor'):.3f}")
        print(f"수 순위 (응수 고정, 위치별 스피어만 평균) {np.mean(x['rho']):.3f} (n={len(x['rho'])}) | 응수 균등 평균 최상위 일치 {mean('top1'):.3f} | 모델 p(o) 가중 {mean('top1w'):.3f} | Q헤드 순위 상관 {np.mean(x['rho_q']):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "equiv":
        equiv(sys.argv[2], sys.argv[3])
    else:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4:])
