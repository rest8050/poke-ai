"""상대 세트 신념(SetBelief) 사전학습: 팀 데이터(Showdown 내보내기)만 쓰고 배틀은 필요 없음.
포켓몬마다 무작위로 일부 정보만 공개한 부분 관측(공개 기술 0~4개, 도구/특성 공개 여부)을 만들고, 정답은 그 포켓몬의 **완전한 세트**(안 드러난 기술/도구/특성).
배틀 로그의 "나중에 드러난 것"과 달리 끝까지 안 쓴 기술/도구도 라벨이라 편향이 없고, 샘플은 매 스텝 새로 뽑으니 사실상 무한.
학습용 팀 풀만 사용 (평가용 풀은 학습에 안 씀, 평가 지표만 따로 계산). 팀 단위로 5%를 떼어 검증.
사용: python src/training/pretrain_belief.py [--steps 12000] [--out checkpoints/belief.pt]
참고(ponytail): 도구 효과로 종족값을 곱하는 인코더 보정(구애 계열 등)은 흉내 내지 않음 / 배틀 상황 입력(HP, 증거)은 기본값이라 미세조정에서 배움"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from poke_env.data import GenData

from src.core.belief import STAT_SCALE, SetBelief
from src.core.tensor_encoder import NUM_DIM, VocabManager

TRAIN_POOLS = [("metamon", "data/team_pool_metamon.json", 0.5), ("randomset", "data/team_pool_randomset_train3.json", 0.2),
               ("rare", "data/team_pool_rare_train.json", 0.1), ("uber", "data/team_pool_randomset_uber_train.json", 0.1),
               ("dagger", "data/team_pool_dagger_tr.json", 0.1)]
EVAL_POOLS = [("holdout(A)", "data/team_pool_metamon_holdout.json"), ("rare(B)", "data/team_pool_rare.json"), ("randomset(C)", "data/team_pool_randomset.json")]
REVEAL_P = [0.25, 0.30, 0.20, 0.15, 0.10]    # 공개 기술 수 0~4개 분포 (대전에서 상대 활성의 공개 기술 수 분포와 비슷)
P_ITEM_KNOWN, P_ABILITY_KNOWN = 0.25, 0.30
STAT_COEF = 0.5                              # 능력치 회귀 손실 가중 (기술 NLL ~3 대비)
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
NATURES = {"adamant": ("atk", "spa"), "bold": ("def", "atk"), "brave": ("atk", "spe"), "calm": ("spd", "atk"), "careful": ("spd", "spa"), "gentle": ("spd", "def"),
           "hasty": ("spe", "def"), "impish": ("def", "spa"), "jolly": ("spe", "spa"), "lax": ("def", "spd"), "lonely": ("atk", "def"), "mild": ("spa", "def"),
           "modest": ("spa", "atk"), "naive": ("spe", "spd"), "naughty": ("atk", "spd"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"), "relaxed": ("def", "spe"),
           "sassy": ("spd", "spe"), "timid": ("spe", "atk")}      # 올리는 능력치, 내리는 능력치 (나머지는 무보정)


def stat_logratio(base, evs, ivs, nature):
    """레벨 100 실제 능력치 / 매치업의 표준 가정(252노력치, 무보정, 개체값 31)의 로그. base/evs/ivs: hp..spe 6개, nature: 이름(없으면 무보정). 표준: 비-HP 2b+99, HP 2b+204"""
    up, down = NATURES.get(nature, (None, None))
    out = []
    for k, n in enumerate(STATS):
        if n == "hp":
            real, std = 2 * base[k] + ivs[k] + evs[k] // 4 + 110, 2 * base[k] + 204
        else:
            real = (2 * base[k] + ivs[k] + evs[k] // 4 + 5) * (1.1 if n == up else 0.9 if n == down else 1.0) // 1
            std = 2 * base[k] + 99
        out.append(math.log(real / std))
    return out


def parse_team(export, vm, dex):
    """Showdown 내보내기 -> 포켓몬 목록 (모르는 종은 건너뜀). 도구/특성/기술이 어휘에 없으면 해당 값은 None/빈칸"""
    unk = {k: vm.unk_id(k) for k in ("item", "ability", "move", "species")}
    mons = []
    for blk in export.strip().split("\n\n"):
        lines = blk.strip().split("\n")
        head = lines[0].split(" @ ")
        name = re.sub(r"\s*\((M|F)\)\s*$", "", head[0].strip())
        m = re.search(r"\(([^()]+)\)\s*$", name)
        sp = norm(m.group(1) if m else name)
        if sp not in dex:
            continue
        e = dex[sp]
        item = vm.get_id("item", norm(head[1])) if len(head) > 1 else None   # 어휘 키는 소문자/영숫자만
        ability = tera = nature = None
        evs, ivs = [0] * 6, [31] * 6
        moves = []
        for ln in lines[1:]:
            if ln.startswith("EVs:") or ln.startswith("IVs:"):
                tgt = evs if ln[0] == "E" else ivs
                for v, n in re.findall(r"(\d+) (HP|Atk|Def|SpA|SpD|Spe)", ln):
                    tgt[STATS.index(n.lower())] = int(v)
            elif ln.strip().endswith(" Nature"):
                nature = norm(ln.strip()[:-len(" Nature")])
            elif ln.startswith("Ability:"):
                ability = vm.get_id("ability", norm(ln.split(":", 1)[1]))
            elif ln.startswith("Tera Type:"):
                tera = vm.get_id("type", norm(ln.split(":", 1)[1]))
            elif ln.startswith("- "):
                mid = vm.get_id("move", norm(ln[2:]))
                if mid < unk["move"] and mid not in moves:
                    moves.append(mid)
        types = [vm.get_id("type", t) for t in e["types"]][:2]
        st = e["baseStats"]
        base6 = [st[k] for k in ("hp", "atk", "def", "spa", "spd", "spe")]
        mons.append(dict(species=vm.get_id("species", sp), stats=[st["hp"] / 255, st["atk"] / 255, st["def"] / 255, st["spa"] / 255, st["spd"] / 255, st["spe"] / 255,
                                                           min(1.0, e.get("weightkg", 0) / 500)],
                         types=types + [0] * (2 - len(types)), item=item if item is not None and item < unk["item"] else None,
                         ability=ability if ability is not None and ability < unk["ability"] else None, moves=moves[:4],
                         stat=stat_logratio(base6, evs, ivs, nature)))
    return mons


def load_pool(path, vm, dex):
    teams = json.load(open(path, encoding="utf-8-sig"))["teams"]
    return [parse_team(t["export"], vm, dex) for t in teams]


def to_arrays(teams, vm, device):
    """팀 목록 -> 포켓몬 단위 텐서 묶음 (team 인덱스 포함)"""
    rows = [(ti, m) for ti, t in enumerate(teams) for m in t]
    N = len(rows)
    hid = {"item": vm.unk_id("item") + 1, "ability": vm.unk_id("ability") + 1}
    A = dict(species=torch.tensor([m["species"] for _, m in rows]), stats=torch.tensor([m["stats"] for _, m in rows], dtype=torch.float32),
             types=torch.tensor([m["types"] for _, m in rows]), team=torch.tensor([ti for ti, _ in rows]),
             item=torch.tensor([-1 if m["item"] is None else m["item"] for _, m in rows]),
             ability=torch.tensor([-1 if m["ability"] is None else m["ability"] for _, m in rows]),
             moves=torch.tensor([m["moves"] + [0] * (4 - len(m["moves"])) for _, m in rows]),
             stat=torch.tensor([m["stat"] for _, m in rows], dtype=torch.float32) / STAT_SCALE)
    return {k: v.to(device) for k, v in A.items()}, N, hid


def make_batch(A, idx, hid, hidden_move_id, gen):
    """부분 관측 입력(opp_cat [B,1,11], opp_num [B,1,NUM_DIM])과 정답을 만듦. 공개 여부는 무작위"""
    B, dev = len(idx), A["species"].device
    moves = A["moves"][idx]                                                      # [B,4] (0 = 없음)
    have = moves > 0
    k = torch.multinomial(torch.tensor(REVEAL_P, device=dev), B, replacement=True, generator=gen)
    k = torch.minimum(k, have.sum(-1))
    score = torch.rand(B, 4, device=dev, generator=gen).masked_fill(~have, -1.0)
    rank = score.argsort(-1, descending=True).argsort(-1)                         # 점수 순위 (0이 가장 높음)
    revealed = have & (rank < k[:, None])
    shown = torch.where(revealed, moves, torch.full_like(moves, hidden_move_id))   # 안 드러난 칸은 숨김 ID
    item, ability = A["item"][idx], A["ability"][idx]
    item_known = (torch.rand(B, device=dev, generator=gen) < P_ITEM_KNOWN) | (item < 0)
    ab_known = (torch.rand(B, device=dev, generator=gen) < P_ABILITY_KNOWN) | (ability < 0)
    cat = torch.zeros(B, 1, 11, dtype=torch.long, device=dev)
    cat[:, 0, 0] = torch.where(item_known & (item >= 0), item, torch.full_like(item, hid["item"]))
    cat[:, 0, 1] = torch.where(ab_known & (ability >= 0), ability, torch.full_like(ability, hid["ability"]))
    cat[:, 0, 2:4] = A["types"][idx]
    cat[:, 0, 5:9] = shown
    cat[:, 0, 10] = A["species"][idx]
    num = torch.zeros(B, 1, NUM_DIM, device=dev)
    num[:, 0, 11:18] = A["stats"][idx]
    num[:, 0, 2] = 1.0                                                           # HP 100%
    num[:, 0, 8] = 1.0                                                           # 레벨 100
    return cat, num, dict(stat=A["stat"][idx], moves=moves, revealed=revealed, item=item, item_hidden=~item_known & (item >= 0), ability=ability, ab_hidden=~ab_known & (ability >= 0))


def belief_loss(logits, tgt, unk_move):
    """안 드러난 정답 기술(완전한 세트 - 공개분)에 대한 소프트맥스 NLL + 숨겨진 도구/특성 CE. 통계(상위 4 적중 등)도 반환"""
    lg = logits["move"][:, 0].float()
    V = lg.size(-1)
    known = torch.zeros_like(lg, dtype=torch.bool)
    known.scatter_(1, torch.where(tgt["revealed"], tgt["moves"], torch.zeros_like(tgt["moves"])), True)
    bad = torch.zeros(V, dtype=torch.bool, device=lg.device); bad[0], bad[unk_move:] = True, True
    lp = F.log_softmax(lg.masked_fill(known | bad, -1e4), -1)
    target = (tgt["moves"] > 0) & ~tgt["revealed"]
    n = target.sum()
    loss = -(lp.gather(1, tgt["moves"]) * target).sum() / n.clamp_min(1)
    top4 = lp.topk(4, -1).indices
    stats = {"move_top4": (((tgt["moves"][:, :, None] == top4[:, None, :]).any(-1) & target).sum() / n.clamp_min(1)).item()}
    pred = logits["stat"][:, 0].float()
    loss = loss + STAT_COEF * F.mse_loss(pred, tgt["stat"])
    stats["stat_mae"] = ((pred - tgt["stat"]).abs().mean() * STAT_SCALE).item()               # 표준 가정과 비교할 기준: stat_base
    stats["stat_base"] = (tgt["stat"].abs().mean() * STAT_SCALE).item()
    for key, hidden in (("item", "item_hidden"), ("ability", "ab_hidden")):
        m = tgt[hidden]
        if m.any():
            lk = F.log_softmax(logits[key][:, 0].float(), -1)[m]
            y = tgt[key][m]
            loss = loss - lk.gather(1, y[:, None]).mean()
            top3 = lk.topk(3, -1).indices
            stats[f"{key}_top1"] = (lk.argmax(-1) == y).float().mean().item()
            stats[f"{key}_top3"] = (top3 == y[:, None]).any(-1).float().mean().item()
    return loss, stats


@torch.no_grad()
def evaluate(model, A, hid, unk_move, gen, n_batches=20, batch=4096, calib=False):
    model.eval()
    N = len(A["species"])
    agg, bins = {}, np.zeros((10, 3))
    for _ in range(n_batches):
        idx = torch.randint(0, N, (batch,), device=A["species"].device, generator=gen)
        cat, num, tgt = make_batch(A, idx, hid, unk_move + 1, gen)
        logits = model(cat, num)
        _, st = belief_loss(logits, tgt, unk_move)
        for k, v in st.items():
            agg.setdefault(k, []).append(v)
        if calib:                                                                # 기술 확률 보정: 상위 4개 예측의 (기대 확률 -> 실제 적중)
            lg = logits["move"][:, 0].float()
            known = torch.zeros_like(lg, dtype=torch.bool); known.scatter_(1, torch.where(tgt["revealed"], tgt["moves"], torch.zeros_like(tgt["moves"])), True)
            bad = torch.zeros(lg.size(-1), dtype=torch.bool, device=lg.device); bad[0], bad[unk_move:] = True, True
            p = torch.softmax(lg.masked_fill(known | bad, -1e4), -1)
            n_unrev = ((tgt["moves"] > 0) & ~tgt["revealed"]).sum(-1, keepdim=True).float()
            tv, ti = p.topk(4, -1)
            marg = (tv * n_unrev).clamp(max=1.0)
            hit = ((ti[:, :, None] == tgt["moves"][:, None, :]) & ((tgt["moves"] > 0) & ~tgt["revealed"])[:, None, :]).any(-1).float()
            ok = n_unrev > 0
            b = (marg.clamp(max=0.999) * 10).long()
            for bi in range(10):
                sel = (b == bi) & ok
                bins[bi] += [sel.sum().item(), marg[sel].sum().item(), hit[sel].sum().item()]
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    if calib:
        tot = bins[:, 0].sum()
        out["ECE"] = float(sum(abs(b[1] - b[2]) for b in bins) / max(1, tot))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="checkpoints/belief.pt")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vm = VocabManager.get_instance("data/vocab.json")
    dex = GenData.from_gen(9).pokedex
    unk_move = vm.unk_id("move")
    t0 = time.time()
    pools = {n: load_pool(p, vm, dex) for n, p, _ in TRAIN_POOLS if os.path.exists(p)}
    weights = {n: w for n, p, w in TRAIN_POOLS if n in pools}
    # 팀 단위 검증 분할 (풀별 5%)
    tr_teams, va = [], {}
    pool_of_team = []
    for n, teams in pools.items():
        cut = int(len(teams) * 0.95)
        tr_teams += teams[:cut]; pool_of_team += [n] * cut
        va[n] = teams[cut:]
    A, N, hid = to_arrays(tr_teams, vm, dev)
    pool_w = torch.tensor([weights[pool_of_team[t]] / sum(1 for x in pool_of_team if x == pool_of_team[t]) for t in range(len(tr_teams))], device=dev)
    sample_w = pool_w[A["team"]]
    print(f"학습 팀 {len(tr_teams)}개 / 포켓몬 {N}마리 (풀별 {({n: len(t) for n, t in pools.items()})}) | 로딩 {time.time() - t0:.0f}s")
    evals = {n: to_arrays(t, vm, dev)[0] for n, t in va.items() if t}
    for n, p in EVAL_POOLS:
        if os.path.exists(p):
            evals[n] = to_arrays(load_pool(p, vm, dex), vm, dev)[0]
    model = SetBelief().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    gen = torch.Generator(device=dev).manual_seed(0)
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.multinomial(sample_w, args.batch, replacement=True, generator=gen)
        cat, num, tgt = make_batch(A, idx, hid, unk_move + 1, gen)
        loss, st = belief_loss(model(cat, num), tgt, unk_move)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == args.steps:
            print(f"[{step}/{args.steps}] 손실 {loss.item():.3f} | 기술 상위4 적중 {st['move_top4']:.3f} | 도구 {st.get('item_top1', 0):.3f} | 특성 {st.get('ability_top1', 0):.3f} | 능력치 오차 {st['stat_mae']:.3f} (표준가정 {st['stat_base']:.3f}) | {time.time() - t0:.0f}s", flush=True)
    print("\n== 검증 (학습에 안 쓴 팀 / 평가용 풀). 지표: 안 드러난 기술의 상위 4 적중률, 숨은 도구/특성의 상위 1/3 적중, 기술 확률 보정 오차(ECE), 능력치 배율 오차(표준 가정 252/무보정과 비교)")
    for n, E in evals.items():
        r = evaluate(model, E, hid, unk_move, gen, calib=True)
        print(f"  {n:14s} 기술 상위4 {r['move_top4']:.3f} | 도구 top1 {r.get('item_top1', 0):.3f} top3 {r.get('item_top3', 0):.3f} | 특성 top1 {r.get('ability_top1', 0):.3f} | ECE {r['ECE']:.3f} | 능력치 오차(로그) {r['stat_mae']:.3f} / 표준가정 {r['stat_base']:.3f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"cfg": model.cfg, "state": {k: v.cpu() for k, v in model.state_dict().items()}}, args.out)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
