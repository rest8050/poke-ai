"""상대의 숨겨진 정보(기술/도구/특성) 예측 보조 손실. 정답은 같은 배틀의 마지막 턴 관측에서 얻음(외부 표/재가공 불필요):
턴 t에 아직 안 보이는데 이 배틀이 끝나기 전에 드러난 기술/도구/특성이 정답. 끝내 안 드러난 것은 모름(음성 라벨 아님)이라
정답 집합에 대한 소프트맥스 CE(양성만)로 계산. 이미 드러난 기술은 후보에서 제외.
정답은 슬롯 번호가 아니라 종으로 맞춤: 인코더는 상대 슬롯을 '본 순서 + 프리뷰로만 본 것'으로 채워서 새로 등장할 때마다 위치가 바뀜.
같은 종이 없으면 폼 이름의 접두사로 맞춤(프리뷰 'urshifu' -> 실제 'urshifurapidstrike'), 마지막 턴에 그 종이 없으면 건너뜀.
필요 입력: b["opp_team_cat"] [N,6,11](0 도구, 1 특성, 5~8 기술, 10 종), b["seq_index"] [N] (결정이 배틀 순서대로 이어 붙어 있음)"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from src.core.matchup_v7 import speed_bucket_ratio
from src.core.model import STAT_SCALE


def vocab_sizes(path="data/vocab.json"):
    v = json.load(open(path, encoding="utf-8"))
    return {k: v[k]["<unk>"] for k in ("move", "item", "ability")}   # <unk> id 이상은 정답으로 쓰지 않음 (unk, 숨김)


_FORM_REL = {}


def _form_rel(device, path="data/vocab.json"):
    """종 어휘 [V,V] bool: 같은 종이거나 한쪽 이름이 다른 쪽의 접두사 (폼 차이: urshifu ~ urshifurapidstrike). 장치별로 한 번만 만듦"""
    if device not in _FORM_REL:
        allsp = json.load(open(path, encoding="utf-8"))["species"]
        sp = {n: i for n, i in allsp.items() if n and not n.startswith("<")}       # 특수 칸(<unk> 등)은 자기 자신과만 같음
        V = max(allsp.values()) + 1
        rel = torch.eye(V, dtype=torch.bool)
        for n, i in sp.items():
            for k in range(1, len(n)):
                j = sp.get(n[:k])
                if j is not None:
                    rel[i, j] = rel[j, i] = True
        _FORM_REL[device] = rel.to(device)
    return _FORM_REL[device]


def align_last(cat, last):
    """결정마다 상대 슬롯 i와 같은 포켓몬이 마지막 턴 관측의 어느 슬롯에 있는지 찾아 그 행을 가져옴 -> (정렬된 last [N,6,C], 찾았는가 [N,6]).
    같은 종을 먼저, 없으면 폼 접두사로 (한 팀에 같은 종은 하나라 겹치지 않음)"""
    sid, lid = cat[..., 10], last[..., 10]
    rel = _form_rel(cat.device)
    V = rel.size(0)
    present = (sid > 0)[..., None] & (lid > 0)[:, None, :]                        # [N,6(지금 i),6(마지막 j)]
    exact = (sid[:, :, None] == lid[:, None, :]) & present
    form = rel[sid.clamp(0, V - 1)[:, :, None], lid.clamp(0, V - 1)[:, None, :]] & present
    match = torch.where(exact.any(-1, keepdim=True), exact, form)
    j = match.float().argmax(-1)                                                  # 첫 번째로 맞는 슬롯
    return last.gather(1, j[..., None].expand(-1, -1, last.size(-1))), match.any(-1)


def true_sets(data, offsets):
    """시퀀스마다 상대 팀의 실제 세트 -> ([G,6,17] int16 (종 순서로 정렬), 찾았는가 [G] bool). 열 0~10 = 같은 판의 반대편 시퀀스 첫 결정의 my_team_cat, 11~16 = 실제 능력치 hp..spe.
    반대편 후보 = 내 팀/상대 팀 종 6마리가 서로 뒤바뀌고 승패가 반대인 시퀀스 (상대 종은 마지막 결정 기준) 중, 내가 본 상대의 공개 기술이 전부 그 팀 기술에 들어 있는 것.
    같은 두 팀의 다른 판도 후보에 들어오지만 팀이 같으면 세트도 같음 -> 후보들의 첫 결정이 전부 같을 때만 씀.
    ponytail: 못 찾는 경우(프리뷰 이름만 남은 폼 'urshifu', 세트에 없는 기술이 드러남(흉내/발버둥), 로드한 데이터에 반대편이 없음)는 예전 라벨(배틀에서 드러난 것)로 남음"""
    first, last = np.asarray(offsets[:-1]), np.asarray(offsets[1:]) - 1
    my = np.concatenate([np.asarray(data["my_team_cat"][first]).astype(np.int16),
                         np.round(np.asarray(data["my_team_num"][first][..., 11:17], dtype=np.float32) * 255).astype(np.int16)], -1)
    my = np.take_along_axis(my, np.argsort(my[..., 10], 1)[..., None], 1)          # 종 순서로 정렬
    opp = np.asarray(data["opp_team_cat"][last]).astype(np.int16)
    opp = np.take_along_axis(opp, np.argsort(opp[..., 10], 1)[..., None], 1)
    win = np.sign(np.asarray(data["outcome"])).astype(int)
    unk = vocab_sizes()["move"]
    by = {}
    for g in range(len(first)):
        by.setdefault((my[g, :, 10].tobytes(), opp[g, :, 10].tobytes(), int(win[g])), []).append(g)
    out, ok = np.zeros_like(my), np.zeros(len(first), bool)
    for g in range(len(first)):
        rev = opp[g, :, 5:9]                                                       # [6,4] 내가 본 상대 기술 (종 순서가 후보의 내 팀과 같음)
        seen = (rev >= 1) & (rev < unk)
        c = [j for j in by.get((opp[g, :, 10].tobytes(), my[g, :, 10].tobytes(), -int(win[g])), ())
             if ((rev[:, :, None] == my[j][:, None, 5:9]).any(-1) | ~seen).all()]
        if c and all(np.array_equal(my[c[0]], my[j]) for j in c[1:]):
            out[g], ok[g] = my[c[0]], True
    return out, ok


def _scarf_id(_c=[]):
    if not _c:
        _c.append(json.load(open("data/vocab.json", encoding="utf-8"))["item"]["choicescarf"])
    return _c[0]


def hidden_loss(logits: dict, b: dict, unk: dict):
    """logits: {"move": [N,6,Vm], "item": [N,6,Vi], "ability": [N,6,Va]} -> (손실 합, 통계 dict)"""
    cat = b["opp_team_cat"].long()
    counts = torch.bincount(b["seq_index"])
    last = cat[(torch.cumsum(counts, 0) - 1)[b["seq_index"]]]                     # 각 결정이 속한 배틀의 마지막 턴 관측
    if "true_set" in b:                                                            # 상대 팀의 실제 세트를 아는 판은 그것이 정답 (true_sets), 모르는 판은 드러난 것만
        last = torch.where(b["true_ok"][b["seq_index"]][:, None, None], b["true_set"][b["seq_index"]], F.pad(last, (0, 6)))
    last, same = align_last(cat, last)                                             # 슬롯 i의 포켓몬이 마지막 턴에 있던 행 (같은 종/폼)
    total, stats = logits["move"].new_zeros(()), {}
    # 기술: 앞으로 드러날 기술들에 대한 CE (이미 드러난 기술은 후보에서 제외)
    cur, fut = cat[..., 5:9], last[..., 5:9]
    lg = logits["move"].float()
    V = lg.size(-1)
    known = torch.zeros_like(lg, dtype=torch.bool)
    known.scatter_(2, torch.where((cur >= 1) & (cur < unk["move"]), cur, torch.zeros_like(cur)).clamp(0, V - 1), True)
    known[..., 0] = False
    lp = F.log_softmax(lg.masked_fill(known, -1e4), -1)
    valid = (fut >= 1) & (fut < unk["move"]) & same[..., None] & ~(fut[..., :, None] == cur[..., None, :]).any(-1)
    nll = -lp.gather(2, fut.clamp(0, V - 1))
    n = valid.sum()
    if n > 0:
        total = total + (nll * valid).sum() / n
        top4 = lp.topk(4, -1).indices                                              # 상위 4개 안에 정답이 있는 비율
        stats["move_top4"] = ((fut[..., :, None] == top4[..., None, :]).any(-1) & valid).sum().item() / n.item()
    stats["move_n"] = int(n)
    # 도구/특성: 지금은 숨겨졌고(숨김 id = unk+1) 나중에 드러난 경우
    for key, col in (("item", 0), ("ability", 1)):
        c, f = cat[..., col], last[..., col]
        ok = (c == unk[key] + 1) & (f >= 1) & (f < unk[key]) & same
        m = ok.sum()
        if m > 0:
            lpk = F.log_softmax(logits[key].float(), -1)
            total = total + (-(lpk.gather(2, f.clamp(0, lpk.size(-1) - 1)[..., None])[..., 0]) * ok).sum() / m
            stats[f"{key}_acc"] = ((lpk.argmax(-1) == f) & ok).sum().item() / m.item()
        stats[f"{key}_n"] = int(m)
    if "stat" in logits and last.size(-1) > 11:                                    # 능력치 배율: 실제 능력치 / 표준 가정(252노력치, 무보정)의 로그 (pretrain_belief.stat_logratio와 같은 척도)
        real, base = last[..., 11:17].float(), b["opp_team_num"][..., 11:17].float() * 255
        ratio = real / torch.cat([2 * base[..., :1] + 204, 2 * base[..., 1:] + 99], -1)
        ok = same[..., None] & (ratio > 0.4) & (ratio < 1.2)                       # 범위 밖 = 폼이 달라 종족값이 안 맞음/인코더가 미리 곱한 부스트에너지 -> 그 능력치만 뺌
        if ok.any():
            err = (logits["stat"].float() - torch.log(ratio.clamp_min(1e-3)) / STAT_SCALE)[ok]
            total = total + 0.5 * (err ** 2).mean()                                # 0.5 = pretrain_belief.STAT_COEF
            stats["stat_mae"] = err.abs().mean().item() * STAT_SCALE
        if "spe" in logits and ok[..., 5].any():                                     # 스피드 세트 구간: 실제 스피드에 가장 가까운 구간(로그 거리)이 정답
            tgt = (torch.log(speed_bucket_ratio(base[..., 5])) - torch.log(ratio[..., 5:6].clamp_min(1e-3))).abs().argmin(-1)
            lg = logits["spe"].float()
            if lg.dim() == 4:                                                        # [N,6,2,K]: 실제 도구가 스카프면 1행, 아니면 0행의 분포를 학습
                lg = torch.where((last[..., 0] == _scarf_id())[..., None], lg[..., 1, :], lg[..., 0, :])
            lg = lg[ok[..., 5]]
            total = total + F.cross_entropy(lg, tgt[ok[..., 5]])
            stats["spe_acc"], stats["spe_n"] = (lg.argmax(-1) == tgt[ok[..., 5]]).float().mean().item(), int(ok[..., 5].sum())
    return total, stats
