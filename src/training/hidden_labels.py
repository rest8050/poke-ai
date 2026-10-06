"""상대의 숨겨진 정보(기술/도구/특성) 예측 보조 손실. 정답은 같은 배틀의 마지막 턴 관측에서 얻음(외부 표/재가공 불필요):
턴 t에 아직 안 보이는데 이 배틀이 끝나기 전에 드러난 기술/도구/특성이 정답. 끝내 안 드러난 것은 모름(음성 라벨 아님)이라
정답 집합에 대한 소프트맥스 CE(양성만)로 계산. 이미 드러난 기술은 후보에서 제외. 마지막 턴에 같은 슬롯이 같은 종이 아닌 경우는 건너뜀.
필요 입력: b["opp_team_cat"] [N,6,11](0 도구, 1 특성, 5~8 기술, 10 종), b["seq_index"] [N] (결정이 배틀 순서대로 이어 붙어 있음)"""
import json

import torch
import torch.nn.functional as F


def vocab_sizes(path="data/vocab.json"):
    v = json.load(open(path, encoding="utf-8"))
    return {k: v[k]["<unk>"] for k in ("move", "item", "ability")}   # <unk> id 이상은 정답으로 쓰지 않음 (unk, 숨김)


def hidden_loss(logits: dict, b: dict, unk: dict):
    """logits: {"move": [N,6,Vm], "item": [N,6,Vi], "ability": [N,6,Va]} -> (손실 합, 통계 dict)"""
    cat = b["opp_team_cat"].long()
    counts = torch.bincount(b["seq_index"])
    last = cat[(torch.cumsum(counts, 0) - 1)[b["seq_index"]]]                     # 각 결정이 속한 배틀의 마지막 턴 관측
    same = (cat[..., 10] > 0) & (last[..., 10] == cat[..., 10])                    # 같은 슬롯에 같은 종
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
    return total, stats
