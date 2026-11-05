"""상대의 숨겨진 정보(기술/도구/특성) 예측 보조 손실. 정답은 같은 배틀의 마지막 턴 관측에서 얻음(외부 표/재가공 불필요):
턴 t에 아직 안 보이는데 이 배틀이 끝나기 전에 드러난 기술/도구/특성이 정답. 끝내 안 드러난 것은 모름(음성 라벨 아님)이라
정답 집합에 대한 소프트맥스 CE(양성만)로 계산. 이미 드러난 기술은 후보에서 제외.
정답은 슬롯 번호가 아니라 종으로 맞춤: 인코더는 상대 슬롯을 '본 순서 + 프리뷰로만 본 것'으로 채워서 새로 등장할 때마다 위치가 바뀜.
같은 종이 없으면 폼 이름의 접두사로 맞춤(프리뷰 'urshifu' -> 실제 'urshifurapidstrike'), 마지막 턴에 그 종이 없으면 건너뜀.
필요 입력: b["opp_team_cat"] [N,6,11](0 도구, 1 특성, 5~8 기술, 10 종), b["seq_index"] [N] (결정이 배틀 순서대로 이어 붙어 있음)"""
import json

import torch
import torch.nn.functional as F


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


def hidden_loss(logits: dict, b: dict, unk: dict):
    """logits: {"move": [N,6,Vm], "item": [N,6,Vi], "ability": [N,6,Va]} -> (손실 합, 통계 dict)"""
    cat = b["opp_team_cat"].long()
    counts = torch.bincount(b["seq_index"])
    last = cat[(torch.cumsum(counts, 0) - 1)[b["seq_index"]]]                     # 각 결정이 속한 배틀의 마지막 턴 관측
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
    return total, stats
