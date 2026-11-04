"""사슬 학습 정답 생성 (docs/chain_design.md): data/dyn 형식(시퀀스를 이어 붙인 행 배열)에서
- 상대 응수 클래스(opp_cls: 0~8 기술 클래스, 9 교체, -1 무효)와 테라 동반 플래그
- 한 턴 뒤 요약 Δ (다음 행의 상태와 현재 상태 차이): 양쪽 12마리 HP 변화, 이번 턴에 새로 기절한 마리, 다음 활성 슬롯(내/상대), 다음 활성의 부스트 5개씩
를 만든다. 열 번호는 tensor_encoder의 팀 수치: 0 활성, 1 기절, 2 HP 비율, 3~7 부스트.
Δ 정답은 "실제로 둔 수와 실제 응수" 쌍에만 의미가 있고(r_valid), 시퀀스의 마지막 행·강제 교체 행·응수를 모르는 행은 뺀다."""
import json
import os

import numpy as np

N_CLS = 10                    # 0~8 기술 클래스, 9 교체
SWITCH_CLS = 9
N_ACT = 14                    # 기술 4 + 테라 기술 4 + 교체 6 (나머지 22칸은 안 쓰는 자리)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def move_class_table(vocab_path=os.path.join(ROOT, "data", "vocab.json"), cls_path=os.path.join(ROOT, "data", "opp_classes.json")):
    """어휘 기술 id -> 클래스 (모르는 기술 -1). tools/gen_opp_classes.js가 만든 이름 표를 어휘 번호로 바꿈"""
    vocab = json.load(open(vocab_path, encoding="utf-8"))["move"]
    cls = json.load(open(cls_path, encoding="utf-8"))
    table = np.full(max(vocab.values()) + 1, -1, np.int8)
    for name, c in cls.items():
        if name in vocab:
            table[vocab[name]] = c
    return table


def build(data, lens, table=None):
    """data: opp_action [N], opp_move [N], kind [N], action_taken [N], my_team_num/opp_team_num [N,6,>=8], opp_team_cat [N,6,>=11]; lens: 시퀀스 길이 [S]
    반환: opp_cls int8 [N], opp_tera int8 [N] (기술 행만 0/1, 나머지 -1), r_valid bool [N], dhp float32 [N,12], faint float32 [N,12],
          act_my int8 [N], act_opp int8 [N] (다음 활성 슬롯, 없으면 -1), boost float32 [N,10]"""
    table = move_class_table() if table is None else table
    N = len(data["opp_action"])
    oa, om = np.asarray(data["opp_action"]), np.asarray(data["opp_move"])
    cls = np.full(N, -1, np.int8)
    tera = np.full(N, -1, np.int8)
    mv = (oa == 0) | (oa == 2)
    cls[mv] = table[np.clip(om[mv], 0, len(table) - 1)]
    cls[oa == 1] = SWITCH_CLS
    tera[mv & (cls >= 0)] = (oa[mv & (cls >= 0)] == 2).astype(np.int8)

    has_next = np.ones(N, bool)
    has_next[np.cumsum(lens) - 1] = False                 # 각 시퀀스의 마지막 행은 다음 상태가 없음
    cur = np.arange(N)
    nxt = np.minimum(cur + 1, N - 1)
    mine, opp = np.asarray(data["my_team_num"], np.float32), np.asarray(data["opp_team_num"], np.float32)
    s0 = np.concatenate([mine, opp], 1)                    # [N,12,F]
    s1 = s0[nxt]
    sp0, sp1 = np.asarray(data["opp_team_cat"])[:, :, 10], np.asarray(data["opp_team_cat"])[nxt][:, :, 10]
    hit = (sp0[:, :, None] == sp1[:, None, :]) & (sp0[:, :, None] > 0)                # 상대 슬롯은 공개 순서가 바뀌면 섞이므로 종 번호로 다음 행을 현재 행 슬롯에 맞춤
    perm = np.where(hit.any(2), hit.argmax(2), np.arange(6)[None])
    s1 = np.concatenate([s1[:, :6], np.take_along_axis(s1[:, 6:], perm[:, :, None], 1)], 1)
    dhp = (s1[:, :, 2] - s0[:, :, 2]).astype(np.float32)
    faint = np.clip(s1[:, :, 1] - s0[:, :, 1], 0, 1).astype(np.float32)           # 이번 턴에 새로 기절한 마리 (이미 기절해 있던 것은 0)

    def active(side):
        act = s1[:, side * 6:(side + 1) * 6, 0]
        idx = act.argmax(1).astype(np.int8)
        return np.where(act.max(1) > 0.5, idx, -1).astype(np.int8), s1[np.arange(N), side * 6 + act.argmax(1), 3:8]

    act_my, b_my = active(0)
    act_opp, b_opp = active(1)
    a = np.asarray(data["action_taken"])
    r_valid = has_next & (np.asarray(data["kind"]) == 0) & (cls >= 0) & (a >= 0) & (a < N_ACT)
    return {"opp_cls": cls, "opp_tera": tera, "r_valid": r_valid, "dhp": dhp, "faint": faint, "act_my": act_my, "act_opp": act_opp,
            "boost": np.concatenate([b_my, b_opp], 1).astype(np.float32)}
