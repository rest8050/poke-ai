"""홀드아웃 결정마다 세계(Foul Play가 표본 낸 상대 세트)별 행동 가치를 저장: npz의 q_mean은 세계를 평균 낸 값이라, 어떤 행동의 이득이 상대 세트에 따라 갈리는지 볼 수 없음.
build_fp_dataset.build()를 그대로 돌리되 action_values를 감싸 세계별 (가중치 p, 행동별 점수/방문 수)를 모으고, 저장된 npz의 q_mean 행과 같은 순서로 맞춰 data/dagger/<이름>_worlds.npz로 저장.
출력: wp [N,W] 세계 가중치 (패딩 0), wq [N,W,22] 세계별 행동 점수 (방문 없으면 NaN), wv [N,W,22] 세계별 방문 수
사용: python tools/world_q.py --decisions "data/dagger/raw/d7/lab_rec_*_h*.jsonl" --npz data/dagger/d7_ho.npz --out data/dagger/d7_ho_worlds.npz"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
import src.training.build_fp_dataset as B

REC = []


def action_values_rec(state, worlds):
    qm, qv = ORIG(state, worlds)
    moves, team = [B._norm(m) for m in state["moves"]], [B._norm(s) for s in state["team_species"]]
    ws = []
    for w in worlds or []:
        s_, v_ = np.zeros(22), np.zeros(22)
        for name, v, s in w.get("o", []):
            i = B._choice_index(name, moves, team)
            if i is not None and state["mask"][i]:
                s_[i] += s; v_[i] += v
        ws.append((float(w.get("p", 1.0)), s_, v_))
    REC.append((qm, ws))
    return qm, qv


ORIG = B.action_values
B.action_values = action_values_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    B.build(B.load_jsonl(args.decisions), B.WORKERS)
    qref = np.load(args.npz)["q_mean"]
    # 버려진 배틀의 결정은 npz에 없으므로, npz 행 순서대로 같은 q_mean을 가진 기록을 앞에서부터 찾아 맞춤
    rows, j = [], 0
    for i in range(len(qref)):
        while j < len(REC) and not np.array_equal(REC[j][0], qref[i], equal_nan=True):
            j += 1
        if j == len(REC):
            raise SystemExit(f"❌ npz {i}번 행에 맞는 기록 없음 (기록 {len(REC)}개)")
        rows.append(REC[j][1]); j += 1
    W = max(len(r) for r in rows)
    wp, wq, wv = np.zeros((len(rows), W)), np.full((len(rows), W, 22), np.nan, np.float32), np.zeros((len(rows), W, 22), np.float32)
    for i, r in enumerate(rows):
        for k, (p, s_, v_) in enumerate(r):
            wp[i, k] = p; wv[i, k] = v_
            wq[i, k] = np.where(v_ > 0, s_ / np.maximum(v_, 1e-9), np.nan)
    np.savez_compressed(args.out, wp=wp, wq=wq, wv=wv)
    print(f"✅ {len(rows)}개 결정, 세계 최대 {W}개 -> {args.out}")


if __name__ == "__main__":
    main()
