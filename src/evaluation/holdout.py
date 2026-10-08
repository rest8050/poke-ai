"""홀드아웃 통합 파일(data/holdout/holdout_all.npz) 읽기/쓰기. 학생 상태(DAgger 라운드별 dg2~dg5…)와 교사 상태(fp)를 한 파일에 모으고,
시퀀스마다 출처 번호(src)와 이름(src_names)을 붙임. 평가 스크립트들이 파일 목록을 따로 들고 있지 않도록 이 모듈만 거침.
- 기준 비교 묶음(LEGACY) = dg2+dg3+dg4+fp: 지금까지 모든 수치(합산 73,517결정)의 기준이라 기본값으로 고정. 새 라운드는 명시해서 부름.
- 파일 만들기/추가: python tools/merge_holdouts.py 이름=경로 [이름=경로 ...]  (같은 이름이면 교체)"""
import numpy as np

PATH = "data/holdout/holdout_all.npz"
OBS = ["my_team_cat", "my_team_num", "my_move_num", "opp_team_cat", "opp_team_num", "opp_move_num", "field_vec"]
KEYS = OBS + ["action_mask", "target"]
LEGACY = ("dg2", "dg3", "dg4", "fp")
MOVE_DIM = 46   # 기술 수치 칸 수: 옛 파일(44)은 0으로 채워 통일 (모델 쪽에서 필요하면 잘라 씀)


def pad_moves(d):
    for k in ("my_move_num", "opp_move_num"):
        if d[k].shape[-1] < MOVE_DIM:
            d[k] = np.concatenate([d[k], np.zeros(d[k].shape[:-1] + (MOVE_DIM - d[k].shape[-1],), d[k].dtype)], -1)
    return d


def load_holdout(groups=LEGACY, path=PATH):
    """시퀀스 리스트 [(그룹 이름, 관측 dict(7키), 행동 마스크, 교사 분포)] — 파일 안 순서대로"""
    z = np.load(path)
    names = [str(n) for n in z["src_names"]]
    want = {names.index(g) for g in groups}
    d = {k: z[k] for k in KEYS + ["lens", "src"]}
    off = np.concatenate([[0], np.cumsum(d["lens"])])
    out = []
    for i in range(len(d["lens"])):
        if int(d["src"][i]) in want:
            s, e = off[i], off[i + 1]
            out.append((names[int(d["src"][i])], {k: d[k][s:e] for k in OBS}, d["action_mask"][s:e], d["target"][s:e]))
    return out


def write_group(name, path_in, path_out=PATH):
    """path_in(npz, 기존 홀드아웃 형식)의 배틀을 name 그룹으로 통합 파일에 넣음 (같은 이름이 있으면 교체)"""
    import os
    z = np.load(path_in)
    new = pad_moves({k: z[k] for k in KEYS + ["lens"]})
    if os.path.exists(path_out):
        old = np.load(path_out)
        names = [str(n) for n in old["src_names"]]
        old_d = {k: old[k] for k in KEYS + ["lens", "src"]}
        if name in names:      # 교체: 그 그룹의 시퀀스를 빼고 나머지를 남김
            gi = names.index(name)
            keep = old_d["src"] != gi
            dec_keep = np.repeat(keep, old_d["lens"])
            for k in KEYS:
                old_d[k] = old_d[k][dec_keep]
            old_d["lens"], old_d["src"] = old_d["lens"][keep], old_d["src"][keep]     # 이름 자리는 그대로 (번호가 바뀌지 않게)
        else:
            names.append(name)
        gi = names.index(name)
        old_d = pad_moves(old_d)
        merged = {k: np.concatenate([old_d[k], new[k]]) for k in KEYS + ["lens"]}
        merged["src"] = np.concatenate([old_d["src"], np.full(len(new["lens"]), gi, np.int8)])
    else:
        names = [name]
        merged = {**new, "src": np.zeros(len(new["lens"]), np.int8)}
    np.savez(path_out, **merged, src_names=np.array(names))
    return names, len(new["lens"]), int(new["lens"].sum())
