"""입력 기여 분리: 같은 모델에서 새로 추가한 입력 블록만 바꿔서 평가하고, 얼마나 떨어지는지로 그 입력을 실제로 쓴 정도를 잰다 (재학습 없음).
변형 (모델 입력만 바꾸고 정답/마스크는 그대로):
  base                 그대로
  monext_zero/shuffle  포켓몬별 추가 수치(MON_EXT 16열: 등장 후 턴, 수면/맹독 나이, 마지막 행동, 쓴 기술 등)를 0으로 / 결정끼리 섞어서
  fieldext_zero/shuffle 필드 확장 블록(휘발성 효과 등, EXT_START 이후)을 0으로 / 섞어서
  itemability_hidden   상대의 도구/특성 입력을 전부 '숨김'으로 (공개된 것과 추론으로 채운 것을 함께 가림)
섞기는 값 분포는 유지하고 그 결정과의 연관만 끊어서, 0으로 만들 때 생기는 분포 이탈을 덜어 줌. 학습 때와 분포가 달라지므로 방향을 보는 용도.
옛 형식 npz(포켓몬 수치 27열, 필드 72열)는 모자란 칸을 0으로 채워 그대로 넣음 (--variants base로 옛 입력 전체를 쓴 경우의 평가).
q_mean/q_vis가 든 npz면 regret도 같이 출력. 홀드아웃 통합 파일은 그룹별로 출력.
사용: python tools/ablate_inputs.py --model v8_full --sets data/holdout/holdout_all.npz,data/dagger/dg5.npz [--variants base,monext_zero]"""
import argparse
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src/training")
import train_fp_distill as T
from src.core.model import model_from_ckpt
from src.core.tensor_encoder import BASE_NUM, EXT_START, FIELD_DIM, NUM_DIM, SLOT_DIM
from src.training.hidden_labels import vocab_sizes

MON_EXT_START = BASE_NUM + SLOT_DIM
VARIANTS = ["base", "monext_zero", "monext_shuffle", "fieldext_zero", "fieldext_shuffle", "itemability_hidden"]


def load_set(path):
    """npz -> (학습 코드의 data 사전, 오프셋, {그룹 이름: 배틀 번호 목록}). 옛 형식은 새 폭으로 0 패딩"""
    z = np.load(path)
    d = {k: z[k] for k in z.files if k not in ("src", "src_names")}
    n, N = len(d["lens"]), len(d["target"])
    d.setdefault("outcome", np.zeros(n, np.float32))
    d.setdefault("action_taken", -np.ones(N, np.int64))
    for k in ("my_team_num", "opp_team_num"):
        d[k] = np.concatenate([d[k], np.zeros(d[k].shape[:-1] + (NUM_DIM - d[k].shape[-1],), d[k].dtype)], -1)
    d["field_vec"] = np.concatenate([d["field_vec"], np.zeros((N, FIELD_DIM - d["field_vec"].shape[1]), d["field_vec"].dtype)], 1)
    src = z["src"] if "src" in z.files else np.zeros(n, int)
    names = [str(g) for g in z["src_names"]] if "src_names" in z.files else [path]
    groups = {names[g]: np.where(src == g)[0].tolist() for g in range(len(names)) if (src == g).any()}
    return d, np.concatenate([[0], np.cumsum(d["lens"])]), groups


def altered(d, name, perm):
    v = dict(d)
    if name == "base":
        return v
    kind, mode = name.rsplit("_", 1)

    def hit(a, sl):
        a = a.copy()
        a[..., sl] = 0 if mode == "zero" else a[perm][..., sl]
        return a
    if kind == "monext":
        for k in ("my_team_num", "opp_team_num"):
            v[k] = hit(d[k], slice(MON_EXT_START, None))
    elif kind == "fieldext":
        v["field_vec"] = hit(d["field_vec"], slice(EXT_START, None))
    elif kind == "itemability":
        sizes = vocab_sizes()
        c = d["opp_team_cat"].copy()
        c[..., 0], c[..., 1] = sizes["item"] + 1, sizes["ability"] + 1      # 숨김 id = <unk> + 1 (hidden_labels.py와 같음)
        v["opp_team_cat"] = c
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    model = model_from_ckpt(f"checkpoints/supervised_v2_fp_{args.model}.pt").to(T.DEVICE).eval()
    run_args = SimpleNamespace(accum=1)
    for path in args.sets.split(","):
        d, off, groups = load_set(path)
        perm = np.random.default_rng(0).permutation(len(d["target"]))
        base = {}
        for name in args.variants.split(","):
            v = altered(d, name, perm)
            for g, idx in groups.items():
                r = T.run(model, v, off, idx, run_args, train=False)
                label = path if len(groups) == 1 else f"{path}#{g}"
                reg = f" | regret {r['reg']:.4f}" if r["reg"] is not None else ""
                if name == "base":
                    base[g] = r
                    delta = ""
                else:
                    b = base.get(g)
                    delta = f"  (일치 {r['agree'] - b['agree']:+.4f}, CE {r['ce'] - b['ce']:+.4f}" + (f", regret {r['reg'] - b['reg']:+.4f}" if r["reg"] is not None and b["reg"] is not None else "") + ")" if b else ""
                print(f"{args.model} {label} n={r['n']} {name:18s} 일치 {r['agree']:.4f} | CE {r['ce']:.4f}{reg}{delta}", flush=True)


if __name__ == "__main__":
    main()
