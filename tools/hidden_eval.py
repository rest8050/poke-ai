"""숨김정보(상대의 숨은 기술/도구/특성 예측) 평가: 모델별로 손실, 기술 상위4 적중, 도구/특성 1위 적중, 그리고 정답 라벨 수를 같은 npz에서 비교.
라벨은 입력의 상대 도구/특성/기술 칸에서 만들어지므로(hidden_labels.py) 인코더가 바뀌면 같은 결정이라도 라벨 수와 난이도가 달라질 수 있음 — 라벨 수(n)를 같이 봐야 손실 비교가 의미 있음.
사용: python tools/hidden_eval.py --models v6_full,v8_full --sets data/holdout/holdout_all.npz,data/holdout/holdout_all_old.npz"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
sys.path.insert(0, "src/training")
import train_fp_distill as T
from ablate_inputs import load_set
from src.core.model import model_from_ckpt
from src.training.hidden_labels import hidden_loss
from src.training.replay_dataset import OBS_KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--sets", required=True)
    args = ap.parse_args()
    for name in args.models.split(","):
        model = model_from_ckpt(f"checkpoints/supervised_v2_fp_{name}.pt").to(T.DEVICE).eval()
        for path in args.sets.split(","):
            d, off, groups = load_set(path)
            for g, idx in groups.items():
                tot = dict(loss=0.0, n=0, top4=0.0, mn=0, item_acc=0.0, item_n=0, ability_acc=0.0, ability_n=0)
                with torch.no_grad():
                    for b in T.make_batches(d, off, idx, T.BATCH_BATTLES, False):
                        b = {k: v.to(T.DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}
                        out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
                        loss, st = hidden_loss({"move": out["hid_move"], "item": out["hid_item"], "ability": out["hid_ability"]}, b, T.UNK)
                        n = len(b["target"])
                        tot["loss"] += loss.item() * n; tot["n"] += n
                        tot["top4"] += st.get("move_top4", 0.0) * st["move_n"]; tot["mn"] += st["move_n"]
                        for k in ("item", "ability"):
                            tot[f"{k}_acc"] += st.get(f"{k}_acc", 0.0) * st[f"{k}_n"]; tot[f"{k}_n"] += st[f"{k}_n"]
                label = path if len(groups) == 1 else f"{path}#{g}"
                print(f"{name} {label} | 손실 {tot['loss'] / max(1, tot['n']):.3f} | 기술 라벨 {tot['mn']} 상위4 {tot['top4'] / max(1, tot['mn']):.3f} | "
                      f"도구 라벨 {tot['item_n']} 1위 {tot['item_acc'] / max(1, tot['item_n']):.3f} | 특성 라벨 {tot['ability_n']} 1위 {tot['ability_acc'] / max(1, tot['ability_n']):.3f}", flush=True)


if __name__ == "__main__":
    main()
