"""
지도학습(모방학습): Metamon 원본 리플레이의 사람 행동을 따라 배우기.

- 리플레이는 학습 중에 DataLoader 워커가 바로 변환 (replay_dataset.py, 저장 공간 불필요)
- 정책: 사람 행동 교차 엔트로피 / 가치: PPO와 같은 의미의 목표로 MSE
- 히스토리 트랜스포머 때문에 배틀 단위로 묶어서 forward_sequences로 학습
- 검증 세트(배틀 단위 분리)의 행동 정확도가 가장 높은 체크포인트를 저장

사용 예:
  python src/training/train_bc.py --max-battles 5000 --epochs 3   # checkpoints/supervised_v2.pt 에서 이어서, 같은 파일에 저장
  python src/training/train_bc.py --min-rating 1600 --max-battles 20000 --workers 8
"""
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir != os.path.dirname(_current_dir):
    if os.path.exists(os.path.join(_current_dir, "data", "vocab.json")):
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        break
    _current_dir = os.path.dirname(_current_dir)
import argparse
import glob
import random
import time
import zlib

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.core.model import DeepPokemonBattleTransformerNet, load_compatible, zero_unseen_id_rows
from src.training.replay_dataset import OBS_KEYS, replay_to_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_logs(pattern, min_rating, max_battles, val_frac, seed):
    """(학습, 검증). 검증은 마지막 샤드에서만 뽑음 → 앞 샤드로 이미 학습한 배틀이 검증에 섞이지 않음"""
    per_file = []
    for path in sorted(p for pat in pattern.split(",") for p in glob.glob(pat)):  # 쉼표로 여러 패턴
        t = pq.read_table(path, columns=["formatid", "rating", "log"]).to_pydict()
        logs = []
        for fmt, rating, log in zip(t["formatid"], t["rating"], t["log"]):
            try:
                ok = fmt == "gen9ou" and int(rating) >= min_rating
            except (TypeError, ValueError):
                ok = False
            if ok:
                logs.append(log)
        random.Random(seed).shuffle(logs)
        per_file.append(logs)
    n_val = max(1, int(min(max_battles, sum(map(len, per_file))) * val_frac))
    val = per_file[-1][:n_val]
    per_file[-1] = per_file[-1][n_val:]
    train = [log for logs in per_file for log in logs]
    random.Random(seed).shuffle(train)
    return train[:max_battles - n_val], val


class ReplayDataset(Dataset):
    """항목 = (리플레이, 시점). 한 판에서 양쪽 플레이어 시점을 모두 사용"""
    def __init__(self, logs):
        # 로그(≈17KB)를 압축해 보관: 윈도우는 워커마다 데이터셋을 통째로 복사하므로 원문이면 워커 수 × 수 GB. 압축하면 1/6 수준
        self.logs = [zlib.compress(l.encode(), 1) for l in logs]

    def __len__(self):
        return 2 * len(self.logs)

    def __getitem__(self, i):
        return replay_to_sequence(zlib.decompress(self.logs[i // 2]).decode(), "p1" if i % 2 == 0 else "p2")


def collate(seqs):
    seqs = [s for s in seqs if s is not None]
    if not seqs:
        return None
    batch = {k: torch.from_numpy(np.concatenate([s[k] for s in seqs])) for k in
             OBS_KEYS + ["action_mask", "action", "value_target"]}
    batch["seq_index"] = torch.cat([torch.full((len(s["action"]),), j) for j, s in enumerate(seqs)])
    batch["time_index"] = torch.cat([torch.arange(len(s["action"])) for s in seqs])
    batch["n_seq"] = len(seqs)
    return batch


def run_epoch(model, ref_model, loader, optimizer, value_coef, kl_coef, train, scheduler=None, id_dropout=0.0, novel_p=0.0):
    model.eval()  # 드롭아웃 끔 (PPO 학습과 동일)
    model.embeddings.id_dropout = id_dropout if train else 0.0  # 도구/특성/기술 ID 임베딩 드롭아웃은 학습 때만
    model.embeddings.novel_p = novel_p if train else 0.0  # 실제 개체를 처음 본 개체로 바꾸는 시뮬레이션도 학습 때만
    tot = {"loss": 0.0, "ce": 0.0, "v": 0.0, "kl": 0.0, "correct": 0, "steps": 0, "batches": 0}
    t0 = time.time()
    for batch in loader:
        if batch is None:
            continue
        b = {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"],
                                          b["n_seq"], action_mask=b["action_mask"])
            ce = F.cross_entropy(out["policy_logits"], b["action"])
            v = F.mse_loss(out["value"].view(-1), b["value_target"])
            loss = ce + value_coef * v

            kl_val = 0.0
            if ref_model is not None and kl_coef > 0:
                with torch.no_grad():
                    ref_out = ref_model.forward_sequences([b[k] for k in OBS_KEYS], b["seq_index"], b["time_index"], b["n_seq"], action_mask=b["action_mask"])
                log_p = F.log_softmax(out["policy_logits"], dim=-1)
                ref_p = F.softmax(ref_out["policy_logits"], dim=-1)
                kl = F.kl_div(log_p, ref_p, reduction="batchmean")
                loss += kl_coef * kl
                kl_val = kl.item()

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        n = len(b["action"])
        tot["loss"] += loss.item() * n
        tot["ce"] += ce.item() * n
        tot["v"] += v.item() * n
        tot["kl"] += kl_val * n
        tot["correct"] += (out["policy_logits"].argmax(-1) == b["action"]).sum().item()
        tot["steps"] += n
        tot["batches"] += 1
        if train and tot["batches"] % 50 == 0:
            print(f"   {tot['batches']} 배치 | 정확도 {tot['correct'] / tot['steps']:.3f} | "
                  f"CE {tot['ce'] / tot['steps']:.3f} | V {tot['v'] / tot['steps']:.3f} | KL {tot['kl'] / tot['steps']:.4f} | "
                  f"LR {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}초")
    s = max(1, tot["steps"])
    return {"acc": tot["correct"] / s, "ce": tot["ce"] / s, "v": tot["v"] / s, "kl": tot["kl"] / s, "steps": tot["steps"]}


def main(args):
    train_logs, val_logs = load_logs(args.parquet, args.min_rating, args.max_battles, args.val_frac, args.seed)
    print(f"리플레이 {len(train_logs) + len(val_logs)}판 (레이팅 ≥ {args.min_rating}) → "
          f"학습 {len(train_logs)} / 검증 {len(val_logs)} (마지막 샤드)")
    kw = dict(batch_size=args.batch_battles, collate_fn=collate, num_workers=args.workers,
              persistent_workers=args.workers > 0, pin_memory=DEVICE.type == "cuda",
              **({"prefetch_factor": 4} if args.workers > 0 else {}))
    train_loader = DataLoader(ReplayDataset(train_logs), shuffle=True, **kw)
    val_kw = dict(kw, num_workers=min(2, args.workers), persistent_workers=False)  # 검증용 워커가 학습 내내 메모리를 잡아먹지 않게
    val_loader = DataLoader(ReplayDataset(val_logs), shuffle=False, **val_kw)
    del train_logs, val_logs  # 원문은 압축본으로 대체됨 → 메모리 반환

    arch = dict(pokemon_embed_dim=args.embed_dim, history_dim=args.history_dim,
                latent_dim=args.latent_dim, num_layers=args.layers, entity_features=not args.no_entity_features)
    model = DeepPokemonBattleTransformerNet(**arch)
    print(f"모델 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 파라미터 {arch}")
    ref_model = None
    if args.init and args.init.lower() != "none" and not os.path.exists(args.init):
        raise SystemExit(f"❌ 시작 체크포인트 {args.init} 가 없습니다. 경로를 확인하거나 --init none 으로 무작위 초기화하세요.")
    if args.init and args.init.lower() != "none":
        expanded, skipped = load_compatible(model, torch.load(args.init, map_location="cpu"))
        print(f"✅ {args.init}에서 시작 (확장 {expanded}, 버림 {skipped})")
        
        if getattr(args, "kl_coef", 0.0) > 0.0:
            ref_model = DeepPokemonBattleTransformerNet(**arch)
            load_compatible(ref_model, torch.load(args.init, map_location="cpu"))
            ref_model.to(DEVICE)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
            print(f"✅ KL 패널티용 Reference 모델 로드 완료 (KL Coef: {args.kl_coef})")

    else:
        print("⚠️ 무작위 초기화로 시작" + (" (reference 모델이 없어 KL 패널티는 꺼짐)" if args.kl_coef > 0 else ""))
    if args.zero_unseen_rows:
        print(f"✅ 학습 팀 풀에 없던 도구/특성/기술의 ID 임베딩 행 {zero_unseen_id_rows(model)}개를 0으로 (특징으로만 표현)")
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 코사인 감쇠: 전체 학습 배치 수에 걸쳐 lr → lr × lr_min_ratio 로 부드럽게 감소 (배치마다 갱신)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(train_loader)), eta_min=args.lr * args.lr_min_ratio)

    before = run_epoch(model, ref_model, val_loader, None, args.value_coef, getattr(args, "kl_coef", 0.0), train=False)
    print(f"시작 전 검증: 정확도 {before['acc']:.3f} | CE {before['ce']:.3f} | V {before['v']:.3f} | KL {before['kl']:.4f}")
    best = before["acc"] if args.init and args.init.lower() != "none" else -1.0  # 시작보다 나아질 때만 덮어씀
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr = run_epoch(model, ref_model, train_loader, optimizer, args.value_coef, getattr(args, "kl_coef", 0.0), train=True,
                       scheduler=scheduler, id_dropout=args.id_dropout, novel_p=args.novel_p)
        va = run_epoch(model, ref_model, val_loader, None, args.value_coef, getattr(args, "kl_coef", 0.0), train=False)
        print(f"학습 정확도 {tr['acc']:.3f} (CE {tr['ce']:.3f}, V {tr['v']:.3f}, KL {tr['kl']:.4f}, {tr['steps']}턴) | "
              f"검증 정확도 {va['acc']:.3f} (CE {va['ce']:.3f}, V {va['v']:.3f}, KL {va['kl']:.4f})")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if va["acc"] > best or (args.save_last and epoch == args.epochs):
            best = max(best, va["acc"])
            torch.save(model.state_dict(), args.out)
            print(f"💾 검증 정확도 최고 → {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/metamon/raw/*.parquet", help="glob 패턴, 쉼표로 여러 개")
    ap.add_argument("--min-rating", type=int, default=1500)
    ap.add_argument("--max-battles", type=int, default=5000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-battles", type=int, default=32, help="배치당 배틀 시점 수 (한 시점 ≈ 20-30턴)")
    ap.add_argument("--lr", type=float, default=1e-4, help="시작 학습률 (코사인 감쇠로 끝까지 줄어듦)")
    ap.add_argument("--lr-min-ratio", type=float, default=0.05, help="마지막 학습률 = lr × 이 값")
    ap.add_argument("--value-coef", type=float, default=0.25)
    ap.add_argument("--kl-coef", type=float, default=0.3, help="기존 가중치 유지(망각 방지)를 위한 KL 패널티 계수")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 8), help="리플레이 변환 워커 수 (기본: 논리 코어 - 8, 이 PC는 12)")
    ap.add_argument("--init", default="checkpoints/supervised_v2.pt",
                    help="시작 체크포인트 (KL reference로도 사용). 'none'이면 무작위 초기화")
    ap.add_argument("--out", default="checkpoints/supervised_v2.pt", help="저장 위치 (기본: 시작 파일에 덮어씀)")
    ap.add_argument("--embed-dim", type=int, default=128, help="포켓몬 벡터 차원 (기존 체크포인트는 128)")
    ap.add_argument("--history-dim", type=int, default=256, help="히스토리 트랜스포머 차원 (기존 256)")
    ap.add_argument("--latent-dim", type=int, default=256, help="융합 후 잠재 벡터 차원 (기존 256)")
    ap.add_argument("--layers", type=int, default=2, help="팀/히스토리 트랜스포머 레이어 수 (기존 2)")
    ap.add_argument("--id-dropout", type=float, default=0.0, help="학습 중 도구/특성/기술 ID 임베딩을 이 확률로 0으로 (특징 표만으로도 두게)")
    ap.add_argument("--novel-p", type=float, default=0.05, help="학습 중 이 확률로 도구/특성/기술을 '처음 본 개체'(<unk>, 특징 0)로 바꿈")
    ap.add_argument("--zero-unseen-rows", action="store_true", help="시작 체크포인트의 미등장 개체 ID 임베딩 행을 0으로")
    ap.add_argument("--no-entity-features", action="store_true", help="도구/특성/기술 특징 표 없이 학습 (대조군)")
    ap.add_argument("--save-last", action="store_true", help="검증 정확도가 안 올라도 마지막 에폭 가중치를 저장")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
