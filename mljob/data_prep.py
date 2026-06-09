"""워크플로우 1번 노드: 데이터 준비.

FashionMNIST 를 내려받아(최초 1회) 결정적으로 서브샘플링한 뒤
train/val/test 텐서를 data/processed/ 에 저장한다.
이후 train/evaluate 노드는 네트워크 없이 이 파일만 읽는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mljob import PROCESSED_DIR
from mljob.telemetry import RunRecorder

RAW_DIR = ".data/raw"

# CPU 에서 빠르게 돌도록 작은 서브셋. (재현성 위해 seed 고정)
N_TRAIN = 4000
N_VAL = 1000
N_TEST = 1000


def _load_fashion_mnist(seed: int):
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST 통계
        ]
    )
    train_full = datasets.FashionMNIST(RAW_DIR, train=True, download=True, transform=tfm)
    test_full = datasets.FashionMNIST(RAW_DIR, train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(seed)

    def take(ds, n):
        idx = torch.randperm(len(ds), generator=g)[:n]
        xs = torch.stack([ds[i][0] for i in idx])
        ys = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
        return xs, ys

    x_tr, y_tr = take(train_full, N_TRAIN + N_VAL)
    x_train, y_train = x_tr[:N_TRAIN], y_tr[:N_TRAIN]
    x_val, y_val = x_tr[N_TRAIN:], y_tr[N_TRAIN:]
    x_test, y_test = take(test_full, N_TEST)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-dir", default=".runs/data_prep")
    args = ap.parse_args()

    out = Path(PROCESSED_DIR)
    out.mkdir(parents=True, exist_ok=True)

    cfg = {"dataset": "FashionMNIST", "seed": args.seed,
           "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST}
    with RunRecorder(args.run_dir, run_id="data_prep", stage="data_prep", config=cfg) as rec:
        rec.log("downloading / loading FashionMNIST ...")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = _load_fashion_mnist(args.seed)
        torch.save({"x": x_train, "y": y_train}, out / "train.pt")
        torch.save({"x": x_val, "y": y_val}, out / "val.pt")
        torch.save({"x": x_test, "y": y_test}, out / "test.pt")
        rec.log(f"saved train={tuple(x_train.shape)} val={tuple(x_val.shape)} "
                f"test={tuple(x_test.shape)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
