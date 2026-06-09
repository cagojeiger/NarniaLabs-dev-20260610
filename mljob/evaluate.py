"""워크플로우 3번 노드: 평가.

학습된 모델을 test 셋으로 평가하고 메트릭을 남긴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mljob import ARTIFACT_DIR, PROCESSED_DIR
from mljob.model import build_model
from mljob.telemetry import RunRecorder, StepRecord


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=".runs/evaluate")
    args = ap.parse_args()

    cfg = {"dataset": "FashionMNIST", "split": "test"}
    with RunRecorder(args.run_dir, run_id="evaluate", stage="evaluate", config=cfg) as rec:
        model = build_model(fault="none")
        state = torch.load(Path(ARTIFACT_DIR) / "model.pt")
        model.load_state_dict(state)
        model.eval()

        d = torch.load(Path(PROCESSED_DIR) / "test.pt")
        loader = DataLoader(TensorDataset(d["x"], d["y"]), batch_size=256)
        loss_fn = nn.CrossEntropyLoss()

        tot, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb)
                tot += float(loss_fn(out, yb)) * len(xb)
                correct += int((out.argmax(1) == yb).sum())
                n += len(xb)
        test_loss, test_acc = tot / n, correct / n
        rec.log_step(StepRecord(step=0, epoch=0, val_loss=test_loss, val_acc=test_acc))
        rec.log(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
