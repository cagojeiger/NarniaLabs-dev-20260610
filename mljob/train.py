"""워크플로우 2번 노드: 학습.  + 실패 시나리오 생성기.

`--fault` 플래그 하나로 정상 학습과 3가지 실패 시나리오를 모두 만든다.
이렇게 코드를 하나로 두면 정상/실패 로그가 **동일한 스키마**로 떨어져
(3) Rust 진단 에이전트가 균일하게 처리할 수 있다.

  --fault none     정상 학습 (happy path; dvc 파이프라인이 사용)
  --fault nan      학습률 과대 -> gradient explosion -> loss=nan (FloatingPointError)
  --fault shape    FC 입력 차원 불일치 -> forward 에서 RuntimeError
  --fault overfit  극소량 데이터 + 과대 epochs -> train loss↓ / val loss↑ (크래시 없음)
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

# fault 별 기본 하이퍼파라미터 프리셋. (CLI 로 덮어쓸 수 있음)
FAULT_PRESETS = {
    "none": dict(lr=0.05, epochs=3, batch_size=64, train_limit=None),
    "nan": dict(lr=12.0, epochs=3, batch_size=64, train_limit=None),
    "shape": dict(lr=0.05, epochs=1, batch_size=64, train_limit=None),
    "overfit": dict(lr=0.05, epochs=40, batch_size=16, train_limit=200),
}


def _load(split: str) -> TensorDataset:
    d = torch.load(Path(PROCESSED_DIR) / f"{split}.pt")
    return TensorDataset(d["x"], d["y"])


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        out = model(xb)
        tot_loss += float(loss_fn(out, yb)) * len(xb)
        correct += int((out.argmax(1) == yb).sum())
        n += len(xb)
    return tot_loss / max(n, 1), correct / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=list(FAULT_PRESETS), default="none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    preset = FAULT_PRESETS[args.fault]
    lr = args.lr if args.lr is not None else preset["lr"]
    epochs = args.epochs if args.epochs is not None else preset["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else preset["batch_size"]
    train_limit = preset["train_limit"]
    run_dir = args.run_dir or (
        "runs/train" if args.fault == "none" else f"logs/scenario_{args.fault}"
    )

    torch.manual_seed(args.seed)

    cfg = {
        "model": "SmallCNN",
        "dataset": "FashionMNIST",
        "optimizer": "SGD",
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": args.seed,
        "train_limit": train_limit,
        "fault_injected": args.fault,
    }
    run_id = "train" if args.fault == "none" else f"scenario_{args.fault}"

    with RunRecorder(run_dir, run_id=run_id, stage="train", config=cfg) as rec:
        # ---- 데이터 ----
        train_ds = _load("train")
        if train_limit is not None:
            train_ds = TensorDataset(
                train_ds.tensors[0][:train_limit], train_ds.tensors[1][:train_limit]
            )
            rec.warn(f"train set limited to {train_limit} samples (fault={args.fault})")
        val_ds = _load("val")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)

        # ---- 모델 ----  (shape fault 는 여기서 만든 모델이 forward 에서 깨진다)
        model = build_model(fault=args.fault)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        rec.log(f"start training epochs={epochs} lr={lr} batch_size={batch_size}")

        step = 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                step += 1
                opt.zero_grad()
                # shape fault: 아래 forward 에서 RuntimeError 발생 -> except 로 기록
                try:
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                except RuntimeError as exc:
                    rec.fail(exc, phase="training_step")
                    return 1

                gnorm = _grad_norm(model)
                opt.step()

                # nan fault: loss 가 비정상이 되는 순간 명시적으로 실패 처리
                if not torch.isfinite(loss):
                    rec.log_step(StepRecord(step=step, epoch=epoch,
                                            train_loss=float(loss), grad_norm=gnorm, lr=lr))
                    exc = FloatingPointError(
                        f"training loss became non-finite (loss={float(loss)}) "
                        f"at epoch={epoch} step={step}; likely exploding gradients "
                        f"(grad_norm={gnorm:.2f}, lr={lr})"
                    )
                    rec.fail(exc, phase="training_step")
                    return 1

                if step % 10 == 0 or step == 1:
                    rec.log_step(StepRecord(step=step, epoch=epoch,
                                            train_loss=float(loss), grad_norm=gnorm, lr=lr))

            # ---- epoch 단위 검증 (overfit 신호가 여기서 드러남) ----
            tr_loss, tr_acc = _evaluate(model, train_loader, loss_fn)
            va_loss, va_acc = _evaluate(model, val_loader, loss_fn)
            rec.log_step(StepRecord(step=step, epoch=epoch, train_loss=tr_loss,
                                    val_loss=va_loss, train_acc=tr_acc, val_acc=va_acc, lr=lr))

        # ---- 정상 종료: 모델 저장 ----
        art = Path(ARTIFACT_DIR)
        art.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), art / "model.pt")
        rec.log(f"saved model -> {art / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
