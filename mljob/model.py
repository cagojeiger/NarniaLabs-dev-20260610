"""작은 CNN 분류기 (FashionMNIST, 1x28x28 -> 10 classes).

CPU 에서 수 초 내 학습되도록 의도적으로 작게 만들었다.
`fault="shape"` 일 때는 FC 입력 차원을 일부러 틀리게 만들어
forward 시점에 RuntimeError(matmul shape mismatch)가 재현되도록 한다.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, *, broken_fc: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        # 정상: 32 * 7 * 7 = 1568.
        # broken_fc=True 이면 일부러 틀린 입력 차원을 줘서 matmul 이 깨지게 한다.
        flat = 32 * 7 * 7
        in_features = (flat // 2) if broken_fc else flat
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(*, fault: str = "none") -> SmallCNN:
    """fault 플래그에 따라 정상/결함 모델을 만든다."""
    return SmallCNN(broken_fc=(fault == "shape"))
