from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        size = math.prod(input_shape)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SimpleCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(inputs), 1))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == channels
            else nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return torch.relu(output + self.shortcut(inputs))


class ResNet18(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        self.channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def _layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.channels, channels, stride)]
        self.channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs)))))
        return self.head(torch.flatten(self.pool(output), 1))


def create_model(
    name: str, input_shape: tuple[int, ...], num_classes: int, **options
) -> nn.Module:
    key = name.lower().replace("-", "").replace("_", "")
    if key == "mlp":
        return MLP(input_shape, num_classes, hidden=int(options.get("hidden", 128)))
    if key in {"cnn", "simplecnn"}:
        if len(input_shape) != 3:
            raise ValueError("CNN models require image-shaped inputs")
        return SimpleCNN(input_shape, num_classes)
    if key == "resnet18":
        if len(input_shape) != 3:
            raise ValueError("ResNet18 requires image-shaped inputs")
        return ResNet18(input_shape, num_classes)
    raise ValueError("Supported models: mlp, simple_cnn, resnet18")

