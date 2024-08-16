"""Module of pytorch models."""
import torch
import torchvision
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        n_features = self.model.fc.in_features
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(n_features, num_classes)

    def forward(self, x) -> torch.Tensor:
        """x.shape=(batch_size, 3, 224, 224)"""
        return self.model(x)


def get_model(name: str, **kwargs) -> nn.Module:
    models: dict[str, nn.Module] = {
        "conv_net": ConvNet,
        "pretrained_res_net": PretrainedResNet,
    }
    try:
        return models[name](**kwargs)
    except KeyError as e:
        raise ValueError(
            f"{name} does not exist. Did you mean: {list(models.keys())}"
        ) from e
