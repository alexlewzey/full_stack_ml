from datetime import datetime, timedelta

import lightning as L  # noqa: N812
import pytest
import torch.nn as nn
from torchvision import transforms

from src.train.train import create_and_train_model
from src.train.utils.loaders import CatVsDogsDataModule
from src.train.utils.models import get_model
from src.utils.core import sample_dir

L.seed_everything(42, workers=True)


def model() -> nn.Module:
    return get_model("pretrained_res_net")


def data_module() -> CatVsDogsDataModule:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return CatVsDogsDataModule(data_dir=sample_dir, transform=transform, num_workers=0)


mock_model = pytest.fixture(model)
mock_data_module = pytest.fixture(data_module)


def test_trainer_fast_dev_run(
    mock_model: nn.Module, mock_data_module: CatVsDogsDataModule
):
    create_and_train_model(
        model=mock_model,
        data_module=mock_data_module,
        lr=1e-3,
        max_epochs=1,
        fast_dev_run=True,
    )


def test_trainer_overfit_batches(
    mock_model: nn.Module, mock_data_module: CatVsDogsDataModule
):
    start = datetime.now()
    trainer = create_and_train_model(
        model=mock_model,
        data_module=mock_data_module,
        lr=1e-3,
        max_epochs=15,
        overfit_batches=1,
        deterministic=True,
    )
    final_accuracy = trainer.callback_metrics["train_acc"].item()
    assert final_accuracy > 0.99
    duration = datetime.now() - start
    assert duration < timedelta(seconds=5)
