from datetime import datetime, timedelta

import lightning as L  # noqa: N812
import pytest

from src.train.train import create_and_train_model
from src.train.utils.loaders import CatVsDogsDataModule
from src.train.utils.preprocessing import transform
from src.utils.core import sample_dir

L.seed_everything(42, workers=True)


def data_module():
    return CatVsDogsDataModule(data_dir=sample_dir, transform=transform, num_workers=9)


mock_data_module = pytest.fixture(data_module)


def test_trainer_fast_dev_run(mock_data_module):
    create_and_train_model(
        data_module=mock_data_module, lr=1e-3, max_epochs=1, fast_dev_run=True
    )


def test_trainer_overfit_batches(mock_data_module):
    start = datetime.now()
    trainer = create_and_train_model(
        data_module=mock_data_module,
        lr=1e-3,
        max_epochs=15,
        overfit_batches=1,
        deterministic=True,
    )
    final_accuracy = trainer.callback_metrics["train_acc"].item()
    assert final_accuracy > 0.99
    assert (datetime.now() - start) < timedelta(seconds=5)
