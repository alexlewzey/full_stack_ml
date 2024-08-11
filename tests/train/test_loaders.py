from pathlib import Path

import pytest
from torch import Tensor
from torchvision import transforms

from src.train.utils.loaders import CatsVsDogsDataset, split_dataset
from src.utils.core import sample_dir


@pytest.fixture
def mock_dataset() -> CatsVsDogsDataset:
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dir = sample_dir / "train"
    return CatsVsDogsDataset(image_dir=train_dir, transform=transform)


class TestCatsVsDogsDataset:
    def test_get_label(self, mock_dataset):
        path_dog = Path("/workspaces/example_cdk/data/train/dog.8011.jpg")
        assert mock_dataset.get_label(path_dog) == "dog"

        path_cat: str = "/workspaces/example_cdk/data/valid/cat.1123.png"
        assert mock_dataset.get_label(path_cat) == "cat"

    def test_len(self, mock_dataset):
        assert len(mock_dataset) == 10

    def test_getitem(self, mock_dataset):
        x, y = mock_dataset[0]
        assert isinstance(x, Tensor)
        assert isinstance(y, int)
        assert list(x.shape) == [3, 32, 32]


def test_split_dataset(mock_dataset):
    train_ds, valid_ds = split_dataset(ds=mock_dataset, pct_train=0.8)
    assert len(train_ds) == 8
    assert len(valid_ds) == 2
