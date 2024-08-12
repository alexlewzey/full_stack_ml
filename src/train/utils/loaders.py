"""Module of pytorch/lightning datasets/data modules."""
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Union

import lightning as L  # noqa: N812
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from . import metadata


class CatsVsDogsDataset(Dataset):
    def __init__(self, image_dir: Union[Path, str], transform: Callable):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.paths = list(self.image_dir.iterdir())

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        path = self.paths[idx]
        label = self.get_label(path)
        y = metadata.ENCODING[label]

        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y

    @staticmethod
    def get_label(path: Path | str) -> str:
        return Path(path).stem.split(".")[0]


class CatVsDogsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        transform: Callable,
        pct_train: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.train_dir = data_dir / "train"
        self.pct_train = pct_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.persistent_workers = True if self.num_workers > 0 else False

    def prepare_data(self) -> None:
        if not self.data_dir.exists() or not self.train_dir.exists():
            self.data_dir.mkdir(exist_ok=True, parents=True)
            kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_path.exists():
                raise Exception(f"{kaggle_path} does not exist, please add!")
            kaggle_path.chmod(0o600)

            command = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "dogs-vs-cats",
                "-p",
                self.data_dir.as_posix(),
            ]
            subprocess.run(command, check=True)  # noqa: S603
            with zipfile.ZipFile(self.data_dir / "dogs-vs-cats.zip", "r") as f:
                f.extractall(self.data_dir)
            with zipfile.ZipFile(self.data_dir / "train.zip", "r") as f:
                f.extractall(self.data_dir)
            assert len(list((self.train_dir).iterdir())) == 25000
            print("Dataset downloaded and extracted successfully.")
        else:
            print("Data already exists. Skipping download.")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            ds = CatsVsDogsDataset(image_dir=self.train_dir, transform=self.transform)
            self.train_ds, self.valid_ds = split_dataset(ds, pct_train=self.pct_train)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )


def split_dataset(ds: Dataset, pct_train: float) -> tuple[Dataset, Dataset]:
    train_size = int(pct_train * len(ds))
    val_size = len(ds) - train_size
    train_ds, valid_ds = random_split(ds, [train_size, val_size])
    return train_ds, valid_ds
