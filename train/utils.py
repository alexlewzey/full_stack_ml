import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Union

import lightning as L  # noqa: N812
import metadata
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch.tuner.tuning import Tuner
from mlflow import MlflowClient
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy


class CatsVsDogsDataset(Dataset):
    def __init__(self, root_dir: Union[Path, str], transform: Callable):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.paths = list(self.root_dir.iterdir())

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
    def get_label(path: Path) -> str:
        return path.stem.split(".")[0]


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

    def prepare_data(self) -> None:
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

    def setup(self, stage: str) -> None:
        if stage == "fit":
            ds = CatsVsDogsDataset(root_dir=self.train_dir, transform=self.transform)
            self.train_ds, self.valid_ds = split_dataset(ds, pct_train=self.pct_train)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class ImageClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        out = self.model.forward(x)
        return out

    def _step(self, batch, batch_idx, set_name: str):
        x, y = batch
        yprob = self.model.forward(x)
        loss = F.cross_entropy(yprob, y)
        self.log(f"{set_name}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        yhat = yprob.argmax(-1)
        acc = self.accuracy(yhat, y)
        self.log(f"{set_name}_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ModelTuner:
    def __init__(
        self,
        trainer: L.Trainer,
        model: L.LightningModule,
        data_module: L.LightningDataModule,
    ):
        self.tuner = Tuner(trainer)
        self.model = model
        self.data_module = data_module

    def find_batch_size(self):
        self.tuner.scale_batch_size(self.model, datamodule=self.data_module)

    def find_learning_rate(self):
        lr_finder = self.tuner.lr_find(self.model, datamodule=self.data_module)
        fig = lr_finder.plot(suggest=True)
        self.lr = lr_finder.suggestion()
        fig.show()


def split_dataset(ds: Dataset, pct_train: float) -> tuple[Dataset, Dataset]:
    train_size = int(pct_train * len(ds))
    val_size = len(ds) - train_size
    train_ds, valid_ds = random_split(ds, [train_size, val_size])
    return train_ds, valid_ds


class MLFlowExperiment:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.df = self.get_experiment_df(experiment_name)

    def get_experiment_df(self, experiment_name: str) -> pd.DataFrame:
        experiments = pd.DataFrame(
            map(dict, self.client.search_experiments())
        ).set_index("name")
        self.experiment_id = experiments.loc[experiment_name, "experiment_id"]
        runs = self.client.search_runs(experiment_ids=self.experiment_id)
        data = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "status": run.info.status,
            }
            run_data.update(run.data.metrics)
            run_data.update(run.data.params)
            data.append(run_data)
        return pd.DataFrame(data)

    def get_best_run_id(self) -> str:
        run = self.df[self.df.status == "FINISHED"].sort_values("valid_loss")
        run_id = run.run_id[0]
        return run_id

    def artifact_path(self, run_id, path: str) -> Path:
        return Path.cwd().parent / f"mlruns/{self.experiment_id}/{run_id}/{path}"
