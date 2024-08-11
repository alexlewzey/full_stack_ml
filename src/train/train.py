import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightning as L  # noqa: N812
import mlflow
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.core import data_dir, root_dir, tmp_dir

from .utils import preprocessing
from .utils.loaders import CatVsDogsDataModule
from .utils.models import ConvNet, ImageClassifier

warnings.filterwarnings("ignore", message="Checkpoint logging is skipped")


@dataclass
class Config:
    experiment_name: str = "cats_vs_dogs"
    batch_size: int = 64
    pct_train: float = 0.8
    lr: float = 1e-5
    max_epochs: int = 50


def cli() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=Config.experiment_name)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--pct_train", type=float, default=Config.pct_train)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max_epochs", type=int, default=Config.max_epochs)
    args = vars(parser.parse_args())

    passed_args: bool = any(
        value != parser.get_default(arg) for arg, value in args.items() if arg != "file"
    )
    if args["file"] and passed_args:
        parser.error("--file cannot be used with any other arguments")
    if args["file"]:
        with Path(args["file"]).open() as f:
            content = json.load(f)
        config = Config(**content)
    else:
        del args["file"]
        config = Config(**args)
    return config


def create_and_train_model(
    data_module: L.LightningDataModule, lr: float, **kwargs
) -> L.Trainer:
    model = ConvNet()
    train_module = ImageClassifier(model=model, lr=lr)
    trainer = L.Trainer(default_root_dir=tmp_dir, **kwargs)
    trainer.fit(train_module, datamodule=data_module)
    return trainer


def train_and_save_model(config: Config) -> None:
    torchscript_path: Path = tmp_dir / "model.torchscript"
    mlflow_dir = (root_dir / "mlruns").as_posix()
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment(experiment_name=config.experiment_name)
    mlflow.pytorch.autolog()

    early_stopping = EarlyStopping("valid_loss")
    model_checkpoint = ModelCheckpoint(
        monitor="valid_loss", filename="cats-vs-dogs-{epoch:03d}-{valid_loss:.3f}"
    )
    callbacks = [
        early_stopping,
        model_checkpoint,
    ]
    with mlflow.start_run():
        data_module = CatVsDogsDataModule(
            data_dir=data_dir,
            batch_size=config.batch_size,
            pct_train=config.pct_train,
            transform=preprocessing.transform,
        )
        create_and_train_model(
            data_module=data_module,
            lr=config.lr,
            max_epochs=config.max_epochs,
            callbacks=callbacks,
        )

        best_model = ImageClassifier.load_from_checkpoint(
            model_checkpoint.best_model_path
        )
        mlflow.pytorch.log_model(best_model, "model")
        mlflow.log_params(data_module.hparams)
        mlflow.log_params(best_model.hparams)

        scripted_model = torch.jit.script(best_model.model)
        scripted_model.save(torchscript_path)
        mlflow.log_artifact(torchscript_path)


def main() -> None:
    config = cli()
    train_and_save_model(config)


if __name__ == "__main__":
    main()
