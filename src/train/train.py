"""CLI script that trains a model and saves the metrics and artifacts to mlflow."""
import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import dagshub
import lightning as L  # noqa: N812
import mlflow
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.core import data_dir, experiment_name, tmp_dir, username

from .utils.loaders import CatVsDogsDataModule
from .utils.models import get_model
from .utils.preprocessing import Transforms, TransformsConfig
from .utils.training import ImageClassifier

logging.getLogger("mlflow").setLevel(logging.ERROR)


default_transforms_config: TransformsConfig = {
    "Resize": {"size": 224},
    "CenterCrop": {"size": 224},
    "ToTensor": {},
    "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}


@dataclass
class Config:
    model: str = "pretrained_res_net"
    batch_size: int = 32
    pct_train: float = 0.8
    lr: float = 1e-4
    max_epochs: int = 50
    transforms_config: TransformsConfig = field(
        default_factory=lambda: default_transforms_config
    )


def cli() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--model", type=str, default=Config.model)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--pct_train", type=float, default=Config.pct_train)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max_epochs", type=int, default=Config.max_epochs)
    parser.add_argument(
        "--transforms_config", type=str, default=Config().transforms_config
    )
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
    model: nn.Module, data_module: L.LightningDataModule, lr: float, **kwargs
) -> L.Trainer:
    train_module = ImageClassifier(model=model, lr=lr)
    trainer = L.Trainer(default_root_dir=tmp_dir, **kwargs)
    trainer.fit(train_module, datamodule=data_module)
    return trainer


def train_and_save_model(config: Config) -> None:
    torchscript_path: Path = tmp_dir / "model.torchscript"
    dagshub.init(repo_owner=username, repo_name="full_stack_ml", mlflow=True)

    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.pytorch.autolog()

    early_stopping = EarlyStopping("valid_loss")
    model_checkpoint = ModelCheckpoint(
        monitor="valid_loss", filename=experiment_name + "-{epoch:03d}-{valid_loss:.3f}"
    )
    callbacks = [
        early_stopping,
        model_checkpoint,
    ]
    with mlflow.start_run():
        transform = Transforms(config.transforms_config)
        model = get_model(config.model)
        data_module = CatVsDogsDataModule(
            data_dir=data_dir,
            batch_size=config.batch_size,
            pct_train=config.pct_train,
            transform=transform.transforms,
        )
        create_and_train_model(
            model=model,
            data_module=data_module,
            lr=config.lr,
            max_epochs=config.max_epochs,
            callbacks=callbacks,
        )

        model = get_model(config.model)
        best_model = ImageClassifier.load_from_checkpoint(
            checkpoint_path=model_checkpoint.best_model_path, model=model
        )
        mlflow.pytorch.log_model(best_model.model, "model")
        mlflow.log_params(data_module.hparams)
        mlflow.log_params(best_model.hparams)
        mlflow.log_params(asdict(config))

        scripted_model = torch.jit.script(best_model.model)
        scripted_model.save(torchscript_path)
        mlflow.log_artifact(torchscript_path)


def main() -> None:
    config = cli()
    train_and_save_model(config)


if __name__ == "__main__":
    main()
