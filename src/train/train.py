# %%
import warnings
from pathlib import Path

import lightning as L  # noqa: N812
import mlflow
import preprocessing
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loaders import CatVsDogsDataModule
from models import ConvNet, ImageClassifier

warnings.filterwarnings("ignore", message="Checkpoint logging is skipped")

# %%
batch_size: int = 64
root_path: str = "."
pct_train: float = 0.8
lr: float = 1e-5
max_epochs: int = 50
experiment_name: str = "cats_vs_dogs"


root_dir = Path(root_path).resolve().parent
tmp_dir = root_dir / "tmp"
tmp_dir.mkdir(exist_ok=True)
data_dir = root_dir / "data"
torchscript_path: Path = tmp_dir / "model.torchscript"

# %%
mlflow_dir = (root_dir / "mlruns").as_posix()
mlflow.set_tracking_uri(f"file:{mlflow_dir}")
mlflow.set_experiment(experiment_name=experiment_name)
mlflow.pytorch.autolog()


# %%
early_stopping = EarlyStopping("valid_loss")
model_checkpoint = ModelCheckpoint(
    monitor="valid_loss", filename="cats-vs-dogs-{epoch:03d}-{valid_loss:.3f}"
)
callbacks = [
    early_stopping,
    model_checkpoint,
]
with mlflow.start_run() as run:
    data_module = CatVsDogsDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        pct_train=pct_train,
        transform=preprocessing.transform,
    )
    model = ConvNet()
    train_module = ImageClassifier(model=model, lr=lr)
    trainer = L.Trainer(
        max_epochs=max_epochs, callbacks=callbacks, default_root_dir=tmp_dir
    )
    trainer.fit(train_module, datamodule=data_module)

    best_model = ImageClassifier.load_from_checkpoint(model_checkpoint.best_model_path)
    mlflow.pytorch.log_model(best_model, "model")
    mlflow.log_params(data_module.hparams)
    mlflow.log_params(best_model.hparams)

    scripted_model = torch.jit.script(best_model.model)
    scripted_model.save(torchscript_path)
    mlflow.log_artifact(torchscript_path)
