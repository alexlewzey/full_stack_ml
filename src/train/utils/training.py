"""Module of classes used to train models."""
import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch.tuner.tuning import Tuner
from torch import Tensor, nn
from torchmetrics import Accuracy


class ImageClassifier(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.accuracy = Accuracy(task="binary")

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, model: nn.Module, map_location=None
    ):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        instance = cls(model=model, **checkpoint["hyper_parameters"])
        instance.load_state_dict(checkpoint["state_dict"])
        return instance

    def forward(self, x: Tensor) -> Tensor:
        out = self.model.forward(x)
        return out

    def _step(self, batch: tuple[Tensor, Tensor], set_name: str):
        x, y = batch
        yprob = self.model.forward(x)
        loss = F.cross_entropy(yprob, y)
        self.log(f"{set_name}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        yhat = yprob.argmax(-1)
        acc = self.accuracy(yhat, y)
        self.log(f"{set_name}_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def configure_optimizers(self) -> torch.optim.Optimizer:
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
