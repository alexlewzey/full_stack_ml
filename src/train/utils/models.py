import lightning as L  # noqa: N812
import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn
from torchmetrics import Accuracy


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
