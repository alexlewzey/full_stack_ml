from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from src.train.utils.metadata import DECODING


class Predictor:
    def __init__(
        self, torchscript_path: str | Path, transform: Callable, device: str = "cpu"
    ):
        self.transform = transform
        self.device = torch.device(device)
        self.model = torch.jit.load(torchscript_path, map_location=self.device).to(
            self.device
        )
        self.model.eval()

    def predict(self, img: Image) -> str:
        img = img.convert("RGB")
        with torch.no_grad():
            x = self.transform(img)[None,].to(self.device)
            prob = self.model(x)
            yhat = prob.argmax(dim=-1).item()
            label = DECODING[yhat]
        return label
