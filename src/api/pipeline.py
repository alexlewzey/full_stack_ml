"""Module that contains `Predictor` which loads model artifact and make predicitons."""


from pathlib import Path

import torch
from PIL import Image

from src.train.utils.metadata import DECODING
from src.train.utils.preprocessing import Transforms


class Pipeline:
    def __init__(
        self, model_path: str | Path, transforms_path: str | Path, device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device).to(
            self.device
        )
        self.model.eval()
        with Path(transforms_path).open() as f:
            self.transforms = Transforms.from_str(f.read())

    def predict(self, img: Image) -> str:
        img = img.convert("RGB")
        with torch.no_grad():
            x = self.transforms(img)[None,].to(self.device)
            prob = self.model(x)
            yhat = prob.argmax(dim=-1).item()
            label = DECODING[yhat]
        return label
