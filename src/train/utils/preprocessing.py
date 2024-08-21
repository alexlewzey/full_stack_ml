"""Module of model preprocessors."""

import ast
import json

from PIL import Image
from torch import Tensor
from torchvision import transforms

TransformsConfig = dict[str, dict]


class Transforms:
    def __init__(self, config):
        self.config = config
        self.transforms = self.build_transforms(config)

    def build_transforms(self, config: TransformsConfig) -> transforms.Compose:
        pipeline = [
            getattr(transforms, method)(**params) for method, params in config.items()
        ]
        return transforms.Compose(pipeline)

    @classmethod
    def from_str(cls, config_str: str) -> transforms.Compose:
        config = ast.literal_eval(config_str)
        return cls(config)

    @classmethod
    def from_json(cls, config_str: str) -> transforms.Compose:
        config = json.loads(config_str)
        return cls(config)

    def __call__(self, img: Image) -> Tensor:
        return self.transforms(img)
