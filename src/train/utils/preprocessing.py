"""Module of model preprocessors."""

from torchvision import transforms

TransformsConfig = dict[str, dict]


def build_transforms(config: TransformsConfig) -> transforms.Compose:
    pipeline = [
        getattr(transforms, method)(**params) for method, params in config.items()
    ]
    return transforms.Compose(pipeline)
