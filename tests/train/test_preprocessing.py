from PIL import Image
from torch import Tensor

from src.train.utils.preprocessing import TransformsConfig, build_transforms
from src.utils.core import sample_dir


def test_transform():
    image_path = sample_dir / "train" / "dog.1.jpg"
    img = Image.open(image_path)
    crop_config: TransformsConfig = {
        "Resize": {"size": 224},
        "CenterCrop": {"size": 224},
        "ToTensor": {},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    transform = build_transforms(crop_config)
    x = transform(img)
    assert isinstance(x, Tensor)
    assert list(x.shape) == [3, 224, 224]

    resize_config: TransformsConfig = {
        "Resize": {"size": [224, 224]},
        "ToTensor": {},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    transform = build_transforms(resize_config)
    x = transform(img)
    assert isinstance(x, Tensor)
    assert list(x.shape) == [3, 224, 224]
