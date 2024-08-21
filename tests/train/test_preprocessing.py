import pytest
from PIL import Image
from torch import Tensor

from src.train.utils.preprocessing import Transforms, TransformsConfig
from src.utils.core import sample_dir


def assert_transform_valid(transforms, img):
    x = transforms(img)
    assert isinstance(x, Tensor)
    assert list(x.shape) == [3, 224, 224]


@pytest.fixture
def mock_img() -> Image:
    image_path = sample_dir / "train" / "dog.1.jpg"
    return Image.open(image_path)


class TestTransforms:
    @staticmethod
    def test_transform_crop(mock_img: Image):
        crop_config: TransformsConfig = {
            "Resize": {"size": 224},
            "CenterCrop": {"size": 224},
            "ToTensor": {},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        transform = Transforms(crop_config)
        assert_transform_valid(transform.transforms, mock_img)

    @staticmethod
    def test_transform_resize(mock_img: Image):
        resize_config: TransformsConfig = {
            "Resize": {"size": [224, 224]},
            "ToTensor": {},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        transform = Transforms(resize_config)
        assert_transform_valid(transform.transforms, mock_img)

    @staticmethod
    def test_from_str(mock_img: Image):
        str_config = "{'Resize': {'size': 224}, 'CenterCrop': {'size': 224}, 'ToTensor': {}, 'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}"  # noqa: E501
        transform = Transforms.from_str(str_config)
        assert_transform_valid(transform.transforms, mock_img)

    @staticmethod
    def test_from_json(mock_img: Image):
        json_config = """{
            "Resize": {
                "size": 224
            },
            "CenterCrop": {
                "size": 224
            },
            "ToTensor": {},
            "Normalize": {
                "mean": [
                    0.485,
                    0.456,
                    0.406
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225
                ]
            }
        }"""
        transform = Transforms.from_json(json_config)
        assert_transform_valid(transform.transforms, mock_img)
