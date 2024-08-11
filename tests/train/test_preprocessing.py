from PIL import Image
from torch import Tensor

from src.train.utils.preprocessing import transform
from src.utils.core import sample_dir


def test_transform():
    image_path = sample_dir / "train" / "dog.1.jpg"
    img = Image.open(image_path)
    x = transform(img)
    assert isinstance(x, Tensor)
    assert list(x.shape) == [3, 32, 32]
