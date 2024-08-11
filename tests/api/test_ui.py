import base64
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.ui import app
from src.utils.core import image_dir

client = TestClient(app)


path_dog_0 = image_dir / "dog_0.png"
path_cat_0 = image_dir / "cat_0.jpg"


def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"hello": "world"}


def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "Cat vs Dog Image Classifier" in response.text
    assert "PyTorch + Lightning + MLflow + FastAPI + HTMX" in response.text
    assert "Full Stack Machine Learning Project" in response.text


def post_image_to_upload(path_img: Path):
    with path_img.open("rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {"image_data": image_base64}
    return client.post("/upload", json=payload)


def test_upload_dog():
    response = post_image_to_upload(path_dog_0)
    assert response.status_code == 200
    assert "It's a <b>dog</b>!" in response.text


def test_upload_cat():
    response = post_image_to_upload(path_cat_0)
    assert response.status_code == 200
    assert "It's a <b>cat</b>!" in response.text


if __name__ == "__main__":
    raise Exception
