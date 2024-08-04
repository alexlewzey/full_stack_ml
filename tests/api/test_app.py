import base64

from fastapi.testclient import TestClient

from api.api import app
from utils.core import image_dir

client = TestClient(app)


path_tmp = image_dir / "example.png"


def test_home():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"hello": "world"}


def test_upload():
    with path_tmp.open("rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    data = {"image_data": img_b64}
    response = client.post("/upload", json=data)
    assert response.status_code == 200
    body = response.json()
    assert body == {"sizes": [3, 1388, 1484]}


def test_predict():
    with path_tmp.open("rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    data = {"image_data": img_b64}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    body = response.json()
    assert body == {"label": "dog"}


if __name__ == "__main__":
    raise Exception
