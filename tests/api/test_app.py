from fastapi.testclient import TestClient

from api.ui import app
from utils.core import image_dir

client = TestClient(app)


path_tmp = image_dir / "example.png"


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


def test_upload():
    # with path_tmp.open("rb") as f:
    #     img_b64 = base64.b64encode(f.read()).decode("utf-8")
    # data = {"image_data": img_b64}
    # response = client.post("/upload", json=data)

    with path_tmp.open("rb") as f:
        files = {"file": f}
        response = client.post("/upload", files=files)

    assert response.status_code == 200
    assert "dog" in response.text


if __name__ == "__main__":
    raise Exception
