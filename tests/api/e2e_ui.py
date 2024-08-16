import base64
import json
from pathlib import Path
from typing import Any

import requests

from src.utils.core import image_dir

lambda_container_url: str = "http://api:8080/2015-03-31/functions/function/invocations"
path_dog_0 = image_dir / "dog_0.png"


def test_healthcheck():
    data: dict[str, Any] = {
        "resource": "/",
        "path": "/healthcheck",
        "httpMethod": "GET",
        "requestContext": {},
    }
    response = requests.get(lambda_container_url, json=data, timeout=10)
    assert response.status_code == 200
    print(response.json())
    assert json.loads(response.json()["body"]) == {"hello": "world"}


def test_index():
    data: dict[str, Any] = {
        "resource": "/",
        "path": "/",
        "httpMethod": "GET",
        "requestContext": {},
    }
    response = requests.get(lambda_container_url, json=data, timeout=10)
    assert response.status_code == 200
    assert "Cat vs Dog Image Classifier" in response.text
    assert "PyTorch + Lightning + MLflow + FastAPI + HTMX" in response.text
    assert "Full Stack Machine Learning Project" in response.text


def post_image_to_upload(path_img: Path) -> requests.models.Response:
    data: dict[str, Any] = {
        "resource": "/",
        "path": "/upload",
        "httpMethod": "POST",
        "requestContext": {},
    }
    with path_img.open("rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    data["body"] = json.dumps({"image_data": image_base64})
    return requests.post(lambda_container_url, json=data, timeout=10)


def test_upload():
    response = post_image_to_upload(path_dog_0)
    print(response.json())
    assert response.status_code == 200
    assert "It's a <b>dog</b>!" in response.json()["body"]
