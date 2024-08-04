import base64
import json
from typing import Any

import requests

from utils.core import image_dir

lambda_container_url: str = "http://api:8080/2015-03-31/functions/function/invocations"
image_path = image_dir / "example.png"


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


def test_upload():
    data: dict[str, Any] = {
        "resource": "/",
        "path": "/upload",
        "httpMethod": "POST",
        "requestContext": {},
    }
    with image_path.open("rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    data["body"] = json.dumps({"image_data": img_b64})
    response = requests.post(lambda_container_url, json=data, timeout=10)
    assert response.status_code == 200
    assert json.loads(response.json()["body"]) == {"sizes": [3, 1388, 1484]}


def test_predict():
    data: dict[str, Any] = {
        "resource": "/",
        "path": "/predict",
        "httpMethod": "POST",
        "requestContext": {},
    }
    with image_path.open("rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    data["body"] = json.dumps({"image_data": img_b64})
    response = requests.post(lambda_container_url, json=data, timeout=10)
    print(response.json())
    assert response.status_code == 200
    assert json.loads(response.json()["body"]) == {"label": "dog"}
