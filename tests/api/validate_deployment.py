import base64
import logging
from pathlib import Path

import boto3
import requests

from src.utils.core import image_dir, logging_level

logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


class ValidateDeployment:
    path_dog_0 = image_dir / "dog_0.png"
    path_cat_0 = image_dir / "cat_0.jpg"
    timeout: int = 60 * 5

    def __init__(self):
        self.url = self.get_apigateway_url()

    def get_apigateway_url(self) -> str:
        cloudformation = boto3.client("cloudformation")
        response = cloudformation.describe_stacks(StackName="CatVsDogStack")
        outputs = response["Stacks"][0]["Outputs"]
        return [
            output["OutputValue"]
            for output in outputs
            if output["OutputKey"] == "ApiGatewayUrl"
        ][0]

    def test_healthcheck(self):
        response = requests.get(f"{self.url}/healthcheck", timeout=self.timeout)
        assert response.status_code == 200
        assert response.json() == {"hello": "world"}

    def test_index(self):
        response = requests.get(f"{self.url}/", timeout=self.timeout)
        assert response.status_code == 200
        assert "Cat vs Dog Image Classifier" in response.text
        assert "PyTorch + Lightning + MLflow + FastAPI + HTMX" in response.text
        assert "Full Stack Machine Learning Project" in response.text

    def post_image_to_upload(self, path_img: Path):
        with path_img.open("rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {"image_data": image_base64}
        return requests.post(f"{self.url}/upload", json=payload, timeout=self.timeout)

    def test_upload_dog(self):
        response = self.post_image_to_upload(self.path_dog_0)
        assert response.status_code == 200
        assert "It's a <b>dog</b>!" in response.text

    def test_upload_cat(self):
        response = self.post_image_to_upload(self.path_cat_0)
        assert response.status_code == 200
        assert "It's a <b>cat</b>!" in response.text

    def validate(self):
        self.test_healthcheck()
        self.test_index()
        self.test_upload_dog()
        self.test_upload_cat()


if __name__ == "__main__":
    validate_deployment = ValidateDeployment()
    try:
        requests.get(validate_deployment.url, timeout=30)
    except Exception:  # noqa: S110
        pass
    validate_deployment.validate()
    logger.info("Validate deployment passed!")
