import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from src.app.stacks.stack import CatVsDogStack


def template():
    app = App()
    stack = CatVsDogStack(app, "test-stack")
    return Template.from_stack(stack)


mock_template = pytest.fixture(autouse=True)(template)


class TestCatVsDogStack:
    @staticmethod
    def test_log_group(mock_template):
        mock_template.has_resource_properties(
            "AWS::Logs::LogGroup", {"RetentionInDays": 365}
        )

    @staticmethod
    def test_lambda(mock_template):
        mock_template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "MemorySize": 512,
                "Timeout": 900,
                "Architectures": ["arm64"],
                "PackageType": "Image",
            },
        )

    @staticmethod
    def test_api_gateway(mock_template):
        mock_template.has_resource_properties(
            "AWS::ApiGateway::RestApi",
            {
                "Name": "FastAPIApiGateway",
                "BinaryMediaTypes": [
                    "image/jpeg",
                    "image/png",
                    "application/octet-stream",
                    "application/pdf",
                ],
            },
        )
