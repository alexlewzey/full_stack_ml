import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_lambda as lambda_
from aws_cdk import App, Stack
from constructs import Construct


class HelloWorldStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        docker_image = ecr_assets.DockerImageAsset(
            self,
            "LambdaDockerImage0",
            directory="/workspaces/example_cdk",
            file="lambda/Dockerfile",
        )

        lambda_function = lambda_.DockerImageFunction(  # noqa: F841
            self,
            "HelloFunction0",
            code=lambda_.DockerImageCode.from_ecr(
                repository=docker_image.repository, tag_or_digest=docker_image.image_tag
            ),
        )


app = App()

HelloWorldStack(app, "HelloWorldStack")
app.synth()
