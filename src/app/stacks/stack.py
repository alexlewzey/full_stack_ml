"""Module that contains the CDK stack i.e. declares the infrastructure to be created in
S3."""
import os

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

from src.utils.core import root_dir


class CatVsDogStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        docker_image = ecr_assets.DockerImageAsset(
            self,
            "LambdaDockerImage",
            directory=root_dir.as_posix(),
            file="src/api/Dockerfile",
            platform=ecr_assets.Platform.LINUX_ARM64,
            build_args={"DAGSHUB_USER_TOKEN": os.environ["DAGSHUB_USER_TOKEN"]},
        )

        log_group = logs.LogGroup(
            self,
            "LambdaLogGroup",
            retention=logs.RetentionDays.ONE_YEAR,
        )

        lambda_function = lambda_.DockerImageFunction(
            self,
            "HelloFunction",
            code=lambda_.DockerImageCode.from_ecr(
                repository=docker_image.repository, tag_or_digest=docker_image.image_tag
            ),
            log_group=log_group,
            architecture=lambda_.Architecture.ARM_64,
            memory_size=512,
            timeout=Duration.seconds(900),
        )

        api = apigateway.LambdaRestApi(  # noqa: F841
            self,
            "FastAPIApiGateway",
            handler=lambda_function,
            proxy=True,
            binary_media_types=[
                "image/jpeg",
                "image/png",
                "application/octet-stream",
                "application/pdf",
            ],
            default_cors_preflight_options={
                "allow_methods": apigateway.Cors.ALL_METHODS,
                "allow_headers": ["Content-Type"],
                "allow_origins": apigateway.Cors.ALL_ORIGINS,
            },
        )

        CfnOutput(self, "ApiGatewayUrl", value=api.url)
