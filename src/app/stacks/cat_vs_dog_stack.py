import aws_cdk.aws_apigateway as apigateway
import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_lambda as lambda_
import aws_cdk.aws_logs as logs
from aws_cdk import Duration, Stack
from constructs import Construct
from utils.core import root_dir


class CatVsDogStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        docker_image = ecr_assets.DockerImageAsset(
            self,
            "LambdaDockerImage",
            directory=root_dir.as_posix(),
            file="api/Dockerfile",
            platform=ecr_assets.Platform.LINUX_ARM64,
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
