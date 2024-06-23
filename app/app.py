import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_
import aws_cdk.aws_logs as logs
import aws_cdk.aws_scheduler as scheduler
from aws_cdk import App, Stack
from constructs import Construct


class HelloWorldStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        docker_image = ecr_assets.DockerImageAsset(
            self,
            "LambdaDockerImage",
            directory="/workspaces/example_cdk",
            file="lambda/Dockerfile",
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        log_group = logs.LogGroup(
            self,
            "LambdaLogGroup",
            retention=logs.RetentionDays.ONE_DAY,
        )

        lambda_function = lambda_.DockerImageFunction(
            self,
            "HelloFunction",
            code=lambda_.DockerImageCode.from_ecr(
                repository=docker_image.repository, tag_or_digest=docker_image.image_tag
            ),
            log_group=log_group,
        )

        scheduler_role = iam.Role(
            self,
            "SchedulerRole",
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"),
        )
        lambda_function.grant_invoke(scheduler_role)

        cfn_schedule = scheduler.CfnSchedule(  # noqa: F841
            self,
            "MyCfnSchedule",
            flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(
                mode="OFF"
            ),
            schedule_expression="rate(1 minute)",
            target=scheduler.CfnSchedule.TargetProperty(
                arn=lambda_function.function_arn,
                role_arn=scheduler_role.role_arn,
                input="{}",
            ),
            name="LambdaScheduler",
        )


app = App()

HelloWorldStack(app, "HelloWorldStack")
app.synth()
