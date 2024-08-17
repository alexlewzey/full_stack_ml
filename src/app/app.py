"""Script that deploys CDK stack to AWS."""
import os

from aws_cdk import App

from .stacks.stack import CatVsDogStack

app = App()
envs = {"DAGSHUB_USER_TOKEN": os.environ.get("DAGSHUB_USER_TOKEN")}
CatVsDogStack(app, "CatVsDogStack", env={"region": "eu-west-2"}, envs=envs)
app.synth()
