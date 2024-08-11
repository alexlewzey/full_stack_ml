from aws_cdk import App

from .stacks.stack import CatVsDogStack

app = App()
CatVsDogStack(app, "CatVsDogStack", env={"region": "eu-west-2"})
app.synth()
