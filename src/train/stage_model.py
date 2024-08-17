"""CLI script that stages trained model i.e. grabs a model artifact from mlflow adds it
to the staging directory and tests it can be loaded for predictions."""
import argparse
from dataclasses import dataclass

import dagshub
from mlflow import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from PIL import Image

from src.api.predictor import MLFlowExperiment, Predictor
from src.utils.core import image_dir, model_name


@dataclass
class Config:
    run_id: str | None = None
    column: str = "valid_loss"
    ascending: bool = True
    experiment_name: str = "cats_vs_dogs"
    environment: str = "prod"


def stage_model(config: Config) -> None:
    dagshub.init(repo_owner="alexlewzey", repo_name="full_stack_ml", mlflow=True)
    client = MlflowClient()
    experiment = MLFlowExperiment(experiment_name=config.experiment_name)
    model_name: str = f"{config.environment}.cats_vs_dogs"
    run_id = (
        config.run_id
        if config.run_id
        else experiment.get_best_run_id(
            column=config.column, ascending=config.ascending
        )
    )
    try:
        client.create_registered_model(model_name)
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise e
    model_uri = RunsArtifactRepository.get_underlying_uri(f"runs:/{run_id}/model")
    version = client.create_model_version(model_name, model_uri, run_id)
    client.set_registered_model_alias(model_name, "champion", version.version)


def test_staged_model(config: Config) -> None:
    predictor = Predictor(experiment_name=config.experiment_name, model_name=model_name)
    img_dog = Image.open(image_dir / "dog_0.png")
    assert predictor.predict(img_dog) == "dog"

    img_cat = Image.open(image_dir / "cat_0.jpg")
    assert predictor.predict(img_cat) == "cat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        default=Config.run_id,
        help="MLfow run_id on the model to be staged. If passed this argument takes precedence over column and ascending.",  # noqa: E501
    )
    parser.add_argument(
        "--column",
        type=str,
        default=Config.column,
        help="Column name of metric used to select run_id.",
    )
    parser.add_argument(
        "--ascending",
        type=bool,
        default=Config.ascending,
        help="Whether to grab the max or min row corresponding to the `column`",
    )
    parser.add_argument("--experiment_name", type=str, default=Config.experiment_name)
    parser.add_argument("--environment", type=str, default=Config.environment)
    args = vars(parser.parse_args())
    if args["run_id"]:
        print("run_id argument detected, column and ascending will be ignored.")
    config = Config(**args)
    stage_model(config)
    test_staged_model(config)
    print("model successfully staged!")


if __name__ == "__main__":
    main()
