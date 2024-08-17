"""Download model and transforms artifacts for the current champion alias model."""
import dagshub
import mlflow
from PIL import Image

from src.api.pipeline import Pipeline
from src.train.utils.experiment import Experiment
from src.utils.core import (
    artifacts_dir,
    experiment_name,
    model_name,
    username,
)

model_path = artifacts_dir / "model.torchscript"
transforms_path = artifacts_dir / "transforms_config.txt"


def download_champion_pipeline(experiment_name: str, model_name: str) -> None:
    dagshub.init(repo_owner=username, repo_name="full_stack_ml", mlflow=True)
    client = mlflow.MlflowClient()
    experiment = Experiment(experiment_name=experiment_name)
    run_id = client.get_model_version_by_alias(model_name, "champion").run_id
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=model_path.name, dst_path=artifacts_dir
    )
    transforms_config_str = experiment.df.set_index("run_id").loc[run_id][
        "transforms_config"
    ]
    with transforms_path.open("w") as f:
        f.write(transforms_config_str)


def validate_download() -> None:
    predictor = Pipeline(model_path=model_path, transforms_path=transforms_path)
    color = (255, 0, 0)
    img = Image.new("RGB", (300, 300), color)
    assert predictor.predict(img) in ("dog", "cat")


if __name__ == "__main__":
    download_champion_pipeline(experiment_name=experiment_name, model_name=model_name)
    validate_download()
