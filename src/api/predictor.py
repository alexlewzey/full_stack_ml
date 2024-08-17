"""Module that contains `Predictor` which loads model artifact and make predicitons."""


from pathlib import Path

import dagshub
import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from PIL import Image
from torch import nn
from torchvision import transforms

from src.train.utils.metadata import DECODING
from src.train.utils.preprocessing import Transforms
from src.utils.core import artifacts_dir, username


class MLFlowExperiment:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.df = self.get_experiment_df(experiment_name)

    def get_experiment_df(self, experiment_name: str) -> pd.DataFrame:
        experiments = pd.DataFrame(
            map(dict, self.client.search_experiments())
        ).set_index("name")
        self.experiment_id = experiments.loc[experiment_name, "experiment_id"]
        runs = self.client.search_runs(experiment_ids=self.experiment_id)
        data = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "status": run.info.status,
            }
            run_data.update(run.data.metrics)
            run_data.update(run.data.params)
            data.append(run_data)
        return pd.DataFrame(data)

    def get_best_run_id(self, column: str, ascending: bool):
        return (
            self.df[self.df.status == "FINISHED"]
            .sort_values(column, ascending=ascending)
            .iloc[0, 0]
        )


class Predictor:
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        model_path = Path(model_path)
        self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, img: Image) -> str:
        img = img.convert("RGB")
        with torch.no_grad():
            x = self.transform.transforms(img)[None,].to(self.device)
            prob = self.model(x)
            yhat = prob.argmax(dim=-1).item()
            label = DECODING[yhat]
        return label


def load_champion_pipeline(
    experiment_name: str, model_name: str
) -> tuple[transforms.Compose, nn.Module]:
    dagshub.init(repo_owner=username, repo_name="full_stack_ml", mlflow=True)
    client = mlflow.MlflowClient()
    experiment = MLFlowExperiment(experiment_name=experiment_name)
    run_id = client.get_model_version_by_alias(model_name, "champion").run_id
    file: str = "model.torchscript"
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=file, dst_path=artifacts_dir
    )
    model = torch.jit.load(artifacts_dir / file)
    transforms_config_str = experiment.df.set_index("run_id").loc[run_id][
        "transforms_config"
    ]
    transforms_ = Transforms.from_str(transforms_config_str)
    return transforms_, model
