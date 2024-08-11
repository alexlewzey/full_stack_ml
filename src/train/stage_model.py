import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src.utils.core import root_dir


@dataclass
class Config:
    run_id: str | None = None
    column: str = "valid_loss"
    ascending: bool = True
    experiment_name: str = "cats_vs_dogs"


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

    def get_artifact_path(self, run_id: str, path: str) -> Path:
        return Path.cwd().parent / f"mlruns/{self.experiment_id}/{run_id}/{path}"

    def get_best_run_id(self, column: str, ascending: bool):
        return (
            self.df[self.df.status == "FINISHED"]
            .sort_values(column, ascending=ascending)
            .iloc[0, 0]
        )


def stage_model(config: Config) -> None:
    mlflow_dir = (root_dir / "mlruns").as_posix()
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    experiment = MLFlowExperiment(experiment_name=config.experiment_name)
    run_id = (
        config.run_id
        if config.run_id
        else experiment.get_best_run_id(
            column=config.column, ascending=config.ascending
        )
    )
    torchscript_path = experiment.get_artifact_path(
        run_id, "artifacts/model.torchscript"
    )
    assert torchscript_path.exists()
    dir_staged = root_dir / "api" / "staged_model"
    dir_staged.mkdir(exist_ok=True, parents=True)
    shutil.copy2(torchscript_path, dir_staged / torchscript_path.name)


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
    args = vars(parser.parse_args())
    if args["run_id"]:
        print("run_id argument detected, column and ascending will be ignored.")
    config = Config(**args)
    stage_model(config)


if __name__ == "__main__":
    main()
