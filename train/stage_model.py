# %%
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient


# %%
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

    def get_best_run_id(self) -> str:
        run = self.df[self.df.status == "FINISHED"].sort_values("valid_loss")
        run_id = run.run_id[0]
        return run_id

    def artifact_path(self, run_id, path: str) -> Path:
        return Path.cwd().parent / f"mlruns/{self.experiment_id}/{run_id}/{path}"


# %%
root_dir = Path.cwd().parent
img_path = root_dir / "images" / "example.png"
mlflow_dir = (root_dir / "mlruns").as_posix()
mlflow.set_tracking_uri(f"file:{mlflow_dir}")
experiment = MLFlowExperiment(experiment_name="cats_vs_dogs")
run_id = experiment.df.sort_values("start_time", ascending=False).iloc[0, 0]
torchscript_path = experiment.artifact_path(run_id, "artifacts/model.torchscript")
assert torchscript_path.exists()
dir_staged = root_dir / "api" / "staged_model"
dir_staged.mkdir(exist_ok=True, parents=True)
shutil.copy2(torchscript_path, dir_staged / torchscript_path.name)
