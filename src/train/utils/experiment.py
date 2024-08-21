import pandas as pd
from mlflow import MlflowClient


class Experiment:
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
