"""Module containing general varibles such as project paths."""
import logging
import os
from pathlib import Path

experiment_name: str = "cats_vs_dogs"
repo_name: str = "full_stack_ml"
username: str = "alexlewzey"
environment: str = os.environ.get("ENVIRONMENT", "prod")
model_name: str = f"{environment}.{experiment_name}"
logging_level = logging.INFO

root_dir = Path(__file__).parent.parent.parent
image_dir = root_dir / "images"
sample_dir = image_dir / "sample" / "data"
tmp_dir = root_dir / "tmp"
tmp_dir.mkdir(exist_ok=True)
data_dir = tmp_dir / "data"
data_dir.mkdir(exist_ok=True)
artifacts_dir = root_dir / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
