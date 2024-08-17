"""Module containing general varibles such as project paths."""
from pathlib import Path

experiment_name: str = "cats_vs_dogs"
repo_name: str = "full_stack_ml"
username: str = "alexlewzey"
environment: str = "prod"
model_name: str = f"{environment}.cats_vs_dogs"

root_dir = Path(__file__).parent.parent.parent
image_dir = root_dir / "images"
sample_dir = image_dir / "sample" / "data"
tmp_dir = root_dir / "tmp"
tmp_dir.mkdir(exist_ok=True)
data_dir = tmp_dir / "data"
data_dir.mkdir(exist_ok=True)
