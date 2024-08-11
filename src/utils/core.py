from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
image_dir = root_dir / "images"
sample_dir = image_dir / "sample" / "data"
tmp_dir = root_dir / "tmp"
tmp_dir.mkdir(exist_ok=True)
data_dir = tmp_dir / "data"
data_dir.mkdir(exist_ok=True)
