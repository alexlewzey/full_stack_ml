"""Script that contains the code for the lambda api which uses fastapi and mangum to map
the http request into lambda proxy."""
import base64
import io
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mangum import Mangum
from PIL import Image
from pydantic import BaseModel

from src.api.predictor import Predictor
from src.train.utils.preprocessing import build_transforms
from src.utils.core import root_dir


class ImageData(BaseModel):
    image_data: str


dir_api = Path(__file__).parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=dir_api / "static"), name="static")
templates = Jinja2Templates(directory=dir_api / "templates")

# todo: load transforms and
with (root_dir / "configs" / "default_config.json").open() as f:
    config = json.load(f)
transform = build_transforms(config["transforms_config"])

torchscript_path = dir_api / "staged_model" / "model.torchscript"
if not torchscript_path.exists():
    print("model.torchscript does not exist, loading default.torchscript")
    torchscript_path = torchscript_path.with_name("default.torchscript")
predictor = Predictor(torchscript_path=torchscript_path, transform=transform)


@app.get("/healthcheck")
async def healthcheck():
    return {"hello": "world"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(request: Request, payload: ImageData):  # noqa: B008
    bytes_ = payload.image_data.encode("utf-8")
    img = Image.open(io.BytesIO(base64.b64decode(bytes_))).convert("RGB")
    label = predictor.predict(img)
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "image_base64": payload.image_data,
            "label": label,
        },
    )


handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)  # noqa: S104
