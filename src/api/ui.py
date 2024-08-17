"""Script that contains the code for the lambda api which uses fastapi and mangum to map
the http request into lambda proxy."""
import base64
import io
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mangum import Mangum
from PIL import Image
from pydantic import BaseModel

from src.api.pipeline import Pipeline
from src.utils.core import artifacts_dir, logging_level

logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


class ImageData(BaseModel):
    image_data: str


dir_api = Path(__file__).parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=dir_api / "static"), name="static")
templates = Jinja2Templates(directory=dir_api / "templates")


def get_pipeline():
    if not hasattr(get_pipeline, "instance"):
        model_path = artifacts_dir / "model.torchscript"
        transforms_path = artifacts_dir / "transforms_config.txt"
        pipeline = Pipeline(model_path=model_path, transforms_path=transforms_path)
        get_pipeline.instance = pipeline  # type: ignore
    return get_pipeline.instance  # type: ignore


@app.get("/healthcheck")
async def healthcheck():
    try:
        return {"hello": "world"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def index(request: Request):
    try:
        return templates.TemplateResponse(request, "index.html")
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload")
async def upload(request: Request, payload: ImageData):  # noqa: B008
    bytes_ = payload.image_data.encode("utf-8")
    img = Image.open(io.BytesIO(base64.b64decode(bytes_))).convert("RGB")
    label = get_pipeline().predict(img)
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
