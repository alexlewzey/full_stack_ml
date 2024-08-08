import base64
import io
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mangum import Mangum
from PIL import Image
from pydantic import BaseModel

from api.predictor import Predictor
from train.preprocessing import transform


class ImageData(BaseModel):
    image_data: str


dir_api = Path(__file__).parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=dir_api / "static"), name="static")
templates = Jinja2Templates(directory=dir_api / "templates")

torchscript_path = dir_api / "staged_model" / "model.torchscript"
assert torchscript_path.exists()
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


@app.get("/test")
async def test(request: Request):
    return templates.TemplateResponse(request, "test.html")


@app.post("/new")
async def new(payload: ImageData):
    return {"image_data": payload.image_data}


handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)  # noqa: S104
