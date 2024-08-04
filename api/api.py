import base64
import io
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mangum import Mangum
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

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


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/healthcheck")
def home():
    return {"hello": "world"}


@app.post("/upload")
def upload_image(data: ImageData):
    contents = base64.b64decode(data.image_data)
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    x = transforms.ToTensor()(img)
    return {"sizes": list(x.shape)}


@app.post("/predict")
def predict(data: ImageData):
    contents = base64.b64decode(data.image_data)
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    label = predictor.predict(img)
    return {"label": label}


handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
