import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
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
async def upload(request: Request, file: UploadFile = File(...)):  # noqa: B008
    contents = await file.read()
    
    img = Image.open(io.BytesIO(base64.b64decode(contents))).convert("RGB")
    label = predictor.predict(img)
    base64_encoded = base64.b64encode(contents).decode("utf-8")
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "base64_encoded": base64_encoded,
            "content_type": file.content_type,
            "label": label,
        },
    )


handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)  # noqa: S104
