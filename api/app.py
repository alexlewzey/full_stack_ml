import base64
import io

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from mangum import Mangum
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms


class ImageData(BaseModel):
    image_data: str


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """

    """  # noqa: E501
    return html_content


@app.get("/healthcheck")
def home():
    return {"hello": "world"}


@app.post("/upload")
def upload_image(data: ImageData):
    contents = base64.b64decode(data.image_data)
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    x = transforms.ToTensor()(img)
    return {"sizes": list(x.shape)}


handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
