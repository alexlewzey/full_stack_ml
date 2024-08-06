import base64
import random
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mangum import Mangum
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
def home():
    return {"hello": "world"}


@app.get("/")
async def root():
    return HTMLResponse(
        """
    <html>
        <head>
            <script src="https://unpkg.com/htmx.org@1.9.6"></script>
        </head>
        <body>
            <h1>Image Upload</h1>
            <form hx-post="/upload" hx-encoding="multipart/form-data" hx-target="#result">
                <input type="file" name="file" accept="image/*">
                <button type="submit">Upload</button>
            </form>
            <div id="result"></div>
        </body>
    </html>
    """  # noqa: E501
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):  # noqa: B008
    contents = await file.read()
    base64_encoded = base64.b64encode(contents).decode("utf-8")
    return HTMLResponse(
        f"""
    <h2>Uploaded Image:</h2>
    <img src="data:image/{file.content_type};base64,{base64_encoded}" alt="Uploaded Image" style="max-width: 300px;">
    <p>Base64 string (first 100 characters):</p>
    <textarea rows="3" cols="50" readonly>{base64_encoded[:100]}...</textarea>
    """  # noqa: E501
    )


@app.get("/read-form")
async def read_form(request: Request):
    return templates.TemplateResponse("read_form.html", {"request": request})


@app.post("/handel-form")
async def handel_form(request: Request, name: str = Form(...)):
    return templates.TemplateResponse(
        "handel_form.html", {"request": request, "name": name}
    )


@app.get("/more-content")
async def more_content():
    n = random.randint(55, 555)
    html = f'<div hx-get="/more-content" hx-trigger="revealed" hx-swap="beforeend">power levels over {n}</div>'  # noqa: E501
    return HTMLResponse(html)


@app.get("/tab1")
async def tab1():
    return HTMLResponse("<div>This is tab 1</div>")


@app.get("/tab2")
async def tab2():
    return HTMLResponse("<div>This is tab 2</div>")


@app.get("/get-time")
def get_time():
    dt = datetime.now().replace(microsecond=0).isoformat()
    return HTMLResponse(f"<div>The time is: {dt}</div>")


handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)  # noqa: S104
