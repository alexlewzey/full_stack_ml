import uvicorn
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

handler = Mangum(app)


@app.get("/")
def home():
    return {"hello": "world"}


@app.get("/items/{id_}")
def items(id_: int, q: str | None = None):
    return {"id_": id_, "q": q}


if __name__ == "__main__":
    uvicorn.run(app)
