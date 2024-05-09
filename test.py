from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware

class Item(BaseModel):
    name: str

app = FastAPI()

origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:4200/",
    "http://localhost:4200/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
    
)

@app.post("/test")
async def create_item(uploadFile: Annotated[UploadFile, Form()]):
    print(uploadFile.filename)
    return {"name": uploadFile.filename}