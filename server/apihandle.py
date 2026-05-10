from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import uuid
import os

from model import generate_image

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GenerateRequest(BaseModel):
    hair_color: str
    eye_color: str
    style: str


@app.post("/generate")
async def generate(req: GenerateRequest):


    return {
       print("gen")
    }


