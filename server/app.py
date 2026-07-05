from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import os
import json
import uuid
import shutil
import time
import random

from texture_transfer import TextureTransfer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
FINISHED_FOLDER = "finished"
SELECTED_FOLDER = "selected"   # precomputed garments
MASKS_FOLDER = "masks"         # precomputed masks (same filenames as selected/)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FINISHED_FOLDER, exist_ok=True)

# Serve the finished (textured) images
app.mount("/finished", StaticFiles(directory=FINISHED_FOLDER), name="finished")

tt = TextureTransfer()


@app.get("/classes")
def get_classes():
    with open("classes.json", "r") as f:
        return json.load(f)


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    request_data: str = Form(...),
):
    start_time = time.time()

    parsed_request = json.loads(request_data)
    request_metadata = parsed_request["request_metadata"]
    user_input = parsed_request["user_input"]
    selected_class = user_input["selected_class"]

    uid = uuid.uuid4()

    # ---------------------------------------
    # SAVE UPLOADED TEXTURE
    # ---------------------------------------
    texture_filename = f"{uid}_texture.jpg"
    texture_path = os.path.join(UPLOAD_FOLDER, texture_filename)

    with open(texture_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # ---------------------------------------
    # PICK A PRECOMPUTED GARMENT FOR THIS CLASS
    # ---------------------------------------
    class_garment_dir = os.path.join(SELECTED_FOLDER, selected_class)
    class_mask_dir = os.path.join(MASKS_FOLDER, selected_class)

    if not os.path.isdir(class_garment_dir):
        return JSONResponse(
            {"status": "error", "message": f"Unknown class: {selected_class}"},
            status_code=404,
        )

    garment_files = [
        f for f in os.listdir(class_garment_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not garment_files:
        return JSONResponse(
            {"status": "error", "message": f"No garments found for {selected_class}"},
            status_code=404,
        )

    chosen_file = random.choice(garment_files)
    garment_path = os.path.join(class_garment_dir, chosen_file)
    mask_path = os.path.join(class_mask_dir, chosen_file)

    if not os.path.exists(mask_path):
        return JSONResponse(
            {"status": "error", "message": f"Mask missing for {chosen_file}"},
            status_code=404,
        )

    # ---------------------------------------
    # APPLY TEXTURE
    # ---------------------------------------
    output_filename = f"{uid}_final.jpg"
    output_path = os.path.join(FINISHED_FOLDER, output_filename)

    tt.apply_texture(
        garment_path=garment_path,
        mask_path=mask_path,
        texture_path=texture_path,
        output_path=output_path,
    )

    generated_image_url = f"http://localhost:8000/finished/{output_filename}"
    processing_time = round(time.time() - start_time, 4)

    response = {
        "status": "success",
        "message": "Generation completed successfully",
        "request_metadata": request_metadata,
        "user_input": {"selected_class": selected_class},
        "upload": {
            "original_filename": image.filename,
            "saved_filename": texture_filename,
        },
        "source_garment": chosen_file,
        "performance": {"processing_time_seconds": processing_time},
        "result": {
            "generated_image": {"url": generated_image_url}
        },
        "errors": [],
        "warnings": [],
    }

    return JSONResponse(response)