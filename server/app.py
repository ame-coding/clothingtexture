from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import os
import json
import uuid
import shutil
import time

import subprocess



app = FastAPI()



# ---------------------------------------
# CORS
# ---------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------
# FOLDERS
# ---------------------------------------

UPLOAD_FOLDER = "uploads"

CLOTHING_FOLDER = "clothing"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Serve clothing images
app.mount(
    "/clothing",
    StaticFiles(directory="clothing"),
    name="clothing"
)



# ---------------------------------------
# GET CLASSES
# ---------------------------------------

@app.get("/classes")
def get_classes():

    with open("classes.json", "r") as file:

        data = json.load(file)

    return data



# ---------------------------------------
# GENERATE
# ---------------------------------------

@app.post("/generate")
async def generate(

    image: UploadFile = File(...),

    request_data: str = Form(...)
):


    # ---------------------------------------
    # START TIMER
    # ---------------------------------------

    start_time = time.time()



    # ---------------------------------------
    # PARSE JSON REQUEST
    # ---------------------------------------

    parsed_request = json.loads(request_data)



    # ---------------------------------------
    # EXTRACT REQUEST DATA
    # ---------------------------------------

    request_metadata = parsed_request["request_metadata"]

    user_input = parsed_request["user_input"]

    generation_settings = parsed_request["generation_settings"]

    model_settings = parsed_request["model_settings"]

    output_settings = parsed_request["output_settings"]



    selected_class = user_input["selected_class"]

    uid = uuid.uuid4()

    # ---------------------------------------
    # SAVE UPLOADED IMAGE
    # ---------------------------------------

    saved_filename = (
        f"{uid}_texture.jpg"
    )

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        saved_filename
    )



    with open(upload_path, "wb") as buffer:

        shutil.copyfileobj(
            image.file,
            buffer
        )



    subprocess.run(
    [
        "python",
        "ganseg.py",
        "--uuid", f"{uid}",
        "--prompt", f"{selected_class}"
    ]
    )

    # ---------------------------------------
    # GENERATED IMAGE
    # ---------------------------------------

    generated_image_url = (
        f"http://localhost:8000/finished/{uid}_final.jpg"
    )



    # ---------------------------------------
    # PERFORMANCE INFO
    # ---------------------------------------

    end_time = time.time()

    processing_time = round(
        end_time - start_time,
        4
    )



    # ---------------------------------------
    # RESPONSE
    # ---------------------------------------

    response = {

        "status": "success",

        "message": "Generation completed successfully",



        "request_metadata": {

            "request_id":
                request_metadata["request_id"],

            "timestamp":
                request_metadata["timestamp"],

            "request_type":
                request_metadata["request_type"],
        },



        "user_input": {

            "selected_class":
                selected_class,
        },



        "upload": {

            "original_filename":
                image.filename,

            "saved_filename":
                saved_filename,

            "upload_path":
                upload_path,
        },



        "generation_settings":
            generation_settings,



        "model_info": {

            "model_name":
                model_settings["model_name"],

            "device":
                model_settings["device"],
        },



        "performance": {

            "processing_time_seconds":
                processing_time,
        },



        "result": {

            "generated_image": {

                "url":
                    generated_image_url,

                "width":
                    300,

                "height":
                    300,
            }
        },



        "errors": [],

        "warnings": []
    }



    return JSONResponse(response)