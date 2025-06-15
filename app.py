from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from ngrok import ngrok
import uvicorn

import os
import glob
import shutil

from gen_t2m import BVHGenerator

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH" , "2y7NukFTBZaGLwKXMgKRGL6PllO_5NC6Lvzevtz6Z8QcU1Zmf")
APPLICATION_PORT = os.getenv("APPLICATION_PORT", 8080)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # logger.info("Setting up Ngrok Tunnel")
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    ngrok.forward(
        addr=APPLICATION_PORT,
    )
    yield
    # logger.info("Tearing Down Ngrok Tunnel")
    ngrok.disconnect()



app = FastAPI(lifespan=lifespan)

app.mount("/mesh/public", StaticFiles(directory="../web/mesh/public"), name="bvh_files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
    expose_headers=["ngrok-skip-browser-warning"]
)

generator = BVHGenerator()


@app.post("/generate-motion")
def generate_motion(request=Body(...)):
    print(request)
    text_prompt = "|".join(request['text_prompt'])
    print(text_prompt)

    try:
        OUTPUT_DIR = "generation/generation_name_nopredlen/animations/0"
        result = generator.create_bvh_from_in(text_prompt)

        if result is True:
            # ✅ Get only the latest .bvh file using max() on an iterator
            print("BVH SEARCH PATH:", OUTPUT_DIR)
            latest_bvh = max(
                glob.iglob(os.path.join(OUTPUT_DIR, "*.bvh")),
                key=os.path.getmtime,
                default=None
            )

            if latest_bvh:
                NEW_BVH_DIR = "../web/mesh/public"
                os.makedirs(NEW_BVH_DIR, exist_ok=True)
                new_bvh_path = os.path.join(NEW_BVH_DIR, os.path.basename(latest_bvh))
                shutil.copyfile(latest_bvh, new_bvh_path)

                print(f"✅ Copied latest BVH: {os.path.basename(latest_bvh)}")
                return {"success": True, "filenames": os.path.basename(latest_bvh)}
            else:
                return {
                    "success": True,
                    "message": "Motion generation successful, but no .bvh file found.",
                    "bvh_file": None
                }
        else:
            return {
                "success": False,
                "message": "Error running the script.",
                "stderr": result.stderr
            }

    except Exception as e:
        print("❌ INTERNAL SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-motion-inbetween")
def add_motion_inbetween(request=Body(...)):
    print(request)
    text_prompt = "|".join(request['text_prompt'])
    insert_time = request.get('insert_time', 2)
    print(text_prompt)
    print("Insert Time:", insert_time)
    try:
        OUTPUT_DIR = "generation/generation_name_nopredlen/animations/0"
        result = generator.add_motion_inbetween(text_prompt , insert_time)

        if result is True:
            # ✅ Get only the latest .bvh file using max() on an iterator
            print("BVH SEARCH PATH:", OUTPUT_DIR)
            latest_bvh = max(
                glob.iglob(os.path.join(OUTPUT_DIR, "*.bvh")),
                key=os.path.getmtime,
                default=None
            )
            latest_bvh = "generation/generation_name_nopredlen/animations/0/sample_between.bvh"
            if latest_bvh:
                NEW_BVH_DIR = "../web/mesh/public"
                os.makedirs(NEW_BVH_DIR, exist_ok=True)
                new_bvh_path = os.path.join(NEW_BVH_DIR, os.path.basename(latest_bvh))
                shutil.copyfile(latest_bvh, new_bvh_path)

                print(f"✅ Copied latest BVH: {os.path.basename(latest_bvh)}")
                return {"success": True, "filenames": os.path.basename(latest_bvh)}
            else:
                return {
                    "success": True,
                    "message": "Motion generation successful, but no .bvh file found.",
                    "bvh_file": None
                }
        else:
            return {
                "success": False,
                "message": "Error running the script.",
                "stderr": result.stderr
            }

    except Exception as e:
        print("❌ INTERNAL SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))