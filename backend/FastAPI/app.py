from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import time
from tqdm import tqdm
from backend.tools.inpaint_tools import create_mask
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.sttn_inpaint import STTNInpaint
from backend.inpaint.video_inpaint import VideoInpaint
from backend.scenedetect import scene_detect
from backend.tools.common_tools import is_video_or_image
from backend.scenedetect.detectors import ContentDetector
from backend.main import SubtitleRemover
from typing import Optional

app = FastAPI()

# Endpoint for video upload and subtitle removal
@app.post("/remove_subtitles/")
async def remove_subtitles(file: UploadFile = File(...), sub_area: Optional[str] = None, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Endpoint to remove subtitles from an uploaded video. Optionally, a sub_area can be specified for subtitle region.
    """
    # Save the uploaded video temporarily
    print(f"File received: {file.filename}")
    print(f"Sub Area: {sub_area}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_video_path = temp_file.name

    # Process the video in the background (for large video files)
    background_tasks.add_task(process_video, temp_video_path, sub_area)

    return {"message": "Video received. Subtitle removal is in progress."}

# Function to handle video processing in the background
def process_video(video_path: str, sub_area: Optional[str]):
    video_path = Path(video_path)

    # Set subtitle area if provided (Convert to tuple if given)
    if sub_area:
        ymin, ymax, xmin, xmax = map(int, sub_area.split(","))
        sub_area_tuple = (ymin, ymax, xmin, xmax)
    else:
        sub_area_tuple = None

    # Create SubtitleRemover object
    sd = SubtitleRemover(video_path, sub_area=sub_area_tuple)
    sd.run()  # Run subtitle removal

    print(f"Subtitle removal completed for {video_path}")

    # Optionally, you can return or store the final result in a specific location.

# Endpoint to fetch the processed video
@app.get("/download_video/{video_filename}")
async def download_video(video_filename: str):
    video_path = Path(f"./processed_videos/{video_filename}")
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4", filename=video_filename)
    else:
        return {"error": "Video file not found!"}

# Additional utility endpoint to check video format (for validation)
@app.get("/check_video/{video_filename}")
async def check_video(video_filename: str):
    video_path = Path(f"./uploaded_videos/{video_filename}")
    if video_path.exists():
        if is_video_or_image(str(video_path)):
            return {"message": f"{video_filename} is a valid video!"}
        else:
            return {"message": f"{video_filename} is not a valid video!"}
    else:
        return {"error": "Video file not found!"}