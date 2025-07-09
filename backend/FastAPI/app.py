from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import os

import tempfile
from backend.tools.common_tools import is_video_or_image
from backend.main import SubtitleRemover
from typing import Optional

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.get("/")
async def root():
    return {"Message": "Visit docs to remove subtitles in your videos!"}

def ensure_processed_videos_dir():
    processed_videos_dir = Path(__file__).resolve().parent / "processed_videos"
    if not processed_videos_dir.exists():
        os.makedirs(processed_videos_dir)
    return processed_videos_dir

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

    video_filename = Path(temp_video_path).name
    status_file = Path(ensure_processed_videos_dir()) / f"{video_filename}.status"  # Status file to track progress
    
    # Initialize the status file
    with open(status_file, 'w') as file:
        file.write("Processing...")

    # Process the video in the background (for large video files)
    background_tasks.add_task(process_video, temp_video_path, status_file, sub_area)

    return {"message": "Video received. Subtitle removal is in progress."}

# Function to handle video processing in the background
def process_video(video_path: str, status_file: Path, sub_area: Optional[str]):
    video_filename = Path(video_path).name
    logging.info(f"Starting subtitle removal for {video_filename}...")
    ensure_processed_videos_dir()

    # Set subtitle area if provided
    if sub_area:
        ymin, ymax, xmin, xmax = map(int, sub_area.split(","))
        sub_area_tuple = (ymin, ymax, xmin, xmax)
    else:
        sub_area_tuple = None

    # Create SubtitleRemover object and run the subtitle removal
    sd = SubtitleRemover(video_path, sub_area=sub_area_tuple)
    sd.run()  # Run subtitle removal

    # After the process is done, update the status to "Completed"
    with open(status_file, 'w') as file:
        file.write("Completed")

    logging.info(f"Subtitle removal completed for {video_filename}.")
    print(f"Subtitle removal completed for {video_path}.")

# Endpoint to track the status of removal process
@app.get("/status/{video_filename}")
async def video_status(video_filename: str):
    status_file = Path(ensure_processed_videos_dir()) / f"{video_filename}.status"
    
    # Check if the status file exists and read its contents
    if status_file.exists():
        with open(status_file, 'r') as file:
            status = file.read()
        return {"status": status}
    else:
        return {"status": "Video is being processed or not found."}

# Endpoint to fetch the processed video
@app.get("/download_video/{video_filename}")
async def download_video(video_filename: str):
    video_path = Path(ensure_processed_videos_dir()) / video_filename
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