from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import os
import shutil
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
        logging.info(f"Created directory: {processed_videos_dir}")
    return processed_videos_dir

# Endpoint for video upload and subtitle removal
@app.post("/remove_subtitles/")
async def remove_subtitles(file: UploadFile = File(...), sub_area: Optional[str] = None, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Endpoint to remove subtitles from an uploaded video. Optionally, a sub_area can be specified for subtitle region.
    """
    original_filename = file.filename
    print(f"File received: {original_filename}")
    print(f"Sub Area: {sub_area}")

    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_video_path = temp_file.name

    # Use the original filename to create a status file
    status_file = Path(ensure_processed_videos_dir()) / f"{original_filename}.status"  # Status file to track progress
    
    # Initialize the status file
    with open(status_file, 'w') as status:
        status.write("Processing...")

    # Process the video in the background (for large video files)
    background_tasks.add_task(process_video, temp_video_path, status_file, sub_area, original_filename)

    processed_video_filename = f"processed_{original_filename}"
    download_url = f"/download_video/{processed_video_filename}"
    return {"message": "Video received. Begin subtitle removal.", "filename": original_filename, "download at": download_url}

# Function to handle video processing in the background
def process_video(video_path: str, status_file: Path, sub_area: Optional[str], original_filename: str):
    logging.info(f"Starting subtitle removal for {original_filename}...")

    # Initialize the status file as "Processing..."
    with open(status_file, 'w') as file:
        file.write("Processing...")

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

    processed_video_path = Path(ensure_processed_videos_dir()) / f"processed_{original_filename}"

    try:
        # Copy the processed video path in self to the target directory
        output_path = Path(sd.video_out_name)

        # Copy the actual processed video instead
        shutil.copy2(output_path, processed_video_path)
        logging.info(f"Subtitle removal completed for {original_filename}. Processed video saved at {processed_video_path}.")
        print(f"Subtitle removal completed for {video_path}.")
        os.remove(video_path)  # Remove the temporary video file
    except Exception as e:
        logging.error(f"Error during file copy or removal: {e}")
        print(f"Error during file copy or removal: {e}")

# Endpoint to track the status of removal process
@app.get("/status/{video_filename}")
async def video_status(video_filename: str):
    status_file = Path(ensure_processed_videos_dir()) / f"{video_filename}.status"
    
    logging.info(f"Checking status for file: {status_file}")

    if not status_file.exists():
        return {"status": "Video Not Found"}

    with open(status_file, 'r') as file:
        status = file.read().strip()

    if status == "Processing...":
        return {"status": "Processing"}
    elif status == "Completed":
        return {"status": "Completed"}
    else:
        return {"status": "Unknown Status"}

# Endpoint to fetch the processed video
@app.get("/download_video/{video_filename}")
async def download_video(video_filename: str):
    # Ensure the processed_videos directory exists
    processed_videos_dir = ensure_processed_videos_dir()
    
    # Construct the full path to the processed video (with 'processed_' prefix)
    video_path = processed_videos_dir / video_filename
    
    # Check if the processed video file exists
    if video_path.exists():
        # Return the processed video file to the user as a download
        return FileResponse(video_path, media_type="video/mp4", filename=video_filename)
    else:
        # Return an error message if the processed video file does not exist
        return {"error": "Processed video file not found!"}