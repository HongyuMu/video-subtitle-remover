from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging
import os
import cv2
import shutil
import tempfile
from backend.main import SubtitleRemover, SubtitleDetect
from typing import Optional
import uvicorn
# from dify_plugin import Plugin, DifyPluginEnv
# from plugin.tools.sub_remover import SubRemoverTool
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from FastAPI.user_models import get_db
# from FastAPI.auth import register_user, validate_api_key

app = FastAPI()

# plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))
# plugin.add_tool(SubRemoverTool)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# class UserCreate(BaseModel):
#     username: str

# @app.post("/register/")
# async def register(user: UserCreate):
#     try:
#         new_user = register_user(user.username)
#         return {"message": "User registered", "api_key": new_user["api_key"]}
#     except Exception:
#         raise HTTPException(status_code=400, detail="User registration failed")

@app.get("/")
async def root():
    return {"Message": "Visit docs to remove subtitles in your videos!"}

def ensure_processed_videos_dir():
    processed_videos_dir = Path(__file__).resolve().parent / "processed_videos"
    if not processed_videos_dir.exists():
        os.makedirs(processed_videos_dir)
        logging.info(f"Created directory: {processed_videos_dir}")
    return processed_videos_dir

@app.post("/find_subtitles/")
async def find_subtitles(file: UploadFile = File(...), api_key: str = None):
    """
    Endpoint to find subtitles in an uploaded video. Returns subtitle frame numbers and coordinates.
    """
    # if not api_key or not validate_api_key(api_key):
    #     raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    original_filename = file.filename
    print(f"File received: {original_filename}")
    
    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_video_path = temp_file.name

    try:
        subtitle_detect = SubtitleDetect(video_path=temp_video_path)
        
        # extract subtitle frames and areas
        subtitle_frame_no_box_dict = subtitle_detect.find_subtitle_frame_no()
        if not subtitle_frame_no_box_dict:
            raise HTTPException(status_code=404, detail="No subtitles found in the video.")
        
        # Unify similar regions across frames
        unified_sub_dict = subtitle_detect.unify_regions(subtitle_frame_no_box_dict)
        complete_subtitle_frame_no_box_dict = subtitle_detect.prevent_missed_detection(unified_sub_dict)

        # Filter out incorrect subtitle areas based on frequency
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        correct_subtitle_frame_no_box_dict = subtitle_detect.filter_mistake_sub_area(complete_subtitle_frame_no_box_dict, fps)

        # The first entry returns the detected subtitle
        first_entry_dict = {frame_no: boxes[0] for frame_no, boxes in correct_subtitle_frame_no_box_dict.items() if boxes}
        sub_frame_no_list_continuous = subtitle_detect.find_continuous_ranges_with_same_mask(first_entry_dict)

        clustered_output = dict()
        for interval in sub_frame_no_list_continuous:
            clustered_output[interval] = first_entry_dict[interval[0]]

        # Return final results after all processing steps
        return {
            "message": "Subtitles found successfully.",
            "subtitle_frames": first_entry_dict,
            "continuous_frame_intervals": sub_frame_no_list_continuous
        }
    
    except Exception as e:
        logging.error(f"Error during subtitle detection: {e}")
        return {"error": "Failed to detect subtitles."}
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except PermissionError as e:
                logging.error(f"Error deleting temp file: {e}")
                pass

# Endpoint for video upload and subtitle removal
@app.post("/remove_subtitles/")
async def remove_subtitles(file: UploadFile = File(...), sub_area: Optional[str] = None, api_key: str = None, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Endpoint to remove subtitles from an uploaded video. Optionally, a sub_area can be specified for subtitle region.
    """
    # if not api_key or not validate_api_key(api_key):
    #     raise HTTPException(status_code=403, detail="Invalid or missing API key")
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
    try:
        logging.info(f"Starting subtitle removal for {original_filename}...")

        # tool_parameters = {
        #     "video_path": video_path,
        #     "sub_area": sub_area
        # }
        
        # Create an instance of the tool
        # tool = SubRemoverTool()

        # # Invoke the tool
        # result = next(tool._invoke(tool_parameters))
        # logging.info(f"Result from tool: {result['result']}")

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

            if output_path.exists():
                shutil.copy2(output_path, processed_video_path)
                logging.info(f"Subtitle removal completed for {original_filename}. Processed video saved at {processed_video_path}.")
                print(f"Subtitle removal completed for {video_path}.")
                os.remove(video_path)  # Remove the temporary video file
            else:
                logging.error(f"Processed video not found at {output_path}.")
                raise FileNotFoundError(f"Processed video not found: {output_path}")
        except Exception as e:
            logging.error(f"Error during file copy: {e}")
            print(f"Error during file copy: {e}")

    except Exception as e:
        logging.error(f"Error during subtitle removal process for {original_filename}: {e}")
        print(f"Error during subtitle removal process: {e}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

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
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
#     plugin.run()