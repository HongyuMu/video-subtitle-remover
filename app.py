from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import os
import json
import shutil
import tempfile
from backend.main import SubtitleRemover, SubtitleDetect
from typing import Optional
import uvicorn
import cv2
import uuid
import aiohttp
import asyncio
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"Message": "Visit docs to remove subtitles in your videos!"}

PROCESSED_DIR = Path("processed_videos")
PROCESSED_DIR.mkdir(exist_ok=True)
PROCESSED_FILES_DIR = Path("processed_files")
PROCESSED_FILES_DIR.mkdir(exist_ok=True)

TASK_RESULTS = {}

def save_temp_file(upload_file: UploadFile, suffix=".mp4"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload_file.file.read())
        return temp_file.name

async def download_file(url: str, dest_path: str):
    # Make a request to the url
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download file: {url}")
            with open(dest_path, "wb") as f:
                # Write the response to the local file
                while True:
                    chunk = await resp.content.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

# Call the SubtitleDetect class functions to find subtitles
@app.post("/find_subtitles/")
async def find_subtitles(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,
    cloud_ref: Optional[str] = None):
    # Use the original filename (without extension) for the output JSON
    original_name = Path(file.filename).stem if file else "unknown"
    temp_video_path = None

    if url:
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(url, temp_video_path)
    elif cloud_ref:
        # Assuming cloud_ref is a URL or a cloud storage path
        # For simplicity, we'll just use the original name as a placeholder for now
        # In a real scenario, you'd download from cloud_ref to temp_video_path
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(cloud_ref, temp_video_path)
    else:
        temp_video_path = save_temp_file(file)

    try:
        # Detect subtitle locations and intervals
        subtitle_detect = SubtitleDetect(video_path=temp_video_path)
        subtitle_frame_no_box_dict = subtitle_detect.find_subtitle_frame_no()
        if not subtitle_frame_no_box_dict:
            raise HTTPException(status_code=404, detail="No subtitles found in the video.")

        unified_sub_dict = subtitle_detect.unify_regions(subtitle_frame_no_box_dict)
        complete_subtitle_frame_no_box_dict = subtitle_detect.prevent_missed_detection(unified_sub_dict)
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Filter out mistake subtitle areas by checking the fps
        correct_subtitle_frame_no_box_dict = subtitle_detect.filter_mistake_sub_area(complete_subtitle_frame_no_box_dict, fps)

        # Get the first entry of each subtitle area as the true subtitle
        first_entry_dict = {frame_no: boxes[0] for frame_no, boxes in correct_subtitle_frame_no_box_dict.items() if boxes}
        sub_frame_no_list_continuous = subtitle_detect.find_continuous_ranges_with_same_mask(first_entry_dict)
        distinct_coords = [
            first_entry_dict[elapse[0]] if elapse[0] in first_entry_dict else None
            for elapse in sub_frame_no_list_continuous
        ]
        json_content = {
            "distinct_coordinates": distinct_coords,
            "frame_intervals": sub_frame_no_list_continuous
        }
        # Save as original_filename_sub.json
        json_file_path = PROCESSED_FILES_DIR / f"{original_name}_sub.json"
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file, indent=4)

        # Store results
        task_id = str(uuid.uuid4())
        TASK_RESULTS[task_id] = {
            "distinct_coords": distinct_coords,
            "frame_intervals": sub_frame_no_list_continuous,
            "original_filename": original_name,
            "video_path": temp_video_path  # Make sure to keep this file until user is done!
        }
        return {"task_id": task_id}
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# Draw subtitle boxes on the video for users to visualize and adjust later
def draw_subtitle_boxes(frame, distinct_coords, frame_intervals, current_frame_idx):
    """
    Draws rectangles for all subtitle regions active at the current frame.
    """
    for coord, (start, end) in zip(distinct_coords, frame_intervals):
        if coord is None:
            continue
        if start <= current_frame_idx <= end:
            xmin, xmax, ymin, ymax = coord  # adjust order if needed
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return frame

@app.get("/show_subtitle_box/{task_id}")
async def show_subtitle_box(task_id: str, frame_idx: int = 0):
    """
    Returns a single video frame with subtitle boxes drawn, for the given frame index.
    """
    # Retrieve the result from TASK_RESULTS
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    video_path = result.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    distinct_coords = result["distinct_coords"]
    frame_intervals = result["frame_intervals"]

    # Open the video and get the requested frame
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise HTTPException(status_code=400, detail=f"frame_idx must be between 0 and {total_frames-1}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=404, detail="Frame not found")

    # Draw the boxes
    frame_with_boxes = draw_subtitle_boxes(frame, distinct_coords, frame_intervals, frame_idx)

    # Encode as PNG for web display
    _, buffer = cv2.imencode('.png', frame_with_boxes)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@app.post("/remove_subtitles/")
async def remove_subtitles(
    file: UploadFile = File(...),
    json_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()):
    temp_video_path = save_temp_file(file)
    temp_json_path = save_temp_file(json_file, suffix=".json")
    # Use the original filename (without extension) for output and status files
    original_stem = Path(file.filename).stem
    processed_video_path = PROCESSED_DIR / f"processed_{original_stem}.mp4"
    status_file = PROCESSED_DIR / f"{original_stem}.status"

    # Start background task, pass original_stem for naming
    background_tasks.add_task(
        process_video,
        temp_video_path,
        temp_json_path,
        processed_video_path,
        status_file
    )
    return {
        "message": "Video and JSON received. Subtitle removal started.",
        "status_url": f"/status/{status_file.name}",
        "download_url": f"/download_video/{processed_video_path.name}"
    }

# Call the SubtitleRemover class to remove subtitles
def process_video(video_path, json_path, output_path, status_file):
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        coords = json_data.get("distinct_coordinates")
        intervals = json_data.get("frame_intervals")
        sd = SubtitleRemover(video_path, distinct_coords=coords, frame_intervals=intervals)       
        sd.run()
        shutil.copy2(sd.video_out_name, output_path)
        with open(status_file, 'w') as f:
            f.write("Completed")
    except Exception as e:
        with open(status_file, 'w') as f:
            f.write(f"Error: {e}")
    finally:
        for path in [video_path, json_path]:
            if os.path.exists(path):
                os.remove(path)


@app.get("/status/{status_filename}")
async def get_status(status_filename: str):
    status_path = PROCESSED_DIR / status_filename
    if not status_path.exists():
        return JSONResponse(content={"status": "Not Found"})
    with open(status_path, 'r') as f:
        status = f.read().strip()
    return JSONResponse(content={"status": status})


@app.get("/download_video/{video_filename}")
async def download_video(video_filename: str):
    video_path = PROCESSED_DIR / video_filename
    if not video_path.exists():
        return JSONResponse(content={"error": "Processed video file not found!"})
    return FileResponse(video_path, media_type="video/mp4", filename=video_filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)