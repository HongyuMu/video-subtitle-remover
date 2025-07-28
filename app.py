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
import time
import psutil
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"Message": "Visit docs to remove subtitles in your videos!"}

# Use absolute paths to avoid working directory issues
PROCESSED_DIR = Path(os.getcwd()) / "processed_videos"
PROCESSED_DIR.mkdir(exist_ok=True)
PROCESSED_FILES_DIR = Path(os.getcwd()) / "processed_files"
PROCESSED_FILES_DIR.mkdir(exist_ok=True)

TASK_RESULTS = {}

def check_memory_usage():
    """Check current memory usage and warn if approaching limits"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Assume 1024MB limit
        memory_limit_mb = 1024
        memory_percentage = (memory_mb / memory_limit_mb) * 100
        
        print(f"Memory usage: {memory_mb:.1f}MB ({memory_percentage:.1f}%)")
        
        if memory_percentage > 80:
            print(f"WARNING: Memory usage at {memory_percentage:.1f}%")
            return False
        return True
    except Exception:
        # If psutil not available, continue
        return True

def cleanup_old_tasks():
    """Clean up old tasks and their associated files to prevent memory leaks"""
    import time
    current_time = time.time()
    # Keep tasks for 30 minutes (3600 seconds) - shorter for memory efficiency
    max_age = 3600
    
    tasks_to_remove = []
    for task_id, result in TASK_RESULTS.items():
        # Check if task has a timestamp (add one if not present)
        if 'timestamp' not in result:
            result['timestamp'] = current_time
        elif current_time - result['timestamp'] > max_age:
            tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        result = TASK_RESULTS.pop(task_id)
        # Clean up the video file
        video_path = result.get('video_path')
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Cleaned up old task: {task_id}")
            except Exception:
                pass  # Ignore cleanup errors

def save_temp_file(upload_file: UploadFile, suffix=".mp4"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload_file.file.read())
        return temp_file.name

async def download_file(url: str, dest_path: str, max_size_mb: int = 50):
    """Download file with size limit for Coze compatibility"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download file: {url}")
            
            # Check content length if available
            content_length = resp.headers.get('content-length')
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                raise Exception(f"File too large: {int(content_length) // (1024*1024)}MB > {max_size_mb}MB")
            
            with open(dest_path, "wb") as f:
                downloaded_size = 0
                while True:
                    chunk = await resp.content.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    downloaded_size += len(chunk)
                    if downloaded_size > max_size_mb * 1024 * 1024:
                        raise Exception(f"File too large: {downloaded_size // (1024*1024)}MB > {max_size_mb}MB")
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
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(cloud_ref, temp_video_path)
    else:
        temp_video_path = save_temp_file(file)

    try:
        # Check memory before processing
        if not check_memory_usage():
            raise HTTPException(status_code=503, detail="Server memory limit reached. Please try again later.")
        
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
            "video_path": temp_video_path,
            "timestamp": time.time()
        }
        return {"task_id": task_id}

    except Exception as e:
        print("Error in find_subtitles: ", e)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e


@app.get("/subtitle_intervals/{task_id}")
async def get_subtitle_intervals(task_id: str):
    """
    Returns all intervals and their current rectangles for a given task_id.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    intervals = []
    distinct_coords = result.get("distinct_coords", [])
    frame_intervals = result.get("frame_intervals", [])
    # Optionally, track which intervals have been edited (not implemented yet)
    for idx, (coords, frame_range) in enumerate(zip(distinct_coords, frame_intervals)):
        intervals.append({
            "interval_idx": idx,
            "frame_range": frame_range,
            "coords": coords,
            # "edited": False  # Could be added if you want to track edits
        })
    return {"intervals": intervals}


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
async def show_subtitle_box(
    task_id: str, 
    frame_idx: int = 0,
    edit_mode: bool = False,
    interval_idx: int = None,
    xmin: int = None,
    xmax: int = None,
    ymin: int = None,
    ymax: int = None
):
    """
    Returns a single video frame with subtitle boxes drawn, for the given frame index.
    Supports interactive editing of bounding boxes by interval indices.
    """
    # Clean up old tasks first
    cleanup_old_tasks()
    
    # Retrieve the result from TASK_RESULTS
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    video_path = result.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    distinct_coords = result["distinct_coords"]
    frame_intervals = result["frame_intervals"]

    # Check memory before loading video
    if not check_memory_usage():
        raise HTTPException(status_code=503, detail="Server memory limit reached. Please try again later.")
    
    # Handle coordinate updates if in edit mode
    if edit_mode and interval_idx is not None and all(v is not None for v in [xmin, xmax, ymin, ymax]):
        if 0 <= interval_idx < len(distinct_coords):
            # Update the coordinates for the specific interval
            distinct_coords[interval_idx] = (xmin, xmax, ymin, ymax)
            # Update the stored result
            TASK_RESULTS[task_id]["distinct_coords"] = distinct_coords
            print(f"Updated coordinates for interval {interval_idx}: ({xmin}, {xmax}, {ymin}, {ymax})")
    
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


class EditSubtitleBoxRequest(BaseModel):
    interval_idx: int
    coords: list  # [xmin, xmax, ymin, ymax]

@app.post("/edit_subtitle_box/{task_id}")
async def edit_subtitle_box(task_id: str, req: EditSubtitleBoxRequest):
    """
    Update the rectangle for a specific interval in distinct_coords.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    distinct_coords = result.get("distinct_coords", [])
    if not (0 <= req.interval_idx < len(distinct_coords)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")
    # Update the rectangle
    distinct_coords[req.interval_idx] = req.coords
    # Optionally, mark as edited
    # result.setdefault("edited_intervals", set()).add(req.interval_idx)
    TASK_RESULTS[task_id]["distinct_coords"] = distinct_coords
    return {"message": f"Interval {req.interval_idx} updated.", "coords": req.coords}


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