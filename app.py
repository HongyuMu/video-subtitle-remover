from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from pathlib import Path
import os
import json
import shutil
import tempfile
import multiprocessing
from backend.main import SubtitleExtractor
from backend.tools import subtitle_ocr

from numpy._core.numeric import False_
from backend.main import SubtitleRemover, SubtitleDetect
import backend.config as config
from typing import Optional, List
import uvicorn
import cv2
import uuid
import aiohttp
import asyncio
import io
import time
import psutil
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from backend.main import find_smallest_bounding_box

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
    """Check current system memory usage and warn if it's high."""
    try:
        # Get system-wide memory usage
        virtual_memory = psutil.virtual_memory()
        memory_percent = virtual_memory.percent
        
        print(f"System memory usage: {memory_percent:.1f}%")
        
        if memory_percent > 90:
            print(f"WARNING: System memory usage is high at {memory_percent:.1f}%")
        
        return memory_percent < 90
            
    except Exception as e:
        # If psutil not available or fails, just print a note and continue
        print(f"Could not check system memory: {e}")
        return True

def cleanup_tasks(user_id: Optional[str] = None, all_old: bool = False):
    """
    Cleans up tasks from the TASK_RESULTS dictionary.
    - If user_id is provided, removes all tasks for that user.
    - If all_old is True, removes all tasks older than the max_age.
    """
    tasks_to_remove = []
    if user_id:
        tasks_to_remove = [
            task_id for task_id, result in TASK_RESULTS.items()
            if result.get("user_id") == user_id
        ]
    elif all_old:
        current_time = time.time()
        max_age = 1800  # 30 minutes
        tasks_to_remove = [
            task_id for task_id, result in TASK_RESULTS.items()
            if current_time - result.get('timestamp', current_time) > max_age
        ]

    cleaned_files = []
    for task_id in tasks_to_remove:
        result = TASK_RESULTS.pop(task_id, None)
        if result:
            original_stem = result.get("original_filename", "unknown")
            # Define all possible temporary files
            paths_to_check = [
                result.get('video_path'),
                PROCESSED_DIR / f"processed_{original_stem}.mp4",
                PROCESSED_DIR / f"{original_stem}.status",
                PROCESSED_FILES_DIR / f"{original_stem}_sub.json"
            ]
            for path in paths_to_check:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        cleaned_files.append(str(path))
                    except OSError as e:
                        print(f"Error removing file {path}: {e}")
    
    return len(tasks_to_remove), cleaned_files

@app.post("/cleanup/{user_id}")
async def manual_cleanup(user_id: str):
    """
    Manually triggers cleanup for a specific user's tasks.
    """
    print(f"Starting cleanup for user: {user_id}")
    cleaned_count, cleaned_files = cleanup_tasks(user_id=user_id)
    
    if cleaned_count > 0:
        response_message = f"Successfully cleaned up {cleaned_count} tasks and associated files for user {user_id}."
        print(response_message)
        if cleaned_files:
            print("Removed files:")
            for f in cleaned_files:
                print(f" - {f}")
        return {"message": response_message, "cleaned_files": cleaned_files}
    else:
        response_message = f"No tasks found to clean up for user {user_id}."
        print(response_message)
        return {"message": response_message}


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
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,
    cloud_ref: Optional[str] = None,
    user_id: Optional[str] = None):
    
    original_name = Path(file.filename).stem if file else "unknown_file"
    temp_video_path = None

    if user_id is None:
        user_id = str(uuid.uuid4())

    try:
        if url:
            temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
            await download_file(url, temp_video_path)
            if original_name == "unknown_file":
                original_name = Path(url).stem
        elif cloud_ref:
            temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
            await download_file(cloud_ref, temp_video_path)
            if original_name == "unknown_file":
                original_name = Path(cloud_ref).stem
        elif file:
            temp_video_path = save_temp_file(file)
        else:
            raise HTTPException(status_code=400, detail="No video file or URL provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process input video: {e}")

    # Create a unique ID for this detection task
    task_id = str(uuid.uuid4())
    status_file = PROCESSED_DIR / f"detect_{task_id}.status"

    # Store a placeholder to indicate the task is running
    TASK_RESULTS[task_id] = {
        "status": "Detecting",
        "timestamp": time.time(),
        "user_id": user_id,
        "original_filename": original_name,
        "status_file": str(status_file),
        "video_path": temp_video_path, # Store path for cleanup
    }

    # Start the long-running detection process in the background
    background_tasks.add_task(
        detect_subtitles_task,
        task_id=task_id,
        temp_video_path=temp_video_path,
        status_file=str(status_file)
    )

    return {
        "message": "Subtitle detection started.",
        "task_id": task_id
    }


@app.get("/subtitle_intervals/{task_id}", include_in_schema=False)
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
    
    # Getting the intervals and their current rectangles in base 0
    for idx, (coords, frame_range) in enumerate(zip(distinct_coords, frame_intervals)):
        intervals.append({
            "interval_idx": idx,
            "frame_range": frame_range,
            "coords": coords,
        })
    return {"intervals": intervals}


# Draw subtitle boxes on the video for users to visualize and adjust later
def draw_subtitle_boxes(frame, distinct_coords, frame_intervals, current_frame_idx):
    """
    Draws rectangles for all subtitle regions active at the current frame.
    Expands the box by 10 pixels to ensure full coverage.
    """
    frame_height, frame_width, _ = frame.shape
    for coord, (start, end) in zip(distinct_coords, frame_intervals):
        if coord is None:
            continue
        if start <= current_frame_idx <= end:
            xmin, xmax, ymin, ymax = coord
            
            # Expand by 10 pixels in each direction, clamping to video dimensions
            expanded_xmin = max(0, xmin - 10)
            expanded_xmax = min(frame_width, xmax + 10)
            expanded_ymin = max(0, ymin - 10)
            expanded_ymax = min(frame_height, ymax + 10)
            
            cv2.rectangle(frame, (expanded_xmin, expanded_ymin), (expanded_xmax, expanded_ymax), (0, 255, 0), 2)
    return frame

@app.get("/show_subtitle_box/{task_id}", include_in_schema=False)
async def show_subtitle_box(
    task_id: str, 
    frame_idx: int = 0,
    draw_box: bool = True):
    """
    Returns a single video frame. 
    If draw_box is true, it includes the subtitle boxes.
    """
    # Clean up old tasks first
    cleanup_tasks(all_old=True)
    
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

    # Draw the boxes only if requested by the client
    if draw_box:
        frame_with_boxes = draw_subtitle_boxes(frame, distinct_coords, frame_intervals, frame_idx)
    else:
        frame_with_boxes = frame

    # Encode as PNG for web display
    _, buffer = cv2.imencode('.png', frame_with_boxes)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


class AdjustSubtitleBoxRequest(BaseModel):
    interval_idx: int
    x: Optional[int] = Field(None, description="New top-left x-coordinate of the box")
    y: Optional[int] = Field(None, description="New top-left y-coordinate of the box")
    width: Optional[int] = Field(None, description="New width of the box")
    height: Optional[int] = Field(None, description="New height of the box")

@app.post("/adjust_box/{task_id}", include_in_schema=False)
async def adjust_box(task_id: str, req: AdjustSubtitleBoxRequest):
    """
    Adjust a subtitle box for a specific interval using slider-like properties (x, y, width, height).
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    video_width = result.get("video_width")
    video_height = result.get("video_height")
    if not video_width or not video_height:
        raise HTTPException(status_code=400, detail="Video dimensions not found for this task.")

    distinct_coords = result.get("distinct_coords", [])
    if not (0 <= req.interval_idx < len(distinct_coords)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    # Get current box properties
    xmin, xmax, ymin, ymax = distinct_coords[req.interval_idx]
    current_x, current_y = xmin, ymin
    current_width, current_height = xmax - xmin, ymax - ymin

    # Use new values if provided, otherwise keep current values
    new_x = req.x if req.x is not None else current_x
    new_y = req.y if req.y is not None else current_y
    new_width = req.width if req.width is not None else current_width
    new_height = req.height if req.height is not None else current_height

    # Validate that the new box is within video boundaries
    if not (0 <= new_x < video_width and 0 <= new_y < video_height):
        raise HTTPException(status_code=400, detail="Box coordinates must be within the video frame.")
    if not (new_x + new_width <= video_width and new_y + new_height <= video_height):
        raise HTTPException(status_code=400, detail="Box dimensions exceed video boundaries.")

    # Update coordinates
    new_coords = (new_x, new_x + new_width, new_y, new_y + new_height)
    distinct_coords[req.interval_idx] = new_coords
    TASK_RESULTS[task_id]["distinct_coords"] = distinct_coords

    return {
        "message": f"Interval {req.interval_idx} adjusted successfully.",
        "new_coords": new_coords
    }


@app.get("/editor/{task_id}", response_class=HTMLResponse, include_in_schema=False)
async def get_editor(task_id: str, request: Request, format: Optional[str] = None):
    """
    Serve the HTML editor page for a given task.
    - By default, serves HTML to browser clients (who send an `Accept: text/html` header).
    - Use the query parameter `?format=json` to force a JSON response with the editor URL.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        return HTMLResponse(content="<h1>Task not found</h1>", status_code=404)
    
    # Allow forcing JSON response via query parameter
    if format and format.lower() == 'json':
        editor_url = str(request.url_for('get_editor', task_id=task_id))
        return JSONResponse(content={
            "message": "Task is ready for editing.",
            "editor_url": editor_url
        })
    
    # Check the Accept header for browser-like requests
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        # Browser client, serve the HTML page
        return templates.TemplateResponse("editor.html", {"request": request, "task_id": task_id})
    else:
        # API client, return a JSON response with the link
        editor_url = str(request.url_for('get_editor', task_id=task_id))
        return JSONResponse(content={
            "message": "Task is ready for editing.",
            "editor_url": editor_url
        })


# Sends the task info to the editor page (video dimensions, filename, user_id)
@app.get("/task_info/{task_id}", include_in_schema=True)
async def get_task_info(task_id: str, request: Request):
    """
    Returns metadata for a given task, including video dimensions.
    Also used for polling detection status.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = result.get("status")

    if status == "Detecting":
        status_file_path = result.get("status_file")
        if status_file_path and os.path.exists(status_file_path):
            try:
                with open(status_file_path, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                pass # Fallback if file is empty or corrupt
        return {"status": "Detecting...", "progress": 0}

    elif status == "Completed":
        return {
            "status": "Completed",
            "video_width": result.get("video_width"),
            "video_height": result.get("video_height"),
            "original_filename": result.get("original_filename"),
            "user_id": result.get("user_id"),
            "fps": result.get("fps"),
            "total_frames": result.get("total_frames"),
            "editor_url": str(request.url_for('get_editor', task_id=task_id))
        }

    elif status == "Error":
        return {"status": "Error", "message": result.get("error", "An unknown error occurred.")}
    
    return {"status": "Unknown"}


# 
@app.post("/process_task/{task_id}", include_in_schema=False)
async def process_task(task_id: str, background_tasks: BackgroundTasks):
    """
    Starts the subtitle removal process for a task that has been edited.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    video_path = result.get("video_path")
    original_stem = result.get("original_filename", "unknown")
    
    # Create a temporary JSON file with the latest coordinates
    temp_json_path = f"/tmp/{uuid.uuid4()}.json"
    with open(temp_json_path, "w") as f:
        json.dump({
            "distinct_coordinates": result.get("distinct_coords"),
            "frame_intervals": result.get("frame_intervals")
        }, f, indent=4)

    processed_video_path = PROCESSED_DIR / f"processed_{original_stem}.mp4"
    status_file = PROCESSED_DIR / f"{original_stem}.status"

    background_tasks.add_task(
        process_video,
        video_path,
        temp_json_path,
        processed_video_path,
        status_file
    )
    print(f"[PROCESS] Started background task for {original_stem}. status_file={status_file}, output={processed_video_path}")
    
    return {
        "message": "Video processing started.",
        "status_url": f"/status/{status_file.name}",
        "download_url": f"/download_video/{processed_video_path.name}"
    }


def detect_subtitles_task(task_id: str, temp_video_path: str, status_file: str):
    """
    A background task that runs the entire subtitle detection pipeline.
    It updates the shared TASK_RESULTS dictionary upon completion.
    """
    try:
        # Initialize status file for frontend polling
        with open(status_file, 'w') as f:
            json.dump({"status": "Detecting...", "progress": 0}, f)
        
        # --- Start of detection logic ---
        subtitle_detect = SubtitleDetect(video_path=temp_video_path)
        subtitle_frame_no_box_dict = subtitle_detect.find_subtitle_frame_no(status_file_path=status_file)
        if not subtitle_frame_no_box_dict:
            raise ValueError("No subtitles found in the video.")

        unified_sub_dict = subtitle_detect.unify_regions(subtitle_frame_no_box_dict)
        complete_subtitle_frame_no_box_dict = subtitle_detect.prevent_missed_detection(unified_sub_dict)
        
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        correct_subtitle_frame_no_box_dict = subtitle_detect.filter_mistake_sub_area(complete_subtitle_frame_no_box_dict, fps)
        first_entry_dict = {frame_no: boxes[0] for frame_no, boxes in correct_subtitle_frame_no_box_dict.items() if boxes}
        sub_frame_no_list_continuous = subtitle_detect.find_continuous_ranges_with_same_mask(first_entry_dict)
        
        merged_intervals = []
        if sub_frame_no_list_continuous:
            merged_intervals.append(sub_frame_no_list_continuous[0])
            for i in range(1, len(sub_frame_no_list_continuous)):
                last_start, last_end = merged_intervals[-1]
                current_start, current_end = sub_frame_no_list_continuous[i]
                if (current_start - last_end <= 3):
                    merged_intervals[-1] = (last_start, current_end)
                else:
                    merged_intervals.append((current_start, current_end))
        sub_frame_no_list_continuous = merged_intervals

        distinct_coords = []
        for start, end in sub_frame_no_list_continuous:
            coords_in_interval = [first_entry_dict[i] for i in range(start, end + 1) if i in first_entry_dict]
            if coords_in_interval:
                unified_box = find_smallest_bounding_box(coords_in_interval)
                distinct_coords.append(unified_box)
            else:
                distinct_coords.append(None)
        # --- End of detection logic ---

        # Update the task result with the final data
        if task_id in TASK_RESULTS:
            TASK_RESULTS[task_id].update({
                "status": "Completed",
                "distinct_coords": distinct_coords,
                "frame_intervals": sub_frame_no_list_continuous,
                "video_width": video_width,
                "video_height": video_height,
                "fps": fps,
                "total_frames": total_frames,
            })

        # Final status file update
        with open(status_file, 'w') as f:
            json.dump({"status": "Completed"}, f)
            
    except Exception as e:
        error_message = f"Error: {e}"
        print(f"Error in detect_subtitles_task for task {task_id}: {error_message}")
        if task_id in TASK_RESULTS:
            TASK_RESULTS[task_id].update({"status": "Error", "error": str(e)})
        with open(status_file, 'w') as f:
            json.dump({"status": error_message}, f)

# Call the SubtitleRemover class to remove subtitles
def process_video(video_path, json_path, output_path, status_file):
    try:
        # Immediately write a "Processing" status to the file so the frontend knows work has started.
        with open(status_file, 'w') as f:
            json.dump({"status": "Processing..."}, f)

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        coords = json_data.get("distinct_coordinates")
        intervals = json_data.get("frame_intervals")
        sd = SubtitleRemover(video_path, distinct_coords=coords, frame_intervals=intervals)
        # Let the remover write incremental progress into the same status file
        try:
            sd.status_file_path = str(status_file)
        except Exception:
            pass
        sd.run()
        shutil.copy2(sd.video_out_name, output_path)

        # Once successfully completed, update the status file.
        with open(status_file, 'w') as f:
            json.dump({"status": "Completed"}, f)
    except Exception as e:
        # If an error occurs, record it in the status file.
        with open(status_file, 'w') as f:
            json.dump({"status": f"Error: {e}"}, f)
    finally:
        # Keep original video so the editor can continue streaming it after processing.
        # Only remove the temporary JSON file here; video cleanup is handled by manual cleanup.
        for path in [json_path]:
            if os.path.exists(path):
                os.remove(path)


@app.get("/status/{status_filename}", include_in_schema=False)
async def get_status(status_filename: str):
    status_path = PROCESSED_DIR / status_filename
    if not status_path.exists():
        # This is expected before the background task creates the file. Frontend will retry.
        return JSONResponse(content={"status": "Not Found"}, status_code=404)
    
    try:
        with open(status_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            # File is empty, meaning processing is just starting.
            return JSONResponse(content={"status": "Processing..."})
        
        # New format: status file contains a JSON object.
        status_data = json.loads(content)
        return JSONResponse(content=status_data)
    except json.JSONDecodeError:
        # Backwards compatibility for old format where the file was just a string.
        return JSONResponse(content={"status": content})
    except Exception as e:
        return JSONResponse(content={"status": f"Error reading status file: {e}"}, status_code=500)
        

@app.get("/download_video/{video_filename}", include_in_schema=False)
async def download_video(video_filename: str):
    video_path = PROCESSED_DIR / video_filename
    if not video_path.exists():
        return JSONResponse(content={"error": "Processed video file not found!"})
    return FileResponse(video_path, media_type="video/mp4", filename=video_filename)


@app.get("/stream_video/{task_id}", include_in_schema=False)
async def stream_video(task_id: str):
    """Streams the video file for the editor's <video> tag."""
    result = TASK_RESULTS.get(task_id)
    if not result or not result.get("video_path") or not os.path.exists(result["video_path"]):
        raise HTTPException(status_code=404, detail="Video for this task not found.")
    
    video_path = result["video_path"]
    return FileResponse(video_path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


class AdjustIntervalRequest(BaseModel):
    interval_idx: int
    start_frame: int
    end_frame: int

@app.post("/adjust_interval/{task_id}", include_in_schema=False)
async def adjust_interval(task_id: str, req: AdjustIntervalRequest):
    """
    Adjusts the start and end frame for a specific interval and updates its neighbors
    to prevent any overlap.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    frame_intervals = result.get("frame_intervals", [])
    distinct_coords = result.get("distinct_coords", [])

    if not (0 <= req.interval_idx < len(frame_intervals)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    idx = req.interval_idx
    new_start = req.start_frame
    new_end = req.end_frame

    # Basic validation
    if new_start < 0 or new_end < new_start:
         raise HTTPException(status_code=400, detail="Invalid frame range: end must be after start.")

    # Adjust previous interval, if it exists
    if idx > 0:
        prev_start, _ = frame_intervals[idx - 1]
        new_prev_end = new_start - 1
        if new_prev_end < prev_start:
            raise HTTPException(status_code=400, detail=f"Change invalid: start frame collides with previous interval.")
        frame_intervals[idx - 1] = (prev_start, new_prev_end)
        
    # Adjust next interval, if it exists
    if idx < len(frame_intervals) - 1:
        _, next_end = frame_intervals[idx + 1]
        new_next_start = new_end + 1
        if new_next_start > next_end:
            raise HTTPException(status_code=400, detail=f"Change invalid: end frame collides with next interval.")
        frame_intervals[idx + 1] = (new_next_start, next_end)

    # Finally, update the current interval
    frame_intervals[idx] = (new_start, new_end)
    
    TASK_RESULTS[task_id]["frame_intervals"] = frame_intervals

    return {
        "message": f"Interval {idx + 1} and its neighbors were adjusted successfully.",
        "new_intervals": [
            {
                "interval_idx": i,
                "frame_range": interval,
                "coords": coords
            }
            for i, (interval, coords) in enumerate(zip(frame_intervals, distinct_coords))
        ]
    }


class MergeWithPreviousRequest(BaseModel):
    interval_idx: int

@app.post("/merge_with_previous/{task_id}", include_in_schema=False)
async def merge_with_previous(task_id: str, req: MergeWithPreviousRequest):
    """Merges the current interval with the previous one."""
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    frame_intervals = result.get("frame_intervals", [])
    distinct_coords = result.get("distinct_coords", [])
    
    if req.interval_idx <= 0 or req.interval_idx >= len(frame_intervals):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    # Merge the current interval with the previous one
    prev_start, prev_end = frame_intervals[req.interval_idx - 1]
    curr_start, curr_end = frame_intervals[req.interval_idx]
    
    # Create new merged interval
    merged_start = prev_start
    merged_end = curr_end
    
    # Use the bounding box that covers both intervals
    prev_coords = distinct_coords[req.interval_idx - 1]
    curr_coords = distinct_coords[req.interval_idx]
    merged_coords = find_smallest_bounding_box([prev_coords, curr_coords])
    
    # Update the previous interval with merged data
    frame_intervals[req.interval_idx - 1] = (merged_start, merged_end)
    distinct_coords[req.interval_idx - 1] = merged_coords
    
    # Remove the current interval
    frame_intervals.pop(req.interval_idx)
    distinct_coords.pop(req.interval_idx)
    
    # Update task results
    result["frame_intervals"] = frame_intervals
    result["distinct_coords"] = distinct_coords
    
    return {
        "message": f"Interval {req.interval_idx} merged with previous interval.",
        "new_intervals": [
            {
                "interval_idx": idx,
                "frame_range": interval,
                "coords": coords
            }
            for idx, (interval, coords) in enumerate(zip(frame_intervals, distinct_coords))
        ]
    }


class SplitIntervalRequest(BaseModel):
    interval_idx: int
    split_frame: int

@app.post("/split_interval/{task_id}", include_in_schema=False)
async def split_interval(task_id: str, req: SplitIntervalRequest):
    """Splits an interval at a specific frame."""
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    frame_intervals = result.get("frame_intervals", [])
    distinct_coords = result.get("distinct_coords", [])
    
    if not (0 <= req.interval_idx < len(frame_intervals)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    start_frame, end_frame = frame_intervals[req.interval_idx]
    
    if not (start_frame < req.split_frame < end_frame):
        raise HTTPException(status_code=400, detail="Split frame must be within the interval")

    # Create two new intervals: put the split frame into the second part
    first_interval = (start_frame, req.split_frame - 1)
    second_interval = (req.split_frame, end_frame)
    
    # Use the same coordinates for both parts (user can adjust later)
    coords = distinct_coords[req.interval_idx]
    
    # Replace the current interval with the first part
    frame_intervals[req.interval_idx] = first_interval
    
    # Insert the second part after the current interval
    frame_intervals.insert(req.interval_idx + 1, second_interval)
    distinct_coords.insert(req.interval_idx + 1, coords)
    
    # Update task results
    result["frame_intervals"] = frame_intervals
    result["distinct_coords"] = distinct_coords
    
    return {
        "message": f"Interval {req.interval_idx} split at frame {req.split_frame}.",
        "new_intervals": [
            {
                "interval_idx": idx,
                "frame_range": interval,
                "coords": coords
            }
            for idx, (interval, coords) in enumerate(zip(frame_intervals, distinct_coords))
        ]
    }




def merge_intervals(intervals, distinct_coords):
    """
    Merge intervals that have similar coordinates.
    Returns new intervals and distinct_coords with merged similar regions.
    Uses config.PIXEL_TOLERANCE_X and config.PIXEL_TOLERANCE_Y for similarity comparison.
    """
    if not intervals or not distinct_coords:
        return intervals, distinct_coords
    
    # Create a mapping of intervals to their coordinates
    interval_coords = list(zip(intervals, distinct_coords))
    
    # Group similar intervals
    merged_groups = []
    used_indices = set()
    
    for i, (interval1, coords1) in enumerate(interval_coords):
        if i in used_indices:
            continue
            
        # Start a new group
        current_group_indices = [i]
        
        # Find all similar intervals
        for j, (interval2, coords2) in enumerate(interval_coords[i+1:], i+1):
            if j in used_indices:
                continue
                
            if SubtitleDetect.are_similar(coords1, coords2):
                current_group_indices.append(j)
        
        # Merge the group
        if len(current_group_indices) > 1:
            # Sort by start frame
            current_group = [interval_coords[k] for k in current_group_indices]
            current_group.sort(key=lambda x: x[0][0])
            
            # Merge intervals
            merged_start = current_group[0][0][0]
            merged_end = current_group[-1][0][1]
            merged_interval = (merged_start, merged_end)
            
            # Use the bounding box that covers all merged coordinates
            all_coords = [item[1] for item in current_group]
            merged_coords = find_smallest_bounding_box(all_coords)
            
            merged_groups.append((merged_interval, merged_coords))
            used_indices.update(current_group_indices)
        else:
            # Single interval, keep as is
            merged_groups.append((interval1, coords1))
            used_indices.add(i)
    
    # Sort by start frame
    merged_groups.sort(key=lambda x: x[0][0])
    
    # Unpack results
    new_intervals = [group[0] for group in merged_groups]
    new_distinct_coords = [group[1] for group in merged_groups]
    
    return new_intervals, new_distinct_coords

@app.post("/merge_similar_intervals/{task_id}", include_in_schema=False)
async def merge_similar_intervals(task_id: str):
    """
    Merge intervals that have similar coordinates after user edits.
    Uses config.PIXEL_TOLERANCE_X and config.PIXEL_TOLERANCE_Y for similarity comparison.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    distinct_coords = result.get("distinct_coords", [])
    frame_intervals = result.get("frame_intervals", [])
    
    if not distinct_coords or not frame_intervals:
        raise HTTPException(status_code=400, detail="No intervals to merge")
    
    # Merge similar intervals
    new_intervals, new_distinct_coords = merge_intervals(
        frame_intervals, 
        distinct_coords
    )
    
    # Update the task results
    result["frame_intervals"] = new_intervals
    result["distinct_coords"] = new_distinct_coords
    
    # Update the JSON file if it exists
    original_name = result.get("original_filename", "unknown")
    json_file_path = PROCESSED_FILES_DIR / f"{original_name}_sub.json"
    
    if json_file_path.exists():
        json_content = {
            "distinct_coordinates": new_distinct_coords,
            "frame_intervals": new_intervals
        }
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file, indent=4)
    
    return {
        "message": f"Merged {len(frame_intervals)} intervals into {len(new_intervals)} intervals",
        "original_count": len(frame_intervals),
        "merged_count": len(new_intervals),
        "intervals": [
            {
                "interval_idx": idx,
                "frame_range": interval,
                "coords": coords
            }
            for idx, (interval, coords) in enumerate(zip(new_intervals, new_distinct_coords))
        ]
    }


def generate_subtitle_task(video_path: str, srt_path: str, txt_path: str):
    """
    Background task to perform OCR, generate SRT, and then convert to TXT.
    """
    try:
        raw_subtitle_path = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
        
        # Configure OCR options
        options = {
            'REC_CHAR_TYPE': config.REC_CHAR_TYPE,
            'DROP_SCORE': config.DROP_SCORE,
            'SUB_AREA_DEVIATION_RATE': config.SUB_AREA_DEVIATION_RATE,
            'DEBUG_OCR_LOSS': config.DEBUG_OCR_LOSS
        }

        # Start the asynchronous OCR process
        process, task_queue, progress_queue = subtitle_ocr.async_start(
            video_path,
            raw_subtitle_path,
            None, # No pre-defined sub area
            options
        )

        # Feed frames to the OCR process
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        current_frame_no = 0
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            current_frame_no += 1
            task = (frame_count, current_frame_no, None, None, None, None)
            task_queue.put(task)
            # Skip frames based on extract frequency
            for _ in range(int(fps // config.EXTRACT_FREQUENCY) - 1):
                if cap.isOpened():
                    cap.read()
                    current_frame_no += 1
        
        cap.release()
        task_queue.put((frame_count, -1, None, None, None, None))
        process.join()

        # Generate SRT and then TXT
        extractor = SubtitleExtractor(raw_subtitle_path, video_path)
        generated_srt_path = extractor.generate_subtitle_file()
        extractor.srt2txt(generated_srt_path)

        # Move final txt file to its destination
        final_txt_path = os.path.join(os.path.dirname(generated_srt_path), Path(generated_srt_path).stem + '.txt')
        shutil.move(final_txt_path, txt_path)

    except Exception as e:
        print(f"Error in subtitle generation task: {e}")
    finally:
        # Clean up temporary raw subtitle file
        if os.path.exists(raw_subtitle_path):
            os.remove(raw_subtitle_path)
        if os.path.exists(srt_path):
            os.remove(srt_path)


@app.post("/generate_subtitle_text/{task_id}")
async def generate_subtitle_text(task_id: str, background_tasks: BackgroundTasks):
    """
    Generates and returns a downloadable TXT file with subtitle content.
    """
    result = TASK_RESULTS.get(task_id)
    if not result or not result.get("video_path"):
        raise HTTPException(status_code=404, detail="Video for this task not found.")

    video_path = result["video_path"]
    original_stem = result.get("original_filename", "unknown")
    
    # Define paths for the final SRT and TXT files
    srt_path = PROCESSED_FILES_DIR / f"{original_stem}.srt"
    txt_path = PROCESSED_FILES_DIR / f"{original_stem}.txt"

    # Run the entire OCR and text generation process in the background
    background_tasks.add_task(
        generate_subtitle_task,
        video_path=video_path,
        srt_path=str(srt_path),
        txt_path=str(txt_path)
    )

    # Poll until the txt file is created
    timeout = 300  # 5 minutes timeout
    start_time = time.time()
    while not os.path.exists(txt_path):
        await asyncio.sleep(1)
        if time.time() - start_time > timeout:
            raise HTTPException(status_code=504, detail="Subtitle generation timed out.")

    return FileResponse(
        path=txt_path,
        media_type='text/plain',
        filename=f"{original_stem}_subtitles.txt"
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    uvicorn.run(app, host="0.0.0.0", port=8000)