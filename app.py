from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from pathlib import Path
import os
import json
import shutil
import tempfile
import multiprocessing
from backend.main import SubtitleExtractor

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
import queue
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, Form
from backend.main import find_smallest_bounding_box
import traceback
from fastapi.responses import Response

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return RedirectResponse(url="/upload")

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

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


@app.post("/upload-and-edit")
async def upload_and_edit(
    background_tasks: BackgroundTasks,
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    cloud_ref: Optional[str] = Form(None)
):
    """
    A user-friendly endpoint that accepts a video, starts the detection
    process, and returns a page that polls for completion and then redirects.
    """
    # --- 1. Handle Video Input ---
    temp_video_path = None
    original_name = "unknown_file"

    if len([source for source in [file, url, cloud_ref] if source]) != 1:
        raise HTTPException(status_code=400, detail="Please provide exactly one video source (file, url, or cloud_ref).")

    if file and file.filename:
        original_name = Path(file.filename).stem
        temp_video_path = save_temp_file(file)
    elif url:
        original_name = Path(url).stem
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(url, temp_video_path)
    elif cloud_ref:
        original_name = Path(cloud_ref).stem
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(cloud_ref, temp_video_path)

    # --- 2. Setup and Start Background Task ---
    task_id = str(uuid.uuid4())
    status_file = PROCESSED_DIR / f"detect_{task_id}.status"

    TASK_RESULTS[task_id] = {
        "status": "Detecting",
        "timestamp": time.time(),
        "original_filename": original_name,
        "status_file": str(status_file),
        "video_path": temp_video_path,
    }

    background_tasks.add_task(
        detect_subtitles_task,
        task_id=task_id,
        temp_video_path=temp_video_path,
        status_file=str(status_file)
    )

    return templates.TemplateResponse("processing.html", {
        "request": request,
        "task_id": task_id
    })


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
        fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
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
                "intervals": [
                    {
                        "frame_range": frame_range,
                        "coords": coord,
                        "text": ""
                    }
                    for frame_range, coord in zip(sub_frame_no_list_continuous, distinct_coords)
                ],
                "detected_areas": first_entry_dict, # Keep this for subtitle generation
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
            

@app.get("/subtitle_intervals/{task_id}", include_in_schema=False)
async def get_subtitle_intervals(task_id: str):
    """
    Returns all intervals and their current rectangles for a given task_id.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")

    response_intervals = []
    # The unified data structure is now a list of interval objects
    for idx, interval in enumerate(result.get("intervals", [])):
        frame_range_tuple = interval.get('frame_range', (0, 0))
        # Per editor requirements, convert 1-based frame ranges from backend detection
        # to 0-based for the frontend video player.
        response_intervals.append({
            "interval_idx": idx,
            "frame_range": (frame_range_tuple[0] - 1, frame_range_tuple[1] - 1),
            "coords": interval.get('coords'),
            "text": interval.get('text', '')
        })
    return {"intervals": response_intervals}


# Draw subtitle boxes on the video for users to visualize and adjust later
def draw_subtitle_boxes(frame, intervals, current_frame_idx):
    """
    Draws rectangles for all subtitle regions active at the current frame.
    Expands the box by 10 pixels to ensure full coverage.
    """
    frame_height, frame_width, _ = frame.shape
    for interval in intervals:
        coord = interval['coords']
        start, end = interval['frame_range']
        if coord is None:
            continue
        # Frame ranges are 1-based, current_frame_idx is 0-based
        if start <= (current_frame_idx + 1) <= end:
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
    intervals = result.get("intervals", [])

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
        frame_with_boxes = draw_subtitle_boxes(frame, intervals, frame_idx)
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

    intervals = result.get("intervals", [])
    if not (0 <= req.interval_idx < len(intervals)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    # Get current box properties
    xmin, xmax, ymin, ymax = intervals[req.interval_idx]['coords']
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
    intervals[req.interval_idx]['coords'] = new_coords
    TASK_RESULTS[task_id]["intervals"] = intervals

    return {
        "message": f"Interval {req.interval_idx} adjusted successfully.",
        "new_coords": new_coords
    }


def generate_subtitle_task(task_id: str, status_path: str):
    """
    Background task to perform subtitle extraction and merge text into existing intervals.
    """
    try:
        result = TASK_RESULTS[task_id]
        video_path = result['video_path']
        detected_areas = result.get('detected_areas', {})

        # The SubtitleExtractor handles OCR internally
        extractor = SubtitleExtractor(video_path=video_path, status_path=status_path, detected_areas=detected_areas)
        
        # This method returns a list of (start_frame, end_frame, coords, text)
        generated_subs = extractor._remove_duplicate_subtitle()

        user_intervals = result.get('intervals', [])

        # Match generated text to user intervals based on frame range overlap (IoU)
        for sub_start, sub_end, _, sub_text in generated_subs:
            best_match_idx = -1
            highest_iou = 0.5  # Require at least 50% overlap

            for i, user_interval in enumerate(user_intervals):
                user_range = user_interval['frame_range']
                iou = get_frame_range_iou((int(sub_start), int(sub_end)), user_range)
                
                if iou > highest_iou:
                    highest_iou = iou
                    best_match_idx = i
            
            if best_match_idx != -1:
                # Append text. If multiple subs match, they get concatenated.
                existing_text = user_intervals[best_match_idx].get('text', '')
                user_intervals[best_match_idx]['text'] = f"{existing_text} {sub_text}".strip()

        # Final status update
        result['status'] = 'Generation Complete'
        # The download URL now points to a file with the combined text content.
        result['download_url'] = f"/download_subtitle_text/{task_id}"
        with open(status_path, 'w') as f:
            json.dump({"status": "Generation Complete", "download_url": result['download_url']}, f)

    except Exception as e:
        traceback.print_exc()
        if task_id in TASK_RESULTS:
            TASK_RESULTS[task_id]['status'] = 'Generation Error'
            TASK_RESULTS[task_id]['error'] = traceback.format_exc()
        with open(status_path, 'w') as f:
            json.dump({"status": "Error", "message": str(e)}, f)
    finally:
        # Clean up temporary raw subtitle file if created by extractor
        if 'extractor' in locals() and hasattr(extractor, 'raw_subtitle_path') and extractor.raw_subtitle_path:
            if os.path.exists(extractor.raw_subtitle_path) and "tmp" in extractor.raw_subtitle_path:
                os.remove(extractor.raw_subtitle_path)

@app.post("/generate_subtitle_text/{task_id}")
async def generate_subtitle_text(task_id: str, background_tasks: BackgroundTasks):
    """
    Generates a downloadable TXT file with subtitle content.
    This is now an async operation. Poll /task_info/{task_id} for status.
    """
    result = TASK_RESULTS.get(task_id)
    if not result or not result.get("video_path"):
        raise HTTPException(status_code=404, detail="Video for this task not found.")

    original_stem = result.get("original_filename", "unknown")
    status_path = PROCESSED_FILES_DIR / f"{original_stem}_generation.status"

    # Update task status to generating
    result["status"] = "Generating Subtitles"
    result["generation_status_file"] = str(status_path)

    background_tasks.add_task(
        generate_subtitle_task,
        task_id=task_id,
        status_path=str(status_path),
    )

    return {
        "message": "Subtitle generation started. Poll /task_info/{task_id} for progress.",
        "task_id": task_id
    }


@app.get("/download_subtitle_text/{task_id}")
async def download_subtitle_text(task_id: str):
    """
    Downloads the generated subtitle text file for a given task.
    This now compiles the text from the final interval data.
    """
    result = TASK_RESULTS.get(task_id)
    if not result or result.get("status") != "Generation Complete":
        raise HTTPException(status_code=404, detail="Subtitle text file not ready or task not found.")

    intervals = result.get("intervals", [])
    
    # Create the text content on-the-fly from the interval data
    srt_content = ""
    for i, interval in enumerate(intervals):
        if interval.get("text"):
            start_frame = interval['frame_range'][0]
            end_frame = interval['frame_range'][1]
            
            # This is a placeholder for time conversion; a real implementation would need fps.
            start_time = f"00:00:{start_frame//30:02d},{ (start_frame%30)*33:03d}"
            end_time = f"00:00:{end_frame//30:02d},{ (end_frame%30)*33:03d}"

            srt_content += f"{i+1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{interval['text']}\n\n"

    original_stem = result.get("original_filename", "unknown")
    return Response(
        content=srt_content,
        media_type='text/plain',
        headers={"Content-Disposition": f"attachment; filename={original_stem}_subtitles.txt"}
    )


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

    elif status == "Generating Subtitles":
        status_file_path = result.get("generation_status_file")
        if status_file_path and os.path.exists(status_file_path):
            try:
                with open(status_file_path, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                pass # Fallback if file is empty or corrupt
        return {"status": "Generating Subtitles...", "progress": 0}
    
    elif status == "Generation Complete":
        return {
            "status": "Generation Complete",
            "download_url": f"/download_subtitle_text/{task_id}",
            "video_width": result.get("video_width"),
            "video_height": result.get("video_height"),
            "original_filename": result.get("original_filename"),
            "user_id": result.get("user_id"),
            "fps": result.get("fps"),
            "total_frames": result.get("total_frames"),
            "editor_url": str(request.url_for('get_editor', task_id=task_id))
        }

    elif status == "Error" or status == "Generation Error":
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
            "intervals": result.get("intervals")
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


# Call the SubtitleRemover class to remove subtitles
def process_video(video_path, json_path, output_path, status_file):
    try:
        # Immediately write a "Processing" status to the file so the frontend knows work has started.
        with open(status_file, 'w') as f:
            json.dump({"status": "Processing..."}, f)

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        intervals_data = json_data.get("intervals")
        # Unpack the structure for the SubtitleRemover class
        coords = [item['coords'] for item in intervals_data]
        intervals = [item['frame_range'] for item in intervals_data]
        
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


def get_frame_range_iou(range1, range2):
    """Calculates the Intersection over Union (IoU) for two 1D ranges."""
    start1, end1 = range1
    start2, end2 = range2
    
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    intersection_length = max(0, intersection_end - intersection_start + 1)
    if intersection_length == 0:
        return 0
        
    union_length = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection_length
    
    return intersection_length / union_length if union_length > 0 else 0


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

    intervals = result.get("intervals", [])

    if not (0 <= req.interval_idx < len(intervals)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    idx = req.interval_idx
    new_start = req.start_frame
    new_end = req.end_frame

    # Basic validation
    if new_start < 0 or new_end < new_start:
         raise HTTPException(status_code=400, detail="Invalid frame range: end must be after start.")

    # Adjust previous interval, if it exists
    if idx > 0:
        prev_start, _ = intervals[idx - 1]['frame_range']
        new_prev_end = new_start - 1
        if new_prev_end < prev_start:
            raise HTTPException(status_code=400, detail=f"Change invalid: start frame collides with previous interval.")
        intervals[idx - 1]['frame_range'] = (prev_start, new_prev_end)
        
    # Adjust next interval, if it exists
    if idx < len(intervals) - 1:
        _, next_end = intervals[idx + 1]['frame_range']
        new_next_start = new_end + 1
        if new_next_start > next_end:
            raise HTTPException(status_code=400, detail=f"Change invalid: end frame collides with next interval.")
        intervals[idx + 1]['frame_range'] = (new_next_start, next_end)

    # Finally, update the current interval
    intervals[idx]['frame_range'] = (new_start, new_end)
    
    TASK_RESULTS[task_id]["intervals"] = intervals

    return {
        "message": f"Interval {idx + 1} and its neighbors were adjusted successfully.",
        "new_intervals": [
            {
                "interval_idx": i,
                "frame_range": (interval['frame_range'][0] - 1, interval['frame_range'][1] - 1),
                "coords": interval['coords'],
                "text": interval.get('text', '')
            }
            for i, (interval) in enumerate(intervals)
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

    intervals = result.get("intervals", [])
    
    if req.interval_idx <= 0 or req.interval_idx >= len(intervals):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    # Get the intervals to merge
    prev_interval = intervals[req.interval_idx - 1]
    curr_interval = intervals[req.interval_idx]

    # Create new merged interval by extending the end of the previous one
    prev_start, _ = prev_interval['frame_range']
    _, curr_end = curr_interval['frame_range']
    merged_range = (prev_start, curr_end)
    
    # Use the bounding box that covers both intervals
    merged_coords = find_smallest_bounding_box([prev_interval['coords'], curr_interval['coords']])
    
    # Merge text content
    merged_text = f"{prev_interval.get('text', '')} {curr_interval.get('text', '')}".strip()

    # Update the previous interval with merged data
    intervals[req.interval_idx - 1]['frame_range'] = merged_range
    intervals[req.interval_idx - 1]['coords'] = merged_coords
    intervals[req.interval_idx - 1]['text'] = merged_text
    
    # Remove the current interval that was merged
    intervals.pop(req.interval_idx)
    
    # Update task results
    result["intervals"] = intervals
    
    return {
        "message": f"Interval {req.interval_idx + 1} merged with previous interval.",
        "new_intervals": [
            {
                "interval_idx": idx,
                "frame_range": (interval['frame_range'][0] - 1, interval['frame_range'][1] - 1),
                "coords": interval['coords'],
                "text": interval.get('text', '')
            }
            for idx, interval in enumerate(intervals)
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

    intervals = result.get("intervals", [])
    
    if not (0 <= req.interval_idx < len(intervals)):
        raise HTTPException(status_code=400, detail="Invalid interval_idx")

    # The split_frame from the frontend is 0-based, but our intervals are 1-based.
    # No conversion is needed for the check, but +1 is needed for the new start.
    start_frame, end_frame = intervals[req.interval_idx]['frame_range']
    
    if not (start_frame <= req.split_frame < end_frame):
        raise HTTPException(status_code=400, detail="Split frame must be within the interval (but not the last frame)")

    # Create two new intervals.
    # Frontend sends 0-based frame to split AT. So, 1st part ends at split_frame. 2nd part starts at split_frame + 1.
    # Backend intervals are 1-based. So, 1st part is [start, split_frame]. 2nd is [split_frame+1, end]
    first_interval_range = (start_frame, req.split_frame)
    second_interval_range = (req.split_frame + 1, end_frame)
    
    # Use the same coordinates and text for both parts (user can adjust later)
    original_interval = intervals[req.interval_idx]
    coords = original_interval['coords']
    text = original_interval.get('text', '') # Keep original text for both
    
    # Replace the current interval with the first part
    intervals[req.interval_idx]['frame_range'] = first_interval_range
    
    # Create a new interval object for the second part
    new_interval_obj = {
        "frame_range": second_interval_range,
        "coords": coords,
        "text": text
    }
    # Insert the second part after the current interval
    intervals.insert(req.interval_idx + 1, new_interval_obj)
    
    # Update task results
    result["intervals"] = intervals
    
    return {
        "message": f"Interval split at frame {req.split_frame}.",
        "new_intervals": [
            {
                "interval_idx": idx,
                "frame_range": (interval['frame_range'][0] - 1, interval['frame_range'][1] - 1),
                "coords": interval['coords'],
                "text": interval.get('text', '')
            }
            for idx, interval in enumerate(intervals)
        ]
    }




def merge_intervals(intervals):
    """
    Merge intervals that have similar coordinates.
    Returns new intervals list with merged similar regions.
    Uses config.PIXEL_TOLERANCE_X and config.PIXEL_TOLERANCE_Y for similarity comparison.
    """
    if not intervals:
        return intervals
    
    # Group similar intervals
    merged_groups = []
    used_indices = set()
    
    for i, interval1 in enumerate(intervals):
        if i in used_indices:
            continue
            
        # Start a new group
        current_group_indices = [i]
        
        # Find all similar intervals
        for j, interval2 in enumerate(intervals[i+1:], i+1):
            if j in used_indices:
                continue
                
            if SubtitleDetect.are_similar(interval1['coords'], interval2['coords']):
                current_group_indices.append(j)
        
        # Merge the group
        if len(current_group_indices) > 1:
            current_group = [intervals[k] for k in current_group_indices]
            current_group.sort(key=lambda x: x['frame_range'][0])
            
            # Merge intervals
            merged_start = current_group[0]['frame_range'][0]
            merged_end = current_group[-1]['frame_range'][1]
            merged_range = (merged_start, merged_end)
            
            # Use the bounding box that covers all merged coordinates
            all_coords = [item['coords'] for item in current_group]
            merged_coords = find_smallest_bounding_box(all_coords)

            # Combine text from all merged intervals
            merged_text = " ".join(item.get('text', '') for item in current_group).strip()
            
            merged_groups.append({
                "frame_range": merged_range,
                "coords": merged_coords,
                "text": merged_text
            })
            used_indices.update(current_group_indices)
        else:
            # Single interval, keep as is
            merged_groups.append(interval1)
            used_indices.add(i)
    
    # Sort by start frame
    merged_groups.sort(key=lambda x: x['frame_range'][0])
    
    return merged_groups

@app.post("/merge_similar_intervals/{task_id}", include_in_schema=False)
async def merge_similar_intervals(task_id: str):
    """
    Merge intervals that have similar coordinates after user edits.
    Uses config.PIXEL_TOLERANCE_X and config.PIXEL_TOLERANCE_Y for similarity comparison.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    intervals = result.get("intervals", [])
    
    if not intervals:
        raise HTTPException(status_code=400, detail="No intervals to merge")
    
    original_count = len(intervals)
    # Merge similar intervals
    new_intervals = merge_intervals(intervals)
    
    # Update the task results
    result["intervals"] = new_intervals
    
    # Update the JSON file if it exists
    original_name = result.get("original_filename", "unknown")
    json_file_path = PROCESSED_FILES_DIR / f"{original_name}_sub.json"
    
    if json_file_path.exists():
        json_content = { "intervals": new_intervals }
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file, indent=4)
    
    return {
        "message": f"Merged {original_count} intervals into {len(new_intervals)} intervals",
        "original_count": original_count,
        "merged_count": len(new_intervals),
        "new_intervals": [
            {
                "interval_idx": idx,
                "frame_range": (interval['frame_range'][0] - 1, interval['frame_range'][1] - 1),
                "coords": interval['coords'],
                "text": interval.get('text', '')
            }
            for idx, interval in enumerate(new_intervals)
        ]
    }


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    uvicorn.run(app, host="0.0.0.0", port=8002)