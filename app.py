from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import os
import json
import shutil
import tempfile
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
    """Check current memory usage and warn if approaching limits"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Assume 4096MB limit
        memory_limit_mb = 4096
        memory_percentage = (memory_mb / memory_limit_mb) * 100
        
        print(f"Memory usage: {memory_mb:.1f}MB ({memory_percentage:.1f}%)")
        
        if memory_percentage > 80:
            print(f"WARNING: Memory usage at {memory_percentage:.1f}%")
            return False
        return True
    except Exception:
        # If psutil not available, continue
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

    cleaned_count = 0
    for task_id in tasks_to_remove:
        result = TASK_RESULTS.pop(task_id, None)
        if result:
            video_path = result.get('video_path')
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError as e:
                    print(f"Error removing file {video_path}: {e}")
            cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} tasks.")
    
    return cleaned_count

class CleanupRequest(BaseModel):
    user_id: str

@app.post("/cleanup")
async def manual_cleanup(request: CleanupRequest):
    """
    Manually triggers cleanup for a specific user's tasks.
    """
    cleaned_count = cleanup_tasks(user_id=request.user_id)
    if cleaned_count > 0:
        return {"message": f"Successfully cleaned up {cleaned_count} tasks for user {request.user_id}."}
    else:
        return {"message": f"No tasks found to clean up for user {request.user_id}."}


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
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,
    cloud_ref: Optional[str] = None,
    user_id: Optional[str] = None):
    # Use the original filename (without extension) for the output JSON
    original_name = Path(file.filename).stem if file else "unknown"
    temp_video_path = None

    if user_id is None:
        user_id = str(uuid.uuid4())

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

        complete_subtitle_frame_no_box_dict = subtitle_detect.prevent_missed_detection(subtitle_frame_no_box_dict)
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Filter out mistake subtitle areas by checking the fps
        correct_subtitle_frame_no_box_dict = subtitle_detect.filter_mistake_sub_area(complete_subtitle_frame_no_box_dict, fps)

        # Instead of using first_entry_dict, find the most representative subtitle area for each frame
        def get_most_representative_box(boxes):
            """Get the most representative subtitle box from a list of boxes."""
            if not boxes:
                return None
            if len(boxes) == 1:
                return boxes[0]
            
            # Calculate area for each box and find the largest one (likely main subtitle)
            box_areas = []
            for box in boxes:
                xmin, xmax, ymin, ymax = box
                area = (xmax - xmin) * (ymax - ymin)
                box_areas.append((area, box))
            
            # Return the box with largest area
            return max(box_areas, key=lambda x: x[0])[1]

        representative_boxes_dict = {
            frame_no: get_most_representative_box(boxes) 
            for frame_no, boxes in correct_subtitle_frame_no_box_dict.items() 
            if boxes
        }
        
        # Create initial intervals using the relaxed continuous ranges method
        initial_intervals = subtitle_detect.find_continuous_ranges(representative_boxes_dict)
        print(f"[DEBUG] Initial intervals: {len(initial_intervals)}")
        
        # Now create intelligent merged intervals based on subtitle similarity
        def create_intelligent_intervals(intervals, boxes_dict):
            """Create merged intervals based on subtitle box similarity and proximity."""
            if not intervals:
                return [], []
            
            # Calculate representative box for each interval
            interval_data = []
            for start, end in intervals:
                # Get all boxes in this interval
                interval_boxes = [
                    boxes_dict[i] for i in range(start, end + 1) 
                    if i in boxes_dict and boxes_dict[i] is not None
                ]
                
                if interval_boxes:
                    # Use the bounding box that encompasses all boxes in the interval
                    representative_box = find_smallest_bounding_box(interval_boxes)
                    interval_data.append(((start, end), representative_box))
            
            if not interval_data:
                return [], []
            
            # Apply intelligent merging
            merged_intervals = [interval_data[0][0]]
            merged_coords = [interval_data[0][1]]
            
            for i in range(1, len(interval_data)):
                current_interval, current_box = interval_data[i]
                last_interval = merged_intervals[-1]
                last_box = merged_coords[-1]
                
                current_start, current_end = current_interval
                last_start, last_end = last_interval
                
                # Calculate gap between intervals
                gap = current_start - last_end
                
                # Check if boxes are similar using the existing similarity function
                boxes_similar = SubtitleDetect.are_similar(last_box, current_box) if last_box and current_box else False
                
                # Merging criteria:
                # 1. Very small gap (1-3 frames) - likely same subtitle with minor detection gaps
                # 2. Small gap (4-10 frames) + similar boxes - same subtitle with brief pause
                # 3. Medium gap (11-20 frames) + very similar boxes - same subtitle with longer pause
                should_merge = False
                
                if gap <= 3:
                    should_merge = True
                elif gap <= 10 and boxes_similar:
                    should_merge = True
                elif gap <= 20 and boxes_similar and last_box and current_box:
                    # Check for very similar boxes (nearly identical positions)
                    xmin1, xmax1, ymin1, ymax1 = last_box
                    xmin2, xmax2, ymin2, ymax2 = current_box
                    
                    very_similar = (
                        abs(xmin1 - xmin2) <= 15 and
                        abs(xmax1 - xmax2) <= 15 and
                        abs(ymin1 - ymin2) <= 15 and
                        abs(ymax1 - ymax2) <= 15
                    )
                    
                    if very_similar:
                        should_merge = True
                
                if should_merge:
                    # Merge with previous interval
                    merged_intervals[-1] = (last_start, current_end)
                    if last_box and current_box:
                        merged_coords[-1] = find_smallest_bounding_box([last_box, current_box])
                    else:
                        merged_coords[-1] = last_box or current_box
                else:
                    # Keep as separate interval
                    merged_intervals.append(current_interval)
                    merged_coords.append(current_box)
            
            return merged_intervals, merged_coords
        
        # Apply intelligent interval creation
        sub_frame_no_list_continuous, distinct_coords = create_intelligent_intervals(
            initial_intervals, representative_boxes_dict
        )
        
        print(f"[DEBUG] After intelligent merging: {len(sub_frame_no_list_continuous)} intervals")

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
        editor_url = str(request.url_for('get_editor', task_id=task_id))
        TASK_RESULTS[task_id] = {
            "distinct_coords": distinct_coords,
            "frame_intervals": sub_frame_no_list_continuous,
            "original_filename": original_name,
            "video_path": temp_video_path,
            "video_width": video_width,
            "video_height": video_height,
            "timestamp": time.time(),
            "user_id": user_id,
        }
        return {"task_id": task_id, "user_id": user_id, "editor_url": editor_url}

    except Exception as e:
        print("Error in find_subtitles: ", e)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e


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


@app.get("/editor/{task_id}", response_class=HTMLResponse, include_in_schema=True)
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
@app.get("/task_info/{task_id}", include_in_schema=False)
async def get_task_info(task_id: str):
    """
    Returns metadata for a given task, including video dimensions.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "video_width": result.get("video_width"),
        "video_height": result.get("video_height"),
        "original_filename": result.get("original_filename"),
        "user_id": result.get("user_id")
    }

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
    
    return {
        "message": "Video processing started.",
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


@app.get("/download_video/{video_filename}", include_in_schema=False)
async def download_video(video_filename: str):
    video_path = PROCESSED_DIR / video_filename
    if not video_path.exists():
        return JSONResponse(content={"error": "Processed video file not found!"})
    return FileResponse(video_path, media_type="video/mp4", filename=video_filename)


def merge_intervals(intervals, distinct_coords):
    """
    Merge intervals using intelligent logic that considers subtitle box similarity and frame gaps.
    This uses the same merging criteria as the find_subtitles function for consistency.
    """
    if not intervals or not distinct_coords:
        return intervals, distinct_coords
    
    # Sort intervals by start frame
    sorted_pairs = sorted(zip(intervals, distinct_coords), key=lambda x: x[0][0])
    
    if not sorted_pairs:
        return [], []
    
    merged_intervals = [sorted_pairs[0][0]]
    merged_coords = [sorted_pairs[0][1]]
    
    for i in range(1, len(sorted_pairs)):
        current_interval, current_box = sorted_pairs[i]
        last_interval = merged_intervals[-1]
        last_box = merged_coords[-1]
        
        current_start, current_end = current_interval
        last_start, last_end = last_interval
        
        # Calculate gap between intervals
        gap = current_start - last_end
        
        # Check if boxes are similar using the existing similarity function
        boxes_similar = SubtitleDetect.are_similar(last_box, current_box) if last_box and current_box else False
        
        # Use the same merging criteria as in find_subtitles:
        # 1. Very small gap (1-3 frames) - likely same subtitle with minor detection gaps
        # 2. Small gap (4-10 frames) + similar boxes - same subtitle with brief pause
        # 3. Medium gap (11-20 frames) + very similar boxes - same subtitle with longer pause
        should_merge = False
        
        if gap <= 3:
            should_merge = True
        elif gap <= 10 and boxes_similar:
            should_merge = True
        elif gap <= 20 and boxes_similar and last_box and current_box:
            # Check for very similar boxes (nearly identical positions)
            xmin1, xmax1, ymin1, ymax1 = last_box
            xmin2, xmax2, ymin2, ymax2 = current_box
            
            very_similar = (
                abs(xmin1 - xmin2) <= 15 and
                abs(xmax1 - xmax2) <= 15 and
                abs(ymin1 - ymin2) <= 15 and
                abs(ymax1 - ymax2) <= 15
            )
            
            if very_similar:
                should_merge = True
        
        if should_merge:
            # Merge with previous interval
            merged_intervals[-1] = (last_start, current_end)
            if last_box and current_box:
                merged_coords[-1] = find_smallest_bounding_box([last_box, current_box])
            else:
                merged_coords[-1] = last_box or current_box
        else:
            # Keep as separate interval
            merged_intervals.append(current_interval)
            merged_coords.append(current_box)
    
    return merged_intervals, merged_coords

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)