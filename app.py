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
from backend.tools.ocr import OcrRecogniser, compare_ocr_result

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Global Model Management ---
class ModelManager:
    subtitle_detector = None
    ocr_recogniser = None

@app.on_event("startup")
async def load_models():
    """Load deep learning models on application startup."""
    print("Loading models into memory...")
    ModelManager.subtitle_detector = SubtitleDetect(video_path=None) # Initialized without a video
    ModelManager.ocr_recogniser = OcrRecogniser()
    print("Models loaded successfully.")

# ---------------------------------

@app.get("/")
async def root():
    return {"Message": "Visit docs to remove subtitles in your videos!"}

# Use absolute paths to avoid working directory issues
PROCESSED_DIR = Path(os.getcwd()) / "processed_videos"
PROCESSED_DIR.mkdir(exist_ok=True)
PROCESSED_FILES_DIR = Path(os.getcwd()) / "processed_files"
PROCESSED_FILES_DIR.mkdir(exist_ok=True)

TASK_RESULTS = {}

def check_memory_usage(return_stats=False):
    """Check current memory usage and warn if approaching limits. Can also return stats."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Assume 4096MB limit
        memory_limit_mb = 4096
        memory_percentage = (memory_mb / memory_limit_mb) * 100
        
        print(f"Memory usage: {memory_mb:.1f}MB ({memory_percentage:.1f}%)")

        if return_stats:
            return {
                "memory_mb": round(memory_mb, 1),
                "memory_percentage": round(memory_percentage, 1)
            }
        
        if memory_percentage > 80:
            print(f"WARNING: Memory usage at {memory_percentage:.1f}%")
            return False
        return True
    except Exception:
        # If psutil not available, continue
        if return_stats:
            return {"memory_mb": -1, "memory_percentage": -1}
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
    Manually triggers cleanup for a specific user's tasks and reports memory usage.
    """
    cleaned_count = cleanup_tasks(user_id=request.user_id)
    memory_after_cleanup = check_memory_usage(return_stats=True)
    
    if cleaned_count > 0:
        return {
            "message": f"Successfully cleaned up {cleaned_count} tasks for user {request.user_id}.",
            "memory_after_cleanup": memory_after_cleanup
        }
    else:
        return {
            "message": f"No tasks found to clean up for user {request.user_id}.",
            "memory_after_cleanup": memory_after_cleanup
        }


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
@app.post("/debug_raw_detection/")
async def debug_raw_detection(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = None,
    cloud_ref: Optional[str] = None,
    user_id: Optional[str] = None):
    """
    DEBUG ENDPOINT: Shows raw output from find_subtitle_frame_no without any post-processing.
    This helps debug what the core detection function actually finds.
    """
    # Use the original filename (without extension) for the output JSON
    original_name = Path(file.filename).stem if file else "debug_raw"
    temp_video_path = None

    if user_id is None:
        user_id = str(uuid.uuid4())

    if url:
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(url, temp_video_path)
    elif cloud_ref:
        temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
        await download_file(cloud_ref, temp_video_path)
    else:
        temp_video_path = save_temp_file(file)

    try:
        # Check memory before processing
        if not check_memory_usage():
            raise HTTPException(status_code=503, detail="Server memory limit reached. Please try again later.")
        
        # Use the global subtitle detector and update its video path
        subtitle_detect = ModelManager.subtitle_detector
        subtitle_detect.video_path = temp_video_path
        
        # *** THIS IS THE RAW OUTPUT FROM find_subtitle_frame_no ***
        raw_subtitle_frame_no_box_dict = subtitle_detect.find_subtitle_frame_no()
        
        if not raw_subtitle_frame_no_box_dict:
            raise HTTPException(status_code=404, detail="No subtitles found in the video.")

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"[DEBUG] Raw detection found subtitles in {len(raw_subtitle_frame_no_box_dict)} frames")
        
        # Create simple intervals directly from the raw detection (no merging at all)
        simple_intervals = []
        simple_coords = []
        
        # Convert frame-by-frame detections to intervals
        frame_numbers = sorted(raw_subtitle_frame_no_box_dict.keys())
        for frame_no in frame_numbers:
            boxes = raw_subtitle_frame_no_box_dict[frame_no]
            if boxes:
                # For debugging, just take the first box from each frame
                first_box = boxes[0]
                simple_intervals.append((frame_no, frame_no))  # Single frame intervals
                simple_coords.append(first_box)
        
        print(f"[DEBUG] Created {len(simple_intervals)} single-frame intervals from raw detection")

        json_content = {
            "distinct_coordinates": simple_coords,
            "frame_intervals": simple_intervals,
            "debug_info": {
                "raw_frames_with_subtitles": len(raw_subtitle_frame_no_box_dict),
                "total_boxes_found": sum(len(boxes) for boxes in raw_subtitle_frame_no_box_dict.values()),
                "frame_numbers": frame_numbers[:20],  # First 20 frame numbers for inspection
                "sample_boxes": [raw_subtitle_frame_no_box_dict[frame_numbers[i]] for i in range(min(5, len(frame_numbers)))]
            }
        }
        
        # Save debug output
        json_file_path = PROCESSED_FILES_DIR / f"{original_name}_debug_raw.json"
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file, indent=4)

        # Store results for editor
        task_id = str(uuid.uuid4())
        editor_url = str(request.url_for('get_editor', task_id=task_id)) + "?debug=true"
        TASK_RESULTS[task_id] = {
            "distinct_coords": simple_coords,
            "frame_intervals": simple_intervals,
            "original_filename": f"{original_name}_debug_raw",
            "video_path": temp_video_path,
            "video_width": video_width,
            "video_height": video_height,
            "timestamp": time.time(),
            "user_id": user_id,
            "debug_info": json_content["debug_info"]  # Store debug info in task results
        }
        return {
            "task_id": task_id, 
            "user_id": user_id, 
            "editor_url": editor_url,
            "debug_info": json_content["debug_info"]
        }

    except Exception as e:
        print("Error in debug_raw_detection: ", e)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e


# @app.post("/find_subtitles/")
# async def find_subtitles(
#     request: Request,
#     file: Optional[UploadFile] = File(None),
#     url: Optional[str] = None,
#     cloud_ref: Optional[str] = None,
#     user_id: Optional[str] = None):
#     # Use the original filename (without extension) for the output JSON
#     original_name = Path(file.filename).stem if file else "unknown"
#     temp_video_path = None

#     if user_id is None:
#         user_id = str(uuid.uuid4())

#     if url:
#         temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
#         await download_file(url, temp_video_path)
#     elif cloud_ref:
#         # Assuming cloud_ref is a URL or a cloud storage path
#         temp_video_path = f"/tmp/{uuid.uuid4()}.mp4"
#         await download_file(cloud_ref, temp_video_path)
#     else:
#         temp_video_path = save_temp_file(file)

#     try:
#         # Check memory before processing
#         if not check_memory_usage():
#             raise HTTPException(status_code=503, detail="Server memory limit reached. Please try again later.")
        
#         # Use the global subtitle detector and update its video path
#         subtitle_detect = ModelManager.subtitle_detector
#         subtitle_detect.video_path = temp_video_path
        
#         subtitle_frame_no_box_dict = subtitle_detect.find_subtitle_frame_no()
#         if not subtitle_frame_no_box_dict:
#             raise HTTPException(status_code=404, detail="No subtitles found in the video.")

#         complete_subtitle_frame_no_box_dict = subtitle_detect.prevent_missed_detection(subtitle_frame_no_box_dict)
#         cap = cv2.VideoCapture(temp_video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         cap.release()

#         # Filter out mistake subtitle areas by checking the fps
#         correct_subtitle_frame_no_box_dict = subtitle_detect.filter_mistake_sub_area(complete_subtitle_frame_no_box_dict, fps)

#         # Instead of using first_entry_dict, find the most representative subtitle area for each frame
#         def get_most_representative_box(boxes):
#             """Get the most representative subtitle box from a list of boxes, with validation."""
#             if not boxes:
#                 return None
#             if len(boxes) == 1:
#                 box = boxes[0]
#                 # Validate that this is a meaningful subtitle box
#                 xmin, xmax, ymin, ymax = box
#                 width = xmax - xmin
#                 height = ymax - ymin
#                 area = width * height
                
#                 # Filter out boxes that are too small or have wrong aspect ratio
#                 # Typical subtitles should be wider than they are tall
#                 if area < 500 or height > width or width < 50 or height < 15:
#                     return None
#                 return box
            
#             # Calculate area for each box and find valid subtitle candidates
#             valid_boxes = []
#             for box in boxes:
#                 xmin, xmax, ymin, ymax = box
#                 width = xmax - xmin
#                 height = ymax - ymin
#                 area = width * height
                
#                 # Filter out invalid boxes
#                 if area >= 500 and height <= width and width >= 50 and height >= 15:
#                     valid_boxes.append((area, box))
            
#             if not valid_boxes:
#                 return None
                
#             # Return the box with largest area among valid candidates
#             return max(valid_boxes, key=lambda x: x[0])[1]

#         # Get representative boxes with better validation
#         representative_boxes_dict = {}
#         for frame_no, boxes in correct_subtitle_frame_no_box_dict.items():
#             if boxes:
#                 representative_box = get_most_representative_box(boxes)
#                 if representative_box is not None:  # Only include frames with valid subtitle boxes
#                     representative_boxes_dict[frame_no] = representative_box
        
#         print(f"[DEBUG] Frames with valid subtitles: {len(representative_boxes_dict)} out of {len(correct_subtitle_frame_no_box_dict)}")
        
#         # Use the global OCR recogniser
#         ocr_recogniser = ModelManager.ocr_recogniser
#         ocr_cache = {}
        
#         # Create initial intervals using the relaxed continuous ranges method
#         initial_intervals = subtitle_detect.find_continuous_ranges_with_same_mask(representative_boxes_dict)
#         print(f"[DEBUG] Initial intervals: {len(initial_intervals)}")
        
#         # Now create intelligent merged intervals based on coordinate similarity with OCR validation
#         def create_coordinate_based_intervals(intervals, boxes_dict, video_cap, ocr_recogniser, ocr_cache):
#             """Create merged intervals based primarily on coordinate similarity, with optional OCR validation."""
#             if not intervals:
#                 return [], []
            
#             # Calculate representative box for each interval
#             interval_data = []
#             for start, end in intervals:
#                 # Get all boxes in this interval
#                 interval_boxes = [
#                     boxes_dict[i] for i in range(start, end + 1) 
#                     if i in boxes_dict and boxes_dict[i] is not None
#                 ]
                
#                 if interval_boxes:
#                     # Use the bounding box that encompasses all boxes in the interval
#                     representative_box = find_smallest_bounding_box(interval_boxes)
                    
#                     # Validate the merged box makes sense as a subtitle
#                     xmin, xmax, ymin, ymax = representative_box
#                     width = xmax - xmin
#                     height = ymax - ymin
#                     area = width * height
                    
#                     # Only include intervals with reasonable subtitle dimensions
#                     if area >= 500 and width >= 50 and height >= 15:
#                         # Get representative frame from middle of interval for OCR
#                         mid_frame = (start + end) // 2
#                         interval_data.append(((start, end), representative_box, mid_frame))
            
#             if not interval_data:
#                 return [], []
            
#             print(f"[DEBUG] Valid interval data: {len(interval_data)}")
            
#             # Apply coordinate-based intelligent merging with OCR validation
#             merged_intervals = [interval_data[0][0]]
#             merged_coords = [interval_data[0][1]]
            
#             for i in range(1, len(interval_data)):
#                 current_interval, current_box, current_frame = interval_data[i]
#                 last_interval = merged_intervals[-1]
#                 last_box = merged_coords[-1]
#                 _, _, last_frame = interval_data[i-1]
                
#                 current_start, current_end = current_interval
#                 last_start, last_end = last_interval
                
#                 # Calculate gap between intervals
#                 gap = current_start - last_end
                
#                 should_merge = False
#                 merge_reason = ""
                
#                 # Strategy 1: Very small gaps (1-2 frames) - always merge (detection jitter)
#                 if gap <= 2:
#                     should_merge = True
#                     merge_reason = f"small gap ({gap})"
                
#                 # Strategy 2: Coordinate-based similarity analysis
#                 elif gap <= 20:  # Only consider merging for reasonable gaps
#                     # Check coordinate similarity using enhanced geometric analysis
#                     coords_similar = analyze_coordinate_similarity(last_box, current_box)
                    
#                     if coords_similar['high_similarity']:
#                         should_merge = True
#                         merge_reason = f"high coordinate similarity (gap={gap})"
#                     elif coords_similar['medium_similarity'] and gap <= 10:
#                         should_merge = True
#                         merge_reason = f"medium coordinate similarity (gap={gap})"
#                     elif coords_similar['y_position_match'] and gap <= 5:
#                         # Same Y-position (horizontal subtitle line) with small gap
#                         should_merge = True  
#                         merge_reason = f"same Y-position (gap={gap})"
#                     elif coords_similar['medium_similarity'] and gap <= 15:
#                         # For medium similarity with larger gaps, use OCR as validation
#                         try:
#                             # Get frames for OCR validation
#                             video_cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame - 1)
#                             ret1, frame1 = video_cap.read()
                            
#                             video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
#                             ret2, frame2 = video_cap.read()
                            
#                             if ret1 and ret2:
#                                 # Crop frames to subtitle regions
#                                 last_xmin, last_xmax, last_ymin, last_ymax = last_box
#                                 current_xmin, current_xmax, current_ymin, current_ymax = current_box
                                
#                                 frame1_cropped = frame1[last_ymin:last_ymax, last_xmin:last_xmax]
#                                 frame2_cropped = frame2[current_ymin:current_ymax, current_xmin:current_xmax]
                                
#                                 # Use OCR to validate the coordinate-based decision
#                                 texts_similar = compare_ocr_result(
#                                     ocr_recogniser, frame1_cropped, last_frame, 
#                                     frame2_cropped, current_frame, ocr_cache, 
#                                     threshold=0.75  # Slightly lower threshold since coordinates are already similar
#                                 )
                                
#                                 if texts_similar:
#                                     should_merge = True
#                                     merge_reason = f"medium coordinate similarity + OCR validation (gap={gap})"
#                                 else:
#                                     merge_reason = f"medium coordinate similarity but different OCR text (gap={gap})"
                            
#                         except Exception as e:
#                             print(f"[WARNING] OCR validation failed: {e}")
#                             # Without OCR validation, be more conservative
#                             if coords_similar['high_similarity']:
#                                 should_merge = True
#                                 merge_reason = f"high coordinate similarity without OCR (gap={gap})"
                
#                 if should_merge:
#                     print(f"[DEBUG] Merging frames {last_start}-{last_end} and {current_start}-{current_end}: {merge_reason}")
#                     # Merge with previous interval
#                     merged_intervals[-1] = (last_start, current_end)
#                     if last_box and current_box:
#                         merged_coords[-1] = find_smallest_bounding_box([last_box, current_box])
#                     else:
#                         merged_coords[-1] = last_box or current_box
#                 else:
#                     print(f"[DEBUG] NOT merging frames {last_start}-{last_end} and {current_start}-{current_end}: {merge_reason if merge_reason else 'coordinates too different'}")
#                     # Keep as separate interval
#                     merged_intervals.append(current_interval)
#                     merged_coords.append(current_box)
            
#             return merged_intervals, merged_coords
        
#         def analyze_coordinate_similarity(box1, box2):
#             """
#             Enhanced coordinate similarity analysis based on corner proximity, not just center points.
#             This is more robust to changes in box size and aligns with user feedback.
#             """
#             if not box1 or not box2:
#                 return {'high_similarity': False, 'medium_similarity': False, 'y_position_match': False}
            
#             xmin1, xmax1, ymin1, ymax1 = box1
#             xmin2, xmax2, ymin2, ymax2 = box2
            
#             # Calculate dimensions
#             width1, height1 = xmax1 - xmin1, ymax1 - ymin1
#             width2, height2 = xmax2 - xmin2, ymax2 - ymin2
            
#             # --- Positional Difference Analysis (Corner-based) ---
#             ymin_diff = abs(ymin1 - ymin2)
#             ymax_diff = abs(ymax1 - ymax2)
#             xmin_diff = abs(xmin1 - xmin2)
#             xmax_diff = abs(xmax1 - xmax2)
            
#             # --- Size and Shape Analysis ---
#             area1, area2 = width1 * height1, width2 * height2
#             area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
            
#             # Classification criteria
#             high_similarity = (
#                 ymin_diff <= 10 and ymax_diff <= 10 and  # Very close Y position
#                 xmin_diff <= 25 and xmax_diff <= 25 and  # Close X position
#                 area_ratio >= 0.9  # Similar size (user-tuned)
#             )
            
#             medium_similarity = (
#                 ymin_diff <= 20 and ymax_diff <= 20 and  # Moderately close Y position
#                 xmin_diff <= 50 and xmax_diff <= 50 and  # Moderately close X position
#                 area_ratio >= 0.75 # Somewhat similar size
#             )
            
#             # Checks if boxes are on the same horizontal line
#             y_position_match = (ymin_diff <= 15 and ymax_diff <= 15)
            
#             return {
#                 'high_similarity': high_similarity,
#                 'medium_similarity': medium_similarity, 
#                 'y_position_match': y_position_match,
#                 'metrics': {
#                     'ymin_diff': ymin_diff,
#                     'ymax_diff': ymax_diff,
#                     'xmin_diff': xmin_diff,
#                     'xmax_diff': xmax_diff,
#                     'area_ratio': area_ratio
#                 }
#             }
        
#         # Apply coordinate-based interval merging with optional OCR validation
#         cap_for_ocr = cv2.VideoCapture(temp_video_path)  # Separate video capture for OCR validation
#         sub_frame_no_list_continuous, distinct_coords = create_coordinate_based_intervals(
#             initial_intervals, representative_boxes_dict, cap_for_ocr, ocr_recogniser, ocr_cache
#         )
#         cap_for_ocr.release()  # Clean up the OCR video capture
        
#         print(f"[DEBUG] After coordinate-based intelligent merging: {len(sub_frame_no_list_continuous)} intervals")

#         json_content = {
#             "distinct_coordinates": distinct_coords,
#             "frame_intervals": sub_frame_no_list_continuous
#         }
#         # Save as original_filename_sub.json
#         json_file_path = PROCESSED_FILES_DIR / f"{original_name}_sub.json"
#         with open(json_file_path, "w") as json_file:
#             json.dump(json_content, json_file, indent=4)

#         # Store results
#         task_id = str(uuid.uuid4())
#         editor_url = str(request.url_for('get_editor', task_id=task_id))
#         TASK_RESULTS[task_id] = {
#             "distinct_coords": distinct_coords,
#             "frame_intervals": sub_frame_no_list_continuous,
#             "original_filename": original_name,
#             "video_path": temp_video_path,
#             "video_width": video_width,
#             "video_height": video_height,
#             "timestamp": time.time(),
#             "user_id": user_id,
#         }
#         return {"task_id": task_id, "user_id": user_id, "editor_url": editor_url}

#     except Exception as e:
#         print("Error in find_subtitles: ", e)
#         if temp_video_path and os.path.exists(temp_video_path):
#             os.remove(temp_video_path)
#         raise e


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

@app.get("/task_debug_info/{task_id}", include_in_schema=False)
async def get_task_debug_info(task_id: str):
    """
    Returns debug information for a given task if available.
    """
    result = TASK_RESULTS.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    debug_info = result.get("debug_info")
    if not debug_info:
        raise HTTPException(status_code=404, detail="Debug information not available for this task")
    
    return {"debug_info": debug_info}

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
    
    # Validate that coordinates represent actual subtitles
    def is_valid_subtitle_box(coords):
        """Check if coordinates represent a valid subtitle box."""
        if not coords:
            return False
        xmin, xmax, ymin, ymax = coords
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        
        # Filter out invalid boxes (too small, wrong aspect ratio, etc.)
        return area >= 500 and height <= width and width >= 50 and height >= 15
    
    # Filter out intervals with invalid subtitle boxes
    valid_pairs = []
    for interval, coords in zip(intervals, distinct_coords):
        if is_valid_subtitle_box(coords):
            valid_pairs.append((interval, coords))
    
    if not valid_pairs:
        return [], []
    
    # Sort intervals by start frame
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[0][0])
    
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
        
        # Use the same conservative merging criteria as in find_subtitles:
        # 1. Very small gap (1-2 frames) - likely same subtitle with detection gaps
        # 2. Small gap (3-5 frames) + very similar boxes - same subtitle with brief pause
        # 3. No merging for gaps > 5 frames unless boxes are nearly identical
        should_merge = False
        
        if gap <= 2:
            # Very small gaps are likely detection inconsistencies
            should_merge = True
        elif gap <= 5 and boxes_similar:
            # Small gaps with similar boxes - brief pauses in same subtitle
            should_merge = True
        elif gap <= 10 and last_box and current_box:
            # Only merge larger gaps if boxes are nearly identical (same subtitle)
            xmin1, xmax1, ymin1, ymax1 = last_box
            xmin2, xmax2, ymin2, ymax2 = current_box
            
            # Very strict similarity check
            nearly_identical = (
                abs(xmin1 - xmin2) <= 10 and
                abs(xmax1 - xmax2) <= 10 and
                abs(ymin1 - ymin2) <= 10 and
                abs(ymax1 - ymax2) <= 10
            )
            
            if nearly_identical:
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