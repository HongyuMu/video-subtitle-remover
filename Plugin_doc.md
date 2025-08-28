# Video Subtitle Remover Plugin - User Guide

## 1. At a Glance

### What it does?

The Video Subtitle Remover is a tool designed to automatically detect and remove hardcoded subtitles from videos. It uses Optical Character Recognition (OCR) to identify text areas, allows for user adjustments, and then inpaints the video to erase the subtitles seamlessly. It can also extract the detected subtitle text into SRT or TXT files.

### Who is it for?

This tool is primarily for content creators, video editors, and individuals who need to repurpose or re-edit video content where the original subtitle-free version is not available.

### Core Functions

- **Subtitle Detection:** Automatically identifies the location of subtitles in each frame.
- **Text Extraction (OCR):** Recognizes and extracts the subtitle text.
- **Interactive Editor:** Provides a web-based UI to review and adjust the detected subtitle bounding boxes and time intervals.
- **Video Inpainting:** Employs advanced inpainting models to fill in the areas where subtitles have been removed.
- **Audio Preservation:** Retains the original audio track in the final processed video.

## 2. Architecture Overview

The application is built as a web service using the **FastAPI** framework. The backend, written in Python, handles all the heavy lifting of video processing.

### Key Dependencies:

- **Backend Framework:** FastAPI
- **Video Processing:** OpenCV, FFmpeg
- **Machine Learning:** PyTorch, PaddlePaddle
- **OCR Engine:** PaddleOCR
- **Inpainting Models:** ProPainter, STTN, LaMa
- **GPU Acceleration:** CUDA/CuDNN (via `onnxruntime-gpu`) or DirectML on Windows.

The system is designed to process videos in a task-based manner. When a video is uploaded, a unique task is created. The subtitle detection, extraction, and inpainting processes are run as background tasks, allowing the user to interact with the web UI without blocking.

## 3. Requirements

To run this application, the following dependencies must be installed on your system:

- **NVIDIA GPU:** For optimal performance, an NVIDIA GPU is highly recommended.
- **CUDA & cuDNN:** Required for GPU acceleration with PyTorch and ONNX Runtime.
- **FFmpeg:** Must be available in the system's PATH for audio extraction and merging.
- **Python:** Version 3.11.

## 4. User Flow

The process of removing subtitles follows a straightforward workflow:

1.  **Upload:** The user uploads a video file through the web interface.
2.  **Set Universal Box (Optional):** The user can define a single rectangular area where all subtitles appear. This skips the auto-detection phase and is useful for videos with static subtitle positions.
3.  **Detect & Extract:** The backend processes the video to detect the precise location and timing of subtitles. The subtitle text is also extracted via OCR.
4.  **Edit UI:** The user is redirected to an editor page. This interface displays the video with the detected subtitle boxes overlaid and shows the corresponding time intervals. Here, the user can:
    - Adjust the size and position of each subtitle box.
    - Modify the start and end times for each subtitle interval.
    - Merge, split, or delete intervals.
5.  **Inpainting:** Once the user is satisfied with the adjustments, they start the inpainting process. The backend uses the final subtitle areas to generate a mask and removes them from the video frames.
6.  **Download:** After processing is complete, the user can download the final, subtitle-free video. The extracted subtitle text can also be downloaded as a `.txt` file.

## 5. Configuration & Models

The application's behavior can be customized by editing the `backend/config.py` file.

### Inpainting Models

You can choose one of three inpainting models, each with its own strengths:

-   **ProPainter (`InpaintMode.PROPAINTER`)**
    -   **Pros:** Delivers the highest quality results, especially for videos with complex motion and backgrounds.
    -   **Cons:** Very resource-intensive, requiring a significant amount of GPU memory. It is the slowest of the three models.
-   **STTN (`InpaintMode.STTN`)**
    -   **Pros:** The fastest model, suitable for quick processing. Performs well on videos with relatively static backgrounds (e.g., interviews, presentations).
    -   **Cons:** May produce lower quality results on videos with dynamic scenes or complex textures.
-   **LaMa (`InpaintMode.LAMA`)**
    -   **Pros:** Excellent for static images and performs reasonably well on videos with minimal motion.
    -   **Cons:** Can produce unstable or mosaic-like artifacts in videos with significant camera or object movement.

### Key Configuration Parameters (`backend/config.py`)

-   `MODE`: Set the default inpainting model (e.g., `InpaintMode.PROPAINTER`).
-   `REC_CHAR_TYPE`: Sets the language for OCR (e.g., `'ch'` for Chinese, `'en'` for English).
-   `THRESHOLD_TEXT_SIMILARITY`: Controls how similar two pieces of text must be to be considered duplicates during the extraction phase.
-   `PIXEL_TOLERANCE_Y` & `PIXEL_TOLERANCE_X`: Defines the pixel deviation allowed when grouping nearby subtitle boxes into a single region.
-   `PROPAINTER_MAX_LOAD_NUM`: The number of frames to process in a single batch for ProPainter. Lower this value if you encounter out-of-memory errors.
-   `DROP_SCORE`: The threshold of confidence for subtitle detection, default to be 0.75. Increase for more accurate detection, but might filter out blurred real subtitles in some cases.

## 6. Advanced Topics

*(Under Construction)*

-   Security and Privacy
-   Installation and Deployment
-   API Integration
-   Runtime Memory Information
-   Running Costs
-   Compatibility Issues
