[简体中文](README.md) | English

## Project Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

Video-subtitle-remover (VSR) is an AI-based software that removes hardcoded subtitles from videos. It mainly implements the following functionalities:

- **Lossless resolution**: Removes hardcoded subtitles from videos and generates files without subtitles.
- Fills in the removed subtitle text area using a powerful AI algorithm model (non-adjacent pixel filling and mosaic removal).
- Supports custom subtitle positions by only removing subtitles in the defined location (input position).
- Supports automatic removal of all text throughout the entire video (without inputting a position).
- Supports multi-selection of images for batch removal of watermark text.

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png" alt="demo.png"/></p>

> Download the .zip package directly, extract, and run it. If it cannot run, follow the tutorial below to try installing the conda environment and running the source code.

**Download Links:**

Windows GPU Version v1.1.0 (GPU):

- Baidu Cloud Disk: <a href="https://pan.baidu.com/s/1zR6CjRztmOGBbOkqK8R1Ng?pwd=vsr1">vsr_windows_gpu_v1.1.0.zip</a> Extraction Code: **vsr1**

- Google Drive: <a href="https://drive.google.com/drive/folders/1NRgLNoHHOmdO4GxLhkPbHsYfMOB_3Elr?usp=sharing">vsr_windows_gpu_v1.1.0.zip</a>


**Pre-built Package Comparison**:

| Pre-built Package Name          | Python | Paddle | Torch | Environment                       | Supported Compute Capability Range |
|----------------------------------|--------|--------|--------|-----------------------------------|------------------------------------|
| `vse-windows-directml.7z`        | 3.12   | 3.0.0 | 2.4.1 | Windows without Nvidia GPU         | Universal                         |
| `vse-windows-nvidia-cuda-11.8.7z`| 3.12   | 3.0.0 | 2.7.0 | CUDA 11.8                         | 3.5 – 8.9                          |
| `vse-windows-nvidia-cuda-12.6.7z`| 3.12   | 3.0.0 | 2.7.0 | CUDA 12.6                         | 5.0 – 8.9                          |
| `vse-windows-nvidia-cuda-12.8.7z`| 3.12   | 3.0.0 | 2.7.0 | CUDA 12.8                         | 5.0 – 9.0+                          |

> NVIDIA provides a list of supported compute capabilities for each GPU model. You can refer to the following link: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) to check which CUDA version is compatible with your GPU.

**Docker Versions:**
```shell
  # Nvidia 10, 20, 30 Series Graphics Cards
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda11.8

  # Nvidia 40 Series Graphics Cards
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda12.6

  # Nvidia 50 Series Graphics Cards
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda12.8

  # AMD / Intel Dedicated or Integrated Graphics
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-directml

  # Demo video, input
  /vsr/test/test.mp4
  docker cp vsr:/vsr/test/test_no_sub.mp4 ./
```

## Demonstration

- GUI:

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="demo2.gif"/></p>

- <a href="https://b23.tv/guEbl9C">Click to view demo video👇</a>

<p style="text-align:center;"><a href="https://b23.tv/guEbl9C"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/></a></p>

## Source Code Usage Instructions

#### 1. Install Python

Please ensure that you have installed Python 3.12+.

- Windows users can go to the [Python official website](https://www.python.org/downloads/windows/) to download and install Python.
- MacOS users can install using Homebrew:
  ```shell
  brew install python@3.12
  ```
- Linux users can install via the package manager, such as on Ubuntu/Debian:
  ```shell
  sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
  ```

#### 2. Install Dependencies

It is recommended to use a virtual environment to manage project dependencies to avoid conflicts with the system environment.

(1) Create and activate the virtual environment:
```shell
python -m venv videoEnv
```

- Windows:
```shell
videoEnv\\Scripts\\activate
```
- MacOS/Linux:
```shell
source videoEnv/bin/activate
```

#### 3. Create and Activate Project Directory

Change to the directory where your source code is located:
```shell
cd <source_code_directory>
```
> For example, if your source code is in the `tools` folder on the D drive and the folder name is `video-subtitle-remover`, use:
> ```shell
> cd D:/tools/video-subtitle-remover-main
> ```

#### 4. Install the Appropriate Runtime Environment

This project supports two runtime modes: CUDA (NVIDIA GPU acceleration) and DirectML (AMD, Intel, and other GPUs/APUs).

##### (1) CUDA (For NVIDIA GPU users)

> Make sure your NVIDIA GPU driver supports the selected CUDA version.

- Recommended CUDA 11.8, corresponding to cuDNN 8.6.0.

- Install CUDA:
  - Windows: [Download CUDA 11.8](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe)
  - Linux:
    ```shell
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
  - CUDA is not supported on MacOS.

- Install cuDNN (CUDA 11.8 corresponds to cuDNN 8.6.0):
  - [Windows cuDNN 8.6.0 Download](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip)
  - [Linux cuDNN 8.6.0 Download](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz)
  - Follow the installation guide in the NVIDIA official documentation.

- Install PaddlePaddle GPU version (CUDA 11.8):
  ```shell
  pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```

- Install Torch GPU version (CUDA 11.8):
  ```shell
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install other dependencies:
  ```shell
  pip install -r requirements.txt
  ```

##### (2) DirectML (For AMD, Intel, and other GPU/APU users)

- Suitable for Windows devices with AMD/NVIDIA/Intel GPUs.
- Install ONNX Runtime DirectML version:
  ```shell
  pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install -r requirements.txt
  pip install torch_directml==0.2.5.dev240914
  ```

## Web UI User Instructions

#### 1. Start the Web server

Make sure `fastapi` and `uvicorn` are installed.

```shell
pip install fastapi uvicorn
```

Run from the project root:

```shell
uvicorn app:app --host 0.0.0.0 --port 8002
```

#### 2. Access the upload page

Open your browser and visit `http://evowork.tech:8002` (or `http://localhost:8002` if running locally). You will see the video upload page.

#### 3. Upload and prepare the video

Upload your video file. After the upload completes, you will be redirected to the subtitle editor page. Notice that there is a file size limit of 80MB due to ProPainter's high requirement on GPU memory. Check FAQ #6 for more information.

#### 4. Subtitle region detection and OCR

You will see two subtitle detection options in the top-right of the editor UI:

- **Use Universal Subtitle Box (recommended):** A green box will appear on the video. Play the video and adjust this box to cover the typical subtitle area across the video. Click “Set and Generate Subtitles” when ready.
- **Use Automatic Detection:** Let the backend find subtitle regions automatically.

The backend will then run OCR to detect and extract text, generate SRT subtitles, and embed the SRT into the UI. This may take some time; a progress bar shows real-time status.

#### 5. Adjust subtitles

- **Navigate intervals:** Use “Prev”/“Next” buttons or click a subtitle line on the left scroller panel. The timeline shows the current position.
- **Adjust the text box:** For each interval, a box appears on the video preview. You can:
  - Drag the box or edges directly on the video to resize.
  - Adjust X/Y/Width/Height via sliders or inputs.
- **Adjust interval range:** Modify “Start Frame” and “End Frame” to precisely cover the subtitle duration.
- **Advanced controls:**
  - **Use previous box:** Apply the previous interval’s box to the current one.
  - **Not a subtitle:** Mark an interval to skip during processing.
  - **Split current interval:** Split at the current frame if multiple subtitles exist within one interval.
  - **Merge with previous:** Merge the current interval with the previous one.
  - **Reset box:** Restore the box to initial coordinates.

#### 6. Process the video

When satisfied, click “Finish and Process” and choose an inpainting mode:

- **STTN:** Fast and memory-efficient; suitable for stable scenes.
- **Lama:** Good for anime or still images.
- **ProPainter:** Highest quality, especially for fast-paced videos. It applies multiprocessing acceleration on systems with multiple GPU cores. ProPainter uses significantly more memory/time. For example, a 720p ~2000-frame video may take ~10 minutes.

Processing will start in the backend; a progress bar will show the status.

#### 7. Download results

A download link will be provided for the final processed video.

## Frequently Asked Questions
0. How to view the exact frame I want

Click on the slider below the Timeline section, then you can drag with mouse or move forward/back using keyboard control. You can adjust the step size for multi-frame movements.

1. What to do if you are not satisfied with the removal quality

You can review training methods in the `design` folder and use the code under `backend/tools/train` to train your own models, then replace the old models.

2. Generated subtitles are empty or missing

Try the following to improve results:

- In the editor, use the “Select and Generate” mode and manually cover the subtitle area before generating.
- Increase frame extraction frequency in `backend/config.py` (e.g., from 3 to 5–8):
```python
EXTRACT_FREQUENCY = 6
```
- Loosen detection/recognition thresholds in `backend/config.py`:
```python
DET_DB_BOX_THRESH = 0.5  # default at 0.6
DROP_SCORE = 0.6         # default at 0.75
```
- Ensure the recognition language is correct (`settings.ini` `Language` or use the default auto mode). For higher accuracy, set mode to `accurate`:
```ini
[DEFAULT]
Mode = accurate
```
- For very short intervals being dropped, lower `MIN_INTERVAL_LEN` in `backend/config.py` to avoid over-merge or discard:
```python
MIN_INTERVAL_LEN = 3
```

3. Out of memory when using ProPainter

The project should be able to adjust batch size using a formula involving training resolution, actual resolution, and training batch size. If OOM persists, modify `PROPAINTER_MAX_LOAD_NUM` in `backend/config.py` to recalculate the maximum frames per group.

4. CondaHTTPError

Place the `.condarc` file from the project into the user directory (`C:/Users/<your_username>`). Overwrite if it already exists.

Solution: https://zhuanlan.zhihu.com/p/260034241

5. 7z file extraction error

Solution: upgrade 7-Zip to the latest version.

6. Upload failed or file too large

The default upload size limit is 80 MB. You can adjust it in `app.py` and restart the service (note: 80 MB is close to the VRAM limit for ProPainter on RTX 3090 4-core GPU setups; increase with caution if you plan to use ProPainter):
```python
# app.py
MAX_UPLOAD_MB = 200  # increase to 200 MB
```

