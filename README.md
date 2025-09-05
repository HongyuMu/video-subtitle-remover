简体中文 | [English](README_en.md)

## 项目简介

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)  

Video-subtitle-remover (VSR) 是一款基于AI技术，将视频中的硬字幕去除的软件。
主要实现了以下功能：
- **无损分辨率**将视频中的硬字幕去除，生成去除字幕后的文件
- 通过超强AI算法模型，对去除字幕文本的区域进行填充（非相邻像素填充与马赛克去除）
- 支持自定义字幕位置，仅去除定义位置中的字幕（传入位置）
- 支持全视频自动去除所有文本（不传入位置）
- 支持多选图片批量去除水印文本

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png" alt="demo.png"/></p>

**使用说明：**

- 有使用问题请加群讨论，QQ群：210150985（已满）、806152575（已满）、816881808（已满）、295894827
- 直接下载压缩包解压运行，如果不能运行再按照下面的教程，尝试源码安装conda环境运行

**下载地址：**

Windows GPU版本v1.1.0（GPU）：

- 百度网盘:  <a href="https://pan.baidu.com/s/1zR6CjRztmOGBbOkqK8R1Ng?pwd=vsr1">vsr_windows_gpu_v1.1.0.zip</a> 提取码：**vsr1**

- Google Drive:  <a href="https://drive.google.com/drive/folders/1NRgLNoHHOmdO4GxLhkPbHsYfMOB_3Elr?usp=sharing">vsr_windows_gpu_v1.1.0.zip</a>

**预构建包对比说明**：
|       预构建包名          | Python  | Paddle | Torch | 环境                          | 支持的计算能力范围|
|---------------|------------|--------------|--------------|-----------------------------|----------|
| `vsr-windows-directml.7z`  | 3.12       | 3.0.0       | 2.4.1       | Windows 非Nvidia显卡             | 通用 |
| `vsr-windows-nvidia-cuda-11.8.7z` | 3.12       | 3.0.0        | 2.7.0       | CUDA 11.8   | 3.5 – 8.9 |
| `vsr-windows-nvidia-cuda-12.6.7z` | 3.12       | 3.0.0       | 2.7.0       | CUDA 12.6   | 5.0 – 8.9 |
| `vsr-windows-nvidia-cuda-12.8.7z` | 3.12       | 3.0.0       | 2.7.0       | CUDA 12.8   | 5.0 – 9.0+ |

> NVIDIA官方提供了各GPU型号的计算能力列表，您可以参考链接: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) 查看你的GPU适合哪个CUDA版本

**Docker版本：**
```shell
  # Nvidia 10 20 30系显卡
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda11.8

  # Nvidia 40系显卡
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda12.6

  # Nvidia 50系显卡
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-cuda12.8

  # AMD / Intel 独显 集显
  docker run -it --name vsr --gpus all eritpchy/video-subtitle-remover:1.1.1-directml

  # 演示视频, 输入
  /vsr/test/test.mp4
  docker cp vsr:/vsr/test/test_no_sub.mp4 ./
```

## 演示

- GUI版：

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="demo2.gif"/></p>

- <a href="https://b23.tv/guEbl9C">点击查看演示视频👇</a>

<p style="text-align:center;"><a href="https://b23.tv/guEbl9C"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/></a></p>

## 源码使用说明


#### 1. 安装 Python

请确保您已经安装了 Python 3.12+。

- Windows 用户可以前往 [Python 官网](https://www.python.org/downloads/windows/) 下载并安装 Python。
- MacOS 用户可以使用 Homebrew 安装：
  ```shell
  brew install python@3.12
  ```
- Linux 用户可以使用包管理器安装，例如 Ubuntu/Debian：
  ```shell
  sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
  ```

#### 2. 安装依赖文件

请使用虚拟环境来管理项目依赖，避免与系统环境冲突。

（1）创建虚拟环境并激活
```shell
python -m venv videoEnv
```

- Windows：
```shell
videoEnv\\Scripts\\activate
```
- MacOS/Linux：
```shell
source videoEnv/bin/activate
```

#### 3. 创建并激活项目目录

切换到源码所在目录：
```shell
cd <源码所在目录>
```
> 例如：如果您的源代码放在 D 盘的 tools 文件夹下，并且源代码的文件夹名为 video-subtitle-remover，则输入：
> ```shell
> cd D:/tools/video-subtitle-remover-main
> ```

#### 4. 安装合适的运行环境

本项目支持 CUDA（NVIDIA显卡加速）和 DirectML（AMD、Intel等GPU/APU加速）两种运行模式。

##### (1) CUDA（NVIDIA 显卡用户）

> 请确保您的 NVIDIA 显卡驱动支持所选 CUDA 版本。

- 推荐 CUDA 11.8，对应 cuDNN 8.6.0。

- 安装 CUDA：
  - Windows：[CUDA 11.8 下载](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe)
  - Linux：
    ```shell
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
  - MacOS 不支持 CUDA。

- 安装 cuDNN（CUDA 11.8 对应 cuDNN 8.6.0）：
  - [Windows cuDNN 8.6.0 下载](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip)
  - [Linux cuDNN 8.6.0 下载](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz)
  - 安装方法请参考 NVIDIA 官方文档。

- 安装 PaddlePaddle GPU 版本（CUDA 11.8）：
  ```shell
  pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```
- 安装 Torch GPU 版本（CUDA 11.8）：
  ```shell
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
  ```

- 安装其他依赖
  ```shell
  pip install -r requirements.txt
  ```

##### (2) DirectML（AMD、Intel等GPU/APU加速卡用户）

- 适用于 Windows 设备的 AMD/NVIDIA/Intel GPU。
- 安装 ONNX Runtime DirectML 版本：
  ```shell
  pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install -r requirements.txt
  pip install torch_directml==0.2.5.dev240914
  ```

## Web UI使用说明

本指南将说明如何使用基于Web的编辑器进行更精确的字幕去除。

#### 1. 启动Web服务器

首先，请确保已安装所有依赖项，包括 `fastapi` 和 `uvicorn`。

```shell
pip install fastapi uvicorn
```

然后，从项目根目录运行Web应用程序：

```shell
uvicorn app:app --host 0.0.0.0 --port 8002
```

服务器将在您的本地计算机上运行。

#### 2. 访问上传页面

打开您的Web浏览器并访问 `http://evowork.tech:8002` (如果在本地访问，则为 `http://localhost:8002`)。您将看到视频上传页面。

#### 3. 上传和准备视频

上传您的视频文件。上传完成后，您将被重定向到字幕编辑器页面。

#### 4. 字幕区域检测和OCR

在编辑器中，UI右上角会出现两种字幕检测选项：

*   **通用文本框（推荐）：** 视频上会出现一个绿色框。您可以播放视频，同时调整此文本框以覆盖整个视频中字幕出现的常规区域。这有助于集中OCR过程。设置好后，单击“设置并生成字幕”。
*   **自动检测：** 单击“使用自动检测”让后端自动查找字幕区域。

然后，后端将执行OCR以检测，提取文本，生成SRT格式字幕，并将SRT文件嵌入用户界面。这可能需要一些时间。您可以通过进度条获取实时提取进度。

#### 5. 调整字幕

生成字幕后，为了确保字幕移除的效果，您可以对其进行微调：

*   **导航区间：** 使用“上一个”和“下一个”按钮或单击左侧面板中的字幕，在不同的字幕段之间跳转。时间轴也会显示当前位置。
*   **调整文本框：** 对于每个区间，视频上都会显示一个文本框。您可以使用以下方式调整其位置和大小：
    *   直接在视频预览上单击并拖动框或其边缘。
    *   拖拽X、Y、宽度、高度滑块或输入数值。
*   **调整区间范围：** 您可以更改每个字幕区间的“开始帧”和“结束帧”，以确保它覆盖确切的字幕持续时间。
*   **高级控件：**
    *   **使用上一个文本框：** 在本区间套用上一个区间中字幕的文本框
    *   **不是字幕：** 标记一个区间以便在处理过程中跳过。
    *   **拆分当前区间：** 如果一个区间包含多个不同的字幕，您可以在当前帧处将其拆分。
    *   **与上一个合并：** 将当前区间与前一个区间合并。
    *   **重置文本框：** 重置文本框为初始坐标

#### 6. 处理视频

当您对所有调整都满意后，单击“完成并处理”。将出现一个对话框，要求您选择修复模式：

*   **STTN：** 速度快，内存效率高，适用于画面稳定的视频。
*   **Lama：** 适用于动漫内容或图片处理
*   **ProPainter：** 质量最高，尤其适用于运动剧烈的视频。此模式在具有多个GPU核的系统上可以利用多进程加速处理。它会消耗更多的内存和时间。E.g. 一个720p，约2000帧的视频大约需要10分钟处理完成。

选择模式并确认后，修复过程将在后端开始。进度条将显示状态。

#### 7. 下载结果

处理完成后，将提供最终视频（已去除字幕）的下载链接。

## 常见问题
0. 如何精确到帧地查看字幕内容

点击右上角Timeline标签下的滑块，可以用鼠标拖拽或使用键盘左右方向键移动。可以通过修改Timeline Step一次性移动多帧

1. 对模型去字幕的效果不满意怎么办

可以查看design文件夹里面的训练方法，利用backend/tools/train里面的代码进行训练，然后将训练的模型替换旧模型即可

2. 生成的字幕为空或缺失怎么办

可以按以下步骤排查与优化：

- 在编辑器中使用“通用文本框”模式，手动框住字幕大致区域后再生成字幕。
- 提高帧抓取频率，在 `backend/config.py` 中增大 `EXTRACT_FREQUENCY`（例如从 3 提高到 5-8）：

```python
EXTRACT_FREQUENCY = 6
```

- 放宽检测/识别阈值，在 `backend/config.py` 中适当降低：

```python
DET_DB_BOX_THRESH = 0.5  # 原为 0.6
DROP_SCORE = 0.6         # 原为 0.75
```

- 确认识别语言正确（`settings.ini` 中 `Language` 或使用默认自动模式）；需要更高准确率时将模式设为 `accurate`：

```ini
[DEFAULT]
Mode = accurate
```

- 对于被丢弃的超短区间，可在 `backend/config.py` 中降低 `MIN_INTERVAL_LEN`，以避免过度合并或丢弃：

```python
MIN_INTERVAL_LEN = 3
```

3. ProPainter处理时显存不够怎么办

对于不同的分辨率，本项目理论上可以通过计算测试视频文件的分辨率和每组的帧数调整处理输入视频时的batch_num。如果仍然出现显存不够的情况，您可以修改backend/config.py中的PROPAINTER_MAX_LOAD_NUM来重新计算每一组的最大处理帧数。

4. CondaHTTPError

将项目中的.condarc放在用户目录下(C:/Users/<你的用户名>)，如果用户目录已经存在该文件则覆盖

解决方案：https://zhuanlan.zhihu.com/p/260034241

5. 7z文件解压错误

解决方案：升级7-zip解压程序到最新版本

6. 上传失败或文件过大怎么办

默认上传大小限制为 80MB。您可以在 `app.py` 中调整限制后重启服务（注意：80MB非常接近ProPainter模式下RTX 3090四卡显存的处理上限，所以如果您希望使用ProPainter模式，请谨慎调整上传限制）：

```python
# app.py
MAX_UPLOAD_MB = 200  # 将限制提高到200MB
```

如果使用 URL/云端方式上传（表单字段 `url` 或 `cloud_ref`），同样受此限制控制。过大的文件建议先本地压缩或裁剪后再上传。