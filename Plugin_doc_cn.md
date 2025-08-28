# 视频字幕去除插件 — 用户指南

## 1. 概览

### 它能做什么？

Video Subtitle Remover 是一款用于**自动检测并移除视频内嵌字幕**的工具。它通过 **OCR** 识别文字区域，支持用户在界面中微调检测结果，然后对字幕区域进行修复，实现无痕去字。同时可将检测到的字幕文本导出为 **SRT** 或 **TXT** 文件。

### 适用人群

面向需要二次创作或再编辑、但无法获得无字幕原片的**内容创作者、视频剪辑师、个人用户**等。

### 核心功能

* **字幕检测**：自动识别每帧中的字幕位置。
* **文本提取（OCR）**：识别并导出字幕文本。
* **交互式编辑器**：提供 Web UI 用于复核与调整字幕的**边框与时间区间**。
* **视频修复（Inpainting）**：使用先进的修复模型对去字区域进行填补。
* **音频保留**：在最终输出视频中保留原始音轨。

---

## 2. 架构简介

本应用以 **FastAPI** 构建 Web 服务，后端使用 **Python** 实现核心视频处理能力。

### 主要依赖

* **后端框架**：FastAPI
* **视频处理**：OpenCV，FFmpeg
* **机器学习**：PyTorch，PaddlePaddle
* **OCR引擎**：PaddleOCR
* **修复模型**：ProPainter，STTN，LaMa
* **GPU加速**：CUDA/CuDNN（通过 `onnxruntime-gpu`）或 Windows 上的 DirectML

系统采用**任务化**处理流程：当视频上传后会创建唯一任务，字幕检测 / 文本提取 / 视频修复在**后台任务**中运行，用户可在 Web UI 中无阻塞地查看并编辑。

---

## 3. 环境与依赖要求（Requirements）

* **NVIDIA GPU**：强烈建议使用以获得最佳性能。
* **CUDA & cuDNN**：与 PyTorch / ONNX Runtime 搭配用于 GPU 加速。
* **FFmpeg**：需在系统 `PATH` 中可用（用于音频抽取与合并）。
* **Python**：版本 3.11。

---

## 4. 使用流程（User Flow）

1. **上传**：在 Web 界面上传视频文件。
2. **设置 Universal Box（可选）**：若字幕位置固定，可先定义单一矩形区域，跳过自动检测阶段（适合固定字幕位置的视频）。
3. **检测与提取**：后端检测字幕的**精确位置与时间**，同时通过 OCR 提取字幕文本。
4. **编辑界面**：进入编辑页，叠加显示检测到的字幕框与时间区间，用户可：

   * 调整每个字幕框的尺寸与位置；
   * 修改每段字幕的起止时间；
   * 合并、拆分或删除区间。
5. **视频修复（Inpainting）**：确认调整后开始修复。后端基于最终字幕区域生成 **mask**，对视频逐帧去字与填补。
6. **下载**：任务完成后，可下载**无字幕视频**；提取的字幕文本可另存为 `.txt`（或导出为 SRT）。

---

## 5. 配置与模型（Configuration & Models）

应用的行为可通过 `backend/config.py` 配置。

### 修复模型（Inpainting Models）

* **ProPainter（`InpaintMode.PROPAINTER`）**

  * **优点**：在复杂运动与背景场景中画质最佳。
  * **缺点**：资源开销最大、显存占用高、速度最慢。

* **STTN（`InpaintMode.STTN`）**

  * **优点**：速度最快，适合快速处理；在背景较静态（如访谈、演示）的视频上表现良好。
  * **缺点**：在动态场景或复杂纹理上质量可能不如 ProPainter。

* **LaMa（`InpaintMode.LAMA`）**

  * **优点**：静态图像效果优秀，在低运动视频上表现尚可。
  * **缺点**：在镜头/物体运动明显时，可能出现不稳定或“马赛克”类伪影。

### 关键配置参数（`backend/config.py`）

* `MODE`：默认修复模型（如 `InpaintMode.PROPAINTER`）。
* `REC_CHAR_TYPE`：OCR 语言（如 `'ch'` 表示中文，`'en'` 表示英文）。
* `THRESHOLD_TEXT_SIMILARITY`：提取阶段用于判定重复文本的相似度阈值。
* `PIXEL_TOLERANCE_Y` & `PIXEL_TOLERANCE_X`：在合并相近字幕框时允许的像素偏差。
* `PROPAINTER_MAX_LOAD_NUM`：ProPainter 单批处理的帧数；若出现显存不足，可适当降低。
* `DROP_SCORE`：字幕识别的置信度下限。默认为0.75，增大可以使识别结果更准确，但同时可能丢失较模糊的字幕。

---

## 6. 高级主题（Advanced Topics）

*（建设中）*

* 安全与隐私（Security & Privacy）
* 安装与部署（Installation & Deployment）
* API 集成（API Integration）
* 运行时内存信息（Runtime Memory Information）
* 运行成本（Running Costs）
* 兼容性问题（Compatibility Issues）
