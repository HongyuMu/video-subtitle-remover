import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import onnxruntime as ort
import paddle
import configparser
from pathlib import Path
import pynvml

def get_best_gpu():
    """Selects the GPU with the most free memory. Returns GPU index or None."""
    if pynvml is None:
        print("pynvml module not found, cannot select best GPU.")
        return 0 # Fallback to GPU 0 if pynvml not installed
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return None

        best_gpu_index = -1
        max_free_memory = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if mem_info.free > max_free_memory:
                max_free_memory = mem_info.free
                best_gpu_index = i
        
        pynvml.nvmlShutdown()

        if best_gpu_index != -1:
            return best_gpu_index
            
    except pynvml.NVMLError:
        print("NVIDIA driver not found, cannot select best GPU.")
        return 0 # Fallback to GPU 0
    except Exception as e:
        print(f"An error occurred during GPU selection: {e}")
        return 0 # Fallback to GPU 0
    return None

# 项目版本号
VERSION = "1.1.1"
# ×××××××××××××××××××× [不要改] start ××××××××××××××××××××
logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印

USE_DML = False
device = None

try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        USE_DML = True
        print("Using DirectML device.")
    else:
        raise ImportError
except (ImportError, Exception):
    USE_DML = False
    if torch.cuda.is_available():
        best_gpu_id = get_best_gpu()
        if best_gpu_id is not None:
            device = torch.device(f"cuda:{best_gpu_id}")
            torch.cuda.set_device(device)
            if paddle.is_compiled_with_cuda():
                paddle.set_device(f'gpu:{best_gpu_id}')
        else:
            device = torch.device("cuda:0")
            if paddle.is_compiled_with_cuda():
                paddle.set_device('gpu:0')
            print("No available GPU found, fallback to cuda:0")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    print(f"Using best available GPU: {device}")

if device is None:
    device = torch.device("cpu")
    print("Device selection failed, defaulting to CPU.")

USE_GPU = (device.type != 'cpu') or USE_DML

BASE_DIR = str(Path(os.path.abspath(__file__)).parent)
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
# Default language for OCR. Can be changed to 'en', 'japan', 'korean', etc.
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')

settings_config = configparser.ConfigParser()
MODE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')):
    # 如果没有配置文件，默认使用中文
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini'), mode='w', encoding='utf-8') as f:
        f.write('[DEFAULT]\n')
        f.write('Interface = 简体中文\n')
        f.write('Language = ch\n')
        f.write('Mode = fast')
settings_config.read(MODE_CONFIG_PATH, encoding='utf-8')

interface_config = configparser.ConfigParser()
INTERFACE_KEY_NAME_MAP = {
    '简体中文': 'ch',
    '繁體中文': 'chinese_cht',
    'English': 'en',
    '한국어': 'ko',
    '日本語': 'japan',
    'Tiếng Việt': 'vi',
    'Español': 'es'
}
interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface',
                              f"{INTERFACE_KEY_NAME_MAP[settings_config['DEFAULT']['Interface']]}.ini")
interface_config.read(interface_file, encoding='utf-8')

# 指定ffmpeg可执行程序路径
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# 将ffmpeg添加可执行权限
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Whether to use ONNX for acceleration on non-Nvidia GPUs (DirectML, AMD, Intel, Apple)
ONNX_PROVIDERS = []
if USE_GPU == False:
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        for provider in available_providers:
            if provider in [
                "CPUExecutionProvider"
            ]:
                continue
            if provider not in [
                "DmlExecutionProvider",         # DirectML，适用于 Windows GPU
                "ROCMExecutionProvider",        # AMD ROCm
                "MIGraphXExecutionProvider",    # AMD MIGraphX
                # "VitisAIExecutionProvider",   # AMD VitisAI，适用于 RyzenAI & Windows
                "OpenVINOExecutionProvider",    # Intel GPU
                "MetalExecutionProvider",       # Apple macOS
                "CoreMLExecutionProvider",      # Apple macOS
                "CUDAExecutionProvider",        # Nvidia GPU
            ]:
                print(interface_config['Main']['OnnxExectionProviderNotSupportedSkipped'].format(provider))
                continue
            print(interface_config['Main']['OnnxExecutionProviderDetected'].format(provider))
            ONNX_PROVIDERS.append(provider)
    except ModuleNotFoundError as e:
        print(interface_config['Main']['OnnxRuntimeNotInstall'])
if len(ONNX_PROVIDERS) > 0:
    USE_GPU = True

# --- Language & Model Path Settings ---
# All supported languages
# 设置识别语言
REC_CHAR_TYPE = settings_config['DEFAULT']['Language']

# 设置识别模式
MODE_TYPE = settings_config['DEFAULT']['Mode']
ACCURATE_MODE_ON = False
if MODE_TYPE == 'accurate':
    ACCURATE_MODE_ON = True
if MODE_TYPE == 'fast':
    ACCURATE_MODE_ON = False
if MODE_TYPE == 'auto':
    if USE_GPU:
        ACCURATE_MODE_ON = True
    else:
        ACCURATE_MODE_ON = False
        
# 模型文件目录
# 文本检测模型
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 设置文本识别模型 + 字典
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# V3, V4模型默认图形识别的shape为3, 48, 320
REC_IMAGE_SHAPE = '3,48,320'
REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_det')

LATIN_LANG = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french'
]
ARABIC_LANG = ['ar', 'fa', 'ug', 'ur']
CYRILLIC_LANG = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
]
DEVANAGARI_LANG = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari'
]
OTHER_LANG = [
    'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
    'chinese_cht',
]
MULTI_LANG = LATIN_LANG + ARABIC_LANG + CYRILLIC_LANG + DEVANAGARI_LANG + \
             OTHER_LANG

DET_MODEL_FAST_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')


# 如果设置了识别文本语言类型，则设置为对应的语言
if REC_CHAR_TYPE in MULTI_LANG:
    # 定义文本检测与识别模型
    # 使用快速模式时，调用轻量级模型
    if MODE_TYPE == 'fast':
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # 使用自动模式时，检测有没有使用GPU，根据GPU判断模型
    elif MODE_TYPE == 'auto':
        # 如果使用GPU，则使用大模型
        if USE_GPU:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
            # 英文模式的ch模型识别效果好于fast
            if REC_CHAR_TYPE == 'en':
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'ch_rec')
            else:
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
        else:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
            REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    else:
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # 如果默认版本(V4)没有大模型，则切换为默认版本(V4)的fast模型
    if not os.path.exists(REC_MODEL_PATH):
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # 如果默认版本(V4)既没有大模型，又没有fast模型，则使用V3版本的大模型
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # 如果V3版本没有大模型，则使用V3版本的fast模型
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')

    if REC_CHAR_TYPE in LATIN_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'latin_rec_fast')
    elif REC_CHAR_TYPE in ARABIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'arabic_rec_fast')
    elif REC_CHAR_TYPE in CYRILLIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'cyrillic_rec_fast')
    elif REC_CHAR_TYPE in DEVANAGARI_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'devanagari_rec_fast')

    # 定义图像识别shape
    if MODEL_VERSION == 'V2':
        REC_IMAGE_SHAPE = '3,32,320'
    else:
        REC_IMAGE_SHAPE = '3,48,320'

    # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
    if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=REC_MODEL_PATH)
    # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
    if 'inference.pdiparams' not in (os.listdir(DET_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=DET_MODEL_PATH)
# ×××××××××××××××××××× [不要改] end ××××××××××××××××××××


@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'


# ×××××××××××××××××××× [可以改] start ××××××××××××××××××××
# 是否使用h264编码，如果需要安卓手机分享生成的视频，请打开该选项
USE_H264 = True

# ×××××××××× 通用设置 start ××××××××××
"""
MODE可选算法类型
- InpaintMode.STTN 算法：对于真人视频效果较好，速度快，可以跳过字幕检测
- InpaintMode.LAMA 算法：对于动画类视频效果好，速度一般，不可以跳过字幕检测
- InpaintMode.PROPAINTER 算法： 需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好
"""
# 【设置inpaint算法】
MODE = InpaintMode.PROPAINTER

# ×××××××××× OCR Settings start ××××××××××
# GPU memory limit in MB for OCR processing
GPU_MEMORY_LIMIT = 4096 if USE_GPU else 2048
# For each image, recognize text in up to 6 text boxes simultaneously. The larger the GPU memory, the larger this value can be set.
REC_BATCH_NUM = 12 if USE_GPU else 6
# How many images are recognized in each batch of the DB algorithm, the default is 10
MAX_BATCH_SIZE = 20 if USE_GPU else 10
# Confidence threshold for text detection. Lower values are less strict.
DET_DB_BOX_THRESH = 0.6
# Do not accept subtitles with a confidence level lower than 0.75
DROP_SCORE = 0.75
# Allowed deviation of the subtitle area, 0 means no out-of-bounds allowed, 0.03 means 3% out-of-bounds is allowed
SUB_AREA_DEVIATION_RATE = 0
# Output lost subtitle frames, only valid for Simplified Chinese, Traditional Chinese, Japanese, Korean. Default debug info is output to: video path/loss
DEBUG_OCR_LOSS = False
# Text similarity threshold for deduplication. Higher is stricter.
THRESHOLD_TEXT_SIMILARITY = 0.8
# How many frames to grab per second for OCR
EXTRACT_FREQUENCY = 3
# ×××××××××× OCR Settings end ××××××××××

# 【设置像素点偏差】
# 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
# 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
SUBTITLE_AREA_DEVIATION_PIXEL = 20
# 同于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
THRESHOLD_HEIGHT_DIFFERENCE = 20
# 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
PIXEL_TOLERANCE_Y = 10  # 允许检测框纵向偏差的像素点数
PIXEL_TOLERANCE_X = 10  # 允许检测框横向偏差的像素点数
# ×××××××××× 通用设置 end ××××××××××

# ×××××××××× InpaintMode.STTN算法设置 start ××××××××××
# 以下参数仅适用STTN算法时，才生效
"""
1. STTN_SKIP_DETECTION
含义：是否使用跳过检测
效果：设置为True跳过字幕检测，会省去很大时间，但是可能误伤无字幕的视频帧或者会导致去除的字幕漏了

2. STTN_NEIGHBOR_STRIDE
含义：相邻帧数步长, 如果需要为第50帧填充缺失的区域，STTN_NEIGHBOR_STRIDE=5，那么算法会使用第45帧、第40帧等作为参照。
效果：用于控制参考帧选择的密度，较大的步长意味着使用更少、更分散的参考帧，较小的步长意味着使用更多、更集中的参考帧。

3. STTN_REFERENCE_LENGTH
含义：参数帧数量，STTN算法会查看每个待修复帧的前后若干帧来获得用于修复的上下文信息
效果：调大会增加显存占用，处理效果变好，但是处理速度变慢

4. STTN_MAX_LOAD_NUM
含义：STTN算法每次最多加载的视频帧数量
效果：设置越大速度越慢，但效果越好
注意：要保证STTN_MAX_LOAD_NUM大于STTN_NEIGHBOR_STRIDE和STTN_REFERENCE_LENGTH
"""
STTN_SKIP_DETECTION = True
# 参考帧步长
STTN_NEIGHBOR_STRIDE = 2
# 参考帧长度（数量）
STTN_REFERENCE_LENGTH = 6
# 设置STTN算法最大同时处理的帧数量
STTN_MAX_LOAD_NUM = 60
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN算法设置 end ××××××××××

# ×××××××××× InpaintMode.PROPAINTER算法设置 start ××××××××××
# 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
# 1280x720p视频设置80需要25G显存，设置50需要19G显存
# 720x480p视频设置80需要8G显存，设置50需要7G显存
PROPAINTER_MAX_LOAD_NUM = 50
# ×××××××××× InpaintMode.PROPAINTER算法设置 end ××××××××××

# ×××××××××× InpaintMode.LAMA算法设置 start ××××××××××
# 是否开启极速模式，开启后不保证inpaint效果，仅仅对包含文本的区域文本进行去除
LAMA_SUPER_FAST = False
# ×××××××××× InpaintMode.LAMA算法设置 end ××××××××××
# ×××××××××××××××××××× [可以改] end ××××××××××××××××××××