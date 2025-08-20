import os
from backend import config
import importlib
from paddleocr import PaddleOCR

# --- Singleton Pattern for OCR Model ---
_ocr_recogniser_instance = None

def get_ocr_recogniser():
    """
    Returns a singleton instance of the OcrRecogniser.
    This ensures the model is loaded only once, improving performance.
    """
    global _ocr_recogniser_instance
    if _ocr_recogniser_instance is None:
        _ocr_recogniser_instance = OcrRecogniser()
    return _ocr_recogniser_instance

# 加载文本检测+识别模型
class OcrRecogniser:
    def __init__(self):
        # 获取参数对象
        importlib.reload(config)
        self.recogniser = self.init_model()

    @staticmethod
    def y_round(y):
        y_min = y + 10 - y % 10
        y_max = y - y % 10
        if abs(y - y_min) < abs(y - y_max):
            return y_min
        else:
            return y_max

    def predict(self, image):
        # Note: The 'cls' parameter in PaddleOCR is for text angle classification.
        # Disabling it (cls=False) is generally a good choice for subtitles, which are typically horizontal.
        detection_box, recognise_result, _ = self.recogniser(image, cls=False)
        
        # The result from paddleocr is a list of detection boxes and a list of (text, score) tuples.
        if detection_box:
            # The results are already in the format we need.
            # `detection_box` is a list of boxes, and `recognise_result` is a list of (text, score) tuples.
            return detection_box, recognise_result
        else:
            return [], []

    def init_model(self):
        # Increase GPU memory allocation for better performance with larger batches and high-res video.
        gpu_mem = config.GPU_MEMORY_LIMIT if hasattr(config, 'GPU_MEMORY_LIMIT') else 2048
        
        # Check for ONNX runtime availability to automatically enable it if possible.
        use_onnx_runtime = len(config.ONNX_PROVIDERS) > 0
        try:
            import onnxruntime
        except ImportError:
            use_onnx_runtime = False

        return PaddleOCR(use_gpu=config.USE_GPU,
                         gpu_mem=500,
                         det_algorithm='DB',
                         # 设置文本检测模型路径
                         det_model_dir=self.convertToOnnxModelIfNeeded(config.DET_MODEL_PATH),
                         rec_algorithm='CRNN',
                         # 设置每张图文本框批处理数量
                         rec_batch_num=config.REC_BATCH_NUM,
                         # 设置文本识别模型路径
                         rec_model_dir=self.convertToOnnxModelIfNeeded(config.REC_MODEL_PATH),
                         max_batch_size=config.MAX_BATCH_SIZE,
                         det=True,
                         use_angle_cls=False,
                         drop_score=0,
                         lang=config.REC_CHAR_TYPE,
                         ocr_version=f'PP-OCR{config.MODEL_VERSION.lower()}',
                         rec_image_shape=config.REC_IMAGE_SHAPE,
                         use_onnx=use_onnx_runtime,
                         onnx_providers=config.ONNX_PROVIDERS,
                         debug=False, show_log=False)
    

    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        if not config.ONNX_PROVIDERS:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir


def get_coordinates(dt_box):
    """
    从返回的检测框中获取坐标
    :param dt_box 检测框返回结果
    :return list 坐标点列表
    """
    coordinate_list = list()
    if isinstance(dt_box, list):
        for i in dt_box:
            # The new format from paddleocr.ocr is a list of points
            # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            x_coords = [p[0] for p in i]
            y_coords = [p[1] for p in i]
            xmin = int(min(x_coords))
            xmax = int(max(x_coords))
            ymin = int(min(y_coords))
            ymax = int(max(y_coords))
            coordinate_list.append((xmin, xmax, ymin, ymax))
    return coordinate_list
