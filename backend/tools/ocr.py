import os
from backend import config
import importlib
from paddleocr import PaddleOCR
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

    def predict_batch(self, images):
        """
        Performs OCR on a batch of images by separating detection and recognition.
        Optimized for GPU with larger batch sizes.
        """
        if not images:
            return []

        # Increase batch size for GPU processing
        effective_batch_size = min(len(images), config.MAX_BATCH_SIZE * 2 if config.USE_GPU else config.MAX_BATCH_SIZE)
        
        # Process images in optimized batches
        batch_results = []
        for i in range(0, len(images), effective_batch_size):
            batch_imgs = images[i:i + effective_batch_size]
            
            # 1. Batch Detection with GPU acceleration
            all_dt_boxes, _ = self.recogniser.text_detector(batch_imgs)

            for j, dt_boxes in enumerate(all_dt_boxes):
                if dt_boxes is None or len(dt_boxes) == 0:
                    batch_results.append(([], []))
                    continue
                
                # 2. Prepare images for recognition based on detected boxes
                img = batch_imgs[j]
                img_crop_list = []
                
                for box in dt_boxes:
                    img_crop = self.recogniser.get_rotate_crop_image(img, box)
                    img_crop_list.append(img_crop)

                # 3. Batch Recognition on the crops with larger batch size for GPU
                if not img_crop_list:
                    batch_results.append((dt_boxes.tolist(), []))
                    continue

                # Process recognition in larger batches for GPU efficiency
                rec_batch_size = config.REC_BATCH_NUM * 2 if config.USE_GPU else config.REC_BATCH_NUM
                all_rec_res = []
                
                for k in range(0, len(img_crop_list), rec_batch_size):
                    crop_batch = img_crop_list[k:k + rec_batch_size]
                    rec_res, _ = self.recogniser.text_recognizer(crop_batch)
                    all_rec_res.extend(rec_res)
                
                batch_results.append((dt_boxes.tolist(), all_rec_res))

        return batch_results

    def predict_parallel_frames(self, frames, max_workers=None):
        """
        Process multiple video frames in parallel for faster OCR.
        Each worker processes a subset of frames using batch processing.
        """
        if not frames:
            return []
        
        if max_workers is None:
            # Use 2-4 workers for GPU to avoid memory conflicts, more for CPU
            max_workers = 2 if config.USE_GPU else min(4, len(frames))
        
        # Split frames into chunks for parallel processing
        chunk_size = max(1, len(frames) // max_workers)
        frame_chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
        
        results = [None] * len(frames)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunks for processing
            future_to_chunk = {}
            for chunk_idx, chunk in enumerate(frame_chunks):
                if chunk:  # Only submit non-empty chunks
                    future = executor.submit(self._process_frame_chunk, chunk)
                    future_to_chunk[future] = (chunk_idx, len(chunk))
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_idx, chunk_len = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    # Place results in correct positions
                    start_idx = chunk_idx * chunk_size
                    for i, result in enumerate(chunk_results):
                        if start_idx + i < len(results):
                            results[start_idx + i] = result
                except Exception as e:
                    print(f"Error processing frame chunk {chunk_idx}: {e}")
                    # Fill with empty results for failed chunk
                    start_idx = chunk_idx * chunk_size
                    for i in range(chunk_len):
                        if start_idx + i < len(results):
                            results[start_idx + i] = ([], [])
        
        return [r for r in results if r is not None]
    
    def _process_frame_chunk(self, frame_chunk):
        """
        Process a chunk of frames using batch processing.
        """
        return self.predict_batch(frame_chunk)

    def init_model(self):
        # Increase GPU memory allocation for better performance with larger batches and high-res video.
        gpu_mem = config.GPU_MEMORY_LIMIT if hasattr(config, 'GPU_MEMORY_LIMIT') else 4096
        
        # Check for ONNX runtime availability to automatically enable it if possible.
        use_onnx_runtime = len(config.ONNX_PROVIDERS) > 0
        try:
            import onnxruntime
        except ImportError:
            use_onnx_runtime = False

        # Force GPU usage and set GPU device if available
        use_gpu = config.USE_GPU
        if use_gpu and hasattr(config, 'device') and config.device.type == 'cuda':
            import paddle
            # Set Paddle device to match the selected GPU
            gpu_id = config.device.index if config.device.index is not None else 0
            paddle.set_device(f'gpu:{gpu_id}')
            print(f"OCR using GPU: {gpu_id}")
        elif use_gpu:
            print("OCR using default GPU")
        else:
            print("OCR using CPU")

        return PaddleOCR(use_gpu=use_gpu,
                         gpu_mem=gpu_mem,
                         det_algorithm='DB',
                         # 设置文本检测模型路径
                         det_model_dir=self.convertToOnnxModelIfNeeded(config.DET_MODEL_PATH),
                         rec_algorithm='CRNN',
                         # 设置每张图文本框批处理数量
                         rec_batch_num=config.REC_BATCH_NUM,
                         # 设置文本识别模型路径
                         rec_model_dir=self.convertToOnnxModelIfNeeded(config.REC_MODEL_PATH),
                         max_batch_size=config.MAX_BATCH_SIZE,
                         det_db_box_thresh=config.DET_DB_BOX_THRESH,
                         det=True,
                         use_angle_cls=False,
                         drop_score=config.DROP_SCORE,
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