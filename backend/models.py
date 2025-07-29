import importlib
from backend import config

class ModelManager:
    _text_detector = None

    @classmethod
    def get_text_detector(cls):
        if cls._text_detector is None:
            print("Initializing Text Detector model...")
            import paddle
            paddle.disable_signal_handler()
            from paddleocr.tools.infer import utility
            from paddleocr.tools.infer.predict_det import TextDetector
            
            importlib.reload(config)
            args = utility.parse_args()
            args.det_algorithm = 'DB'
            args.det_model_dir = cls.convertToOnnxModelIfNeeded(config.DET_MODEL_PATH)
            args.use_onnx = len(config.ONNX_PROVIDERS) > 0
            args.onnx_providers = config.ONNX_PROVIDERS
            
            cls._text_detector = TextDetector(args)
            print("Text Detector model loaded.")
        return cls._text_detector

    @staticmethod
    def convertToOnnxModelIfNeeded(model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        import os
        if not config.ONNX_PROVIDERS:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        try:
            import paddle2onnx
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
            paddle2onnx.export(
                model_filename=os.path.join(model_dir, model_filename),
                params_filename=os.path.join(model_dir, params_filename) if params_filename else "",
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
            )
            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during ONNX conversion: {e}")
            return model_dir

# Pre-load the model when the application starts if desired
# get_text_detector() 