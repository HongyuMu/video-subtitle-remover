from paddleocr import PaddleOCR
import backend.config as config
import Levenshtein

def get_coordinates(dt_box):
    """
    Extracts coordinates from a detection box.
    """
    coordinate_list = []
    if isinstance(dt_box, list):
        for i in dt_box:
            i = list(i)
            # Ensure each box has four points
            if len(i) == 4:
                # Extract coordinates and convert to integers
                try:
                    (x1, y1) = int(i[0][0]), int(i[0][1])
                    (x2, y2) = int(i[1][0]), int(i[1][1])
                    (x3, y3) = int(i[2][0]), int(i[2][1])
                    (x4, y4) = int(i[3][0]), int(i[3][1])
                    
                    # Define the bounding box
                    xmin = min(x1, x4)
                    xmax = max(x2, x3)
                    ymin = min(y1, y2)
                    ymax = max(y3, y4)
                    
                    coordinate_list.append((xmin, xmax, ymin, ymax))
                except (ValueError, IndexError):
                    # Handle cases where conversion to int fails or points are missing
                    continue
    return coordinate_list

class OcrRecogniser:
    """
    OCR Recogniser class for predicting text from images.
    """

    def __init__(self):
        """
        Initializes the PaddleOCR model.
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=config.REC_CHAR_TYPE,
            rec_model_dir=config.REC_MODEL_PATH,
            rec_image_shape='3, 48, 320',
            use_gpu=config.USE_GPU
        )

    def predict(self, img):
        """
        Predicts text from a given image.
        """
        result = self.ocr.ocr(img, cls=True)
        if result and result[0]:
            dt_boxes = [line[0] for line in result[0]]
            rec_res = [line[1] for line in result[0]]
            return dt_boxes, rec_res
        return [], []

    def get_area_text(self, ocr_result):
        """
        Extracts text content from OCR results (adapted from subtitle extractor).
        """
        dt_box, rec_res = ocr_result
        if not rec_res:
            return []
        
        text_list = []
        for res in rec_res:
            if res and len(res) >= 2:
                text, confidence = res[0], res[1]
                if confidence > 0.5:  # Only include high-confidence text
                    text_list.append(text)
        return text_list

def compare_ocr_result(ocr1, img1, img1_no, img2, img2_no, result_cache, threshold=0.8):
    """
    Compare OCR results between two images (adapted from subtitle extractor).
    Uses caching to avoid repeated OCR on the same frames.
    """
    if ocr1 is None:
        ocr1 = OcrRecogniser()
    
    # Get or compute OCR result for image 1
    if img1_no in result_cache:
        area_text1 = result_cache[img1_no]['text']
    else:
        dt_box, rec_res = ocr1.predict(img1)
        area_text1 = "".join(ocr1.get_area_text((dt_box, rec_res)))
        result_cache[img1_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}

    # Get or compute OCR result for image 2
    if img2_no in result_cache:
        area_text2 = result_cache[img2_no]['text']
    else:
        dt_box, rec_res = ocr1.predict(img2)
        area_text2 = "".join(ocr1.get_area_text((dt_box, rec_res)))
        result_cache[img2_no] = {'text': area_text2, 'dt_box': dt_box, 'rec_res': rec_res}

    # Clean up old cache entries to prevent memory buildup
    delete_no_list = []
    for no in result_cache:
        if no < min(img1_no, img2_no) - 10:
            delete_no_list.append(no)
    for no in delete_no_list:
        del result_cache[no]

    # Compare text similarity using Levenshtein ratio
    if not area_text1 and not area_text2:
        return True  # Both empty is considered similar
    if not area_text1 or not area_text2:
        return False  # One empty, one with text is dissimilar
    
    # Remove spaces and compare
    text1_clean = area_text1.replace(' ', '')
    text2_clean = area_text2.replace(' ', '')
    similarity = Levenshtein.ratio(text1_clean, text2_clean)
    
    return similarity > threshold

def extract_text_from_frame(ocr_recogniser, frame, bbox):
    """
    Extract text from a specific bounding box region of a frame.
    """
    if ocr_recogniser is None:
        ocr_recogniser = OcrRecogniser()
    
    # Crop the frame to the bounding box
    xmin, xmax, ymin, ymax = bbox
    cropped_frame = frame[ymin:ymax, xmin:xmax]
    
    # Get OCR results
    dt_box, rec_res = ocr_recogniser.predict(cropped_frame)
    text = "".join(ocr_recogniser.get_area_text((dt_box, rec_res)))
    
    return text
