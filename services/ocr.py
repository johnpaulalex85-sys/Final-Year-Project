import logging
import cv2
import easyocr
import numpy as np

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, languages=['en']):
        logger.info(f"Loading EasyOCR model for languages: {languages}...")
        try:
            self.reader = easyocr.Reader(languages, gpu=True)
            logger.info("EasyOCR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR model: {e}")
            self.reader = None

    def detect_text(self, frame: np.ndarray, conf_threshold: float = 0.5, max_dim: int = 480, **kwargs):
        if "min_confidence" in kwargs:
            conf_threshold = kwargs["min_confidence"]

        if self.reader is None or frame is None or frame.size == 0:
            return {"texts": []}

        h, w = frame.shape[:2]
        
        # Calculate scale factor to resize image if larger than max_dim for fast detection
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_frame = frame

        # Convert to grayscale for faster PyTorch tensor conversion & inference
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        try:
            # Fast greedy decoder, disabled contrast adjustment pass, batched inference
            results = self.reader.readtext(
                gray_frame,
                decoder='greedy',
                adjust_contrast=0.0,
                batch_size=8,
                canvas_size=max_dim
            )
        except Exception as e:
            logger.error(f"EasyOCR readtext error: {e}")
            return {"texts": []}

        texts = []
        for (bbox, text, prob) in results:
            if prob >= conf_threshold:
                # Scale bbox back to original frame size if resized
                serializable_bbox = [
                    [int(round(pt[0] / scale)), int(round(pt[1] / scale))]
                    for pt in bbox
                ]
                texts.append({
                    "text": text,
                    "confidence": round(float(prob), 3),
                    "bbox": serializable_bbox
                })

        return {"texts": texts}
