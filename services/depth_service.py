import logging
import cv2
import numpy as np
import base64
from PIL import Image
from transformers import pipeline
import torch
from config import settings

logger = logging.getLogger(__name__)

class DepthService:
    def __init__(self, model_name=None):
        model_name = model_name or settings.DEPTH_MODEL_NAME
        logger.info(f"Loading Depth model {model_name} on {settings.DEVICE}...")
        try:
            device = 0 if settings.DEVICE == "cuda" else -1
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=device)
            logger.info(f"Depth model loaded successfully on {settings.DEVICE}.")
        except Exception as e:
            logger.error(f"Failed to load Depth model: {e}")
            self.pipe = None

    def estimate_depth(self, frame: np.ndarray, detections: list, max_dim: int = 384):
        """
        Estimates relative depth map and relative proximity rankings for detections.
        Preserves uncertainty handling without claiming calibrated metric accuracy.
        Returns a tuple: (distances_list, depth_map_base64)
        """
        if self.pipe is None or frame is None or frame.size == 0:
            return [], None

        try:
            h_orig, w_orig = frame.shape[:2]
            scale = 1.0
            if max(h_orig, w_orig) > max_dim:
                scale = max_dim / float(max(h_orig, w_orig))
                frame_infer = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
            else:
                frame_infer = frame

            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame_infer, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Run inference
            result = self.pipe(pil_image)
            depth_map = result["depth"]  # PIL Image
            depth_array = np.array(depth_map)

            # Resize depth array back to original frame dimensions
            if scale != 1.0:
                depth_array = cv2.resize(depth_array, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # Normalize for visualization
            depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
            _, buffer = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
            depth_base64 = base64.b64encode(buffer).decode('utf-8')

            distances = []

            # Calculate relative proximity for each detection
            for det in detections:
                box = det.get("box")
                if not box or len(box) != 4:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(x1, w_orig - 1))
                y1 = max(0, min(y1, h_orig - 1))
                x2 = max(0, min(x2, w_orig - 1))
                y2 = max(0, min(y2, h_orig - 1))

                if x1 >= x2 or y1 >= y2:
                    continue

                box_depths = depth_array[y1:y2, x1:x2]
                if box_depths.size == 0:
                    continue

                # In Depth-Anything, higher values represent closer objects
                median_depth = float(np.median(box_depths))
                
                # Classify into relative proximity tiers
                if median_depth > 180:
                    proximity_label = "immediate path (~1m)"
                elif median_depth > 120:
                    proximity_label = "close (~1-2m)"
                elif median_depth > 70:
                    proximity_label = "mid-range (~2-4m)"
                else:
                    proximity_label = "distant (>4m)"

                distances.append({
                    "object": det.get("class_name", "unknown"),
                    "distance": proximity_label,
                    "relative_depth_val": round(median_depth, 1),
                    "calibrated": False,
                    "box": box
                })

            return distances, depth_base64

        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return [], None
