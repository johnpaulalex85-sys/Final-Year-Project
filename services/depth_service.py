import logging
import cv2
import numpy as np
import base64
from PIL import Image
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class DepthService:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        logger.info(f"Loading Depth model {model_name}...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=device)
            logger.info(f"Depth model loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load Depth model: {e}")
            self.pipe = None

    def estimate_depth(self, frame: np.ndarray, detections: list):
        """
        Estimates depth from the frame and calculates distances for detections.
        Returns a tuple: (distances_list, depth_map_base64)
        """
        if self.pipe is None:
            return [], None

        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Run inference
            result = self.pipe(pil_image)
            depth_map = result["depth"]  # PIL Image
            
            # Convert PIL Image back to numpy array for processing
            depth_array = np.array(depth_map)
            
            # depth_array contains inverse depth values (larger value = closer)
            # We'll normalize it for visualization
            depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Colorize for visualization
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
            # Convert to base64 for frontend
            _, buffer = cv2.imencode('.jpg', depth_colored)
            depth_base64 = base64.b64encode(buffer).decode('utf-8')
            
            distances = []
            
            # Calculate distance for each YOLO detection
            for det in detections:
                box = det.get("box")
                if not box or len(box) != 4:
                    continue
                    
                x1, y1, x2, y2 = [int(v) for v in box]
                
                # Ensure coordinates are within image bounds
                h, w = depth_array.shape
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Get the depth values in the bounding box
                box_depths = depth_array[y1:y2, x1:x2]
                
                if box_depths.size == 0:
                    continue
                    
                # Calculate average or median depth value
                avg_depth_value = np.median(box_depths)
                
                if avg_depth_value > 0:
                    # Pseudo-distance calculation (since it's relative depth)
                    # depth-anything gives larger values for closer objects (inverse depth)
                    # For an 8-bit depth map (0-255), we use a scale factor to approximate meters
                    scale_factor = 150.0
                    estimated_distance = scale_factor / float(avg_depth_value)
                    
                    # Cap distance to reasonable bounds, e.g., 0.1 to 20 meters
                    estimated_distance = max(0.1, min(estimated_distance, 20.0))
                    
                    distances.append({
                        "object": det.get("class_name", "unknown"),
                        "distance": f"{estimated_distance:.1f} meters",
                        "box": box
                    })

            return distances, depth_base64

        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return [], None
