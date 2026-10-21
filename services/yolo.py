import logging
import time
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional

try:
    from ultralytics import YOLOWorld
except Exception:
    from ultralytics import YOLO as YOLOWorld
from ultralytics import YOLO

from config import settings
from services.tracker import WalkingObjectTracker

logger = logging.getLogger(__name__)

# Complete historical class list preserved for 100% backwards compatibility
DEFAULT_CUSTOM_CLASSES = [
    # People and body-related objects
    "person", "child", "adult", "elderly person",
    "pedestrian", "crowd", "person walking",
    "person sitting", "person lying on the ground",
    "wheelchair user", "person with a cane",
    "guide dog",

    # Indoor structures
    "wall", "door", "open door", "closed door",
    "glass door", "sliding door", "emergency exit",
    "window", "glass window", "wall corner",
    "pillar", "column", "ceiling", "floor",
    "corridor", "hallway", "staircase", "stairs",
    "upstairs", "downstairs", "step", "ramp",
    "handrail", "railing", "elevator", "escalator",
    "doorway", "entrance", "exit",

    # Navigation hazards
    "obstacle", "pothole", "crack in pavement",
    "uneven pavement", "broken pavement",
    "wet floor", "slippery floor", "spill",
    "water puddle", "debris", "rubble",
    "fallen object", "low hanging object",
    "overhead obstacle", "protruding object",
    "low barrier", "road barrier", "construction barrier",
    "open drain", "manhole", "ditch", "curb",
    "steep slope", "loose cable", "electric wire",
    "hanging wire", "broken step",

    # Road and traffic
    "car", "bus", "truck", "van", "motorcycle",
    "scooter", "bicycle", "auto rickshaw",
    "ambulance", "fire engine", "police vehicle",
    "moving vehicle", "parked vehicle",
    "traffic light", "red traffic light",
    "green traffic light", "yellow traffic light",
    "pedestrian crossing", "zebra crossing",
    "road sign", "stop sign", "speed limit sign",
    "direction sign", "street sign", "traffic cone",
    "road divider", "guardrail", "speed bump",
    "railway track", "train", "platform edge",

    # Furniture
    "chair", "office chair", "stool", "table",
    "desk", "dining table", "bed", "sofa",
    "bench", "cabinet", "cupboard", "shelf",
    "drawer", "wardrobe", "bookshelf",
    "reception desk", "counter", "shoe rack",

    # Everyday objects
    "book", "notebook", "paper", "document",
    "poster", "newspaper", "magazine", "pen",
    "pencil", "scissors", "bottle", "water bottle",
    "cup", "glass", "mug", "plate", "bowl",
    "spoon", "fork", "knife", "food container",
    "bag", "backpack", "handbag", "suitcase",
    "wallet", "keys", "umbrella", "watch",
    "spectacles", "sunglasses", "hat", "shoes",
    "slippers", "clothing", "towel",
    "photo frame", "picture frame", "framed picture", "portrait", "painting",

    # Electronic devices
    "cell phone", "smartphone", "laptop",
    "desktop computer", "computer monitor",
    "keyboard", "mouse", "tablet", "television",
    "remote control", "headphones", "earphones",
    "charger", "power bank", "electrical socket",
    "plug", "power cable", "fan", "air conditioner",
    "light switch", "electrical switch",
    "wall clock", "doorbell",

    # Kitchen and household
    "refrigerator", "microwave", "oven", "stove",
    "washing machine", "sink", "tap", "faucet",
    "bucket", "mop", "broom", "dustbin",
    "trash bag", "cleaning equipment",
    "fire extinguisher", "smoke detector",

    # Signs and readable information
    "signboard", "information board", "notice board",
    "nameplate", "house number", "room number",
    "direction board", "exit sign", "warning sign",
    "danger sign", "toilet sign", "restroom sign",
    "emergency sign", "lift sign", "floor number",
    "bus stop sign", "shop sign", "menu board",

    # Accessibility features
    "tactile paving", "braille sign",
    "accessible ramp", "wheelchair ramp",
    "handrail", "accessible toilet sign",
    "elevator button", "pedestrian signal",
    "audible pedestrian signal",

    # Outdoor environment
    "tree", "tree branch", "bush", "plant",
    "flower pot", "grass", "rock", "stone",
    "fence", "gate", "wall gate", "pole",
    "electricity pole", "streetlight",
    "parked bicycle", "parked scooter",
    "construction site", "scaffolding",
    "building", "house", "shop", "bus stop",
    "bridge", "footpath", "sidewalk", "road",

    # Animals
    "dog", "cat", "cow", "goat", "horse",
    "bird", "chicken", "snake",

    # Common places and facilities
    "doorway entrance", "stair landing",
    "hospital bed", "hospital sign",
    "school sign", "office entrance",
    "parking sign", "parking space",
    "ticket counter", "security desk",
]

# High-priority curated walking hazard classes for high-speed YOLO-World execution
STREAMLINED_WALKING_CLASSES = [
    "person", "pedestrian", "car", "bus", "truck", "motorcycle", "bicycle",
    "staircase", "stairs", "step", "curb", "pothole", "obstacle",
    "pole", "tree", "door", "wall", "chair", "table", "dustbin",
    "traffic light", "pedestrian crossing", "dog", "wet floor", "road barrier"
]

class ObjectDetector:
    def __init__(
        self,
        model_name: Optional[str] = None,
        custom_classes: Optional[List[str]] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name or settings.YOLO_MODEL_PATH
        self.imgsz = imgsz or settings.YOLO_IMGSZ
        self.device = device or settings.DEVICE
        self.is_world_model = "world" in self.model_name.lower()
        
        # Determine active classes
        if custom_classes is not None:
            self.custom_classes = custom_classes
        elif self.is_world_model:
            if settings.YOLO_STREAMLINED_CLASSES:
                self.custom_classes = STREAMLINED_WALKING_CLASSES
            else:
                self.custom_classes = DEFAULT_CUSTOM_CLASSES
        else:
            self.custom_classes = None

        logger.info(f"Initializing ObjectDetector with model '{self.model_name}' on device '{self.device}'...")
        t0 = time.time()
        
        try:
            if self.is_world_model:
                try:
                    self.model = YOLOWorld(self.model_name)
                except Exception:
                    self.model = YOLO(self.model_name)
                if self.custom_classes and hasattr(self.model, "set_classes"):
                    self.model.set_classes(self.custom_classes)
                    logger.info(f"YOLO-World set_classes configured ({len(self.custom_classes)} classes).")
            else:
                self.model = YOLO(self.model_name)

            load_time = time.time() - t0
            logger.info(f"YOLO model loaded successfully in {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

        # Lightweight object tracker
        self.tracker = WalkingObjectTracker(
            max_lost=settings.TRACK_MAX_LOST,
            iou_threshold=settings.TRACK_IOU_THRESHOLD,
            smoothing_alpha=settings.TRACK_SMOOTHING_ALPHA
        ) if settings.TRACKING_ENABLED else None

        # Perform warm-up inference if enabled
        if self.model and settings.YOLO_WARMUP:
            self._warmup()

        self.latest_detections = []
        self.running = False
        self.thread = None
        self.camera_manager = None

    def _warmup(self):
        """Runs a synthetic dummy frame through the model to prime JIT kernels."""
        try:
            t0 = time.perf_counter()
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.model.predict(
                    source=dummy,
                    imgsz=self.imgsz,
                    conf=0.5,
                    device=self.device,
                    verbose=False
                )
            warmup_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"YOLO warmup completed in {warmup_ms:.1f}ms.")
        except Exception as e:
            logger.warning(f"YOLO warmup warning: {e}")

    def set_custom_classes(self, classes: List[str]):
        """Dynamically update detectable classes at runtime (YOLO-World only)."""
        self.custom_classes = classes
        if self.model and hasattr(self.model, "set_classes"):
            try:
                t0 = time.time()
                self.model.set_classes(classes)
                logger.info(f"Updated YOLO-World classes ({len(classes)} classes) in {time.time()-t0:.2f}s.")
            except Exception as e:
                logger.error(f"Failed to update YOLO-World classes: {e}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        enable_tracking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Runs high-speed object detection on the input frame.
        Applies non-blocking inference mode, custom thresholds, and temporal tracking.
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        conf = conf_threshold if conf_threshold is not None else settings.YOLO_CONF
        iou = settings.YOLO_IOU

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=conf,
                    iou=iou,
                    device=self.device,
                    verbose=False
                )
        except Exception as e:
            logger.error(f"YOLO predict error: {e}")
            return []

        raw_detections = []
        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                b_conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id]

                raw_detections.append({
                    "box": [round(float(x), 1) for x in xyxy],
                    "confidence": round(b_conf, 3),
                    "class_name": cls_name
                })

        # Apply temporal tracking if enabled
        if enable_tracking and self.tracker is not None:
            tracked_detections = self.tracker.update(raw_detections)
            return tracked_detections

        return raw_detections

    def reset_tracker(self):
        if self.tracker:
            self.tracker.reset()

    def start_live_loop(self, camera_manager, conf_threshold: Optional[float] = None):
        """Continuously process camera frames in a background worker thread."""
        if self.running:
            return
        self.camera_manager = camera_manager
        self.running = True
        import threading
        self.thread = threading.Thread(target=self._loop, args=(conf_threshold,), daemon=True)
        self.thread.start()
        logger.info("Live YOLO detection loop started.")

    def _loop(self, conf_threshold):
        while self.running:
            if self.camera_manager:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    dets = self.detect(frame, conf_threshold)
                    self.latest_detections = dets
            time.sleep(1.0 / max(1, settings.TARGET_FPS))

    def stop_live_loop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Live YOLO detection loop stopped.")
