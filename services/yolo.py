import logging
try:
    from ultralytics import YOLOWorld
except Exception:
    from ultralytics import YOLO as YOLOWorld
from ultralytics import YOLO
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)

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

class ObjectDetector:
    def __init__(self, model_name="yolov8s-worldv2.pt", custom_classes=None):
        logger.info(f"Loading YOLO-World model {model_name}...")
        if custom_classes is None:
            custom_classes = DEFAULT_CUSTOM_CLASSES
        self.custom_classes = custom_classes

        try:
            try:
                self.model = YOLOWorld(model_name)
            except Exception:
                self.model = YOLO(model_name)

            if hasattr(self.model, "set_classes"):
                self.model.set_classes(self.custom_classes)

            logger.info(f"YOLO-World loaded successfully with classes: {self.custom_classes}")
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            self.model = None

        self.latest_detections = []
        self.running = False
        self.thread = None
        self.camera_manager = None

    def set_custom_classes(self, classes: list):
        """Dynamically update the active detectable classes at runtime."""
        self.custom_classes = classes
        if self.model and hasattr(self.model, "set_classes"):
            try:
                self.model.set_classes(classes)
                logger.info(f"Updated YOLO-World classes to: {classes}")
            except Exception as e:
                logger.error(f"Failed to update YOLO-World classes: {e}")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.45):
        if self.model is None:
            return []
            
        try:
            # Resize internal prediction matrix for fast CPU execution (~30ms)
            results = self.model.predict(source=frame, imgsz=384, conf=conf_threshold, verbose=False)
        except Exception as e:
            logger.error(f"YOLO predict error: {e}")
            return []

        detections = []
        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id]
                
                detections.append({
                    "box": [round(x, 2) for x in xyxy],
                    "confidence": round(conf, 3),
                    "class_name": cls_name
                })
                
        return detections

    def start_live_loop(self, camera_manager, conf_threshold=0.4):
        """Continuously process frames in a background thread."""
        if self.running: return
        self.camera_manager = camera_manager
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(conf_threshold,), daemon=True)
        self.thread.start()
        logger.info("Live YOLO detection loop started.")

    def _loop(self, conf_threshold):
        while self.running:
            if self.camera_manager:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # Run detection and store the results instantly
                    dets = self.detect(frame, conf_threshold)
                    self.latest_detections = dets
            # Run at roughly ~20 FPS max to conserve CPU
            time.sleep(0.05) 

    def stop_live_loop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Live YOLO detection loop stopped.")
