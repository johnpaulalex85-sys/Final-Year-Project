from .yolo import ObjectDetector
from .ocr import OCRService
from .face_service import FaceService
from .depth_service import DepthService
from .describe_service import DescribeService
from .qwen_service import QwenService
from .tracker import WalkingObjectTracker
from .hazard_service import HazardPrioritizer
from config import settings

# Global state for services to ensure singletons are loaded properly during app lifespan
services_state = {
    "yolo_detector": None,
    "ocr_service": None,
    "face_service": None,
    "depth_service": None,
    "describe_service": None,
    "qwen_service": None,
    "hazard_prioritizer": None
}

def init_services():
    services_state["yolo_detector"] = ObjectDetector(model_name=settings.YOLO_MODEL_PATH)
    services_state["ocr_service"] = OCRService()
    services_state["face_service"] = FaceService()
    services_state["depth_service"] = DepthService(model_name=settings.DEPTH_MODEL_NAME)
    services_state["describe_service"] = DescribeService()
    services_state["hazard_prioritizer"] = HazardPrioritizer(
        corridor_left_ratio=settings.CORRIDOR_LEFT_RATIO,
        corridor_right_ratio=settings.CORRIDOR_RIGHT_RATIO,
        hazard_cooldown=settings.HAZARD_COOLDOWN_SECONDS,
        urgent_cooldown=settings.URGENT_HAZARD_COOLDOWN_SECONDS
    )

def get_yolo_detector() -> ObjectDetector:
    return services_state.get("yolo_detector")

def get_ocr_service() -> OCRService:
    return services_state.get("ocr_service")

def get_face_service() -> FaceService:
    return services_state.get("face_service")

def get_depth_service() -> DepthService:
    return services_state.get("depth_service")

def get_describe_service() -> DescribeService:
    return services_state.get("describe_service")

def get_hazard_prioritizer() -> HazardPrioritizer:
    return services_state.get("hazard_prioritizer")

def get_qwen_service() -> QwenService:
    if services_state.get("qwen_service") is None:
        try:
            services_state["qwen_service"] = QwenService(model_name=settings.QWEN_MODEL_NAME)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed lazy loading QwenService: {e}")
    return services_state.get("qwen_service")
