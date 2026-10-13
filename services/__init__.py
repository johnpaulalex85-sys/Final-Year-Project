from .yolo import ObjectDetector
from .ocr import OCRService
from .face_service import FaceService
from .depth_service import DepthService
from .describe_service import DescribeService
from .qwen_service import QwenService
from config import settings

# Global state for services to ensure they are loaded properly during app lifespan
services_state = {
    "yolo_detector": None,
    "ocr_service": None,
    "face_service": None,
    "depth_service": None,
    "describe_service": None,
    "qwen_service": None
}

def init_services():
    services_state["yolo_detector"] = ObjectDetector(model_name=settings.YOLO_MODEL_PATH)
    services_state["ocr_service"] = OCRService()
    services_state["face_service"] = FaceService()
    services_state["depth_service"] = DepthService(model_name=settings.DEPTH_MODEL_NAME)
    services_state["describe_service"] = DescribeService()

def get_yolo_detector():
    return services_state.get("yolo_detector")

def get_ocr_service():
    return services_state.get("ocr_service")

def get_face_service():
    return services_state.get("face_service")

def get_depth_service():
    return services_state.get("depth_service")

def get_describe_service():
    return services_state.get("describe_service")

def get_qwen_service():
    if services_state.get("qwen_service") is None:
        try:
            services_state["qwen_service"] = QwenService(model_name=settings.QWEN_MODEL_NAME)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed lazy loading QwenService: {e}")
    return services_state.get("qwen_service")
