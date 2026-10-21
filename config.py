import os
import psutil
import torch
from pydantic_settings import BaseSettings

def get_optimal_torch_threads() -> int:
    """Calculates optimal PyTorch thread count for CPU inference."""
    physical_cores = psutil.cpu_count(logical=False) or 4
    # On hybrid architectures (e.g. Intel 12th Gen P/E cores), limit to physical cores
    # to avoid thread oversubscription and context switching overhead.
    return min(max(2, physical_cores), 8)

def get_default_device() -> str:
    """Returns 'cuda' if GPU is accessible, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vision Assistant API"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Hardware & Compute Configuration
    DEVICE: str = get_default_device()
    TORCH_THREADS: int = get_optimal_torch_threads()

    # YOLO Model Configuration
    # Options: "yolov8n.pt" (Ultra-fast, ~20+ FPS on CPU) or "yolov8s-worldv2.pt" (Custom classes, ~8-10 FPS)
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_IMGSZ: int = 384
    YOLO_CONF: float = 0.40
    YOLO_IOU: float = 0.45
    YOLO_WARMUP: bool = True
    YOLO_STREAMLINED_CLASSES: bool = True  # If using YOLO-World, use curated walking hazards

    # Object Tracking Configuration
    TRACKING_ENABLED: bool = True
    TRACK_MAX_LOST: int = 4
    TRACK_IOU_THRESHOLD: float = 0.30
    TRACK_SMOOTHING_ALPHA: float = 0.65

    # Hazard Detection & Spatial Corridor Configuration
    HAZARD_DETECTION_ENABLED: bool = True
    CORRIDOR_LEFT_RATIO: float = 0.32
    CORRIDOR_RIGHT_RATIO: float = 0.68
    HAZARD_COOLDOWN_SECONDS: float = 3.5
    URGENT_HAZARD_COOLDOWN_SECONDS: float = 2.0

    # Background AI Services Schedules (in processed frame intervals)
    OCR_INTERVAL_FRAMES: int = 30
    FACE_INTERVAL_FRAMES: int = 15
    DEPTH_INTERVAL_FRAMES: int = 20
    RUN_DEPTH_ON_DEMAND_ONLY: bool = False

    # Model Paths
    INSIGHTFACE_MODEL_DIR: str = "models/insightface"
    DEPTH_MODEL_NAME: str = "depth-anything/Depth-Anything-V2-Small-hf"
    QWEN_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"
    WHISPER_MODEL_SIZE: str = "base"

    # Camera Configs
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    TARGET_FPS: int = 15

    # Audio Configs
    TTS_RATE: int = 165
    TTS_VOLUME: float = 1.0
    TTS_MODE: str = "client"  # "client" (browser SpeechSynthesis) or "server" (pyttsx3) or "both"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Apply optimal torch threads on import
if settings.DEVICE == "cpu":
    try:
        torch.set_num_threads(settings.TORCH_THREADS)
    except Exception:
        pass
