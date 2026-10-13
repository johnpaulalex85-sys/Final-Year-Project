import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vision Assistant API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model Paths / Configs
    YOLO_MODEL_PATH: str = "yolov8s-worldv2.pt"
    INSIGHTFACE_MODEL_DIR: str = "models/insightface"
    DEPTH_MODEL_NAME: str = "depth-anything/Depth-Anything-V2-Small-hf"
    QWEN_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"
    WHISPER_MODEL_SIZE: str = "base"
    
    # Camera Configs
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    
    # Audio Configs
    TTS_RATE: int = 150
    TTS_VOLUME: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
