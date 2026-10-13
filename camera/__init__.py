from config import settings
from .capture import CameraManager

# Create a singleton instance based on config settings
camera_manager = CameraManager(
    index=settings.CAMERA_INDEX,
    width=settings.CAMERA_WIDTH,
    height=settings.CAMERA_HEIGHT
)

__all__ = ['camera_manager', 'CameraManager']
