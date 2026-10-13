import cv2
import threading
import logging
import time

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        
        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.current_frame = None

    def start(self):
        if self.running:
            return
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.index)
        
        # Request specific resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera with index {self.index}")
            raise RuntimeError(f"Could not open camera {self.index}")
            
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Read a single frame to ensure it's working before returning
        for _ in range(10):
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                break
            time.sleep(0.1)
            
        logger.info(f"Camera {self.index} started successfully")

    def _update(self):
        """Background thread loop to continuously grab the latest frame."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera")
                time.sleep(0.1)  # prevent busy wait
                continue
                
            # Safely store the latest frame
            with self.lock:
                self.current_frame = frame

    def get_frame(self):
        """Return a copy of the most recent frame."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        """Stop the camera thread and release the device."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Camera {self.index} stopped")
