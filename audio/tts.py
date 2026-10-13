import pyttsx3
import threading
import queue
import logging
import time
from config import settings

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, rate=None, volume=None):
        self.rate = rate or settings.TTS_RATE
        self.volume = volume or settings.TTS_VOLUME
        self.queue = queue.Queue()
        self.thread = None
        self.running = False
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("TTS Manager started")
        
    def _worker(self):
        # pyttsx3 on Windows uses COM, so we must initialize COM in this thread
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass # If not on Windows or missing, just continue
            
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.running = False
            return
            
        while self.running:
            try:
                # Dynamically update engine rate/volume if changed
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)

                # Wait for text to speak
                text = self.queue.get(timeout=0.5)
                if text:
                    logger.debug(f"Speaking: {text}")
                    engine.say(text)
                    engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS engine error: {e}")
                time.sleep(1) # Prevent tight loops on error

    def set_rate(self, rate: int):
        """Updates the speech rate (words per minute)."""
        self.rate = max(50, min(rate, 400))
        logger.info(f"TTS rate set to {self.rate}")

    def set_volume(self, volume: float):
        """Updates the speech volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(volume, 1.0))
        logger.info(f"TTS volume set to {self.volume}")
                
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except ImportError:
            pass

    def speak(self, text: str):
        """Adds text to the speaking queue."""
        if self.running:
            self.queue.put(text)
        else:
            logger.warning("TTS Manager not running. Cannot speak.")
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("TTS Manager stopped")

# Expose a singleton instance
tts_manager = TTSManager()
