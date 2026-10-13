import logging
import cv2
import asyncio
import concurrent.futures
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config import settings
from audio import tts_manager
from services import init_services, get_yolo_detector, get_ocr_service, get_face_service, get_depth_service, get_describe_service
from utils.scene_builder import build_scene_json
import json

# Configure Logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dedicated thread pool executor for heavy background AI tasks (OCR, Face recognition, Depth estimation)
# This ensures that fast YOLO object detection is NEVER delayed or blocked when background models take time.
bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="bg_ai_worker")

# Global state to hold the latest frame and scene data for description generation
latest_state = {
    "frame": None,
    "scene_json": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} version {settings.VERSION}")
    
    # Initialize AI models/services
    init_services()
    
    # Start the audio TTS thread
    tts_manager.start()
        
    yield  # Hand over control to FastAPI to serve requests
    
    # Shutdown
    logger.info("Shutting down services...")
    tts_manager.stop()
    bg_executor.shutdown(wait=False)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered assistant for visually impaired people",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint to receive frames from client and return low-latency YOLO, OCR, and Depth detections."""
    await websocket.accept()
    yolo_detector = get_yolo_detector()
    ocr_service = get_ocr_service()
    face_service = get_face_service()
    depth_service = get_depth_service()
    loop = asyncio.get_running_loop()
    
    # Low latency state caching
    ocr_cache = {"texts": []}
    face_cache = {"faces": []}
    distance_cache = []
    depth_map_cache = None
    
    # In-flight task flags to prevent task queuing and thread starvation
    ocr_busy = False
    face_busy = False
    depth_busy = False
    
    frame_counter = 0
    bg_tasks = set()
    
    try:
        while True:
            # Receive image frame as bytes
            data = await websocket.receive_bytes()
            
            if yolo_detector is None:
                await websocket.send_json({"detections": []})
                continue
                
            # Decode the image bytes into a cv2 frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            frame_counter += 1
            
            # 1. FAST PATH: Run YOLO Object Detection on every frame (~30ms)
            detections = await asyncio.to_thread(yolo_detector.detect, frame, 0.45)
            yolo_raw = list(detections)
            
            # 2. STAGGERED PATH: Face recognition (background task, non-blocking)
            if frame_counter % 3 == 0 and face_service is not None and not face_busy:
                face_busy = True
                async def update_face(f):
                    nonlocal face_cache, face_busy
                    try:
                        res = await loop.run_in_executor(bg_executor, face_service.recognize_face, f, 0.6)
                        face_cache = res
                    except Exception as err:
                        logger.error(f"Background face recognition error: {err}")
                    finally:
                        face_busy = False
                        
                task = asyncio.create_task(update_face(frame.copy()))
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)

            # 3. STAGGERED PATH: OCR text detection (background task, instant non-blocking check)
            if frame_counter % 2 == 0 and ocr_service is not None and not ocr_busy:
                ocr_busy = True
                async def update_ocr(f):
                    nonlocal ocr_cache, ocr_busy
                    try:
                        res = await loop.run_in_executor(bg_executor, ocr_service.detect_text, f, 0.5)
                        ocr_cache = res
                    except Exception as err:
                        logger.error(f"Background OCR error: {err}")
                    finally:
                        ocr_busy = False
                        
                task = asyncio.create_task(update_ocr(frame.copy()))
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)

            # 4. STAGGERED PATH: Depth estimation (background task, non-blocking)
            if frame_counter % 3 == 0 and depth_service is not None and not depth_busy:
                depth_busy = True
                async def update_depth(f, dets):
                    nonlocal distance_cache, depth_map_cache, depth_busy
                    try:
                        dists, dmap = await loop.run_in_executor(bg_executor, depth_service.estimate_depth, f, dets)
                        if dists: distance_cache = dists
                        if dmap: depth_map_cache = dmap
                    except Exception as err:
                        logger.error(f"Background depth estimation error: {err}")
                    finally:
                        depth_busy = False
                        
                task = asyncio.create_task(update_depth(frame.copy(), list(detections)))
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)

            # 5. Merge cached OCR & Face results into detections array
            for item in ocr_cache.get("texts", []):
                bbox = item["bbox"]
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                detections.append({
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": item["confidence"],
                    "class_name": f'"{item["text"]}"'
                })
                
            for item in face_cache.get("faces", []):
                detections.append({
                    "box": item["box"],
                    "confidence": item["confidence"],
                    "class_name": f'Face: {item["name"]}'
                })
                
            # Build structured scene JSON for scene description requests
            scene_json = build_scene_json(
                yolo_raw,
                ocr_cache, 
                face_cache, 
                distance_cache
            )
            
            latest_state["frame"] = frame
            latest_state["scene_json"] = scene_json
            
            # Send high-speed response back to client immediately
            await websocket.send_json({
                "detections": detections,
                "distance": distance_cache,
                "depth_map": depth_map_cache
            })

            
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/audio/speak")
def speak_text(text: str = "Hello, I am your vision assistant."):
    """Test endpoint for Text-to-Speech."""
    tts_manager.speak(text)
    return {"status": "speaking", "text": text}

@app.get("/api/describe_scene")
async def describe_scene_api():
    """Endpoint to generate a description of the current scene."""
    describe_service = get_describe_service()
    if describe_service is None:
        return {"description": "Describe service not initialized.", "error": True}
        
    frame = latest_state.get("frame")
    scene_json = latest_state.get("scene_json")
    
    if frame is None or scene_json is None:
        return {"description": "No active camera feed or scene data.", "error": True}
        
    try:
        # Run Rule-based describe in a threadpool (or direct, it's instant)
        description = describe_service.generate_description(frame, scene_json)
        
        # Speak the description automatically
        tts_manager.speak(description)
        
        return {"description": description, "success": True}
    except Exception as e:
        logger.error(f"Describe scene API error: {e}")
        return {"description": str(e), "error": True}

@app.get("/api/ask_question")
async def ask_question_api(q: str):
    """Endpoint to answer a specific spoken voice question about the current scene."""
    describe_service = get_describe_service()
    if describe_service is None:
        return {"answer": "Describe service not initialized.", "error": True}
        
    frame = latest_state.get("frame")
    scene_json = latest_state.get("scene_json")
    
    if frame is None or scene_json is None:
        return {"answer": "No active camera feed or scene data.", "error": True}
        
    try:
        answer = describe_service.answer_question(q, frame, scene_json)
        tts_manager.speak(answer)
        return {"question": q, "answer": answer, "success": True}
    except Exception as e:
        logger.error(f"Ask question API error: {e}")
        return {"answer": str(e), "error": True}

@app.post("/api/ocr")
async def detect_text_api(file: UploadFile = File(...)):
    """API endpoint to receive an image and return text detections."""
    ocr_service = get_ocr_service()
    if ocr_service is None:
        return {"texts": [], "error": "OCR service not initialized"}
        
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"texts": [], "error": "Failed to decode image"}
            
        # Run OCR in a threadpool to avoid blocking
        result = await asyncio.to_thread(ocr_service.detect_text, frame, 0.5)
        return result
    except Exception as e:
        logger.error(f"OCR API error: {e}")
        return {"texts": [], "error": str(e)}

@app.post("/api/register_face")
async def register_face_api(name: str = Form(...), file: UploadFile = File(...)):
    """API endpoint to register a new face."""
    face_service = get_face_service()
    if face_service is None:
        return {"success": False, "message": "Face service not initialized"}
        
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "message": "Failed to decode image"}
            
        success, message = await asyncio.to_thread(face_service.register_face, name, frame)
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Face registration API error: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/register_face_multi")
async def register_face_multi_api(name: str = Form(...), files: list[UploadFile] = File(...)):
    """API endpoint to register a new face with multiple angles."""
    face_service = get_face_service()
    if face_service is None:
        return {"success": False, "message": "Face service not initialized"}
        
    try:
        frames = []
        for file in files:
            data = await file.read()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            return {"success": False, "message": "Failed to decode any images"}
            
        success, message = await asyncio.to_thread(face_service.register_face_multi, name, frames)
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Multi-face registration API error: {e}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
