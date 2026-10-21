import logging
import cv2
import asyncio
import concurrent.futures
import numpy as np
import time
import psutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from config import settings
from audio import tts_manager
from services import (
    init_services,
    get_yolo_detector,
    get_ocr_service,
    get_face_service,
    get_depth_service,
    get_describe_service,
    get_hazard_prioritizer
)
from utils.scene_builder import build_scene_json

# Configure Logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("VisionAssistantApp")

# Dedicated single-worker background executor for heavy AI tasks (OCR, Face, Depth)
# Using max_workers=1 prevents multi-model thread contention on CPU
bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bg_ai_worker")

# Global state to hold latest frame and scene data
latest_state = {
    "frame": None,
    "scene_json": None,
    "last_detections": [],
    "last_hazards": []
}

# Real-time performance monitor stats
perf_monitor = {
    "frames_received": 0,
    "frames_processed": 0,
    "frames_dropped": 0,
    "total_inference_time": 0.0,
    "avg_inference_ms": 0.0,
    "processed_fps": 0.0,
    "received_fps": 0.0,
    "start_time": time.time(),
    "last_sec_timestamp": time.time(),
    "sec_frames_processed": 0,
    "sec_frames_received": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} on {settings.DEVICE} with {settings.TORCH_THREADS} torch threads")
    init_services()
    tts_manager.start()
    perf_monitor["start_time"] = time.time()
    perf_monitor["last_sec_timestamp"] = time.time()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Vision Assistant services...")
    tts_manager.stop()
    bg_executor.shutdown(wait=False)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Optimized AI assistant for visually impaired people with real-time walking assistance",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/config")
def get_config():
    """Returns current active system configuration."""
    return {
        "model_path": settings.YOLO_MODEL_PATH,
        "imgsz": settings.YOLO_IMGSZ,
        "conf_threshold": settings.YOLO_CONF,
        "iou_threshold": settings.YOLO_IOU,
        "device": settings.DEVICE,
        "torch_threads": settings.TORCH_THREADS,
        "tracking_enabled": settings.TRACKING_ENABLED,
        "hazard_detection_enabled": settings.HAZARD_DETECTION_ENABLED,
        "tts_mode": settings.TTS_MODE
    }

@app.get("/api/metrics")
def get_metrics():
    """Returns live telemetry and resource consumption statistics."""
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "processed_fps": round(perf_monitor["processed_fps"], 1),
        "received_fps": round(perf_monitor["received_fps"], 1),
        "avg_inference_ms": round(perf_monitor["avg_inference_ms"], 1),
        "frames_processed": perf_monitor["frames_processed"],
        "frames_received": perf_monitor["frames_received"],
        "frames_dropped": perf_monitor["frames_dropped"],
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_rss_mb": round(mem.rss / (1024 * 1024), 1),
        "ram_vms_mb": round(mem.vms / (1024 * 1024), 1),
        "uptime_sec": round(time.time() - perf_monitor["start_time"], 1)
    }

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    High-speed, responsive WebSocket endpoint.
    Implements latest-frame buffering, zero backlog queuing, temporal tracking,
    walking hazard corridor prioritization, and telemetry.
    """
    await websocket.accept()
    yolo_detector = get_yolo_detector()
    ocr_service = get_ocr_service()
    face_service = get_face_service()
    depth_service = get_depth_service()
    hazard_prioritizer = get_hazard_prioritizer()
    loop = asyncio.get_running_loop()

    # Low latency state caching
    ocr_cache = {"texts": []}
    face_cache = {"faces": []}
    distance_cache = []
    depth_map_cache = None

    # In-flight flags for background services
    ocr_busy = False
    face_busy = False
    depth_busy = False

    # Latest-frame slot pattern to completely prevent frame queue backlog
    latest_frame_slot = {"data": None, "timestamp": 0.0, "frame_id": 0}
    new_frame_event = asyncio.Event()
    client_connected = True

    # Receiver task: reads frames from WebSocket and updates latest_frame_slot
    async def receiver():
        nonlocal client_connected
        frame_id_counter = 0
        try:
            while client_connected:
                data = await websocket.receive_bytes()
                frame_id_counter += 1
                now = time.time()

                perf_monitor["frames_received"] += 1
                perf_monitor["sec_frames_received"] += 1

                # If the previous frame was not consumed, count as dropped/skipped
                if latest_frame_slot["data"] is not None:
                    perf_monitor["frames_dropped"] += 1

                latest_frame_slot["data"] = data
                latest_frame_slot["timestamp"] = now
                latest_frame_slot["frame_id"] = frame_id_counter
                new_frame_event.set()
        except WebSocketDisconnect:
            client_connected = False
        except Exception as e:
            logger.debug(f"WebSocket receiver disconnected: {e}")
            client_connected = False
        finally:
            new_frame_event.set()

    receiver_task = asyncio.create_task(receiver())

    bg_tasks = set()
    frame_counter = 0

    try:
        while client_connected:
            await new_frame_event.wait()
            new_frame_event.clear()

            if not client_connected:
                break

            # Grab latest frame atomically
            frame_data = latest_frame_slot["data"]
            frame_recv_time = latest_frame_slot["timestamp"]
            frame_id = latest_frame_slot["frame_id"]
            latest_frame_slot["data"] = None

            if frame_data is None or yolo_detector is None:
                continue

            # Decode image bytes
            t_decode_start = time.perf_counter()
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame_counter += 1
            perf_monitor["frames_processed"] += 1
            perf_monitor["sec_frames_processed"] += 1

            # Update FPS calculations every second
            now_sec = time.time()
            if now_sec - perf_monitor["last_sec_timestamp"] >= 1.0:
                elapsed_sec = now_sec - perf_monitor["last_sec_timestamp"]
                perf_monitor["processed_fps"] = perf_monitor["sec_frames_processed"] / elapsed_sec
                perf_monitor["received_fps"] = perf_monitor["sec_frames_received"] / elapsed_sec
                perf_monitor["sec_frames_processed"] = 0
                perf_monitor["sec_frames_received"] = 0
                perf_monitor["last_sec_timestamp"] = now_sec

            # 1. FAST PATH: Run YOLO Object Detection + Temporal Tracking
            t_inf_start = time.perf_counter()
            detections = await asyncio.to_thread(yolo_detector.detect, frame, None, True)
            inf_duration_ms = (time.perf_counter() - t_inf_start) * 1000.0

            # Update rolling average inference time
            perf_monitor["total_inference_time"] += inf_duration_ms
            perf_monitor["avg_inference_ms"] = (
                perf_monitor["total_inference_time"] / perf_monitor["frames_processed"]
            )

            yolo_raw = list(detections)
            h, w = frame.shape[:2]

            # 2. HAZARD PRIORITIZATION & SPATIAL WALKING CORRIDOR ANALYSIS
            hazard_data = {"hazards": [], "spoken_alert": "", "is_urgent": False, "corridor": {"x_min": 0, "x_max": w}}
            if hazard_prioritizer and settings.HAZARD_DETECTION_ENABLED:
                hazard_data = hazard_prioritizer.evaluate_hazards(detections, frame_width=w, frame_height=h, frame=frame)
                
                # If ground surface analysis detected stairs, merge into detections list
                for haz in hazard_data.get("hazards", []):
                    if haz.get("class_name") in ["stairs", "staircase", "steps"] and not any(d.get("class_name") in ["stairs", "staircase", "steps"] for d in detections):
                        detections.append({
                            "track_id": haz.get("track_id", 9901),
                            "box": haz["box"],
                            "confidence": haz["confidence"],
                            "class_name": haz["class_name"]
                        })

                # Speak urgent or proactive hazard alert on server if configured
                alert_text = hazard_data.get("spoken_alert")
                if alert_text and settings.TTS_MODE in ["server", "both"]:
                    tts_manager.speak(alert_text)

            # 3. BACKGROUND SERVICES (Staggered to prevent CPU starvation)
            # Face recognition on scheduled interval or if persons detected
            if frame_counter % settings.FACE_INTERVAL_FRAMES == 0 and face_service is not None and not face_busy:
                face_busy = True
                async def update_face(f):
                    nonlocal face_cache, face_busy
                    try:
                        res = await loop.run_in_executor(bg_executor, face_service.recognize_face, f, 0.6)
                        face_cache = res
                    except Exception as err:
                        logger.debug(f"Face service error: {err}")
                    finally:
                        face_busy = False

                t = asyncio.create_task(update_face(frame.copy()))
                bg_tasks.add(t)
                t.add_done_callback(bg_tasks.discard)

            # OCR detection on scheduled interval
            if frame_counter % settings.OCR_INTERVAL_FRAMES == 0 and ocr_service is not None and not ocr_busy:
                ocr_busy = True
                async def update_ocr(f):
                    nonlocal ocr_cache, ocr_busy
                    try:
                        res = await loop.run_in_executor(bg_executor, ocr_service.detect_text, f, 0.5)
                        ocr_cache = res
                    except Exception as err:
                        logger.debug(f"OCR service error: {err}")
                    finally:
                        ocr_busy = False

                t = asyncio.create_task(update_ocr(frame.copy()))
                bg_tasks.add(t)
                t.add_done_callback(bg_tasks.discard)

            # Depth estimation on scheduled interval (if not on-demand only)
            if not settings.RUN_DEPTH_ON_DEMAND_ONLY and frame_counter % settings.DEPTH_INTERVAL_FRAMES == 0 and depth_service is not None and not depth_busy:
                depth_busy = True
                async def update_depth(f, dets):
                    nonlocal distance_cache, depth_map_cache, depth_busy
                    try:
                        dists, dmap = await loop.run_in_executor(bg_executor, depth_service.estimate_depth, f, dets)
                        if dists: distance_cache = dists
                        if dmap: depth_map_cache = dmap
                    except Exception as err:
                        logger.debug(f"Depth service error: {err}")
                    finally:
                        depth_busy = False

                t = asyncio.create_task(update_depth(frame.copy(), list(detections)))
                bg_tasks.add(t)
                t.add_done_callback(bg_tasks.discard)

            # 4. Merge cached OCR & Face detections
            combined_detections = list(detections)
            for item in ocr_cache.get("texts", []):
                bbox = item.get("bbox", [])
                if len(bbox) == 4:
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    combined_detections.append({
                        "box": [min(xs), min(ys), max(xs), max(ys)],
                        "confidence": item["confidence"],
                        "class_name": f'"{item["text"]}"'
                    })

            for item in face_cache.get("faces", []):
                combined_detections.append({
                    "box": item["box"],
                    "confidence": item["confidence"],
                    "class_name": f'Face: {item["name"]}'
                })

            # Update global state for description/query APIs
            scene_json = build_scene_json(yolo_raw, ocr_cache, face_cache, distance_cache)
            latest_state["frame"] = frame
            latest_state["scene_json"] = scene_json
            latest_state["last_detections"] = combined_detections
            latest_state["last_hazards"] = hazard_data.get("hazards", [])

            # Compute latency metrics
            total_server_latency_ms = (time.time() - frame_recv_time) * 1000.0

            # 5. Send enriched response back to client immediately
            await websocket.send_json({
                "frame_id": frame_id,
                "server_timestamp": time.time(),
                "server_latency_ms": round(total_server_latency_ms, 1),
                "inference_time_ms": round(inf_duration_ms, 1),
                "processed_fps": round(perf_monitor["processed_fps"], 1),
                "received_fps": round(perf_monitor["received_fps"], 1),
                "dropped_frames": perf_monitor["frames_dropped"],
                "detections": combined_detections,
                "hazards": hazard_data.get("hazards", []),
                "hazard_alert": hazard_data.get("spoken_alert", ""),
                "is_urgent": hazard_data.get("is_urgent", False),
                "corridor": hazard_data.get("corridor"),
                "distance": distance_cache,
                "depth_map": depth_map_cache
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected from detection WebSocket")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        client_connected = False
        receiver_task.cancel()
        if yolo_detector:
            yolo_detector.reset_tracker()

@app.get("/audio/speak")
def speak_text(text: str = "Hello, I am your vision assistant."):
    """Endpoint for Text-to-Speech playback."""
    tts_manager.speak(text)
    return {"status": "speaking", "text": text}

@app.get("/api/describe_scene")
async def describe_scene_api():
    """Generates an audio scene description for the visually impaired user."""
    describe_service = get_describe_service()
    if describe_service is None:
        return {"description": "Describe service not initialized.", "error": True}

    frame = latest_state.get("frame")
    scene_json = latest_state.get("scene_json")

    if frame is None or scene_json is None:
        return {"description": "No active camera feed or scene data.", "error": True}

    try:
        description = describe_service.generate_description(frame, scene_json)
        if settings.TTS_MODE in ["server", "both"]:
            tts_manager.speak(description)
        return {"description": description, "success": True}
    except Exception as e:
        logger.error(f"Describe scene API error: {e}")
        return {"description": str(e), "error": True}

@app.get("/api/ask_question")
async def ask_question_api(q: str):
    """Answers a spoken voice query based on current camera surroundings."""
    describe_service = get_describe_service()
    if describe_service is None:
        return {"answer": "Describe service not initialized.", "error": True}

    frame = latest_state.get("frame")
    scene_json = latest_state.get("scene_json")

    if frame is None or scene_json is None:
        return {"answer": "No active camera feed or scene data.", "error": True}

    try:
        answer = describe_service.answer_question(q, frame, scene_json)
        if settings.TTS_MODE in ["server", "both"]:
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
