# Vision Assistant: Build Summary & Documentation

## 1. Project Overview
The Vision Assistant is a real-time, AI-powered sensory augmentation tool designed to help visually impaired individuals. It leverages your mobile phone's camera to "see" the world, transmits the video securely to a PC-hosted backend, runs advanced object detection (YOLOv8), and provides spoken audio feedback about the environment.

---

## 2. Technical Stack
* **Backend:** Python 3.11, FastAPI, Uvicorn, WebSockets
* **Computer Vision:** OpenCV (`cv2`), Ultralytics YOLOv8
* **Audio / TTS:** `pyttsx3` (Text-to-Speech)
* **Frontend:** HTML5, Vanilla JS, CSS Glassmorphism, WebRTC (`getUserMedia`), Canvas API
* **Network:** `ngrok` (for secure HTTPS tunneling to mobile devices)

---

## 3. Data Flow Architecture
The system operates on a highly optimized, low-latency loop:
1. **Capture:** The HTML5 `<video>` element on the mobile phone captures live frames using the phone's back camera.
2. **Transmit:** The JavaScript frontend compresses the current frame into a JPEG Blob and pushes it over a WebSocket connection to the backend.
3. **Process:** The FastAPI backend receives the binary data, decodes it using OpenCV, and offloads it to a background thread where YOLOv8 runs inference.
4. **Respond:** The backend calculates bounding box coordinates, class names, and confidence scores, sending them back to the phone as a JSON payload.
5. **Render:** The frontend's HTML5 `<canvas>` clears itself and redraws the new bounding boxes perfectly aligned over the live video.
6. **Act:** If the user presses "Analyze Scene (Speak)", the frontend takes the most confident object detection and fires a request to `/audio/speak`, causing the backend to speak the result out loud.

---

## 4. Codebase Breakdown

### `app.py` (The Core Server)
This is the central nervous system of the application, built with **FastAPI**.
- **Lifecycle Management:** Uses `@asynccontextmanager` to safely start and stop the Text-to-Speech background thread when the server boots up.
- **WebSocket Endpoint (`/ws/detect`):** The heavy lifter. It maintains an open connection with the phone. When it receives image bytes, it decodes them (`np.frombuffer` -> `cv2.imdecode`), runs `yolo_detector.detect()` in a non-blocking `asyncio.to_thread` pool, and sends the JSON results back.
- **Static Hosting:** Mounts the `static/` directory to serve the frontend web page when you visit the root `/` URL.

### `static/index.html` (The Mobile Frontend)
A fully responsive, dark-mode web application designed to run on a phone browser.
- **`startCamera()`:** Uses WebRTC (`navigator.mediaDevices.getUserMedia`) to request permission for the phone's environment (back) camera.
- **`connectWebSocket()`:** Establishes the real-time two-way connection to `/ws/detect`.
- **`sendFrame()`:** Uses an invisible `<canvas>` to grab a snapshot of the video, convert it to a lightweight JPEG, and send it over the socket. It uses `requestAnimationFrame` to only send the next frame *after* the previous one was processed, preventing network congestion.

### `services/yolo.py` (The AI Engine)
Wraps the `ultralytics` YOLO model to keep the code clean.
- **`detect()`:** Takes an OpenCV image array, runs `model.predict()`, and parses the complex PyTorch tensors into a clean, simple Python dictionary containing `box` (coordinates), `confidence`, and `class_name` (e.g., "person", "chair").

### `audio/tts_manager.py` (The Voice)
Manages the `pyttsx3` text-to-speech engine.
- Because TTS engines are strictly synchronous and block the main thread while talking, this manager uses a dedicated Python `threading.Thread` and a `queue.Queue`. 
- When the API calls `speak()`, it just drops the text into the queue instantly, allowing the server to keep processing video frames while the background thread does the actual talking.

### `config.py`
Uses `pydantic-settings` to manage environment variables and project configuration (like version numbers and debug modes) in a type-safe way.

---

## 5. Why WebSockets instead of MJPEG?
Initially, the PC webcam captured video, and the server streamed MJPEG images to the browser. Switching to the phone camera required reversing this. WebSockets were chosen over standard HTTP POST requests because:
1. They avoid the overhead of opening and closing HTTP connections 30 times a second.
2. They allow true two-way real-time communication, minimizing lag between the camera moving and the blue boxes updating on the screen.
