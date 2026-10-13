# Vision Assistant API

An AI-powered assistant for visually impaired people, built with Python and FastAPI.

## Features

- **FastAPI** Backend for scalable API endpoints.
- **Computer Vision** powered by:
  - OpenCV (Camera interfacing)
  - Ultralytics YOLOv8 (Object Detection)
  - EasyOCR (Optical Character Recognition)
  - InsightFace (Face Recognition)
  - Depth Anything V2 (Depth Estimation)
  - Qwen2-VL (Vision-Language Model for scene understanding)
- **Audio Processing**:
  - Whisper (Speech-to-Text)
  - pyttsx3 (Text-to-Speech)

## Project Structure

```
├── api/             # FastAPI routes and endpoints
├── assets/          # Static assets (images, icons)
├── audio/           # Audio processing (TTS, STT)
├── camera/          # Camera interface and frame processing
├── models/          # Model weights and configuration
├── outputs/         # Generated output files
├── services/        # Business logic and model inference wrappers
├── utils/           # Helper functions and utilities
├── app.py           # FastAPI application entry point
├── config.py        # Pydantic settings and configuration
└── requirements.txt # Python dependencies
```

## Setup Instructions

1. **Create a virtual environment (Python 3.11 required):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```
"# Friday-" 
"# Final-Year-Project" 
