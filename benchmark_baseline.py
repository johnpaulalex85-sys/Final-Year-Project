import time
import os
import sys
import psutil
import torch
import cv2
import numpy as np
from ultralytics import YOLO
try:
    from ultralytics import YOLOWorld
except Exception:
    YOLOWorld = YOLO

def benchmark():
    print(f"=== SYSTEM INFO ===")
    print(f"CPU count (logical): {psutil.cpu_count(logical=True)}")
    print(f"CPU count (physical): {psutil.cpu_count(logical=False)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Thread count: {torch.get_num_threads()}")
    
    # Check sample video
    video_path = "sample_test_input.mp4"
    if not os.path.exists(video_path):
        print("sample_test_input.mp4 not found, creating dummy frame")
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
    else:
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(60):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        print(f"Loaded {len(frames)} frames from {video_path}")
    
    sample_frame = frames[0] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
    
    from services.yolo import DEFAULT_CUSTOM_CLASSES
    print(f"Number of DEFAULT_CUSTOM_CLASSES: {len(DEFAULT_CUSTOM_CLASSES)}")
    
    # 1. Benchmark YOLO-World (yolov8s-worldv2.pt) with all 127 classes
    print("\n--- 1. Testing yolov8s-worldv2.pt (Existing baseline) ---")
    t0 = time.time()
    model_world = YOLOWorld("yolov8s-worldv2.pt")
    if hasattr(model_world, "set_classes"):
        model_world.set_classes(DEFAULT_CUSTOM_CLASSES)
    load_time_world = time.time() - t0
    print(f"yolov8s-worldv2 load + set_classes time: {load_time_world:.3f}s")
    
    # Warmup
    _ = model_world.predict(source=sample_frame, imgsz=384, conf=0.45, verbose=False)
    
    # Test resolutions with yolov8s-worldv2
    for imgsz in [320, 384, 416, 512]:
        latencies = []
        for f in frames[:20]:
            t_start = time.perf_counter()
            _ = model_world.predict(source=f, imgsz=imgsz, conf=0.45, verbose=False)
            latencies.append((time.perf_counter() - t_start) * 1000)
        avg_lat = np.mean(latencies)
        fps = 1000.0 / avg_lat
        print(f"yolov8s-worldv2 (127 classes) @ imgsz={imgsz}: {avg_lat:.1f} ms/frame ({fps:.1f} FPS)")
        
    # Test yolov8s-worldv2 with streamlined classes (e.g. 25 walking-critical classes)
    STREAMLINED_CLASSES = [
        "person", "car", "bus", "truck", "motorcycle", "bicycle",
        "staircase", "stairs", "step", "curb", "pothole", "obstacle",
        "pole", "tree", "door", "chair", "table", "traffic light",
        "pedestrian crossing", "dog", "wall", "trash bin", "wet floor", "road barrier"
    ]
    if hasattr(model_world, "set_classes"):
        model_world.set_classes(STREAMLINED_CLASSES)
        # Warmup
        _ = model_world.predict(source=sample_frame, imgsz=384, conf=0.45, verbose=False)
        latencies = []
        for f in frames[:20]:
            t_start = time.perf_counter()
            _ = model_world.predict(source=f, imgsz=384, conf=0.45, verbose=False)
            latencies.append((time.perf_counter() - t_start) * 1000)
        avg_lat = np.mean(latencies)
        fps = 1000.0 / avg_lat
        print(f"yolov8s-worldv2 ({len(STREAMLINED_CLASSES)} classes) @ imgsz=384: {avg_lat:.1f} ms/frame ({fps:.1f} FPS)")

    # 2. Benchmark yolov8n.pt (Standard Nano YOLO model)
    if os.path.exists("yolov8n.pt"):
        print("\n--- 2. Testing yolov8n.pt (Lightweight Alternative) ---")
        t0 = time.time()
        model_nano = YOLO("yolov8n.pt")
        load_time_nano = time.time() - t0
        print(f"yolov8n load time: {load_time_nano:.3f}s")
        
        # Warmup
        _ = model_nano.predict(source=sample_frame, imgsz=384, conf=0.45, verbose=False)
        
        for imgsz in [320, 384, 416, 512]:
            latencies = []
            for f in frames[:20]:
                t_start = time.perf_counter()
                _ = model_nano.predict(source=f, imgsz=imgsz, conf=0.45, verbose=False)
                latencies.append((time.perf_counter() - t_start) * 1000)
            avg_lat = np.mean(latencies)
            fps = 1000.0 / avg_lat
            print(f"yolov8n (COCO 80 classes) @ imgsz={imgsz}: {avg_lat:.1f} ms/frame ({fps:.1f} FPS)")

    # 3. Test Thread count tuning
    print("\n--- 3. Testing Thread Optimization ---")
    current_threads = torch.get_num_threads()
    for num_t in [2, 4, psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)]:
        if num_t is None or num_t <= 0: continue
        torch.set_num_threads(num_t)
        latencies = []
        for f in frames[:15]:
            t_start = time.perf_counter()
            with torch.inference_mode():
                _ = model_world.predict(source=f, imgsz=384, conf=0.45, verbose=False)
            latencies.append((time.perf_counter() - t_start) * 1000)
        avg_lat = np.mean(latencies)
        print(f"Threads={num_t}: {avg_lat:.1f} ms/frame ({1000.0/avg_lat:.1f} FPS)")
    torch.set_num_threads(current_threads)

    # 4. Memory and CPU usage during inference
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"\nProcess Memory RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
    print(f"Process Memory VMS: {mem_info.vms / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    benchmark()
