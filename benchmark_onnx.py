import time
import os
import torch
import numpy as np
from ultralytics import YOLO

def test_onnx():
    print("Testing ONNX export and inference for yolov8n.pt...")
    model = YOLO("yolov8n.pt")
    
    # Export to onnx if not exists
    onnx_path = "yolov8n.onnx"
    if not os.path.exists(onnx_path):
        print("Exporting yolov8n to ONNX (imgsz=384)...")
        model.export(format="onnx", imgsz=384, dynamic=False, simplify=True)
    
    if os.path.exists(onnx_path):
        import onnxruntime as ort
        print(f"ONNX file exists: {onnx_path}")
        print("Available providers:", ort.get_available_providers())
        
        # Test Ultralytics with ONNX model
        onnx_model = YOLO(onnx_path, task="detect")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Warmup
        _ = onnx_model.predict(source=dummy, imgsz=384, conf=0.45, verbose=False)
        
        latencies = []
        for _ in range(25):
            t0 = time.perf_counter()
            _ = onnx_model.predict(source=dummy, imgsz=384, conf=0.45, verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(latencies)
        print(f"yolov8n.onnx @ imgsz=384: {avg:.1f} ms ({1000.0/avg:.1f} FPS)")
    else:
        print("ONNX export was not created.")

if __name__ == "__main__":
    test_onnx()
