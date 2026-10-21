import time
import cv2
import numpy as np
import torch

def test_services():
    print("Testing individual service latencies on Intel Core i7-1255U...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy, "EXIT SIGN AHEAD", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 1. OCR Service
    try:
        from services.ocr import OCRService
        t0 = time.time()
        ocr = OCRService()
        init_time = time.time() - t0
        print(f"OCR load time: {init_time:.2f}s")
        
        t0 = time.perf_counter()
        res = ocr.detect_text(dummy, conf_threshold=0.5)
        dt = (time.perf_counter() - t0) * 1000
        print(f"OCR inference latency: {dt:.1f}ms (detected: {len(res.get('texts', []))} texts)")
    except Exception as e:
        print(f"OCR test error: {e}")

    # 2. Face Service
    try:
        from services.face_service import FaceService
        t0 = time.time()
        face = FaceService()
        init_time = time.time() - t0
        print(f"FaceService load time: {init_time:.2f}s")
        
        t0 = time.perf_counter()
        res = face.recognize_face(dummy)
        dt = (time.perf_counter() - t0) * 1000
        print(f"Face inference latency: {dt:.1f}ms")
    except Exception as e:
        print(f"Face test error: {e}")

    # 3. Depth Service
    try:
        from services.depth_service import DepthService
        t0 = time.time()
        depth = DepthService()
        init_time = time.time() - t0
        print(f"DepthService load time: {init_time:.2f}s")
        
        dets = [{"box": [100, 100, 300, 300], "class_name": "person", "confidence": 0.8}]
        t0 = time.perf_counter()
        dists, _ = depth.estimate_depth(dummy, dets)
        dt = (time.perf_counter() - t0) * 1000
        print(f"Depth inference latency: {dt:.1f}ms (results: {dists})")
    except Exception as e:
        print(f"Depth test error: {e}")

    # 4. Describe Service
    try:
        from services.describe_service import DescribeService
        from utils.scene_builder import build_scene_json
        desc = DescribeService()
        scene = build_scene_json(
            [{"box": [200, 100, 400, 400], "class_name": "person", "confidence": 0.85}],
            {"texts": [{"text": "EXIT", "confidence": 0.9, "bbox": [[100, 50], [180, 50], [180, 90], [100, 90]]}]},
            {"faces": [{"name": "Alex", "box": [250, 120, 350, 220], "confidence": 0.88}]},
            [{"object": "person", "distance": "1.5 meters", "box": [200, 100, 400, 400]}]
        )
        t0 = time.perf_counter()
        d_text = desc.generate_description(dummy, scene)
        dt = (time.perf_counter() - t0) * 1000
        print(f"DescribeService latency: {dt:.2f}ms -> '{d_text}'")
    except Exception as e:
        print(f"Describe test error: {e}")

if __name__ == "__main__":
    test_services()
