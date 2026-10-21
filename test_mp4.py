import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestMP4Pipeline")

# Import project modules
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

def draw_annotations(frame, yolo_dets, face_cache, ocr_cache, distance_cache, description="", hazards=None, corridor=None, metrics=None):
    """Annotates video frame with corridor rails, hazard badges, track IDs, labels, distances, and scene HUD."""
    annotated = frame.copy()
    h, w, _ = annotated.shape
    hazards = hazards or []

    # 0. Draw Walking Corridor Guide Rails
    if corridor and "x_min" in corridor and "x_max" in corridor:
        x_min = corridor["x_min"]
        x_max = corridor["x_max"]
        # Draw dotted / dashed vertical guide lines
        for y_step in range(0, h, 20):
            cv2.line(annotated, (x_min, y_step), (x_min, min(h, y_step + 10)), (255, 200, 50), 1)
            cv2.line(annotated, (x_max, y_step), (x_max, min(h, y_step + 10)), (255, 200, 50), 1)
        cv2.putText(annotated, "WALKING CORRIDOR", ((x_min + x_max) // 2 - 70, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 50), 1)

    urgent_track_ids = set(haz.get("track_id") for haz in hazards if haz.get("is_urgent"))
    hazard_track_ids = set(haz.get("track_id") for haz in hazards)

    # 1. Draw YOLO Object Detections (with Tracking & Hazard Urgency)
    for det in yolo_dets:
        box = det.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id")
        is_approaching = det.get("is_approaching", False)

        is_urgent = (track_id in urgent_track_ids) if track_id else False
        is_hazard = (track_id in hazard_track_ids) if track_id else False

        # Color scheme: Red for Urgent, Amber for Hazard, Cyan for standard
        if is_urgent:
            box_color = (0, 0, 255)  # BGR Red
            label_bg = (0, 0, 220)
        elif is_hazard:
            box_color = (0, 165, 255)  # BGR Orange/Amber
            label_bg = (0, 140, 230)
        else:
            box_color = (255, 191, 0)  # BGR Cyan/Blue
            label_bg = (255, 191, 0)

        # Match distance
        dist_str = ""
        for d in distance_cache:
            if d.get("object") == cls_name:
                dist_str = f" | {d.get('distance')}"
                break

        track_str = f"[#{track_id}] " if track_id else ""
        appr_str = " ^APPR" if is_approaching else ""
        label = f"{track_str}{cls_name} {int(conf * 100)}%{dist_str}{appr_str}"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2 if not is_urgent else 3)

        # Text label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 6, max(22, y1)), label_bg, -1)
        text_color = (255, 255, 255) if is_urgent or is_hazard else (0, 0, 0)
        cv2.putText(annotated, label, (x1 + 3, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 1)

    # 2. Draw Recognized Faces (Purple)
    for face in face_cache.get("faces", []):
        box = face.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        name = face.get("name", "Unknown")
        label = f"Face: {name}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 140, max(22, y1)), (255, 0, 255), -1)
        cv2.putText(annotated, label, (x1 + 3, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # 3. Draw OCR Text Detections (Yellow)
    for item in ocr_cache.get("texts", []):
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            text = item.get("text", "")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f'"{text}"', (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 4. HUD Top Bar for Telemetry and Scene Status
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    hud_fps = f"{metrics.get('fps', 0):.1f} FPS" if metrics else ""
    hud_inf = f"Infer: {metrics.get('inf_ms', 0):.1f}ms" if metrics else ""
    hud_hazards = f"Hazards: {len(hazards)}" if hazards else "Path: Clear"
    cv2.putText(annotated, f"Vision Assistant v2.0 | {hud_fps} | {hud_inf} | {hud_hazards}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 1)

    if description:
        cv2.putText(annotated, f"AI: {description[:75]}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return annotated


def create_dummy_sample_video(output_path="sample_test_input.mp4", duration_sec=5, fps=24):
    """Creates a sample test MP4 video with moving shapes/text if no input file is provided."""
    logger.info(f"No MP4 input file supplied. Creating a sample test video: {output_path}")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create dark background
        frame = np.full((h, w, 3), (30, 20, 10), dtype=np.uint8)
        
        # Draw moving rectangle (simulating a moving obstacle / object)
        x_pos = int((i / total_frames) * (w - 150))
        cv2.rectangle(frame, (x_pos, 180), (x_pos + 120, 320), (50, 180, 50), -1)
        cv2.putText(frame, "MOVING OBJECT", (x_pos + 5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Add sample text for OCR
        cv2.putText(frame, "VISION ASSISTANT TEST", (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        out.write(frame)
        
    out.release()
    logger.info(f"Sample test video created at {output_path}")
    return output_path


def test_pipeline_on_mp4(video_path: str, output_path: str = "outputs/mp4_test_output.mp4", display_window: bool = False):
    """Main testing routine for running the full AI vision assistant pipeline smoothly on an MP4 video."""
    import threading
    import queue

    if not os.path.exists(video_path):
        logger.warning(f"Video file '{video_path}' not found!")
        video_path = create_dummy_sample_video("sample_test_input.mp4")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Opened Video: {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize all AI services
    logger.info("Initializing AI pipeline models...")
    start_init_time = time.time()
    init_services()
    logger.info(f"Models initialized in {time.time() - start_init_time:.2f} seconds.")

    yolo_detector = get_yolo_detector()
    ocr_service = get_ocr_service()
    face_service = get_face_service()
    depth_service = get_depth_service()
    describe_service = get_describe_service()
    hazard_prioritizer = get_hazard_prioritizer()

    # Shared thread-safe state for AI detections
    state_lock = threading.Lock()
    ai_state = {
        "yolo_dets": [],
        "face_cache": {"faces": []},
        "ocr_cache": {"texts": []},
        "distance_cache": [],
        "description": "",
        "hazards": [],
        "corridor": {"x_min": int(width * 0.32), "x_max": int(width * 0.68)},
        "inf_ms": 0.0
    }
    
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Background AI Worker Thread
    def ai_worker():
        ai_frame_count = 0
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_to_process, f_count = item
            ai_frame_count = f_count

            # Downscale frame for fast AI inference if width > 640
            h_orig, w_orig = frame_to_process.shape[:2]
            if w_orig > 640:
                scale = 640.0 / w_orig
                frame_small = cv2.resize(frame_to_process, (640, int(h_orig * scale)))
            else:
                scale = 1.0
                frame_small = frame_to_process

            # 1. YOLO Object Detection with Temporal Tracking
            y_dets = []
            inf_ms = 0.0
            if yolo_detector:
                t0 = time.perf_counter()
                raw_dets = yolo_detector.detect(frame_small, enable_tracking=True)
                inf_ms = (time.perf_counter() - t0) * 1000.0
                # Rescale boxes back to original video dimensions
                for d in raw_dets:
                    box = d.get("box", [])
                    if len(box) == 4 and scale != 1.0:
                        box = [round(v / scale, 2) for v in box]
                    y_dets.append({
                        "track_id": d.get("track_id"),
                        "class_name": d.get("class_name"),
                        "confidence": d.get("confidence"),
                        "box": box,
                        "is_approaching": d.get("is_approaching", False)
                    })

            # 2. Hazard Prioritization
            haz_res = {"hazards": [], "corridor": {"x_min": int(w_orig * 0.32), "x_max": int(w_orig * 0.68)}}
            if hazard_prioritizer:
                haz_res = hazard_prioritizer.evaluate_hazards(y_dets, frame_width=w_orig, frame_height=h_orig)

            # 3. Face Recognition (Staggered every 15th frame)
            f_cache = ai_state["face_cache"]
            if ai_frame_count % 15 == 0 and face_service:
                try:
                    f_cache = face_service.recognize_face(frame_small, threshold=0.6)
                except Exception as e:
                    logger.debug(f"Face error: {e}")

            # 4. OCR Text Detection (Staggered every 30th frame)
            o_cache = ai_state["ocr_cache"]
            if ai_frame_count % 30 == 0 and ocr_service:
                try:
                    o_cache = ocr_service.detect_text(frame_small, min_confidence=0.5)
                except Exception as e:
                    logger.debug(f"OCR error: {e}")

            # 5. Depth Estimation (Staggered every 20th frame)
            d_cache = ai_state["distance_cache"]
            if ai_frame_count % 20 == 0 and depth_service:
                try:
                    dists, _ = depth_service.estimate_depth(frame_small, y_dets)
                    if dists:
                        d_cache = dists
                except Exception as e:
                    logger.debug(f"Depth error: {e}")

            # 6. Scene Description (Every 40th AI frame)
            desc = ai_state["description"]
            if ai_frame_count % 40 == 1 and describe_service:
                scene_json = build_scene_json(y_dets, o_cache, f_cache, d_cache)
                desc = describe_service.generate_description(frame_small, scene_json)

            # Update shared state safely
            with state_lock:
                ai_state["yolo_dets"] = y_dets
                ai_state["face_cache"] = f_cache
                ai_state["ocr_cache"] = o_cache
                ai_state["distance_cache"] = d_cache
                ai_state["description"] = desc
                ai_state["hazards"] = haz_res.get("hazards", [])
                ai_state["corridor"] = haz_res.get("corridor")
                ai_state["inf_ms"] = inf_ms

            frame_queue.task_done()

    # Start AI worker thread
    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    frame_count = 0
    start_time = time.time()
    target_frame_duration = 1.0 / (fps if fps > 0 else 24.0)

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Send frame to background AI queue if space available
            if not frame_queue.full():
                frame_queue.put((frame.copy(), frame_count))

            # Retrieve current AI state safely
            with state_lock:
                yolo_dets = list(ai_state["yolo_dets"])
                face_cache = dict(ai_state["face_cache"])
                ocr_cache = dict(ai_state["ocr_cache"])
                distance_cache = list(ai_state["distance_cache"])
                current_description = ai_state["description"]
                hazards = list(ai_state.get("hazards", []))
                corridor = ai_state.get("corridor")
                inf_ms = ai_state.get("inf_ms", 0.0)

            # Annotate current frame
            elapsed_sec = time.time() - start_time
            live_fps = frame_count / elapsed_sec if elapsed_sec > 0 else 0.0
            annotated_frame = draw_annotations(
                frame,
                yolo_dets,
                face_cache,
                ocr_cache,
                distance_cache,
                description=current_description,
                hazards=hazards,
                corridor=corridor,
                metrics={"fps": live_fps, "inf_ms": inf_ms}
            )

            writer.write(annotated_frame)

            # Live Smooth Video Preview Window
            if display_window:
                try:
                    cv2.imshow("Vision Assistant MP4 Live Detection", annotated_frame)
                    # Smooth delay matching target video FPS
                    elapsed_frame_time = time.time() - loop_start
                    delay_ms = max(1, int((target_frame_duration - elapsed_frame_time) * 1000))
                    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                        logger.info("User interrupted test via window key 'q'.")
                        break
                except Exception:
                    logger.warning("cv2.imshow GUI window not available. Continuing video processing in background...")
                    display_window = False

            if frame_count % 30 == 0 or frame_count == total_frames:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Playing video: {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")

    finally:
        stop_event.set()
        ai_thread.join(timeout=1.0)
        cap.release()
        writer.release()
        if display_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    logger.info("=" * 60)
    logger.info("TEST PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Total Frames Processed: {frame_count}")
    logger.info(f"Total Processing Time: {total_time:.2f} seconds")
    logger.info(f"Average Pipeline FPS:   {avg_fps:.2f} FPS")
    logger.info(f"Annotated Video Saved:  {os.path.abspath(output_path)}")
    logger.info("=" * 60)


def select_input_video(default_path="test.mp4"):
    """Prompts the user to pick an MP4 video file using a file dialog or terminal input."""
    # Try opening a GUI File Dialog if tkinter is available
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw() # Hide main tkinter window
        root.attributes('-topmost', True)
        logger.info("Opening File Chooser dialog to select an MP4 video file...")
        selected_file = filedialog.askopenfilename(
            title="Select Input MP4 Video File",
            filetypes=[("MP4 Video Files", "*.mp4"), ("All Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
        )
        root.destroy()
        if selected_file and os.path.exists(selected_file):
            logger.info(f"Selected Video File: {selected_file}")
            return selected_file
    except Exception as e:
        logger.debug(f"GUI File Dialog not available: {e}")

    # Fallback to Terminal Prompt
    print("\n" + "=" * 60)
    print(" VISION ASSISTANT - MP4 LIVE DETECTION TEST")
    print("=" * 60)
    user_input = input("Enter path to your MP4 video file (or press Enter for default/sample): ").strip().strip('"').strip("'")
    if user_input and os.path.exists(user_input):
        return user_input
    
    return default_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Vision Assistant pipeline with an MP4 video file.")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input MP4 video file")
    parser.add_argument("--output", "-o", type=str, default="outputs/mp4_test_output.mp4", help="Path for output annotated MP4 video file")
    parser.add_argument("--no-show", action="store_true", help="Disable live OpenCV preview window")

    args = parser.parse_args()

    # Determine input video path
    video_file = args.input
    if not video_file:
        video_file = select_input_video()

    # Display live window by default unless --no-show is explicitly passed
    show_live_preview = not args.no_show

    test_pipeline_on_mp4(
        video_path=video_file, 
        output_path=args.output, 
        display_window=show_live_preview
    )

