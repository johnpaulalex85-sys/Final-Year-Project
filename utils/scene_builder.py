import datetime
import json

def _is_inside(inner_box, outer_box):
    if not inner_box or not outer_box or len(inner_box) != 4 or len(outer_box) != 4:
        return False
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[2], outer_box[2])
    yB = min(inner_box[3], outer_box[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    innerArea = max(0, inner_box[2] - inner_box[0]) * max(0, inner_box[3] - inner_box[1])
    if innerArea <= 0:
        return False
    return (interArea / innerArea) > 0.5

def build_scene_json(yolo_dets, ocr_dets, face_dets, distances):
    """
    Merges outputs from various models into a single structured JSON object.
    """
    # Process YOLO objects
    objects = []
    if yolo_dets:
        for det in yolo_dets:
            if det.get("class_name", "").startswith('"') or det.get("class_name", "").startswith("Face: "):
                continue
            objects.append({
                "class_name": det.get("class_name"),
                "confidence": det.get("confidence"),
                "box": det.get("box")
            })

    # Process faces (filtering out printed faces inside books, notebooks, posters, screens, frames)
    printed_face_classes = {
        "book", "notebook", "paper", "poster", "tv", "television", "screen",
        "laptop", "photo frame", "picture frame", "framed picture", "portrait", "painting"
    }
    faces = []
    if face_dets and "faces" in face_dets:
        for face in face_dets["faces"]:
            f_box = face.get("box")
            is_printed = False
            if f_box:
                for obj in objects:
                    if obj.get("class_name") in printed_face_classes:
                        if _is_inside(f_box, obj.get("box")):
                            is_printed = True
                            break
            if not is_printed:
                faces.append({
                    "name": face.get("name"),
                    "confidence": face.get("confidence"),
                    "box": face.get("box")
                })

    # Process text
    texts = []
    if ocr_dets and "texts" in ocr_dets:
        for text_obj in ocr_dets["texts"]:
            # OCR bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]. Convert to [x1, y1, x2, y2]
            bbox = text_obj.get("bbox", [])
            if len(bbox) == 4:
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                box = [min(xs), min(ys), max(xs), max(ys)]
            else:
                box = []
                
            texts.append({
                "text": text_obj.get("text"),
                "confidence": text_obj.get("confidence"),
                "box": box
            })

    scene_data = {
        "objects": objects,
        "faces": faces,
        "texts": texts,
        "distances": distances if distances else [],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    return scene_data
