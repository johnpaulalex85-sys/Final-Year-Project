import logging

logger = logging.getLogger(__name__)

def _calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if not boxA or not boxB or len(boxA) != 4 or len(boxB) != 4:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator <= 0:
        return 0.0
    return interArea / denominator

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

def _get_direction(box, frame_width=640):
    """Determines spatial direction relative to user: 'on your left', 'directly ahead', or 'on your right'."""
    if not box or len(box) != 4 or frame_width <= 0:
        return "ahead"
    
    center_x = (box[0] + box[2]) / 2.0
    left_boundary = frame_width * 0.35
    right_boundary = frame_width * 0.65
    
    if center_x < left_boundary:
        return "on your left"
    elif center_x > right_boundary:
        return "on your right"
    else:
        return "directly ahead"

def _get_distance_for_box(box, distances, obj_class=""):
    """Finds distance for a bounding box using IoU matching or class matching fallback."""
    if not box or not distances:
        return ""
    
    best_match_dist = ""
    best_iou = 0.0
    
    for d in distances:
        d_box = d.get("box")
        if not d_box:
            continue
            
        # Check IoU overlap
        iou = _calculate_iou(box, d_box)
        if iou > best_iou:
            best_iou = iou
            best_match_dist = d.get("distance", "")
            
    # Fallback to matching class name if IoU overlap was low
    if not best_match_dist and obj_class:
        for d in distances:
            if d.get("object", "").lower() == obj_class.lower():
                return d.get("distance", "")

def _apply_nms(objects, iou_thresh=0.4):
    """Filters out overlapping bounding boxes, keeping the highest-confidence object."""
    if not objects:
        return []
    sorted_objs = sorted(objects, key=lambda x: x.get("confidence", 0.0), reverse=True)
    keep = []
    for obj in sorted_objs:
        box = obj.get("box")
        if not box:
            continue
        overlap = False
        for kept in keep:
            if _calculate_iou(box, kept.get("box")) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(obj)
    return keep

CANONICAL_CLASS_MAP = {
    "office chair": "chair",
    "dining chair": "chair",
    "folding chair": "chair",
    "stool": "chair",
    "bench": "chair",
    "dining table": "table",
    "desk": "table",
    "coffee table": "table",
    "side table": "table",
    "restaurant table": "table",
    "reception desk": "table",
    "counter": "table",
    "magazine": "notebook",
    "newspaper": "notebook",
    "paper": "document",
    "open door": "door",
    "closed door": "door",
    "glass door": "door",
    "sliding door": "door",
    "doorway entrance": "doorway",
    "elevator door": "elevator",
    "stairs": "staircase",
    "step": "staircase",
    "stairs outside": "staircase",
    "upstairs": "staircase",
    "downstairs": "staircase",
    "stair landing": "staircase",
    "water bottle": "bottle",
    "bottle of water": "bottle",
    "smartphone": "cell phone",
    "mobile phone": "cell phone",
    "computer mouse": "mouse",
    "desktop computer": "computer",
    "computer monitor": "monitor",
    "handbag": "bag",
    "trash bag": "bag",
    "school bag": "bag",
    "grocery bag": "bag",
    "plastic bag": "bag",
    "backpack": "bag",
    "suitcase": "bag",
    "trash can": "dustbin",
    "cardboard box": "box",
    "white cane": "walking cane",
    "person using a walking cane": "person with a cane",
    "street lamp": "streetlight",
    "emergency exit sign": "exit sign",
    "electrical socket": "light switch",
    "electrical switch": "light switch",
    "red traffic light": "traffic light",
    "green traffic light": "traffic light",
    "yellow traffic light": "traffic light",
    "zebra crossing": "pedestrian crossing",
    "moving vehicle": "vehicle",
    "parked vehicle": "vehicle",
    "person walking": "person",
    "person sitting": "person",
    "child": "person",
    "adult": "person",
    "elderly person": "person",
    "pedestrian": "person",
    "picture frame": "photo frame",
    "framed picture": "photo frame",
    "portrait": "photo frame",
}

def _normalize_class(cls_name):
    return CANONICAL_CLASS_MAP.get(str(cls_name).lower().strip(), cls_name)

class DescribeService:
    def __init__(self):
        logger.info("Initializing Enhanced Rule-Based Describe Service")
        
    def generate_description(self, frame, scene_json: dict) -> str:
        """
        Generates an accurate, natural audio description of the scene including:
        - Recognized people and unknown faces
        - Objects grouped with counts, spatial direction (left/center/right), and distance
        - Detected text in reading order
        """
        try:
            faces = scene_json.get("faces", [])
            objects = scene_json.get("objects", [])
            texts = scene_json.get("texts", [])
            distances = scene_json.get("distances", [])
            
            # Normalize class names to canonical terms to prevent log flickering
            normalized_objects = []
            for obj in objects:
                obj_copy = dict(obj)
                obj_copy["class_name"] = _normalize_class(obj.get("class_name", "object"))
                normalized_objects.append(obj_copy)
            objects = normalized_objects

            # Apply NMS to eliminate overlapping duplicate object boxes
            objects = _apply_nms(objects, iou_thresh=0.4)
            
            frame_width = 640
            if frame is not None and hasattr(frame, "shape"):
                frame_width = frame.shape[1]

            # Filter out faces that are printed inside objects (books, notebooks, papers, posters, screens, frames)
            printed_face_classes = {
                "book", "notebook", "paper", "poster", "tv", "television", "screen",
                "laptop", "photo frame", "picture frame", "framed picture", "portrait", "painting"
            }
            valid_faces = []
            for face in faces:
                f_box = face.get("box")
                is_printed = False
                if f_box:
                    for obj in objects:
                        if obj.get("class_name") in printed_face_classes:
                            if _is_inside(f_box, obj.get("box")):
                                is_printed = True
                                break
                if not is_printed:
                    valid_faces.append(face)
            faces = valid_faces

            # Filter out "person" objects if a face was already detected inside the box
            filtered_objects = []
            for obj in objects:
                if obj.get("class_name") == "person":
                    has_face = False
                    for face in faces:
                        if _is_inside(face.get("box"), obj.get("box")):
                            has_face = True
                            break
                    if has_face:
                        continue
                filtered_objects.append(obj)
            objects = filtered_objects

            parts = []

            # 1. Describe Recognized Faces & Real People
            if faces:
                known_faces = []
                for f in faces:
                    name = f.get("name", "Unknown")
                    if name.lower() != "unknown" and not name.startswith("Person "):
                        box = f.get("box")
                        dist = _get_distance_for_box(box, distances)
                        dir_str = _get_direction(box, frame_width)
                        
                        info = name
                        if dist:
                            info += f" {dir_str} at {dist}"
                        else:
                            info += f" {dir_str}"
                        known_faces.append(info)

                unknown_faces_count = len([f for f in faces if f.get("name", "").lower() == "unknown" or f.get("name", "").startswith("Person ")])

                face_parts = []
                if known_faces:
                    face_parts.append(", ".join(known_faces))
                if unknown_faces_count > 0:
                    face_parts.append(f"{unknown_faces_count} unrecognized person{'s' if unknown_faces_count > 1 else ''}")

                if face_parts:
                    parts.append(f"I see {' and '.join(face_parts)}.")

            # 2. Describe Objects with Counts, Spatial Direction & Distance
            if objects:
                # Group objects by (class_name, direction)
                grouped = {}
                for obj in objects:
                    cls_name = obj.get("class_name", "object")
                    box = obj.get("box")
                    direction = _get_direction(box, frame_width)
                    dist = _get_distance_for_box(box, distances, cls_name)

                    key = (cls_name, direction)
                    if key not in grouped:
                        grouped[key] = {"count": 0, "distances": []}
                    grouped[key]["count"] += 1
                    if dist:
                        grouped[key]["distances"].append(dist)

                # Format object phrases (limit to top 4 to keep audio description concise)
                obj_phrases = []
                for (cls_name, direction), data in list(grouped.items())[:4]:
                    count = data["count"]
                    dists = data["distances"]
                    
                    # Pluralization
                    name_str = f"{count} {cls_name}s" if count > 1 else f"a {cls_name}"
                    
                    if dists:
                        obj_phrases.append(f"{name_str} {direction} at {dists[0]}")
                    else:
                        obj_phrases.append(f"{name_str} {direction}")

                if obj_phrases:
                    parts.append(f"Nearby, there is {', '.join(obj_phrases)}.")

            # 3. Describe Detected Text (Reading Order)
            if texts:
                sorted_texts = sorted(texts, key=lambda t: (t.get("box", [0, 0, 0, 0])[1] // 15, t.get("box", [0, 0, 0, 0])[0]))
                valid_words = []
                for t in sorted_texts:
                    w = t.get("text", "").strip()
                    if w and len(w) > 1:  # ignore single noise characters
                        valid_words.append(w)

                if valid_words:
                    readable_str = " ".join(valid_words[:10])  # limit to first 10 words
            # 0. Check for physical elevation / stairs hazard in frame
            stair_detected = False
            if any(o.get("class_name") in ["stairs", "staircase", "steps"] for o in objects):
                stair_detected = True
            elif frame is not None:
                from services.hazard_service import HazardPrioritizer
                hp = HazardPrioritizer()
                s_res = hp.detect_stairs_and_elevation(frame, frame_width * 0.3, frame_width * 0.7)
                if s_res:
                    stair_detected = True

            if stair_detected:
                parts.insert(0, "Caution, flight of stairs directly ahead. Proceed with care.")
            elif frame is not None:
                from services.hazard_service import HazardPrioritizer
                hp = HazardPrioritizer()
                d_res = hp.detect_door_and_doorway(frame)
                if d_res:
                    door_side = d_res.get("door", {}).get("side", "right")
                    parts.insert(0, f"There is an open doorway directly ahead with an open door on your {door_side}.")

            if not parts:
                return "No distinct obstacles identified in your immediate view."

            return " ".join(parts)

        except Exception as e:
            logger.error(f"Enhanced description generation error: {e}")
            return "Unable to describe current scene."

    def answer_question(self, question: str, frame, scene_json: dict) -> str:
        """
        Analyzes the user's spoken question against real-time scene detections (YOLO, OCR, InsightFace, Depth)
        and returns a direct, natural language spoken answer.
        """
        if not question or not isinstance(question, str):
            return self.generate_description(frame, scene_json)

        q = question.lower().strip()
        scene_json = scene_json or {}
        faces = scene_json.get("faces", [])
        objects = scene_json.get("objects", [])
        texts = scene_json.get("texts", [])
        distances = scene_json.get("distances", [])

        frame_width = 640
        if frame is not None and hasattr(frame, "shape"):
            frame_width = frame.shape[1]

        # 1. Text Reading / OCR Queries
        if any(kw in q for kw in ["read", "text", "sign", "writing", "book", "word", "written", "label", "say"]):
            if texts:
                sorted_texts = sorted(texts, key=lambda t: (t.get("box", [0, 0, 0, 0])[1] // 15, t.get("box", [0, 0, 0, 0])[0]))
                words = [t.get("text", "").strip() for t in sorted_texts if t.get("text", "").strip()]
                if words:
                    readable_text = " ".join(words[:15])
                    return f"The text in view reads: '{readable_text}'."
            return "I do not see any readable text in the current camera view."

        # 2. Person / Face Recognition Queries
        if any(kw in q for kw in ["who", "person", "people", "face", "friend", "someone", "anybody"]):
            if faces:
                known = [f.get("name") for f in faces if f.get("name") and f.get("name").lower() != "unknown" and not f.get("name").startswith("Person ")]
                unknown_count = len([f for f in faces if not f.get("name") or f.get("name").lower() == "unknown" or f.get("name").startswith("Person ")])
                
                parts = []
                if known:
                    parts.append(f"I see {', '.join(known)}")
                if unknown_count > 0:
                    parts.append(f"{unknown_count} unrecognized person{'s' if unknown_count > 1 else ''}")
                
                return f"{' and '.join(parts)} in front of you."
            
            # Check YOLO person detections fallback
            person_objs = [o for o in objects if o.get("class_name") == "person"]
            if person_objs:
                return f"I see {len(person_objs)} person{'s' if len(person_objs) > 1 else ''} in front of you."
                
            return "No people are detected in the current view."

        # 3. Path / Obstacle / Safety Queries
        if any(kw in q for kw in ["safe", "path", "clear", "obstacle", "walk", "danger", "block", "front", "ahead", "stairs", "step"]):
            # Check for stairs in frame or objects
            if any(o.get("class_name") in ["stairs", "staircase", "steps"] for o in objects):
                return "Caution, there is a staircase directly ahead in your walking path. Please proceed carefully."
            elif frame is not None:
                from services.hazard_service import HazardPrioritizer
                hp = HazardPrioritizer()
                s_res = hp.detect_stairs_and_elevation(frame, frame_width * 0.3, frame_width * 0.7)
                if s_res:
                    return "Caution, a flight of stairs is detected directly ahead in your walking path."

            ahead_objs = []
            for obj in objects:
                direction = _get_direction(obj.get("box"), frame_width)
                if direction == "directly ahead":
                    cls = obj.get("class_name", "object")
                    dist = _get_distance_for_box(obj.get("box"), distances, cls)
                    dist_str = f" at {dist}" if dist else ""
                    ahead_objs.append(f"a {cls}{dist_str}")

            if ahead_objs:
                return f"Caution, directly ahead there is {', '.join(ahead_objs[:3])}."
            return "No obstacles are identified directly in front of you. Always use your cane to verify ground conditions."

        # 3b. Door / Doorway / Entrance Queries
        if any(kw in q for kw in ["door", "doorway", "entrance", "exit", "portal", "passage"]):
            # Check if door/doorway is in objects
            door_objs = [o for o in objects if any(d_kw in o.get("class_name", "").lower() for d_kw in ["door", "doorway", "entrance"])]
            if door_objs:
                obj = door_objs[0]
                cls = obj.get("class_name", "doorway")
                box = obj.get("box")
                direction = _get_direction(box, frame_width)
                dist = _get_distance_for_box(box, distances, cls)
                dist_str = f" at {dist}" if dist else ""
                return f"Yes, there is a {cls} {direction}{dist_str}."

            # Physical frame doorway check
            if frame is not None:
                from services.hazard_service import HazardPrioritizer
                hp = HazardPrioritizer()
                door_res = hp.detect_door_and_doorway(frame)
                if door_res:
                    door_side = door_res.get("door", {}).get("side", "right")
                    return f"There is an open doorway directly ahead with an open door on your {door_side}. The walking passage through the center appears open."

            return "I do not see any door or doorway directly visible in the current camera view."

        # 4. Count Queries ("how many ...")
        if "how many" in q:
            all_items = {}
            for obj in objects:
                cls = obj.get("class_name", "object")
                all_items[cls] = all_items.get(cls, 0) + 1
            for face in faces:
                name = face.get("name", "person")
                all_items["person"] = all_items.get("person", 0) + 1

            for cls, count in all_items.items():
                if cls in q:
                    return f"There {'is' if count == 1 else 'are'} {count} {cls}{'' if count == 1 else 's'} in view."
            
            if objects or faces:
                summary = ", ".join([f"{c} {name}{'s' if c > 1 else ''}" for name, c in all_items.items()])
                return f"In total, I see {summary}."
            return "No items or people are currently detected."

        # 5. Specific Object Location Query ("where is the bottle", "is there a chair")
        for obj in objects:
            cls = obj.get("class_name", "").lower()
            if cls and cls in q:
                box = obj.get("box")
                direction = _get_direction(box, frame_width)
                dist = _get_distance_for_box(box, distances, cls)
                dist_str = f" at {dist}" if dist else ""
                return f"Yes, I see a {cls} {direction}{dist_str}."

        # 6. Fallback to general description generator
        return self.generate_description(frame, scene_json)
