import time
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config import settings

logger = logging.getLogger(__name__)

# Hazard Category Taxonomy with base severity weights
HAZARD_SEVERITY_MAP = {
    # Elevation / Falling / Step Hazards (Critical severity)
    "stairs": 0.95,
    "staircase": 0.95,
    "steps": 0.90,
    "step": 0.90,
    "downstairs": 0.95,
    "upstairs": 0.90,
    "curb": 0.85,
    "kerb": 0.85,
    "raised pavement": 0.80,
    "roadside edge": 0.85,
    "ramp": 0.70,
    "steep slope": 0.85,
    
    # Walking-Surface Hazards
    "pothole": 0.85,
    "hole in the ground": 0.85,
    "uneven pavement": 0.80,
    "broken pavement": 0.80,
    "cracked sidewalk": 0.75,
    "loose paving stone": 0.80,
    "construction debris": 0.80,
    "rubble": 0.80,
    "open drain": 0.90,
    "drain hole": 0.85,
    "manhole": 0.85,
    "open manhole": 0.95,
    "water puddle": 0.75,
    "flooded pavement": 0.85,
    "wet floor": 0.80,
    "slippery floor": 0.80,
    
    # Moving Vehicles & Transport (Critical severity)
    "car": 0.90,
    "taxi": 0.90,
    "van": 0.90,
    "bus": 0.95,
    "truck": 0.95,
    "motorcycle": 0.90,
    "scooter": 0.85,
    "electric scooter": 0.85,
    "bicycle": 0.85,
    "moving vehicle": 0.95,
    "auto rickshaw": 0.90,
    "cyclist": 0.85,
    "person on a bicycle": 0.85,
    "person on a motorcycle": 0.90,
    
    # Fixed Obstacles
    "pole": 0.80,
    "lamp post": 0.80,
    "electric pole": 0.80,
    "electricity pole": 0.80,
    "traffic sign": 0.75,
    "signpost": 0.75,
    "bollard": 0.80,
    "bollard post": 0.80,
    "road barrier": 0.80,
    "traffic barrier": 0.80,
    "construction barrier": 0.85,
    "fence": 0.75,
    "gate": 0.75,
    "wall": 0.70,
    "pillar": 0.75,
    "tree": 0.75,
    "tree trunk": 0.75,
    "low-hanging branch": 0.85,
    "obstacle": 0.80,

    # Head & Upper Body Height Hazards
    "low ceiling": 0.90,
    "overhanging obstacle": 0.90,
    "hanging sign": 0.85,
    "hanging wire": 0.90,
    "protruding object": 0.85,
    "open cabinet door": 0.80,
    "open door": 0.75,
    "door": 0.65,
    "doorway": 0.60,
    "door frame": 0.60,
    "closed door": 0.75,
    "glass door": 0.85,
    "sliding door": 0.70,
    "entrance": 0.60,
    "exit": 0.60,

    # Objects Blocking Walking Path
    "chair": 0.65,
    "table": 0.65,
    "bench": 0.65,
    "stool": 0.60,
    "box": 0.65,
    "cardboard box": 0.65,
    "crate": 0.70,
    "trash bin": 0.60,
    "dustbin": 0.60,
    "garbage bag": 0.60,
    "suitcase": 0.65,
    "backpack": 0.60,
    "shopping cart": 0.70,
    "trolley": 0.70,
    "wheelchair": 0.70,
    "stroller": 0.70,
    "parked bicycle": 0.70,
    "parked motorcycle": 0.75,

    # Road & Pedestrian Infrastructure
    "traffic cone": 0.70,
    "roadwork": 0.85,
    "excavation": 0.95,
    "road divider": 0.80,
    "median barrier": 0.80,
    "railway crossing barrier": 0.95,

    # Dropped or Misplaced Ground Obstacles
    "brick": 0.75,
    "stone": 0.75,
    "rock": 0.75,
    "wooden plank": 0.75,
    "metal pipe": 0.80,
    "cable": 0.80,
    "hose": 0.75,
    "fallen branch": 0.80,
    "fallen tree": 0.90,

    # People & Animals
    "person": 0.70,
    "pedestrian": 0.70,
    "child": 0.75,
    "dog": 0.70,
    "cat": 0.60,
    "cow": 0.85,
    "goat": 0.70,
    "horse": 0.85,
    "stray animal": 0.75
}

class HazardPrioritizer:
    """
    Lightweight, deterministic walking hazard prioritization layer.
    Classifies hazards by walking corridor location, scale proximity, expansion velocity,
    and manages rate-limited urgent audio alerts.
    """
    def __init__(
        self,
        corridor_left_ratio: float = 0.32,
        corridor_right_ratio: float = 0.68,
        hazard_cooldown: float = 3.5,
        urgent_cooldown: float = 2.0
    ):
        self.corridor_left_ratio = corridor_left_ratio
        self.corridor_right_ratio = corridor_right_ratio
        self.hazard_cooldown = hazard_cooldown
        self.urgent_cooldown = urgent_cooldown
        
        # Track history of announced hazards to prevent repetitive audio noise
        # Key: (class_name, direction), Value: timestamp
        self.announcement_history: Dict[Tuple[str, str], float] = {}
        self.last_urgent_announcement_time: float = 0.0

    def detect_stairs_and_elevation(
        self,
        frame: np.ndarray,
        x_min: float,
        x_max: float
    ) -> Optional[Dict[str, Any]]:
        """
        Fast, deterministic geometric surface analyzer for detecting flights of stairs,
        steps, and elevation drop-offs in the walking corridor without relying solely on bounding-box models.
        """
        if frame is None or frame.size == 0:
            return None
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Analyze both standard corridor and center lane (which avoids side handrail vertical edges)
            corridors = [
                (int(x_min), int(x_max)),
                (int(w * 0.28), int(w * 0.72))
            ]
            
            for c_x1, c_x2 in corridors:
                if c_x2 <= c_x1:
                    continue
                y_start = int(h * 0.30)
                corridor = gray[y_start:, c_x1:c_x2]
                if corridor.size == 0:
                    continue

                sobel_y = np.abs(cv2.Sobel(corridor, cv2.CV_32F, 0, 1, ksize=3))
                sobel_x = np.abs(cv2.Sobel(corridor, cv2.CV_32F, 1, 0, ksize=3))

                h_energy = float(np.mean(sobel_y))
                v_energy = float(np.mean(sobel_x)) + 1e-5
                ratio = h_energy / v_energy

                row_means = sobel_y.mean(axis=1)
                thresh = np.percentile(row_means, 82)
                strong_rows = np.where(row_means > thresh)[0]

                diffs = np.diff(strong_rows)
                step_diffs = diffs[diffs > 8]
                spacing_std = float(np.std(step_diffs)) if len(step_diffs) >= 4 else 999.0

                # Staircase criteria: Strong horizontal edge energy, horizontal dominance, and regular step spacing
                if h_energy >= 32.0 and ratio >= 1.15 and spacing_std <= 20.0 and len(step_diffs) >= 4:
                    return {
                        "class_name": "stairs",
                        "confidence": round(min(0.95, 0.70 + (h_energy / 150.0)), 2),
                        "box": [c_x1, int(h * 0.20), c_x2, int(h * 0.95)],
                        "is_urgent": True
                    }
        except Exception as e:
            logger.debug(f"Stair detection error: {e}")
        return None

    def detect_door_and_doorway(
        self,
        frame: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Fast architectural analyzer for detecting indoor doorways, open/closed door frames,
        and open door panels in the walking path without relying on bounding box models.
        """
        if frame is None or frame.size == 0:
            return None
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            y_top = int(h * 0.18)
            y_mid = int(h * 0.60)
            roi = gray[y_top:y_mid, :]

            sobel_x = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3))
            col_energy = sobel_x.mean(axis=0)
            kernel = np.ones(11) / 11
            smoothed = np.convolve(col_energy, kernel, mode='same')

            # Left door jamb (15% to 45% width)
            left_zone = smoothed[int(w * 0.15):int(w * 0.45)]
            left_idx = int(w * 0.15) + int(np.argmax(left_zone))
            left_peak = float(smoothed[left_idx])

            # Right door jamb (55% to 85% width)
            right_zone = smoothed[int(w * 0.55):int(w * 0.85)]
            right_idx = int(w * 0.55) + int(np.argmax(right_zone))
            right_peak = float(smoothed[right_idx])

            width_ratio = (right_idx - left_idx) / w

            # Must have two strong vertical jambs and valid doorway proportions (35% to 70% frame width)
            if left_peak >= 35.0 and right_peak >= 35.0 and 0.35 <= width_ratio <= 0.70:
                has_right_door = (w - right_idx) >= int(w * 0.12)
                has_left_door = left_idx >= int(w * 0.12)
                door_side = "right" if has_right_door else ("left" if has_left_door else "none")

                return {
                    "doorway": {
                        "class_name": "doorway",
                        "confidence": 0.90,
                        "box": [left_idx, int(h * 0.18), right_idx, int(h * 0.88)],
                        "direction": "directly ahead",
                        "is_open": True
                    },
                    "door": {
                        "class_name": "open door",
                        "confidence": 0.85,
                        "box": [right_idx, int(h * 0.18), min(w, right_idx + int(w * 0.25)), int(h * 0.88)] if has_right_door else [max(0, left_idx - int(w * 0.25)), int(h * 0.18), left_idx, int(h * 0.88)],
                        "direction": f"on your {door_side}" if door_side != "none" else "directly ahead",
                        "side": door_side
                    }
                }
        except Exception as e:
            logger.debug(f"Doorway detection error: {e}")
        return None

    def evaluate_hazards(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame_width: int = 640,
        frame_height: int = 480,
        frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates tracked objects and physical surface geometry in the walking corridor.
        Returns prioritized hazards and any proactive spoken audio alert.
        """
        now = time.time()
        corridor_x_min = frame_width * self.corridor_left_ratio
        corridor_x_max = frame_width * self.corridor_right_ratio
        
        evaluated_hazards = []

        # 0. Check physical surface geometry for stairs / steps in walking corridor
        if frame is not None:
            stair_res = self.detect_stairs_and_elevation(frame, corridor_x_min, corridor_x_max)
            if stair_res:
                evaluated_hazards.append({
                    "track_id": 9901,
                    "class_name": "stairs",
                    "confidence": stair_res["confidence"],
                    "box": stair_res["box"],
                    "direction": "directly ahead",
                    "direction_key": "center",
                    "in_corridor": True,
                    "proximity": "immediate",
                    "is_approaching": False,
                    "is_urgent": True,
                    "priority_score": 0.98
                })
            else:
                # 0b. Check physical doorway and door structure
                door_res = self.detect_door_and_doorway(frame)
                if door_res:
                    if "doorway" in door_res:
                        dw = door_res["doorway"]
                        evaluated_hazards.append({
                            "track_id": 9902,
                            "class_name": "doorway",
                            "confidence": dw["confidence"],
                            "box": dw["box"],
                            "direction": dw["direction"],
                            "direction_key": "center",
                            "in_corridor": True,
                            "proximity": "approaching",
                            "is_approaching": False,
                            "is_urgent": False,
                            "priority_score": 0.60
                        })
                    if "door" in door_res:
                        dr = door_res["door"]
                        evaluated_hazards.append({
                            "track_id": 9903,
                            "class_name": dr["class_name"],
                            "confidence": dr["confidence"],
                            "box": dr["box"],
                            "direction": dr["direction"],
                            "direction_key": dr.get("side", "right"),
                            "in_corridor": False,
                            "proximity": "approaching",
                            "is_approaching": False,
                            "is_urgent": False,
                            "priority_score": 0.70
                        })

        for obj in tracked_objects:
            cls_name = obj.get("class_name", "").lower().strip()
            box = obj.get("box", [0, 0, 0, 0])
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            
            # 1. Determine Spatial Direction & Corridor Intersection
            in_corridor = (x1 < corridor_x_max and x2 > corridor_x_min)
            if cx < corridor_x_min:
                direction = "on your left"
                direction_key = "left"
            elif cx > corridor_x_max:
                direction = "on your right"
                direction_key = "right"
            else:
                direction = "directly ahead"
                direction_key = "center"

            # 2. Check if category is a potential walking hazard
            base_severity = HAZARD_SEVERITY_MAP.get(cls_name)
            if base_severity is None:
                # Check for substring matches (e.g. "stairs" in "stair landing")
                for key, sev in HAZARD_SEVERITY_MAP.items():
                    if key in cls_name:
                        base_severity = sev
                        break
                        
            if base_severity is None:
                continue  # Not a walking hazard

            # 3. Compute Proximity / Urgency Score
            # Ground contact proximity: y2 near bottom of frame
            y_norm = y2 / float(frame_height)
            h_norm = h / float(frame_height)
            area_norm = (w * h) / float(frame_width * frame_height)
            
            # Position multiplier: central walking corridor is weighted significantly higher
            corridor_weight = 1.35 if in_corridor else 0.85
            
            # Proximity rank
            if y_norm > 0.82 or h_norm > 0.55:
                proximity_desc = "immediate"
                proximity_score = 1.0
            elif y_norm > 0.65 or h_norm > 0.30:
                proximity_desc = "close"
                proximity_score = 0.75
            else:
                proximity_desc = "ahead"
                proximity_score = 0.45

            # Approaching boost
            is_approaching = obj.get("is_approaching", False)
            approach_weight = 1.30 if is_approaching else 1.0

            # Composite Hazard Priority Score [0.0 - 1.0+]
            priority_score = base_severity * proximity_score * corridor_weight * approach_weight
            
            # Urgent criteria: High priority + in corridor or approaching vehicle
            is_urgent = (
                (priority_score >= 0.75 and in_corridor) or
                (is_approaching and in_corridor) or
                (cls_name in ["stairs", "staircase", "step", "curb", "pothole"] and in_corridor and proximity_score >= 0.70) or
                (cls_name in ["car", "bus", "truck", "motorcycle", "bicycle"] and is_approaching)
            )

            evaluated_hazards.append({
                "track_id": obj.get("track_id"),
                "class_name": cls_name,
                "confidence": obj.get("confidence"),
                "box": box,
                "direction": direction,
                "direction_key": direction_key,
                "in_corridor": in_corridor,
                "proximity": proximity_desc,
                "is_approaching": is_approaching,
                "is_urgent": is_urgent,
                "priority_score": round(priority_score, 3)
            })

        # Sort hazards by priority score descending
        evaluated_hazards.sort(key=lambda h: h["priority_score"], reverse=True)

        # 4. Determine Spoken Audio Alert
        spoken_alert = ""
        is_alert_urgent = False

        if evaluated_hazards:
            top_hazard = evaluated_hazards[0]
            cls = top_hazard["class_name"]
            direction = top_hazard["direction"]
            direction_key = top_hazard["direction_key"]
            is_urgent = top_hazard["is_urgent"]
            is_approaching = top_hazard["is_approaching"]
            proximity = top_hazard["proximity"]
            
            # Construct concise, natural alert phrase
            alert_phrase = self._construct_alert_phrase(cls, direction, proximity, is_approaching, is_urgent)
            
            # Check cooldowns
            cache_key = (cls, direction_key)
            last_spoken = self.announcement_history.get(cache_key, 0.0)
            cooldown = self.urgent_cooldown if is_urgent else self.hazard_cooldown

            # Urgent hazard can interrupt non-urgent speech
            allow_speech = False
            if is_urgent and (now - self.last_urgent_announcement_time >= self.urgent_cooldown):
                allow_speech = True
            elif not is_urgent and (now - last_spoken >= cooldown):
                allow_speech = True

            if allow_speech:
                spoken_alert = alert_phrase
                is_alert_urgent = is_urgent
                self.announcement_history[cache_key] = now
                if is_urgent:
                    self.last_urgent_announcement_time = now

        return {
            "hazards": evaluated_hazards,
            "spoken_alert": spoken_alert,
            "is_urgent": is_alert_urgent,
            "corridor": {
                "x_min": int(corridor_x_min),
                "x_max": int(corridor_x_max)
            }
        }

    def _construct_alert_phrase(
        self,
        cls: str,
        direction: str,
        proximity: str,
        is_approaching: bool,
        is_urgent: bool
    ) -> str:
        """Constructs concise walking assistance audio alerts."""
        # Special Elevation / Drop-off phrases
        if cls in ["stairs", "staircase", "step", "steps", "downstairs", "upstairs"]:
            return f"Caution, steps {direction}."
        if cls in ["curb", "kerb", "raised pavement", "roadside edge", "steep slope"]:
            return f"Caution, curb or slope {direction}."
        if cls in ["pothole", "hole in the ground", "uneven pavement", "broken pavement", "cracked sidewalk", "loose paving stone"]:
            return f"Caution, uneven surface {direction}."
        if cls in ["open drain", "drain hole", "manhole", "open manhole", "excavation"]:
            return f"Warning, open ground hazard {direction}."
        if cls in ["wet floor", "slippery floor", "water puddle", "flooded pavement"]:
            return f"Caution, wet surface {direction}."
        if cls in ["low ceiling", "overhanging obstacle", "hanging sign", "hanging wire", "low-hanging branch"]:
            return f"Caution, overhead hazard {direction}."

        # Approaching vehicle / bicycle phrases
        if cls in ["car", "taxi", "van", "bus", "truck", "motorcycle", "scooter", "electric scooter", "moving vehicle", "auto rickshaw", "person on a motorcycle"] and is_approaching:
            return f"Vehicle approaching {direction}."
        if cls in ["bicycle", "cyclist", "person on a bicycle"] and is_approaching:
            return f"Bicycle approaching {direction}."

        # Approaching person
        if cls in ["person", "pedestrian", "child"] and is_approaching and direction == "directly ahead":
            return "Person approaching ahead."

        # General obstacle in path
        if direction == "directly ahead":
            if proximity == "immediate":
                return f"Stop, {cls} directly in your path."
            else:
                return f"{cls.capitalize()} directly ahead."

        return f"{cls.capitalize()} {direction}."

    def clear_history(self):
        self.announcement_history.clear()
        self.last_urgent_announcement_time = 0.0
