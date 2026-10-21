import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from config import settings

logger = logging.getLogger(__name__)

# Hazard Category Taxonomy with base severity weights
HAZARD_SEVERITY_MAP = {
    # Elevation / Falling / Step Hazards (Critical severity)
    "stairs": 0.95,
    "staircase": 0.95,
    "step": 0.90,
    "curb": 0.85,
    "pothole": 0.85,
    "open drain": 0.90,
    "manhole": 0.85,
    "wet floor": 0.80,
    "slippery floor": 0.80,
    
    # Moving Vehicles & Transport (Critical severity)
    "car": 0.90,
    "bus": 0.95,
    "truck": 0.95,
    "motorcycle": 0.90,
    "bicycle": 0.85,
    "moving vehicle": 0.95,
    "auto rickshaw": 0.90,
    "van": 0.90,
    
    # Direct Walking Path Obstacles (High severity when central)
    "pole": 0.80,
    "electricity pole": 0.80,
    "tree": 0.75,
    "obstacle": 0.80,
    "road barrier": 0.80,
    "construction barrier": 0.85,
    "fence": 0.75,
    "wall": 0.70,
    "door": 0.65,
    "chair": 0.65,
    "table": 0.65,
    "dustbin": 0.60,
    "trash bin": 0.60,
    
    # People (Moderate severity, elevated if approaching or central)
    "person": 0.70,
    "pedestrian": 0.70,
    "dog": 0.70
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

    def evaluate_hazards(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame_width: int = 640,
        frame_height: int = 480
    ) -> Dict[str, Any]:
        """
        Evaluates a set of tracked objects in the current frame and returns
        prioritized hazards and any proactive spoken audio alert.
        """
        now = time.time()
        corridor_x_min = frame_width * self.corridor_left_ratio
        corridor_x_max = frame_width * self.corridor_right_ratio
        
        evaluated_hazards = []

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
        if cls in ["stairs", "staircase", "step"]:
            return f"Caution, steps {direction}."
        if cls in ["curb", "pothole"]:
            return f"Caution, {cls} {direction}."
        if cls in ["open drain", "manhole"]:
            return f"Warning, open drain {direction}."
        if cls in ["wet floor", "slippery floor"]:
            return f"Caution, wet floor {direction}."

        # Approaching vehicle / bicycle phrases
        if cls in ["car", "bus", "truck", "motorcycle", "moving vehicle"] and is_approaching:
            return f"Vehicle approaching {direction}."
        if cls == "bicycle" and is_approaching:
            return f"Bicycle approaching {direction}."

        # Approaching person
        if cls == "person" and is_approaching and direction == "directly ahead":
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
