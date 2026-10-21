import numpy as np
from typing import List, Dict, Any, Optional

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union between two boxes [x1, y1, x2, y2]."""
    if not boxA or not boxB or len(boxA) != 4 or len(boxB) != 4:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0.0:
        return 0.0
    return interArea / denom

class TrackedObject:
    def __init__(self, track_id: int, detection: Dict[str, Any], smoothing_alpha: float = 0.65):
        self.track_id = track_id
        self.class_name = detection.get("class_name", "object")
        self.confidence = float(detection.get("confidence", 0.0))
        self.box = list(detection.get("box", [0, 0, 0, 0]))
        self.smoothed_box = list(self.box)
        self.smoothing_alpha = smoothing_alpha
        
        self.hits = 1
        self.lost_count = 0
        self.age = 1
        
        # History of box areas and bottom-Y coordinate (ground contact) for movement analysis
        # Stores tuples: (area, bottom_y)
        area = max(1.0, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))
        self.history = [(area, self.box[3])]
        self.is_approaching = False
        self.expansion_ratio = 1.0

    def update(self, detection: Dict[str, Any]):
        raw_box = detection.get("box", [0, 0, 0, 0])
        self.confidence = float(detection.get("confidence", 0.0))
        self.class_name = detection.get("class_name", self.class_name)
        self.hits += 1
        self.lost_count = 0
        self.age += 1

        # Smooth bounding box with exponential moving average to prevent camera jitter
        # while staying responsive to motion
        alpha = self.smoothing_alpha
        self.smoothed_box = [
            round(alpha * raw_box[i] + (1.0 - alpha) * self.smoothed_box[i], 2)
            for i in range(4)
        ]
        self.box = list(raw_box)

        # Track expansion and approaching dynamic
        area = max(1.0, (self.smoothed_box[2] - self.smoothed_box[0]) * (self.smoothed_box[3] - self.smoothed_box[1]))
        bottom_y = self.smoothed_box[3]
        self.history.append((area, bottom_y))
        if len(self.history) > 12:
            self.history.pop(0)

        # Evaluate if object is approaching:
        # Successive observations show growing bounding box area and descending ground contact
        if len(self.history) >= 4:
            past_area, past_by = self.history[0]
            curr_area, curr_by = self.history[-1]
            self.expansion_ratio = curr_area / past_area if past_area > 0 else 1.0
            
            # Approaching condition: area expanded by at least 25% and bottom moved down
            if self.expansion_ratio >= 1.25 and curr_by >= past_by + 5:
                self.is_approaching = True
            elif self.expansion_ratio <= 0.90:
                self.is_approaching = False
        else:
            self.is_approaching = False

    def mark_missed(self):
        self.lost_count += 1
        self.age += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "box": [round(x, 1) for x in self.smoothed_box],
            "raw_box": [round(x, 1) for x in self.box],
            "hits": self.hits,
            "lost_count": self.lost_count,
            "is_approaching": self.is_approaching,
            "expansion_ratio": round(self.expansion_ratio, 2)
        }

class WalkingObjectTracker:
    """
    Lightweight, high-speed multi-object tracker designed for mobile walking camera feeds.
    Provides temporal consistency, track ID persistence across brief motion blur dropouts,
    velocity/expansion smoothing, and duplicate detection removal.
    """
    def __init__(self, max_lost: int = 4, iou_threshold: float = 0.30, smoothing_alpha: float = 0.65):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha
        self.tracks: List[TrackedObject] = []
        self.next_track_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections and not self.tracks:
            return []

        # Filter out invalid or non-spatial detections
        valid_dets = []
        for d in detections:
            box = d.get("box")
            if box and len(box) == 4 and (box[2] > box[0]) and (box[3] > box[1]):
                valid_dets.append(d)

        num_tracks = len(self.tracks)
        num_dets = len(valid_dets)

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            # Compute IoU matching matrix
            iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(valid_dets):
                    # Prefer matching identical or compatible classes
                    cls_match = 1.0 if track.class_name == det.get("class_name") else 0.5
                    iou_matrix[t_idx, d_idx] = calculate_iou(track.smoothed_box, det["box"]) * cls_match

            # Greedy bipartite matching
            while True:
                max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                self.tracks[t_idx].update(valid_dets[d_idx])
                # Invalidate row and column
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Mark unmatched existing tracks as missed
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Add new tracks for unmatched detections
        for d_idx, det in enumerate(valid_dets):
            if d_idx not in matched_dets:
                new_track = TrackedObject(self.next_track_id, det, self.smoothing_alpha)
                self.next_track_id += 1
                self.tracks.append(new_track)

        # Remove dead tracks that exceeded max_lost frames
        self.tracks = [t for t in self.tracks if t.lost_count <= self.max_lost]

        # Only return active tracks (either detected in current frame or confirmed tracks with lost_count <= 2)
        active_outputs = []
        for track in self.tracks:
            # Skip unconfirmed tracks that were only seen once and then immediately lost
            if track.hits < 2 and track.lost_count > 0:
                continue
            active_outputs.append(track.to_dict())

        return active_outputs

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
