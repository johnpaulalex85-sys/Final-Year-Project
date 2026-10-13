import os
import json
import logging
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceService:
    def __init__(self, db_path="outputs/faces.json"):
        self.db_path = db_path
        self.database = {}
        
        logger.info("Loading InsightFace model...")
        try:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            self.app = None
            
        self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    self.database = {name: np.array(emb, dtype=np.float32) for name, emb in data.items()}
                logger.info(f"Loaded {len(self.database)} faces from database.")
            except Exception as e:
                logger.error(f"Failed to load face database: {e}")
                self.database = {}
        else:
            self.database = {}

    def save_database(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, "w") as f:
                data = {name: emb.tolist() for name, emb in self.database.items()}
                json.dump(data, f)
            logger.info(f"Saved {len(self.database)} faces to database.")
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")

    def register_face(self, name: str, frame: np.ndarray):
        if self.app is None:
            return False, "Face model not initialized"

        faces = self.app.get(frame)
        if not faces:
            return False, "No face detected in image"

        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        self.database[name] = largest_face.embedding
        self.save_database()
        return True, f"Face registered for {name}"

    def register_face_multi(self, name: str, frames: list):
        if self.app is None:
            return False, "Face model not initialized"

        embeddings = []
        for frame in frames:
            faces = self.app.get(frame)
            if faces:
                # Get largest face in this frame
                largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                embeddings.append(largest_face.embedding)

        if not embeddings:
            return False, "No face detected in any of the provided images"

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        # Re-normalize to length 1
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        self.database[name] = avg_embedding
        self.save_database()
        return True, f"Face registered for {name} with {len(embeddings)} angles"

    def recognize_face(self, frame: np.ndarray, threshold: float = 0.6):
        if self.app is None:
            return {"faces": []}

        try:
            faces = self.app.get(frame)
        except Exception as e:
            logger.error(f"InsightFace inference error: {e}")
            return {"faces": []}

        results = []
        for face in faces:
            best_match = "unknown"
            best_sim = -1.0
            
            emb1 = face.embedding
            norm1 = np.linalg.norm(emb1)
            
            if norm1 > 0:
                for name, emb2 in self.database.items():
                    norm2 = np.linalg.norm(emb2)
                    if norm2 > 0:
                        sim = np.dot(emb1, emb2) / (norm1 * norm2)
                        if sim > best_sim:
                            best_sim = sim
                            best_match = name

            confidence = round(float(best_sim), 3) if best_sim != -1.0 else 0.0
            if best_sim < threshold:
                best_match = f"Person {len(self.database) + 1}"
                self.database[best_match] = face.embedding
                self.save_database()
                logger.info(f"Auto-registered new face as {best_match}")

            bbox = [int(x) for x in face.bbox]
            results.append({
                "name": best_match,
                "confidence": confidence,
                "box": bbox
            })
            
        return {"faces": results}
