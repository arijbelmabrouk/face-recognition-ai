import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pickle
import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# ── Configuration ─────────────────────────────────────────────────────
MODEL       = "Facenet512"
DETECTOR    = "opencv"
THRESHOLD   = 0.70
DB_PATH     = "face_db/embeddings.pkl"
UNKNOWN_DIR = "unknown_captures/"

# ── Database helpers ──────────────────────────────────────────────────
def load_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def save_db(db: dict):
    os.makedirs("face_db", exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

# ── Core recognition function ─────────────────────────────────────────
def recognize(image_bytes: bytes) -> dict:

    # Decode bytes → image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Stage 1 + 2: detect face + generate embedding
    try:
        result = DeepFace.represent(
            img_path          = frame,
            model_name        = MODEL,
            detector_backend  = DETECTOR,
            enforce_detection = True
        )
    except Exception:
        return {
            "status":     "NO_FACE",
            "name":       None,
            "confidence": 0.0
        }

    query_vector = np.array(result[0]["embedding"]).reshape(1, -1)

    # Stage 3: compare against enrolled employees
    db         = load_db()
    best_name  = None
    best_score = 0.0

    for name, stored_vector in db.items():
        score = cosine_similarity(
            query_vector,
            stored_vector.reshape(1, -1)
        )[0][0]
        if score > best_score:
            best_score = score
            best_name  = name

    confidence = round(best_score * 100, 2)

    # Decision
    if best_score >= THRESHOLD:
        return {
            "status":     "AUTHORIZED",
            "name":       best_name,
            "confidence": confidence
        }
    else:
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{UNKNOWN_DIR}/unknown_{ts}.jpg"
        cv2.imwrite(path, frame)
        return {
            "status":     "UNKNOWN",
            "name":       None,
            "confidence": confidence,
            "photo_path": path
        }

# ── Enroll a new employee ─────────────────────────────────────────────
def enroll(name: str, image_bytes: bytes) -> dict:

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.represent(
            img_path          = frame,
            model_name        = MODEL,
            detector_backend  = DETECTOR,
            enforce_detection = True
        )
    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }

    vector = np.array(result[0]["embedding"])
    db     = load_db()

    if name in db:
        # Average with existing vector for better stability
        db[name] = np.mean([db[name], vector], axis=0)
    else:
        db[name] = vector

    save_db(db)

    return {
        "success": True,
        "name":    name,
        "message": f"{name} enrolled successfully"
    }