# Face Recognition API — AI Module
### Système Intelligent de Gestion des Accès Physiques
**ISI — GLSI 1ère Ingénieur | 2025–2026**  
**Team:** Arij Belmabrouk · Wassim Lourimi · Yasmine Hsayri

---

## Overview

This module is the AI brain of the access control system. It exposes a REST API that handles face enrollment and real-time face recognition. It is designed as an independent microservice — the Flask backend calls it internally, and the ESP32 never communicates with it directly.

**Stack:**
- Face Detection → OpenCV (Haar Cascade)
- Face Recognition → FaceNet512 via DeepFace
- Similarity Metric → Cosine Similarity
- API Framework → Flask
- Face Database → embeddings.pkl (512-dimensional vectors)

---

## Project Structure

```
face_ai/
├── dataset/                  ← enrollment photos per person
│   └── arij_belmabrouk/
│       ├── photo_1.jpg
│       └── ...
├── face_db/
│   └── embeddings.pkl        ← face vector database
├── unknown_captures/         ← photos of unrecognized people
├── config.py                 ← all settings in one place
├── recognize.py              ← core AI pipeline
├── enroll.py                 ← bulk enrollment from dataset
├── app.py                    ← Flask API server (port 5001)
├── collect_photos.py         ← webcam photo capture tool
├── test_webcam.py            ← live camera test (no API)
├── test_api.py               ← API test script
└── requirements.txt
```

---

## Configuration

All settings are in `config.py`:

```python
MODEL       = "Facenet512"           # recognition model
DETECTOR    = "opencv"               # face detector
THRESHOLD   = 0.70                   # minimum similarity to authorize
DB_PATH     = "face_db/embeddings.pkl"
UNKNOWN_DIR = "unknown_captures/"
```

---

## API Endpoints

### `GET /health`
Check if the API is running.

**Response:**
```json
{
  "status": "running",
  "model": "FaceNet512",
  "detector": "opencv"
}
```

---

### `POST /enroll`
Enroll a new employee. Generates a 512-dimensional face embedding from the provided photo and saves it to the database.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| name | string | Employee name (e.g. arij_belmabrouk) |
| image | file | JPEG photo of the employee's face |

**Response:**
```json
{
  "success": true,
  "name": "arij_belmabrouk",
  "message": "arij_belmabrouk enrolled successfully"
}
```

---

### `POST /recognize`
Recognize a face from a door camera image. Runs the full AI pipeline and returns a decision with confidence score.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| image | file | JPEG photo from the door camera |

**Response (authorized):**
```json
{
  "status": "AUTHORIZED",
  "name": "arij_belmabrouk",
  "confidence": 94.1,
  "timestamp": "2026-04-23T01:11:47"
}
```

**Response (unknown):**
```json
{
  "status": "UNKNOWN",
  "name": null,
  "confidence": 23.4,
  "photo_path": "unknown_captures/unknown_20260423_011147.jpg",
  "timestamp": "2026-04-23T01:11:47"
}
```

**Response (no face detected):**
```json
{
  "status": "NO_FACE",
  "name": null,
  "confidence": 0.0
}
```

---

## Full Pipeline — Who Does What

### Part 1 — Adding an Employee

```
Admin takes photo on mobile app
         ↓
Mobile App
  → sends photo + name to Flask Backend
         ↓
Flask Backend
  → saves employee record (name, email) to PostgreSQL
  → forwards photo + name to Face API /enroll
         ↓
Face API /enroll
  → opencv detects the face in the photo
  → FaceNet512 converts face to 512 numbers
  → saves those 512 numbers to embeddings.pkl
  → returns "success" to Flask Backend
         ↓
Flask Backend
  → tells mobile app "employee added successfully"
```

### Part 2 — Someone at the Door

```
Person stands at door
         ↓
PIR sensor detects presence
  → tells ESP32 "someone is here"
         ↓
ESP32-CAM
  → captures a JPEG photo
  → sends it to Flask Backend /api/door/access
         ↓
Flask Backend
  → receives the photo
  → forwards it to Face API /recognize
         ↓
Face API /recognize  ← THIS IS WHERE THE AI HAPPENS
  → opencv detects face in the photo
  → FaceNet512 converts detected face to 512 numbers
  → compares those 512 numbers against EVERY employee in embeddings.pkl
  → finds the closest match using cosine similarity
  → if similarity > 70% → AUTHORIZED
  → if similarity < 70% → UNKNOWN
  → returns {status, name, confidence} to Flask Backend
         ↓
Flask Backend
  → saves access log to PostgreSQL (who, when, authorized or not)
  → if AUTHORIZED → tells ESP32 "open door"
  → if UNKNOWN → saves photo + sends Firebase notification to admin
         ↓
ESP32
  → if AUTHORIZED → activates motor → door opens
  → if UNKNOWN → door stays closed
         ↓
Admin mobile app
  → receives push notification if UNKNOWN
  → can see full access log anytime
```

---

## How /recognize Works Internally

The recognition happens in 3 sub-steps all inside the Face API:

```
Step 1 — Detection (opencv)
  "Where is the face in this image?"
  → finds the face bounding box
  → crops just the face region

Step 2 — Embedding (FaceNet512)
  "What does this face look like mathematically?"
  → converts the face crop to 512 numbers
  → this is the "query vector"

Step 3 — Comparison (cosine similarity)
  "Which enrolled employee does this match?"
  → takes query vector
  → compares it against every vector in embeddings.pkl
  → finds the closest one
  → returns the name + confidence score
```

---

## Component Responsibilities

| Component | Job |
|-----------|-----|
| ESP32-CAM | Takes the photo at the door |
| Flask Backend | Middleman — coordinates everything |
| Face API | The brain — pure AI, no business logic |
| embeddings.pkl | Face database — 512 numbers per employee |
| PostgreSQL | Stores employees, logs, events |
| Firebase | Sends push notifications to the admin |
| Mobile App | Admin interface — add employees, see logs, get alerts |

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/face-recognition-api.git
cd face-recognition-api

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create required folders
mkdir face_db unknown_captures

# 5. Run the API
python app.py
```

The API will start on `http://localhost:5001`

---

## Enrolling Employees

**Option 1 — From existing photos (bulk):**
```bash
python enroll.py
```

**Option 2 — Live webcam capture:**
```bash
python collect_photos.py   # capture photos
python enroll.py           # generate embeddings
```

**Option 3 — Via API directly:**
```bash
curl -X POST http://localhost:5001/enroll \
  -F "name=arij_belmabrouk" \
  -F "image=@path/to/photo.jpg"
```

---

## Testing

```bash
# Test all endpoints
python test_api.py

# Test live webcam recognition
python test_webcam.py

# Test individual endpoints with curl
curl http://localhost:5001/health
curl -X POST http://localhost:5001/recognize -F "image=@photo.jpg"
```

---

## Performance

| Metric | Result |
|--------|--------|
| Recognition accuracy | 91–95% confidence |
| Detection model | OpenCV Haar Cascade |
| Recognition model | FaceNet512 (99.65% on LFW benchmark) |
| Embedding size | 512 dimensions |
| Similarity threshold | 0.70 (70%) |
| Response time | ~1–2 seconds per request |

---

## Architecture Decision — Why Microservice

This Face API is intentionally separate from the Flask backend following the **API Gateway pattern**:

- ESP32 never communicates directly with the Face API
- Flask backend acts as the single entry point for all IoT devices
- Face API can be updated or replaced without touching the backend
- Each service is independently deployable on Render

```
ESP32 → Flask Backend (port 5000) → Face API (port 5001)
                    ↓
               PostgreSQL
                    ↓
                Firebase
```
