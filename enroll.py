from deepface import DeepFace
import numpy as np
import pickle
import os

MODEL   = "Facenet512"
DB_PATH = "face_db/embeddings.pkl"

def enroll_all():
    db = {}

    for person_name in os.listdir("dataset"):
        folder = os.path.join("dataset", person_name)
        if not os.path.isdir(folder):
            continue

        vectors = []
        print(f"\n👤 Enrolling: {person_name}")

        for photo_file in os.listdir(folder):
            photo_path = os.path.join(folder, photo_file)
            try:
                result = DeepFace.represent(
                    img_path    = photo_path,
                    model_name  = MODEL,
                    detector_backend= "opencv",
                    enforce_detection = True
                )
                vectors.append(np.array(result[0]["embedding"]))
                print(f"  ✅ {photo_file}")
            except Exception as e:
                print(f"  ⚠️  Skipped {photo_file} — {e}")

        if vectors:
            # Average all photo vectors → one stable fingerprint per person
            db[person_name] = np.mean(vectors, axis=0)
            print(f"  → Stored fingerprint from {len(vectors)} photos")

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"\n🎉 Enrolled {len(db)} people → saved to {DB_PATH}")

enroll_all()