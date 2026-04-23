import requests

BASE_URL = "http://localhost:5001"

# ── Test 1: Health ────────────────────────────────────────────────────
print("Testing /health...")
r = requests.get(f"{BASE_URL}/health")
print(r.json())
print()

# ── Test 2: Enroll ────────────────────────────────────────────────────
print("Testing /enroll...")
with open("dataset/arij/photo_1.jpg", "rb") as f:
    r = requests.post(
        f"{BASE_URL}/enroll",
        data  = {"name": "arij_belmabrouk"},
        files = {"image": f}
    )
print(r.json())
print()

# ── Test 3: Recognize ─────────────────────────────────────────────────
print("Testing /recognize...")
with open("dataset/arij/photo_5.jpg", "rb") as f:
    r = requests.post(
        f"{BASE_URL}/recognize",
        files = {"image": f}
    )
print(r.json())
print()