import cv2
import numpy as np
from recognize import recognize
import time
import os
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PAUSE_AFTER_DECISION = 5

# Shared state between threads
latest_frame   = None
result_ready   = False
latest_result  = None
processing     = False
lock           = threading.Lock()

def recognition_worker():
    global latest_frame, result_ready, latest_result, processing
    while True:
        time.sleep(0.1)
        with lock:
            if latest_frame is None or processing or result_ready:
                continue
            frame_to_process = latest_frame.copy()
            processing = True

        _, buffer = cv2.imencode('.jpg', frame_to_process)
        result = recognize(buffer.tobytes())

        with lock:
            processing = False
            if result['status'] != "NO_FACE":
                latest_result = result
                result_ready  = True

# Start recognition in background thread
thread = threading.Thread(target=recognition_worker, daemon=True)
thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera ready — show your face\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Always update latest frame for the background thread
    with lock:
        latest_frame = frame.copy()

    # Check if background thread found a face
    with lock:
        got_result = result_ready
        result     = latest_result

    if got_result:
        # ── Decision made ───────────────────────────────────────────
        status     = result['status']
        name       = result.get('name', '—')
        confidence = result.get('confidence', 0)

        print("─────────────────────────────")
        print(f"  Status:     {status}")
        print(f"  Name:       {name}")
        print(f"  Confidence: {confidence}%")
        print("─────────────────────────────\n")

        if status == "AUTHORIZED":
            color   = (0, 255, 0)
            message = f"AUTHORIZED — {name} ({confidence}%)"
            print(f"🚪 Door OPENS for {name}")
        else:
            color   = (0, 0, 255)
            message = f"ACCESS DENIED ({confidence}%)"
            print("🔒 Door stays CLOSED")

        # Show frozen result for PAUSE seconds — still smooth
        pause_end = time.time() + PAUSE_AFTER_DECISION
        while time.time() < pause_end:
            result_frame = frame.copy()
            cv2.putText(result_frame, message, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            remaining = int(pause_end - time.time()) + 1
            cv2.putText(result_frame, f"Closing in {remaining}s", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Door Camera", result_frame)
            cv2.waitKey(30)

        break

    # No result yet — show smooth live feed
    cv2.putText(frame, "Scanning...", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Door Camera", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print("\n✅ Session complete — camera closed.")