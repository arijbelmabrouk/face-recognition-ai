import cv2
import os

def collect(name, total=10):
    folder = f"dataset/{name}"
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"\n📸 Collecting photos for: {name}")
    print("Press SPACE to capture | Q to quit\n")

    while count < total:
        ret, frame = cap.read()
        display = frame.copy()
        cv2.putText(display,
                    f"{name} — {count}/{total} captured — SPACE to take photo",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Collect Photos", display)

        key = cv2.waitKey(1)
        if key == ord(' '):
            path = f"{folder}/photo_{count+1}.jpg"
            cv2.imwrite(path, frame)
            print(f"  ✅ Saved photo {count+1}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! {count} photos saved for {name}\n")

# Change these names to match your team
collect("arij_belmabrouk",  total=10)
collect("wassim_lourimi",   total=10)
collect("yasmine_hsayri",   total=10)