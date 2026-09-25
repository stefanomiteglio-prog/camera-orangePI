import cv2, time
import mediapipe as mp
import config

CAMERA_PATH = config.CAMERA_INDEX
print("Uso camera:", CAMERA_PATH)

cap = cv2.VideoCapture(CAMERA_PATH)
if not cap.isOpened():
    print("ERRORE: impossibile aprire la camera")
else:
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    n = 30
    t0 = time.time()
    read_fail = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            read_fail += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
    t1 = time.time()
    elapsed = t1 - t0
    processed = n - read_fail
    if elapsed > 0 and processed > 0:
        print(f"{processed}/{n} frame processati in {elapsed:.2f}s -> {processed/elapsed:.2f} FPS")
    print(f"Frame falliti: {read_fail}")
    cap.release()
