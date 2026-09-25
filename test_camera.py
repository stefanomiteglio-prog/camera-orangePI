import cv2, time
import mediapipe as mp
import config

CAMERA_PATH = config.CAMERA_INDEX
print("Uso camera:", CAMERA_PATH)

cap = cv2.VideoCapture(CAMERA_PATH)
if not cap.isOpened():
    print("ERRORE: impossibile aprire la camera")
else:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if getattr(config, "CAMERA_WIDTH", None):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    if getattr(config, "CAMERA_HEIGHT", None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if getattr(config, "CAMERA_FPS", None):
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera configurata: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

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
