import time
import cv2
import numpy as np
from rknn.api import RKNN

MODEL_PATH = "yolov8s-pose.rknn"
TARGET = "rk3588"
CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN5100-video-index0"
INPUT_SIZE = 640
CONF_THRESH = 0.5
NMS_THRESH = 0.45

SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]


def letterbox(img, size=640, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top


def nms(boxes, scores, thresh):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou <= thresh]
    return keep


def decode(output, conf_thresh=CONF_THRESH):
    # output shape: (1, 56, 8400) -> (8400, 56)
    out = output[0].transpose(1, 0)
    conf = out[:, 4]
    mask = conf >= conf_thresh
    out = out[mask]
    if len(out) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3))

    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = out[:, 4]
    kpts = out[:, 5:].reshape(-1, 17, 3)

    keep = nms(boxes, scores, NMS_THRESH)
    return boxes[keep], scores[keep], kpts[keep]


print("Caricamento modello RKNN...")
rknn = RKNN(verbose=False)
ret = rknn.load_rknn(MODEL_PATH)
if ret != 0:
    print("ERRORE load_rknn"); exit(ret)

ret = rknn.init_runtime(target=TARGET)
if ret != 0:
    print("ERRORE init_runtime"); exit(ret)
print("Modello pronto.")

cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    print("ERRORE apertura camera")
    exit(1)

frame_count = 0
t0 = time.time()
last_save = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame non letto")
            continue

        img, ratio, dx, dy = letterbox(frame, INPUT_SIZE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = rknn.inference(inputs=[rgb])
        boxes, scores, kpts = decode(outputs[0])

        for b, s, kp in zip(boxes, scores, kpts):
            x1 = int((b[0] - dx) / ratio); y1 = int((b[1] - dy) / ratio)
            x2 = int((b[2] - dx) / ratio); y2 = int((b[3] - dy) / ratio)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{s:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            pts = kp.copy()
            pts[:, 0] = (pts[:, 0] - dx) / ratio
            pts[:, 1] = (pts[:, 1] - dy) / ratio
            for x, y, c in pts:
                if c > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            for sk in SKELETON:
                p1, p2 = pts[sk[0]-1], pts[sk[1]-1]
                if p1[2] > 0.5 and p2[2] > 0.5:
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 128, 0), 2)

        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            print(f"FPS: {frame_count/elapsed:.2f} | persone rilevate: {len(boxes)}")
            frame_count = 0
            t0 = time.time()

        if time.time() - last_save >= 2:
            cv2.imwrite("pose_test_out.jpg", frame)
            last_save = time.time()

except KeyboardInterrupt:
    print("Interrotto.")
finally:
    cap.release()
    rknn.release()
