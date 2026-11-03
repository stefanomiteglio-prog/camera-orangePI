import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

POSE_SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                 [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

COCO_CLASSES = ("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush")


def input_wh(size):
    """640 -> (640, 640); (640, 480) resta (larghezza, altezza)."""
    return (size, size) if isinstance(size, int) else tuple(size)


def letterbox(img, size=640, color=114):
    tw, th = input_wh(size)
    h, w = img.shape[:2]
    # Frame grande quanto l'input (es. 640x480 su modello 640x480): nessun resize né bordo
    if (w, h) == (tw, th):
        return img, 1.0, 0, 0
    r = min(th / h, tw / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    # Con 640x480 su modello 640x640 r == 1: niente resize, solo il bordo (copyMakeBorder, in C++)
    resized = img if (nh, nw) == (h, w) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas = cv2.copyMakeBorder(resized, top, th - nh - top, left, tw - nw - left,
                                cv2.BORDER_CONSTANT, value=(color, color, color))
    return canvas, r, left, top


def preprocess(frame, size=640):
    """
    Input NPU pronto (RGB letterbox NHWC) + parametri per riportare i box sul frame.
    Calcolato UNA volta per frame e condiviso da tutti i modelli con lo stesso input_size;
    la conversione colore è fatta sul frame originale (più piccolo del 640x640).
    """
    size = input_wh(size)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img, ratio, dx, dy = letterbox(rgb, size)
    return np.expand_dims(img, axis=0), ratio, dx, dy, size


def _nms(boxes, scores, nms_thresh, class_ids=None):
    """
    NMS greedy. Con class_ids è per classe in un solo passaggio: i box di classi
    diverse vengono traslati lontano tra loro, così non si sovrappongono mai.
    """
    if class_ids is not None:
        # float64: in float32 l'offset farebbe perdere precisione alle coordinate
        boxes = boxes.astype(np.float64) + (class_ids.astype(np.float64) * 8192.0)[:, None]
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0]); yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2]); yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter/(areas[i]+areas[rest]-inter+1e-6)
        idxs = rest[iou <= nms_thresh]
    return keep


def _make_rknnlite(model_path, core_mask):
    rknn = RKNNLite(verbose=False)
    assert rknn.load_rknn(model_path) == 0, f"load_rknn fallito per {model_path}"
    assert rknn.init_runtime(core_mask=core_mask) == 0, f"init_runtime fallito per {model_path}"
    return rknn


def core_mask_from(value):
    """0/1/2 -> singolo core NPU del RK3588; "auto" (o altro) -> scelta del runtime."""
    masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
    return masks.get(value, RKNNLite.NPU_CORE_AUTO)


class _RknnModel:
    """Base comune: contesto RKNN, preprocessing condiviso e tempo medio di inferenza NPU."""

    def __init__(self, model_path, core_mask, conf_thresh, nms_thresh, input_size):
        self.rknn = _make_rknnlite(model_path, core_mask)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_wh(input_size)   # (larghezza, altezza)
        self.npu_ms = 0.0   # media mobile del solo tempo NPU (senza pre/post-processing)

    def _run(self, frame, prep):
        if prep is None or prep[4] != self.input_size:
            prep = preprocess(frame, self.input_size)
        t0 = time.perf_counter()
        outputs = self.rknn.inference(inputs=[prep[0]])
        ms = (time.perf_counter() - t0) * 1000.0
        self.npu_ms = ms if self.npu_ms == 0 else self.npu_ms * 0.9 + ms * 0.1
        return outputs, prep[1], prep[2], prep[3]

    def release(self):
        self.rknn.release()


class PoseModel(_RknnModel):
    def __init__(self, model_path, core_mask=RKNNLite.NPU_CORE_1, conf_thresh=0.5, nms_thresh=0.45, input_size=640):
        super().__init__(model_path, core_mask, conf_thresh, nms_thresh, input_size)

    def infer(self, frame, prep=None):
        outputs, ratio, dx, dy = self._run(frame, prep)
        if len(outputs) > 1:
            return self._infer_split(outputs, ratio, dx, dy)
        # Export Ultralytics standard (FP16): output unico gia' decodificato
        o = outputs[0][0]                       # (56, 8400)
        # Filtro sulla confidenza PRIMA della trasposizione: si copiano solo le colonne valide
        idx = np.nonzero(o[4] >= self.conf_thresh)[0]
        results = []
        if len(idx) == 0:
            return results
        out = o[:, idx].T
        cx, cy, w, h = out[:,0], out[:,1], out[:,2], out[:,3]
        boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
        scores = out[:,4]
        kpts = out[:,5:].reshape(-1,17,3)
        keep = _nms(boxes, scores, self.nms_thresh)
        for i in keep:
            b = boxes[i]
            x1 = (b[0]-dx)/ratio; y1=(b[1]-dy)/ratio; x2=(b[2]-dx)/ratio; y2=(b[3]-dy)/ratio
            kp = kpts[i].copy()
            kp[:,0] = (kp[:,0]-dx)/ratio
            kp[:,1] = (kp[:,1]-dy)/ratio
            results.append({"box":[x1,y1,x2,y2], "score": float(scores[i]), "keypoints": kp})
        return results

    def _infer_split(self, outputs, ratio, dx, dy):
        """
        Head separata INT8 (export_onnx_split.py): per ramo box (1,64,h,w),
        cls (1,1,h,w) e keypoint grezzi (1,51,h,w), decodificati solo sulle celle candidate.
        """
        parts = []
        for i in range(len(outputs) // 3):
            reg, cls, kpt = outputs[3*i:3*i+3]
            part = _decode_branch(reg, cls, self.input_size[1], self.conf_thresh)
            if part is None:
                continue
            boxes, scores, _, idx = part
            gh, gw = kpt.shape[2:]
            stride = self.input_size[1] / gh
            k = kpt[0].reshape(17, 3, gh * gw)[:, :, idx]            # (17, 3, k)
            col = (idx % gw).astype(np.float32)
            row = (idx // gw).astype(np.float32)
            # Come Ultralytics: (offset*2 + ancora - 0.5) * stride, con ancora = cella + 0.5
            kx = (k[:, 0] * 2.0 + col) * stride
            ky = (k[:, 1] * 2.0 + row) * stride
            kv = 1.0 / (1.0 + np.exp(-k[:, 2]))
            parts.append((boxes, scores, np.stack([kx, ky, kv], axis=-1).transpose(1, 0, 2)))
        results = []
        if not parts:
            return results
        boxes = np.concatenate([p[0] for p in parts])
        scores = np.concatenate([p[1] for p in parts])
        kpts = np.concatenate([p[2] for p in parts])
        for i in _nms(boxes, scores, self.nms_thresh):
            b = boxes[i]
            x1 = (b[0]-dx)/ratio; y1=(b[1]-dy)/ratio; x2=(b[2]-dx)/ratio; y2=(b[3]-dy)/ratio
            kp = kpts[i].copy()
            kp[:,0] = (kp[:,0]-dx)/ratio
            kp[:,1] = (kp[:,1]-dy)/ratio
            results.append({"box":[x1,y1,x2,y2], "score": float(scores[i]), "keypoints": kp})
        return results


WEAPON_CLASSES = ("Gun", "explosion", "grenade", "knife")


class WeaponModel(_RknnModel):
    """
    Modello armi/minacce (weapon_threat.rknn): head YOLOv8 a output singolo
    (1, 4+num_classi, 8400) con box cx,cy,w,h e score per classe gia' decodificati.
    """
    def __init__(self, model_path, core_mask=RKNNLite.NPU_CORE_2, conf_thresh=0.35, nms_thresh=0.45, input_size=640, classes=WEAPON_CLASSES):
        super().__init__(model_path, core_mask, conf_thresh, nms_thresh, input_size)
        self.classes = tuple(classes)

    def infer(self, frame, prep=None):
        outputs, ratio, dx, dy = self._run(frame, prep)
        if len(outputs) > 1:
            # Head separata INT8 (export_onnx_split.py): stessa decodifica del detector COCO
            return _decode_detect(outputs, self.input_size[1], self.conf_thresh, self.nms_thresh,
                                  self.classes, ratio, dx, dy)
        # Export Ultralytics standard (FP16): output unico gia' decodificato
        o = outputs[0][0]                       # (4+nc, 8400)
        class_scores = o[4:4 + len(self.classes)]
        scores = class_scores.max(axis=0)
        idx = np.nonzero(scores >= self.conf_thresh)[0]
        results = []
        if len(idx) == 0:
            return results
        cls_ids = class_scores[:, idx].argmax(axis=0); scores = scores[idx]
        cx, cy, w, h = o[0, idx], o[1, idx], o[2, idx], o[3, idx]
        boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
        keep = _nms(boxes, scores, self.nms_thresh)
        for i in keep:
            b = boxes[i]
            x1 = (b[0]-dx)/ratio; y1=(b[1]-dy)/ratio; x2=(b[2]-dx)/ratio; y2=(b[3]-dy)/ratio
            results.append({"box":[x1,y1,x2,y2], "score": float(scores[i]), "class_id": int(cls_ids[i]), "class_name": self.classes[int(cls_ids[i])]})
        return results


def _decode_branch(reg, cls, input_h, conf_thresh):
    """
    Un ramo della head YOLOv8 (rknn_model_zoo o export_onnx_split.py): reg (1,4*16,h,w)
    in formato DFL, cls (1,nc,h,w) gia' passato per la sigmoide.
    Soglia sulle classi PRIMA della DFL: softmax solo sulle celle candidate invece
    che su tutte le 8400 (la parte più costosa del post-processing su CPU).
    """
    _, nc, gh, gw = cls.shape
    c = cls[0].reshape(nc, gh * gw)
    cmax = c.max(axis=0)
    idx = np.nonzero(cmax >= conf_thresh)[0]
    if len(idx) == 0:
        return None
    mc = reg.shape[1] // 4
    r = reg[0].reshape(4, mc, gh * gw)[:, :, idx]                   # (4, 16, k)
    e = np.exp(r - r.max(axis=1, keepdims=True))
    acc = np.arange(mc, dtype=np.float32).reshape(1, mc, 1)
    d = (e * acc).sum(axis=1) / e.sum(axis=1)                       # (4, k)
    col = (idx % gw).astype(np.float32) + 0.5
    row = (idx // gw).astype(np.float32) + 0.5
    stride = input_h / gh
    boxes = np.stack([col - d[0], row - d[1], col + d[2], row + d[3]], axis=1) * stride
    return boxes, cmax[idx], c[:, idx].argmax(axis=0), idx


def _decode_detect(outputs, input_h, conf_thresh, nms_thresh, class_names, ratio, dx, dy):
    """
    Output a rami: 3 per ramo nel model zoo (box DFL, classi, score_sum ignorato),
    2 per ramo con export_onnx_split.py (box DFL, classi).
    """
    per_branch = len(outputs) // 3
    parts = [_decode_branch(outputs[per_branch*i], outputs[per_branch*i+1], input_h, conf_thresh)
             for i in range(3)]
    parts = [p for p in parts if p is not None]
    results = []
    if not parts:
        return results
    boxes = np.concatenate([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    classes = np.concatenate([p[2] for p in parts])
    keep = _nms(boxes, scores, nms_thresh, class_ids=classes)
    for k in keep:
        x1,y1,x2,y2 = boxes[k]
        x1=(x1-dx)/ratio; y1=(y1-dy)/ratio; x2=(x2-dx)/ratio; y2=(y2-dy)/ratio
        c = int(classes[k])
        results.append({"box":[x1,y1,x2,y2], "score": float(scores[k]), "class_id": c, "class_name": class_names[c]})
    return results


class DetectModel(_RknnModel):
    def __init__(self, model_path, core_mask=RKNNLite.NPU_CORE_0, conf_thresh=0.25, nms_thresh=0.45, input_size=640, classes=COCO_CLASSES):
        super().__init__(model_path, core_mask, conf_thresh, nms_thresh, input_size)
        self.classes = classes

    def infer(self, frame, prep=None):
        outputs, ratio, dx, dy = self._run(frame, prep)
        return _decode_detect(outputs, self.input_size[1], self.conf_thresh, self.nms_thresh,
                              self.classes, ratio, dx, dy)


class SimpleTracker:
    def __init__(self, max_lost=15, iou_thresh=0.3):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    @staticmethod
    def _iou(b1, b2):
        xx1=max(b1[0],b2[0]); yy1=max(b1[1],b2[1]); xx2=min(b1[2],b2[2]); yy2=min(b1[3],b2[3])
        w=max(0,xx2-xx1); h=max(0,yy2-yy1)
        inter=w*h
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

    def update(self, detections):
        assigned = set()
        for tid, tr in list(self.tracks.items()):
            best_iou, best_j = 0, -1
            for j, det in enumerate(detections):
                if j in assigned:
                    continue
                iou = self._iou(tr["box"], det["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_thresh:
                detections[best_j]["track_id"] = tid
                self.tracks[tid] = {"box": detections[best_j]["box"], "lost": 0}
                assigned.add(best_j)
            else:
                tr["lost"] += 1
                if tr["lost"] > self.max_lost:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in assigned:
                tid = self.next_id
                self.next_id += 1
                det["track_id"] = tid
                self.tracks[tid] = {"box": det["box"], "lost": 0}

        return detections

    def release(self):
        pass
