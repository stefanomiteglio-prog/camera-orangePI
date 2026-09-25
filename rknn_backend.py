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


def letterbox(img, size=640, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top


def _make_rknnlite(model_path, core_mask):
    rknn = RKNNLite(verbose=False)
    assert rknn.load_rknn(model_path) == 0, f"load_rknn fallito per {model_path}"
    assert rknn.init_runtime(core_mask=core_mask) == 0, f"init_runtime fallito per {model_path}"
    return rknn


class PoseModel:
    def __init__(self, model_path, core_mask=RKNNLite.NPU_CORE_1, conf_thresh=0.5, nms_thresh=0.45, input_size=640):
        self.rknn = _make_rknnlite(model_path, core_mask)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

    def _nms(self, boxes, scores):
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
            w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
            inter = w*h
            area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            area_r = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
            iou = inter/(area_i+area_r-inter+1e-6)
            idxs = rest[iou <= self.nms_thresh]
        return keep

    def infer(self, frame):
        img, ratio, dx, dy = letterbox(frame, self.input_size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])
        out = outputs[0][0].transpose(1, 0)
        conf = out[:, 4]
        mask = conf >= self.conf_thresh
        out = out[mask]
        results = []
        if len(out) == 0:
            return results
        cx, cy, w, h = out[:,0], out[:,1], out[:,2], out[:,3]
        boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
        scores = out[:,4]
        kpts = out[:,5:].reshape(-1,17,3)
        keep = self._nms(boxes, scores)
        for i in keep:
            b = boxes[i]
            x1 = (b[0]-dx)/ratio; y1=(b[1]-dy)/ratio; x2=(b[2]-dx)/ratio; y2=(b[3]-dy)/ratio
            kp = kpts[i].copy()
            kp[:,0] = (kp[:,0]-dx)/ratio
            kp[:,1] = (kp[:,1]-dy)/ratio
            results.append({"box":[x1,y1,x2,y2], "score": float(scores[i]), "keypoints": kp})
        return results

    def release(self):
        self.rknn.release()


def _dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    e = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = e / np.sum(e, axis=2, keepdims=True)
    acc = np.arange(mc).reshape(1,1,mc,1,1).astype(np.float32)
    return (y * acc).sum(2)


def _box_process(position, img_size=640):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1,1,grid_h,grid_w); row = row.reshape(1,1,grid_h,grid_w)
    grid = np.concatenate((col,row), axis=1)
    stride = np.array([img_size//grid_h, img_size//grid_w]).reshape(1,2,1,1)
    position = _dfl(position)
    box_xy = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy


def _sp_flatten(x):
    ch = x.shape[1]
    return x.transpose(0,2,3,1).reshape(-1, ch)


class DetectModel:
    def __init__(self, model_path, core_mask=RKNNLite.NPU_CORE_0, conf_thresh=0.25, nms_thresh=0.45, input_size=640, classes=COCO_CLASSES):
        self.rknn = _make_rknnlite(model_path, core_mask)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.classes = classes

    def _nms_boxes(self, boxes, scores):
        x, y = boxes[:,0], boxes[:,1]
        w, h = boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]
        areas = w*h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]]); yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i]+w[i], x[order[1:]]+w[order[1:]]); yy2 = np.minimum(y[i]+h[i], y[order[1:]]+h[order[1:]])
            w1 = np.maximum(0.0, xx2-xx1+1e-5); h1 = np.maximum(0.0, yy2-yy1+1e-5)
            inter = w1*h1
            ovr = inter/(areas[i]+areas[order[1:]]-inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds+1]
        return np.array(keep, dtype=int)

    def infer(self, frame):
        img, ratio, dx, dy = letterbox(frame, self.input_size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])
        boxes_l, scores_l, classes_conf_l = [], [], []
        for i in range(3):
            boxes_l.append(_box_process(outputs[3*i], self.input_size))
            classes_conf_l.append(outputs[3*i+1])
            scores_l.append(np.ones_like(outputs[3*i+1][:, :1, :, :], dtype=np.float32))

        boxes = np.concatenate([_sp_flatten(b) for b in boxes_l])
        classes_conf = np.concatenate([_sp_flatten(c) for c in classes_conf_l])
        scores = np.concatenate([_sp_flatten(s) for s in scores_l])

        box_conf = scores.reshape(-1)
        class_max = np.max(classes_conf, axis=-1)
        classes = np.argmax(classes_conf, axis=-1)
        pos = np.where(class_max * box_conf >= self.conf_thresh)
        boxes = boxes[pos]; classes = classes[pos]; final_scores = (class_max*box_conf)[pos]

        results = []
        if len(boxes) == 0:
            return results
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]; s = final_scores[inds]
            keep = self._nms_boxes(b, s)
            for k in keep:
                x1,y1,x2,y2 = b[k]
                x1=(x1-dx)/ratio; y1=(y1-dy)/ratio; x2=(x2-dx)/ratio; y2=(y2-dy)/ratio
                results.append({"box":[x1,y1,x2,y2], "score": float(s[k]), "class_id": int(c), "class_name": self.classes[int(c)]})
        return results

    def release(self):
        self.rknn.release()


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
