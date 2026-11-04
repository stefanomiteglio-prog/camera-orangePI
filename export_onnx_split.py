"""
Export ONNX "a head separata" dei modelli YOLOv8 (detect e pose) per la
quantizzazione INT8 su NPU RK3588 (vedi convert_rknn_int8.py).

Perché non basta l'export standard di Ultralytics: quello concatena in un unico
tensore (1, 4+nc[+51], 8400) coordinate in pixel (0-640) e confidenze (0-1).
In INT8 un tensore ha una sola scala, quindi le confidenze perderebbero quasi
tutta la precisione. Qui la head viene "tagliata" prima della decodifica e ogni
ramo (stride 8/16/32) esce con tensori distinti, ciascuno con la propria scala:

    detect:  box_i (1, 64, h, w) logit DFL, cls_i (1, nc, h, w) sigmoide
    pose:    box_i, cls_i (1, 1, h, w) sigmoide, kpt_i (1, 51, h, w) grezzi

La decodifica (DFL, griglia, keypoint) la fa rknn_backend.py su CPU, solo sulle
celle sopra soglia.

Gira su qualsiasi sistema con ultralytics installato (anche Windows):
    pip install ultralytics onnx
    python export_onnx_split.py yolov8s-pose.pt
    python export_onnx_split.py weapon_threat.pt
    python export_onnx_split.py yolov8n.pt
"""
import argparse
import inspect

import torch
import torch.nn as nn
from ultralytics import YOLO


class SplitHeadModel(nn.Module):
    """Stesso grafo del modello Ultralytics, ma con la head fermata prima della decodifica."""

    def __init__(self, det_model):
        super().__init__()
        self.layers = det_model.model
        self.save = det_model.save
        self.head = self.layers[-1]
        self.is_pose = hasattr(self.head, "cv4")

    def forward(self, x):
        # Come DetectionModel._predict_once, escluso l'ultimo layer (la head)
        y = []
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        feats = [x if j == -1 else y[j] for j in self.head.f]

        outs = []
        for i, f in enumerate(feats):
            outs.append(self.head.cv2[i](f))
            outs.append(self.head.cv3[i](f).sigmoid())
            if self.is_pose:
                outs.append(self.head.cv4[i](f))
        return tuple(outs)


def main():
    ap = argparse.ArgumentParser(description="Export ONNX a head separata per RKNN INT8")
    ap.add_argument("weights", help="pesi Ultralytics .pt (yolov8 detect o pose)")
    ap.add_argument("--width", type=int, default=640, help="larghezza input (multiplo di 32)")
    ap.add_argument("--height", type=int, default=640, help="altezza input (multiplo di 32; 480 sconsigliato, vedi README)")
    ap.add_argument("--output", help="file .onnx di uscita (default: <pesi>_split.onnx)")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    if args.width % 32 or args.height % 32:
        raise SystemExit("Larghezza e altezza devono essere multipli di 32.")

    yolo = YOLO(args.weights)
    det_model = yolo.model.float().fuse().eval()
    model = SplitHeadModel(det_model).eval()

    # L'ordine delle classi deve coincidere con config.py (WEAPON_CLASSES per il modello armi)
    names = det_model.names
    print(f"Classi ({len(names)}): {[names[i] for i in sorted(names)]}")
    if model.is_pose and tuple(model.head.kpt_shape) != (17, 3):
        raise SystemExit(f"kpt_shape {model.head.kpt_shape} non supportato (atteso 17x3).")

    per_branch = ["box", "cls", "kpt"] if model.is_pose else ["box", "cls"]
    output_names = [f"{n}{i}" for i in range(len(model.head.cv2)) for n in per_branch]
    output = args.output or args.weights.rsplit(".", 1)[0] + "_split.onnx"

    dummy = torch.zeros(1, 3, args.height, args.width)
    export_kw = dict(opset_version=args.opset, input_names=["images"], output_names=output_names)
    # Exporter TorchScript classico: quello "dynamo" (default da torch 2.9) produce grafi
    # con opset recenti che rknn-toolkit2 non supporta
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kw["dynamo"] = False
    with torch.no_grad():
        shapes = [tuple(o.shape) for o in model(dummy)]
        torch.onnx.export(model, dummy, output, **export_kw)

    for name, shape in zip(output_names, shapes):
        print(f"  {name}: {shape}")
    print(f"ONNX salvato: {output} (input {args.width}x{args.height})")


if __name__ == "__main__":
    main()
