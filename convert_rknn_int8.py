"""
Conversione ONNX (da export_onnx_split.py) -> RKNN INT8 per RK3588.

Richiede rknn-toolkit2 (NON la versione lite dell'Orange Pi), disponibile per
Linux x86_64 e aarch64: su Windows usare WSL2 con Ubuntu.
    pip install rknn-toolkit2 opencv-python

La quantizzazione si calibra su immagini reali delle telecamere: le foto in
snapshots/ e i fotogrammi delle clip in clips/ dell'Orange Pi vanno benissimo
(stessa inquadratura e luce del campo). Ne bastano 100-300, varie (giorno/sera,
con e senza persone).

    python convert_rknn_int8.py yolov8s-pose_split.onnx --calib snapshots clips
    -> yolov8s-pose_split.rknn (da copiare sull'Orange Pi)
"""
import argparse
import os
import random

import cv2
from rknn.api import RKNN

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXT = (".mp4", ".avi", ".mkv")


def letterbox(img, width, height, color=114):
    """Stesso letterbox di rknn_backend.preprocess, così la calibrazione vede le immagini di runtime."""
    h, w = img.shape[:2]
    r = min(width / w, height / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (width - nw) // 2, (height - nh) // 2
    return cv2.copyMakeBorder(img, top, height - nh - top, left, width - nw - left,
                              cv2.BORDER_CONSTANT, value=(color, color, color))


def collect_frames(paths, frames_per_video):
    """Elenco di (sorgente, indice_frame): immagini singole e fotogrammi campionati dai video."""
    items = []
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))]
        else:
            files.append(p)
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXT:
            items.append((f, None))
        elif ext in VIDEO_EXT:
            cap = cv2.VideoCapture(f)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > 0:
                step = max(1, n // frames_per_video)
                items += [(f, i) for i in range(0, n, step)][:frames_per_video]
    return items


def read_frame(src, index):
    if index is None:
        return cv2.imread(src)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def build_dataset(paths, out_dir, width, height, max_images, frames_per_video, seed=0):
    items = collect_frames(paths, frames_per_video)
    if not items:
        raise SystemExit(f"Nessuna immagine/video di calibrazione in {paths}")
    random.Random(seed).shuffle(items)

    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for src, index in items:
        if len(lines) >= max_images:
            break
        frame = read_frame(src, index)
        if frame is None:
            continue
        path = os.path.abspath(os.path.join(out_dir, f"calib_{len(lines):04d}.jpg"))
        cv2.imwrite(path, letterbox(frame, width, height))
        lines.append(path)

    dataset_txt = os.path.join(out_dir, "dataset.txt")
    with open(dataset_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Calibrazione: {len(lines)} immagini {width}x{height} in {out_dir}")
    return dataset_txt, lines


def main():
    ap = argparse.ArgumentParser(description="ONNX a head separata -> RKNN INT8 (RK3588)")
    ap.add_argument("onnx", help="modello .onnx da export_onnx_split.py")
    ap.add_argument("--calib", nargs="+", required=True, help="cartelle o file (immagini/video) di calibrazione")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--frames-per-video", type=int, default=10)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--output", help="file .rknn di uscita (default: <onnx>.rknn)")
    ap.add_argument("--analyze", action="store_true",
                    help="confronta float e INT8 strato per strato sulla prima immagine (lento)")
    args = ap.parse_args()

    base = args.onnx.rsplit(".", 1)[0]
    output = args.output or base + ".rknn"
    dataset_txt, images = build_dataset(args.calib, base + "_calib", args.width, args.height,
                                        args.max_images, args.frames_per_video)

    rknn = RKNN(verbose=False)
    # Input a runtime: RGB uint8 NHWC (rknn_backend.preprocess); la normalizzazione 0-1 la fa la NPU
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform=args.target, quantized_algorithm="normal", quantized_method="channel")
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx fallito"
    assert rknn.build(do_quantization=True, dataset=dataset_txt) == 0, "build INT8 fallito"
    if args.analyze:
        rknn.accuracy_analysis(inputs=[images[0]], output_dir=base + "_analysis")
        print(f"Analisi precisione in {base}_analysis/error_analysis.txt")
    assert rknn.export_rknn(output) == 0, "export_rknn fallito"
    rknn.release()
    print(f"RKNN INT8 salvato: {output} ({os.path.getsize(output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
