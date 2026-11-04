"""
Addestra il modello coltelli/armi per l'NPU dell'Orange Pi (sostituto di weapon_threat).

Gira sul PC con GPU NVIDIA (pip install -r requirements-train.txt), sul dataset
creato da prepare_weapon_dataset.py:

    python train_weapon_model.py --data datasets/weapon_knife/data.yaml
    python train_weapon_model.py --data ... --model yolov8m.pt --epochs 150   # più preciso, più lento sulla NPU

Alla fine:
  - weapon_knife.pt             pesi migliori (copia di runs/weapon/<nome>/weapon/best.pt)
  - weapon_knife_split.onnx     export a head separata per la conversione INT8
                                (export_onnx_split.py, input 640x480 come le telecamere)
e stampa precision/recall per classe. Poi, su Linux/WSL2:
    python convert_rknn_int8.py weapon_knife_split.onnx \\
        --calib datasets/weapon_knife/images/val snapshots clips
e su config.py: WEAPON_RKNN_PATH / WEAPON_CLASSES / WEAPON_CLASS_THRESHOLDS
(vedi commento in config.py).

Perché yolov8s e non n: i coltelli sono oggetti sottili; la "s" ha circa il
triplo dei parametri e sull'NPU in INT8 a 640x480 resta intorno ai 20 ms.
L'head deve essere YOLOv8 (o YOLO11), che export_onnx_split.py sa tagliare.
"""
import argparse
import os
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Training modello coltelli/armi (YOLOv8)")
    ap.add_argument("--data", required=True, help="data.yaml da prepare_weapon_dataset.py")
    ap.add_argument("--model", default="yolov8s.pt", help="pesi di partenza (pre-addestrati COCO)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = automatico in base alla VRAM")
    ap.add_argument("--device", default="0", help="'0' = prima GPU, 'cpu' per prova (lentissimo)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="weapon_knife")
    ap.add_argument("--output", default="weapon_knife.pt")
    ap.add_argument("--no-export", action="store_true", help="non lanciare export_onnx_split.py")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    # Augmentation pensate per riprese da telecamera USB a distanza ravvicinata:
    # - hsv/luminosità ampie: luce mista interna/esterna, sera;
    # - degrees: il coltello in mano è inclinato in ogni modo;
    # - flipud basso: un coltello capovolto resta un coltello;
    # - mosaic + scale: varietà di dimensioni; close_mosaic stabilizza le ultime epoche;
    # - con "albumentations" installato Ultralytics aggiunge blur/median blur/CLAHE,
    #   utili contro la sfocatura di movimento a 15-20 FPS.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs/weapon",
        name=args.name,
        exist_ok=True,
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        degrees=15.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, flipud=0.1,
        mosaic=1.0, mixup=0.1,
        plots=True,
    )

    best = os.path.join("runs", "weapon", args.name, "weights", "best.pt")
    if not os.path.isfile(best):
        raise SystemExit(f"best.pt non trovato in {best}")
    shutil.copy2(best, args.output)
    print(f"\nPesi migliori copiati in {args.output}")

    # Validazione finale: precision/recall per classe
    metrics = YOLO(args.output).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)
    names = metrics.names
    print("\n========== VALIDAZIONE ==========")
    print(f"mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f}")
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{names[int(c)]:8s} precision {p:.3f}  recall {r:.3f}  AP50 {ap50:.3f}  AP50-95 {ap:.3f}")
    print(f"Curva F1/confidenza (soglia ottimale per WEAPON_CLASS_THRESHOLDS): "
          f"{os.path.join('runs', 'weapon', args.name, 'BoxF1_curve.png')}")

    if not args.no_export:
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([sys.executable, os.path.join(here, "export_onnx_split.py"), args.output], check=True)


if __name__ == "__main__":
    main()
