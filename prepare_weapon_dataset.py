"""
Costruisce il dataset YOLO per il nuovo modello coltelli/armi (train_weapon_model.py).

Unisce più fonti in un'unica cartella con classi uniformi:
    0 = knife   (coltelli da cucina, lame, pugnali)
    1 = gun     (pistole/armi da fuoco; escludibile con --classes knife)

Fonti supportate (tutte opzionali, combinabili):
  --coco N         COCO 2017: immagini con "knife" (lo stesso dominio dei coltelli
                   da cucina), via FiftyOne, senza account.
  --openimages N   Open Images V7: "Kitchen knife", "Knife", "Dagger" (+ "Handgun"),
                   via FiftyOne, senza account.
  --negatives N    COCO 2017 con oggetti che si confondono con un coltello
                   (forchetta, cucchiaio, forbici, telefono, spazzolino,
                   telecomando) e SENZA coltelli: etichetta vuota = hard negative.
  --voc DIR        Dataset in formato Pascal VOC (.xml), es. OD-WeaponDetection
                   "Knife_detection" (Univ. Granada, CC-BY-4.0):
                   https://github.com/ari-dasci/OD-WeaponDetection
  --roboflow W/P/V Dataset di Roboflow Universe (workspace/progetto/versione),
                   scaricato in formato YOLOv8. Serve ROBOFLOW_API_KEY (gratuita).
                   Es. ai-0jtbr/knife-detection-hgvy2/2 (~9000 immagini).
  --extra DIR      Qualsiasi cartella YOLO con data.yaml (export Roboflow/CVAT,
                   oppure weapon_samples/ raccolti dall'Orange Pi e RIVISTI).

I nomi di classe delle fonti vengono mappati con parole chiave (knife, dagger,
blade, pistol, gun, ...); --map "Nome=knife" o "Nome=none" forza la mappatura.
Un'immagine con un box di arma non mappabile viene scartata (un coltello non
etichettato insegnerebbe al modello a ignorarlo).

Esempio completo (PC con GPU, pip install -r requirements-train.txt):
    python prepare_weapon_dataset.py --out datasets/weapon_knife \\
        --coco 4000 --openimages 3000 --negatives 2000 \\
        --voc datasets/_downloads/OD-WeaponDetection/Knife_detection \\
        --roboflow ai-0jtbr/knife-detection-hgvy2/2 \\
        --extra weapon_samples
"""
import argparse
import hashlib
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

KNIFE_WORDS = ("knife", "knives", "dagger", "blade", "machete", "cleaver", "coltello", "pisau", "cuchillo")
GUN_WORDS = ("gun", "pistol", "handgun", "revolver", "firearm", "rifle", "shotgun", "pistola")
# Classi composte che NON sono l'oggetto (box su mano+coltello, scena di violenza, ...)
DROP_WORDS = ("holding", "violence", "person", "hand", "people")

COCO_NEGATIVE_CLASSES = ["fork", "spoon", "scissors", "cell phone", "toothbrush", "remote"]
OPENIMAGES_KNIFE = ["Kitchen knife", "Knife", "Dagger"]
OPENIMAGES_GUN = ["Handgun"]


class ClassMapper:
    def __init__(self, classes, overrides):
        self.classes = list(classes)
        self.overrides = {k.strip().lower(): v.strip().lower() for k, v in overrides.items()}

    def __call__(self, name):
        """Nome di classe della fonte -> indice nel nostro dataset, oppure None (box scartato)."""
        n = name.strip().lower()
        if n in self.overrides:
            target = self.overrides[n]
            return self.classes.index(target) if target in self.classes else None
        # Parole intere: "hand" scarta "holding hand" ma non "handgun"
        if set(re.split(r"[^a-z]+", n)) & set(DROP_WORDS):
            return None
        if any(w in n for w in KNIFE_WORDS):
            return self.classes.index("knife") if "knife" in self.classes else None
        if any(w in n for w in GUN_WORDS) or n == "weapon":
            return self.classes.index("gun") if "gun" in self.classes else None
        return None

    def is_weapon(self, name):
        n = name.strip().lower()
        return any(w in n for w in KNIFE_WORDS + GUN_WORDS) or n == "weapon"


class DatasetWriter:
    """Scrive immagini + etichette YOLO, con split train/val stabile (hash del nome)."""

    def __init__(self, out, classes, val_ratio):
        self.out = out
        self.classes = classes
        self.val_ratio = val_ratio
        self.names = set()
        self.stats = defaultdict(Counter)
        for split in ("train", "val"):
            os.makedirs(os.path.join(out, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    def add(self, source, image_path, boxes):
        """boxes: lista di (classe, cx, cy, w, h) normalizzati 0-1 (vuota = negativo)."""
        if not os.path.isfile(image_path):
            self.stats[source]["immagine_mancante"] += 1
            return
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(image_path))[0])
        name = f"{source}_{stem}"
        i = 1
        while name in self.names:
            name = f"{source}_{stem}_{i}"
            i += 1
        self.names.add(name)

        h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "val" if h < self.val_ratio else "train"
        ext = os.path.splitext(image_path)[1].lower()
        shutil.copy2(image_path, os.path.join(self.out, "images", split, name + ext))

        lines = []
        for cls, cx, cy, w, h_ in boxes:
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            w, h_ = min(max(w, 0.0), 1.0), min(max(h_, 0.0), 1.0)
            if w * h_ <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h_:.6f}")
            self.stats[source][self.classes[cls]] += 1
        with open(os.path.join(self.out, "labels", split, name + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        self.stats[source]["immagini"] += 1
        if not lines:
            self.stats[source]["negativi"] += 1

    def finish(self):
        data_yaml = os.path.join(self.out, "data.yaml")
        with open(data_yaml, "w") as fh:
            fh.write(f"path: {os.path.abspath(self.out)}\n")
            fh.write("train: images/train\nval: images/val\n")
            fh.write(f"nc: {len(self.classes)}\nnames: {self.classes}\n")
        print("\n========== RIEPILOGO DATASET ==========")
        total = Counter()
        for source, c in self.stats.items():
            total.update(c)
            print(f"{source:18s} " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
        print(f"{'TOTALE':18s} " + ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
        print(f"data.yaml: {data_yaml}")
        return data_yaml


# ---------------------------------------------------------------- FiftyOne
def _fiftyone_detections(dataset):
    import fiftyone as fo
    schema = dataset.get_field_schema(embedded_doc_type=fo.Detections)
    if not schema:
        raise SystemExit(f"Nessun campo Detections nel dataset {dataset.name}")
    return list(schema)


def _det_to_yolo(det):
    x, y, w, h = det.bounding_box   # FiftyOne: [x_top_left, y_top_left, w, h] relativi
    return x + w / 2, y + h / 2, w, h


def _flag(det, name):
    try:
        return bool(getattr(det, name))
    except AttributeError:
        return False


def import_fiftyone(writer, mapper, source, zoo_name, split, classes, max_samples, negatives_of=None):
    """
    Importa un dataset dello zoo FiftyOne. negatives_of: se valorizzato, tiene
    SOLO le immagini senza nessuna di quelle classi e le scrive come negativi.
    """
    import fiftyone.zoo as foz

    print(f"\n[{source}] download/caricamento {zoo_name} ({split}), classi {classes}, max {max_samples}...")
    ds = foz.load_zoo_dataset(
        zoo_name, split=split, label_types=["detections"], classes=classes,
        max_samples=max_samples, only_matching=negatives_of is None,
        dataset_name=f"treeeyes-{source}-{split}", drop_existing_dataset=True,
    )
    fields = _fiftyone_detections(ds)
    for sample in ds.iter_samples(progress=True):
        dets = []
        for f in fields:
            value = sample[f]
            if value is not None:
                dets += value.detections
        if negatives_of is not None:
            labels = {d.label.lower() for d in dets}
            if labels & {n.lower() for n in negatives_of}:
                continue
            writer.add(source, sample.filepath, [])
            continue
        boxes, skip = [], False
        for d in dets:
            if _flag(d, "iscrowd") or _flag(d, "IsGroupOf"):
                skip = skip or mapper.is_weapon(d.label)
                continue
            cls = mapper(d.label)
            if cls is None:
                skip = skip or mapper.is_weapon(d.label)
                continue
            boxes.append((cls,) + _det_to_yolo(d))
        if boxes and not skip:
            writer.add(source, sample.filepath, boxes)


# ------------------------------------------------------------- YOLO / VOC
def _load_names(data_yaml):
    with open(data_yaml, encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml
        names = yaml.safe_load(text).get("names", [])
    except ImportError:
        # Senza PyYAML: "names: ['a', 'b']" oppure lista/dizionario a blocchi
        m = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            names = [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
        else:
            block = re.search(r"^names:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
            lines = block.group(1).splitlines() if block else []
            names = [re.sub(r"^\s*(-|\d+:)\s*", "", l).strip().strip("'\"") for l in lines if l.strip()]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def _image_for_label(label_path):
    parts = label_path.replace("\\", "/").split("/")
    idx = len(parts) - 1 - parts[::-1].index("labels")
    parts[idx] = "images"
    base = os.path.splitext("/".join(parts))[0]
    for ext in IMAGE_EXT + tuple(e.upper() for e in IMAGE_EXT):
        if os.path.isfile(base + ext):
            return base + ext
    return None


def import_yolo_dir(writer, mapper, root, source):
    yamls = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f in ("data.yaml", "data.yml")]
    if not yamls:
        print(f"[{source}] nessun data.yaml in {root}: saltato")
        return
    names = _load_names(yamls[0])
    print(f"\n[{source}] {root}: classi {names} -> " +
          ", ".join(f"{n}->{writer.classes[mapper(n)] if mapper(n) is not None else 'scartata'}" for n in names))
    for dp, _, fs in os.walk(root):
        if "labels" not in dp.replace("\\", "/").split("/"):
            continue
        for f in fs:
            if not f.endswith(".txt"):
                continue
            label_path = os.path.join(dp, f)
            image = _image_for_label(label_path)
            if image is None:
                continue
            boxes, skip = [], False
            with open(label_path) as fh:
                for line in fh:
                    v = line.split()
                    if len(v) < 5:
                        continue
                    src_cls = int(float(v[0]))
                    name = names[src_cls] if src_cls < len(names) else str(src_cls)
                    cls = mapper(name)
                    if cls is None:
                        skip = skip or mapper.is_weapon(name)
                        continue
                    coords = [float(x) for x in v[1:]]
                    if len(coords) == 4:
                        boxes.append((cls,) + tuple(coords))
                    else:   # poligono (export segmentazione) -> box che lo contiene
                        xs, ys = coords[0::2], coords[1::2]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            if not skip:
                writer.add(source, image, boxes)   # vuoto = negativo (es. falso positivo rivisto)


def import_voc(writer, mapper, root, source):
    print(f"\n[{source}] Pascal VOC da {root}...")
    images = {}
    xmls = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXT:
                images.setdefault(stem, os.path.join(dp, f))
            elif ext.lower() == ".xml":
                xmls.append(os.path.join(dp, f))
    for xml_path in xmls:
        try:
            tree = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        fname = tree.findtext("filename") or os.path.basename(xml_path)
        image = images.get(os.path.splitext(os.path.basename(fname))[0]) or \
            images.get(os.path.splitext(os.path.basename(xml_path))[0])
        if image is None:
            continue
        w = float(tree.findtext("size/width") or 0)
        h = float(tree.findtext("size/height") or 0)
        if w <= 0 or h <= 0:
            import cv2
            img = cv2.imread(image)
            if img is None:
                continue
            h, w = img.shape[:2]
        boxes, skip = [], False
        for obj in tree.findall("object"):
            name = obj.findtext("name") or ""
            cls = mapper(name)
            if cls is None:
                skip = skip or mapper.is_weapon(name)
                continue
            bb = obj.find("bndbox")
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            boxes.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        if boxes and not skip:
            writer.add(source, image, boxes)


def download_roboflow(spec, dest_root):
    try:
        ws, project, version = spec.strip("/").split("/")
    except ValueError:
        raise SystemExit(f"--roboflow '{spec}': formato atteso workspace/progetto/versione")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Imposta ROBOFLOW_API_KEY (gratuita su roboflow.com, Settings > API Keys).")
    from roboflow import Roboflow
    location = os.path.join(dest_root, f"{ws}_{project}_{version}")
    if not os.path.isdir(location):
        print(f"\n[roboflow] download {spec} -> {location}")
        Roboflow(api_key=api_key).workspace(ws).project(project).version(int(version)).download(
            "yolov8", location=location)
    return location, f"rf_{project}"[:24]


def main():
    ap = argparse.ArgumentParser(description="Dataset YOLO coltelli/armi per TreeEyes")
    ap.add_argument("--out", default="datasets/weapon_knife")
    ap.add_argument("--classes", nargs="+", default=["knife", "gun"], choices=["knife", "gun"])
    ap.add_argument("--coco", type=int, default=0, help="max immagini COCO con coltello")
    ap.add_argument("--openimages", type=int, default=0, help="max immagini Open Images con coltello (e pistola)")
    ap.add_argument("--negatives", type=int, default=0, help="max immagini COCO di hard negative")
    ap.add_argument("--voc", nargs="*", default=[], help="cartelle Pascal VOC")
    ap.add_argument("--roboflow", nargs="*", default=[], help="workspace/progetto/versione")
    ap.add_argument("--extra", nargs="*", default=[], help="cartelle YOLO con data.yaml")
    ap.add_argument("--map", nargs="*", default=[], help='override mappatura: "Nome=knife" / "Nome=none"')
    ap.add_argument("--val", type=float, default=0.15, help="frazione di validazione")
    ap.add_argument("--clean", action="store_true", help="svuota --out prima di iniziare")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    overrides = dict(m.split("=", 1) for m in args.map)
    mapper = ClassMapper(args.classes, overrides)
    writer = DatasetWriter(args.out, args.classes, args.val)
    downloads = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_downloads")

    if args.coco:
        import_fiftyone(writer, mapper, "coco", "coco-2017", "train", ["knife"], args.coco)
    if args.openimages:
        oi_classes = OPENIMAGES_KNIFE + (OPENIMAGES_GUN if "gun" in args.classes else [])
        import_fiftyone(writer, mapper, "openimages", "open-images-v7", "train", oi_classes, args.openimages)
    if args.negatives:
        import_fiftyone(writer, mapper, "coco_neg", "coco-2017", "train", COCO_NEGATIVE_CLASSES,
                        args.negatives, negatives_of=["knife"])
    for i, root in enumerate(args.voc):
        import_voc(writer, mapper, root, f"voc{i}")
    for spec in args.roboflow:
        location, source = download_roboflow(spec, os.path.join(downloads, "roboflow"))
        import_yolo_dir(writer, mapper, location, source)
    for i, root in enumerate(args.extra):
        import_yolo_dir(writer, mapper, root, f"extra{i}_{os.path.basename(os.path.normpath(root))}"[:24])

    if not writer.names:
        raise SystemExit("Nessuna immagine importata: indica almeno una fonte (--coco, --voc, --extra, ...).")
    writer.finish()


if __name__ == "__main__":
    main()
