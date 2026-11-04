"""
Raccolta automatica di campioni per riaddestrare il modello armi.

Quando un modello vede un'arma candidata (anche sotto la soglia di allarme)
salva il frame RAW e un file di etichette YOLO già compilato (pseudo-etichetta),
nella struttura che prepare_weapon_dataset.py importa direttamente:

    weapon_samples/
        data.yaml
        images/<device>_<ts>.jpg
        labels/<device>_<ts>.txt     # "classe cx cy w h" normalizzati

Prima di usarli per il training vanno RIVISTI (es. con CVAT, Roboflow o
labelImg): correggere i box sbagliati e cancellare quelli falsi. I falsi
positivi con il file .txt svuotato sono preziosi: insegnano al modello cosa NON
è un coltello (hard negative).
"""
import os
import threading

import cv2

import config
from logger import logger

SAMPLE_CLASSES = ("knife", "gun")
# alert_type -> classe del dataset (le altre armi non vengono salvate)
_ALERT_TO_CLASS = {"coltello": 0, "arma_da_fuoco": 1}


class SampleCollector:
    def __init__(self, device_id):
        self.device_id = device_id
        self.enabled = getattr(config, "WEAPON_SAMPLES_ENABLED", False)
        self.root = getattr(config, "WEAPON_SAMPLES_DIR", "weapon_samples")
        self.min_conf = getattr(config, "WEAPON_SAMPLES_MIN_CONF", 0.25)
        self.interval = getattr(config, "WEAPON_SAMPLES_INTERVAL_SECONDS", 2.0)
        self._last = 0.0
        if self.enabled:
            os.makedirs(os.path.join(self.root, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "labels"), exist_ok=True)
            data_yaml = os.path.join(self.root, "data.yaml")
            if not os.path.exists(data_yaml):
                with open(data_yaml, "w") as fh:
                    fh.write(f"nc: {len(SAMPLE_CLASSES)}\nnames: {list(SAMPLE_CLASSES)}\n")

    def maybe_save(self, now, frame, candidates):
        """
        candidates: lista di (alert_type, score, box[x1,y1,x2,y2]) di questo frame.
        Salva al massimo un frame ogni WEAPON_SAMPLES_INTERVAL_SECONDS, in background.
        """
        if not self.enabled or now - self._last < self.interval:
            return
        rows = [(_ALERT_TO_CLASS[t], box) for t, s, box in candidates
                if t in _ALERT_TO_CLASS and s >= self.min_conf]
        if not rows:
            return
        self._last = now
        # Il frame raw non viene mai modificato dalla pipeline: niente copia
        threading.Thread(target=self._write, args=(now, frame, rows), daemon=True).start()

    def _write(self, now, frame, rows):
        h, w = frame.shape[:2]
        name = f"{self.device_id}_{int(now * 1000)}"
        lines = []
        for cls, (x1, y1, x2, y2) in rows:
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        try:
            cv2.imwrite(os.path.join(self.root, "images", name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(self.root, "labels", name + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
        except Exception as e:
            logger.warning(f"[{self.device_id}] Salvataggio campione armi fallito: {e}")


def cleanup_samples(root, max_count):
    """Tiene al massimo max_count campioni (immagine + etichetta), eliminando i più vecchi."""
    img_dir = os.path.join(root, "images")
    if not os.path.isdir(img_dir):
        return
    try:
        imgs = sorted((os.path.join(img_dir, f) for f in os.listdir(img_dir)), key=os.path.getmtime)
        for img in imgs[:max(0, len(imgs) - max_count)]:
            stem = os.path.splitext(os.path.basename(img))[0]
            for path in (img, os.path.join(root, "labels", stem + ".txt")):
                if os.path.exists(path):
                    os.remove(path)
    except Exception as e:
        logger.warning(f"Pulizia campioni armi in {root} fallita: {e}")
