import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import config
from events import (
    build_mqtt_client,
    publish_event,
    publish_presence,
    start_heartbeat,
    start_handshake_retry,
    notify_telegram,
    upload_snapshot,
)
from hand_gesture import HelpGestureDetector
import hud
import vlm_confirm
from video_clip import ClipRecorder
from logger import logger
from rknn_backend import (PoseModel, DetectModel, WeaponModel, SimpleTracker, POSE_SKELETON, WEAPON_CLASSES,
                          preprocess, core_mask_from)

POSE_RKNN_PATH = getattr(config, "POSE_RKNN_PATH", "yolov8s-pose.rknn")
DETECT_RKNN_PATH = getattr(config, "DETECT_RKNN_PATH", "rknn_test/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn")
WEAPON_RKNN_PATH = getattr(config, "WEAPON_RKNN_PATH", "weapon_threat.rknn")
WEAPON_CLASS_MAP = getattr(config, "WEAPON_CLASS_MAP", {})
ALERT_MIN_CONF = getattr(config, "ALERT_MIN_CONF", 0.90)
RKNN_TARGET = getattr(config, "RKNN_TARGET", "rk3588")


def cleanup_old_files(directory: str, max_files: int):
    """
    Rimuove i file più vecchi nella cartella specificata se il totale supera max_files,
    evitando la saturazione dello storage locale su SD / eMMC.
    """
    if not os.path.exists(directory):
        return
    try:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        if len(files) > max_files:
            files.sort(key=os.path.getmtime)
            num_to_delete = len(files) - max_files
            for f in files[:num_to_delete]:
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Impossibile rimuovere vecchio file {f}: {e}")
            logger.info(f"Pulizia storage '{directory}': rimossi {num_to_delete} file obsoleti.")
    except Exception as e:
        logger.warning(f"Errore durante pulizia storage in {directory}: {e}")


def _process_event_async(device_id, zone, danger_type, confidence, snapshot_path, raw_frame, mqtt_client):
    """
    Gestisce l'upload dello snapshot HTTP e l'invio dell'allarme MQTT in background,
    a nome della telecamera (device_id/zona) che ha generato l'evento,
    utilizzando il frame raw pulito (senza bounding box o artefatti HUD).
    """
    # 1. Carica snapshot JPG al backend per ottenere l'URL pubblico (frame_url)
    frame_url = upload_snapshot(raw_frame if raw_frame is not None else snapshot_path, device_id=device_id)

    description = ""
    # 2. VLM confirmation se abilitato (saltato per 'segnale_aiuto' per garantire reattività immediata)
    if config.VLM_ENABLED and danger_type != "segnale_aiuto":
        confirmed, description = vlm_confirm.confirm(snapshot_path, danger_type)
        if not confirmed:
            logger.info(f"[{device_id}] Evento {danger_type} scartato dal VLM (falso positivo): {description}")
            return

    # 3. Pubblica allarme istantaneo via MQTT sul topic parco/<device_id>/camera con frame_url
    publish_event(
        mqtt_client=mqtt_client,
        danger_type=danger_type,
        confidence=confidence,
        frame_url=frame_url,
        description=description,
        device_id=device_id,
    )

    # 4. Notifica Telegram opzionale
    label = f"{danger_type} - {description}" if description else danger_type
    notify_telegram(f"⚠️ TreeEyes [{zone}] - {label} (Camera: {device_id})")


# --- Assegnazione automatica telecamere ("camera_index": "auto") ----------
# Una zona "auto" prende la prima telecamera USB libera in /dev/v4l/by-id,
# di qualunque modello. Le zone "auto" vengono servite nell'ordine di
# config.CAMERAS: una zona non prende una telecamera finché quelle prima di
# lei ne sono senza, così con una sola telecamera si accende sempre la prima.
# I path scritti esplicitamente in config non vengono mai assegnati in automatico.
V4L_BY_ID_DIR = "/dev/v4l/by-id"
_auto_lock = threading.Lock()
_auto_zones = []        # zone "auto" in ordine di priorità
_auto_claims = {}       # path telecamera -> zona che la sta usando
_fixed_sources = set()  # path espliciti in config


def init_auto_cameras(cameras):
    _auto_zones[:] = [c.get("zone", c["device_id"]) for c in cameras if c["camera_index"] == "auto"]
    _fixed_sources.clear()
    _fixed_sources.update(c["camera_index"] for c in cameras if c["camera_index"] != "auto")


def _claim_auto_camera(zone):
    with _auto_lock:
        for z in _auto_zones:
            if z == zone:
                break
            if z not in _auto_claims.values():
                return None
        try:
            names = sorted(os.listdir(V4L_BY_ID_DIR))
        except OSError:
            names = []
        # Solo il nodo di cattura (index0): index1 è il nodo metadati UVC
        for name in names:
            path = os.path.join(V4L_BY_ID_DIR, name)
            if name.endswith("-video-index0") and path not in _fixed_sources and path not in _auto_claims:
                _auto_claims[path] = zone
                return path
    return None


def _release_auto_camera(path):
    with _auto_lock:
        _auto_claims.pop(path, None)


def open_capture(cam_cfg):
    source = cam_cfg["camera_index"]
    # Path V4L2 assente = telecamera scollegata: inutile tentare l'apertura
    if isinstance(source, str) and source.startswith("/dev/") and not os.path.exists(source):
        raise RuntimeError(f"Dispositivo non presente: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Impossibile aprire la sorgente video: {source}")

    # Imposta formato MJPG se disponibile (riduce il carico del bus USB)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    width = cam_cfg.get("width", getattr(config, "CAMERA_WIDTH", None))
    height = cam_cfg.get("height", getattr(config, "CAMERA_HEIGHT", None))
    fps = cam_cfg.get("fps", getattr(config, "CAMERA_FPS", None))

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        f"[{cam_cfg.get('zone', '?')}] Telecamera '{cam_cfg['device_id']}' configurata: "
        f"{actual_w}x{actual_h} @ {actual_fps:.1f} FPS ({source})"
    )

    return cap


class VideoCaptureThread:
    """
    Thread di acquisizione video non-bloccante a latenza zero.
    Svuota costantemente il buffer hardware del driver V4L2 mantenendo
    disponibile esclusivamente il frame più recente per la pipeline.
    Un'istanza per telecamera.
    """
    def __init__(self, cam_cfg):
        self.cam_cfg = cam_cfg
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.consecutive_failures = 0
        self.worker_thread = None
        # True mentre la telecamera è assente: evita di ripetere lo stesso log
        # a ogni tentativo di riconnessione (uno al secondo).
        self.waiting = False
        # Sorgente effettivamente aperta (per "auto" il path assegnato)
        self.source = None
        self.auto = cam_cfg["camera_index"] == "auto"
        self.last_frame_time = 0.0
        self._init_cap()

    def _init_cap(self):
        zone = self.cam_cfg.get("zone", "?")
        cfg = self.cam_cfg
        try:
            if self.auto:
                path = _claim_auto_camera(zone)
                if path is None:
                    raise RuntimeError(f"nessuna telecamera libera in {V4L_BY_ID_DIR}")
                cfg = dict(cfg, camera_index=path)
            self.source = cfg["camera_index"]
            self.cap = open_capture(cfg)
            self.consecutive_failures = 0
            if self.waiting:
                logger.info(f"[{zone}] Telecamera ricollegata.")
            self.waiting = False
        except Exception as e:
            if not self.waiting:
                logger.error(f"[{zone}] Telecamera non disponibile ({e}): in attesa che venga collegata...")
            self.waiting = True
            self.cap = None
            self._release_source()

    def _release_source(self):
        if self.auto and self.source is not None:
            _release_auto_camera(self.source)
            self.source = None

    def _drop_cap(self):
        """Rilascia la capture e scarta l'ultimo frame (telecamera persa)."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self._release_source()
        with self.lock:
            self.frame = None
        self.new_frame_event.clear()

    def _device_missing(self):
        source = self.source
        return isinstance(source, str) and source.startswith("/dev/") and not os.path.exists(source)

    def has_signal(self, max_age=3.0):
        """True se la telecamera ha mandato un frame negli ultimi max_age secondi."""
        return time.time() - self.last_frame_time < max_age

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._loop, daemon=True)
        self.worker_thread.start()
        return self

    def _loop(self):
        zone = self.cam_cfg.get("zone", "?")
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._init_cap()
                continue

            try:
                ok, frame = self.cap.read()
            except Exception as e:
                logger.error(f"[{zone}] Errore lettura frame: {e}")
                ok, frame = False, None

            if not ok or frame is None or frame.size == 0:
                self.consecutive_failures += 1
                # Device sparito da /dev = cavo staccato: inutile attendere altri errori
                if self.consecutive_failures >= 10 or self._device_missing():
                    if not self.waiting:
                        logger.warning(f"[{zone}] Telecamera scollegata: in attesa che venga ricollegata...")
                    self.waiting = True
                    self._drop_cap()
                    time.sleep(1.0)
                else:
                    time.sleep(0.02)
                continue

            self.consecutive_failures = 0
            self.last_frame_time = time.time()
            with self.lock:
                self.frame = frame
            self.new_frame_event.set()

    def read(self, timeout=1.0):
        if self.new_frame_event.wait(timeout=timeout):
            self.new_frame_event.clear()
            with self.lock:
                # Nessuna copia: il thread di cattura sostituisce il frame con un
                # array nuovo a ogni lettura e non modifica mai quello consegnato.
                if self.frame is not None:
                    return True, self.frame
        return False, None

    def release(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._release_source()


class CameraPipeline(threading.Thread):
    """
    Pipeline completa per UNA telecamera, in un thread dedicato.

    Ogni pipeline ha stato indipendente (contatori, cooldown, tracker, clip,
    rilevatore gesto) e pubblica gli allarmi con il proprio device_id.
    I modelli NPU (detect/pose) sono CONDIVISI tra le pipeline e protetti da lock
    per serializzare l'accesso all'acceleratore RK3588.

    La pipeline non chiama mai OpenCV imshow: si limita a produrre l'ultimo frame
    di rendering (con HUD ed etichette), che il thread principale compone e mostra.
    """

    def __init__(self, cam_cfg, mqtt_client, index=0):
        super().__init__(daemon=True)
        self.cam_cfg = cam_cfg
        self.device_id = cam_cfg["device_id"]
        self.zone = cam_cfg.get("zone", self.device_id)
        self.mqtt_client = mqtt_client

        # Modelli NPU PROPRI della pipeline, ciascuno sul core indicato in
        # config.NPU_CORES (una voce per telecamera): inferenze delle due
        # telecamere in parallelo sui 3 core del RK3588, senza lock.
        layouts = getattr(config, "NPU_CORES", [])
        cores = layouts[index] if index < len(layouts) else {}
        logger.info(f"[{self.zone}] Caricamento modelli NPU dedicati (core: {cores or 'auto'})...")
        self.detect_model = DetectModel(DETECT_RKNN_PATH, core_mask=core_mask_from(cores.get("detect")), conf_thresh=config.CONF_THRESHOLD)
        self.pose_model = PoseModel(POSE_RKNN_PATH, core_mask=core_mask_from(cores.get("pose")), conf_thresh=config.CONF_THRESHOLD)

        # Modello armi dedicato (Gun/explosion/grenade/knife). Opzionale: se il
        # file .rknn manca la pipeline prosegue con i soli modelli COCO/pose.
        self.weapon_model = None
        if getattr(config, "WEAPON_ENABLED", True):
            try:
                # Sotto ALERT_MIN_CONF le armi vengono comunque scartate: filtrarle
                # già nel post-processing evita NMS e risultati inutili.
                self.weapon_model = WeaponModel(
                    WEAPON_RKNN_PATH,
                    core_mask=core_mask_from(cores.get("weapon")),
                    conf_thresh=max(getattr(config, "WEAPON_CONF_THRESHOLD", 0.35), ALERT_MIN_CONF),
                    classes=getattr(config, "WEAPON_CLASSES", WEAPON_CLASSES),
                )
            except Exception as e:
                logger.error(f"[{self.zone}] Modello armi '{WEAPON_RKNN_PATH}' non caricato ({e}): rilevamento armi dedicato disattivato.")

        self.capture = None
        self.gesture = HelpGestureDetector()
        self.clip = ClipRecorder(device_id=self.device_id)
        self.tracker = SimpleTracker()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Cadenze inferenza (configurabili)
        self.pose_every = max(1, getattr(config, "POSE_PROCESS_EVERY", 1))
        self.detect_every = max(1, getattr(config, "DETECT_PROCESS_EVERY", 2))
        self.weapon_every = max(1, getattr(config, "WEAPON_PROCESS_EVERY", 2))
        self.hands_every = max(1, getattr(config, "HANDS_PROCESS_EVERY", 3))
        self.hands_only_with_person = getattr(config, "HANDS_ONLY_WITH_PERSON", True)
        # In headless nessuno guarda il video: niente box/HUD da disegnare
        self.draw = not getattr(config, "HEADLESS", False)

        # Conferma a tempo delle armi: danger_type -> istante di primo/ultimo avvistamento
        self.danger_first_seen = {}
        self.danger_last_seen = {}
        self.last_event_time = {}
        self.crowd_start_time = None
        self.presence_frames = 0
        self.last_presence_time = 0.0
        self.frame_idx = 0
        self.last_detections = []
        self.last_weapons = []
        self.last_pose = []

        self._fps = 0.0
        self._last_fps_t = time.time()

        self.latest_display = None
        self.display_version = 0   # incrementato a ogni nuovo frame di display
        self.display_lock = threading.Lock()
        self.running = False
        self.signal_lost = False

    def reset(self):
        self.last_event_time.clear()
        self.danger_first_seen.clear()
        self.danger_last_seen.clear()
        self.crowd_start_time = None
        logger.info(f"[{self.zone}] Reset manuale cooldown/contatori.")

    def _fire_event(self, now, danger_type, confidence, raw_frame):
        cooldown = getattr(config, "EVENT_COOLDOWN_MAP", {}).get(danger_type, config.EVENT_COOLDOWN_SECONDS)
        if now - self.last_event_time.get(danger_type, 0) < cooldown:
            return False

        self.last_event_time[danger_type] = now
        os.makedirs("snapshots", exist_ok=True)
        snapshot_path = f"snapshots/{self.device_id}_{danger_type}_{int(now)}.jpg"
        frame_copy = raw_frame.copy()
        cv2.imwrite(snapshot_path, frame_copy)

        # Avvia buffer di registrazione video (post-evento) con frame pulito
        self.clip.trigger(danger_type)

        threading.Thread(
            target=_process_event_async,
            args=(self.device_id, self.zone, danger_type, confidence, snapshot_path, frame_copy, self.mqtt_client),
            daemon=True,
        ).start()
        logger.info(f"[{self.zone}] Evento innescato: {danger_type} (conf={confidence:.2f})")
        return True

    def get_display(self):
        # Senza copia: ogni frame pubblicato è un array nuovo, mai modificato dopo
        with self.display_lock:
            return self.latest_display

    def stop(self):
        self.running = False

    def camera_online(self):
        """True se la telecamera della zona sta mandando frame."""
        return self.capture is not None and self.capture.has_signal()

    def _on_signal_lost(self):
        """
        Telecamera assente/scollegata: la pipeline resta viva in attesa dei frame.
        Azzera lo stato (così al ritorno non scattano allarmi su dati vecchi) e
        libera il display, che torna al riquadro "IN ATTESA SEGNALE".
        """
        if self.signal_lost:
            return
        self.signal_lost = True
        logger.warning(f"[{self.zone}] Nessun segnale video: pipeline in attesa della telecamera.")
        with self.display_lock:
            self.latest_display = None
            self.display_version += 1
        self.danger_first_seen.clear()
        self.danger_last_seen.clear()
        self.crowd_start_time = None
        self.presence_frames = 0
        self.last_detections = []
        self.last_weapons = []
        self.last_pose = []
        self._fps = 0.0
        self.clip.flush()

    def run(self):
        self.running = True
        self.capture = VideoCaptureThread(self.cam_cfg).start()

        # Un errore su un frame (o su una telecamera) non deve fermare il thread:
        # si logga e il ciclo riparte.
        while self.running:
            try:
                self._run_loop()
            except Exception:
                logger.exception(f"[{self.zone}] Errore nella pipeline, il ciclo riparte.")
                time.sleep(0.5)

        self._cleanup()

    def _run_loop(self):
        while self.running:
            ok, raw_frame = self.capture.read(timeout=1.5)
            if not ok:
                self._on_signal_lost()
                continue
            if self.signal_lost:
                self.signal_lost = False
                logger.info(f"[{self.zone}] Segnale video ripristinato.")

            now = time.time()
            self.frame_idx += 1

            # 1. Aggiorna buffer circolare clip video con il frame raw pulito
            #    (raw_frame non viene più modificato: niente copia)
            self.clip.update(raw_frame, copy=False)

            detected_this_frame = set()
            person_count = 0

            # 2. Inferenze in parallelo su thread dedicati, con cadenze indipendenti.
            #    I risultati vengono riusati nei frame in cui l'inferenza è saltata.
            #    Letterbox + RGB calcolati una volta sola e condivisi dai modelli NPU.
            futures = {}
            run_pose = self.frame_idx % self.pose_every == 0
            run_detect = self.frame_idx % self.detect_every == 0
            # Sfalsata di un frame rispetto a COCO per non sommare i due carichi NPU
            run_weapon = self.weapon_model is not None and (self.frame_idx + 1) % self.weapon_every == 0
            prep = preprocess(raw_frame, self.pose_model.input_size) if (run_pose or run_detect or run_weapon) else None
            if run_pose:
                futures["pose"] = self.executor.submit(self.pose_model.infer, raw_frame, prep)
            if run_detect:
                futures["detect"] = self.executor.submit(self.detect_model.infer, raw_frame, prep)
            if run_weapon:
                futures["weapon"] = self.executor.submit(self.weapon_model.infer, raw_frame, prep)
            if self.frame_idx % self.hands_every == 0:
                # MediaPipe (CPU) solo se nel frame precedente c'era qualcuno
                someone = bool(self.last_pose) or any(d["class_name"] == "person" for d in self.last_detections)
                if someone or not self.hands_only_with_person:
                    small_raw = cv2.resize(raw_frame, (raw_frame.shape[1] // 2, raw_frame.shape[0] // 2))
                    futures["hands"] = self.executor.submit(self.gesture.detect, small_raw, False)
                else:
                    self.gesture.idle()

            if "pose" in futures:
                self.last_pose = futures["pose"].result()
            if "detect" in futures:
                self.last_detections = futures["detect"].result()
            if "weapon" in futures:
                self.last_weapons = futures["weapon"].result()
            gesture_triggered = futures["hands"].result() if "hands" in futures else False

            pose_results = self.last_pose
            detections = self.last_detections

            # 3. Frame di rendering dedicato all'HUD (solo con finestra a schermo)
            draw = self.draw
            display_frame = raw_frame.copy() if draw else None

            # --- Analisi Oggetti Pericolosi (COCO + modello armi dedicato) ---
            # danger_type -> confidenza migliore nel frame: ogni tipo conta una
            # sola volta per frame anche se visto da entrambi i modelli.
            dangers = {}
            danger_dets = []
            for det in detections:
                class_name = det["class_name"].strip()
                if class_name == "person":
                    person_count += 1
                if class_name in config.DANGER_CLASS_MAP and det["score"] > ALERT_MIN_CONF:
                    danger_dets.append((config.DANGER_CLASS_MAP[class_name], class_name, det))
            for det in self.last_weapons:
                class_name = det["class_name"]
                if class_name in WEAPON_CLASS_MAP and det["score"] > ALERT_MIN_CONF:
                    danger_dets.append((WEAPON_CLASS_MAP[class_name], class_name, det))

            for danger_type, class_name, det in danger_dets:
                confidence = det["score"]
                dangers[danger_type] = max(dangers.get(danger_type, 0.0), confidence)
                if not draw:
                    continue
                x1, y1, x2, y2 = map(int, det["box"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"{class_name} {confidence:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # L'arma deve restare visibile per DANGER_CONFIRM_SECONDS prima di
            # segnalare: un riconoscimento isolato non basta (falsi positivi).
            # Buchi più brevi di DANGER_CONFIRM_GAP_SECONDS non azzerano il conteggio.
            confirm = getattr(config, "DANGER_CONFIRM_SECONDS", 3.0)
            gap = getattr(config, "DANGER_CONFIRM_GAP_SECONDS", 0.5)
            for danger_type in list(self.danger_first_seen.keys()):
                if danger_type not in dangers and now - self.danger_last_seen[danger_type] > gap:
                    del self.danger_first_seen[danger_type]
                    del self.danger_last_seen[danger_type]

            danger_held = {}
            for danger_type, confidence in dangers.items():
                detected_this_frame.add(danger_type)
                self.danger_last_seen[danger_type] = now
                held = now - self.danger_first_seen.setdefault(danger_type, now)
                danger_held[danger_type] = held
                if held >= confirm:
                    self._fire_event(now, danger_type, confidence, raw_frame)

            # --- Analisi Pose ---
            tracked = self.tracker.update(pose_results)

            for det in (tracked if draw else ()):
                track_id = det["track_id"]
                x1, y1, x2, y2 = map(int, det["box"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(display_frame, f"ID {track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                for kx, ky, kc in det["keypoints"]:
                    if kc > 0.5:
                        cv2.circle(display_frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)

                for sk in POSE_SKELETON:
                    p1 = det["keypoints"][sk[0] - 1]
                    p2 = det["keypoints"][sk[1] - 1]
                    if p1[2] > 0.5 and p2[2] > 0.5:
                        cv2.line(display_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 128, 0), 2)

            # --- Presenza (accensione luci di zona) ---
            # Persona vista dal detector COCO oppure dal modello pose: basta uno dei due.
            people = max(person_count, len(tracked))
            self.presence_frames = self.presence_frames + 1 if people > 0 else 0
            if (
                getattr(config, "PRESENCE_ENABLED", True)
                and self.presence_frames >= getattr(config, "PRESENCE_MIN_FRAMES", 3)
                and now - self.last_presence_time >= getattr(config, "PRESENCE_REPUBLISH_SECONDS", 20)
            ):
                self.last_presence_time = now
                publish_presence(self.mqtt_client, people, device_id=self.device_id)

            # --- Assembramento ---
            if person_count >= config.CROWD_COUNT_THRESHOLD:
                if self.crowd_start_time is None:
                    self.crowd_start_time = now
                elif now - self.crowd_start_time >= config.CROWD_SECONDS_THRESHOLD:
                    self._fire_event(now, "assembramento", 1.0, raw_frame)
            else:
                self.crowd_start_time = None

            # --- Skeleton + gesto aiuto ---
            if gesture_triggered:
                self._fire_event(now, "segnale_aiuto", 1.0, raw_frame)

            # --- HUD ---
            # FPS smussato
            dt = now - self._last_fps_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else (self._fps * 0.9 + inst * 0.1)
            self._last_fps_t = now

            if self.frame_idx % 300 == 0:
                npu = ", ".join(
                    f"{name} {m.npu_ms:.1f}ms"
                    for name, m in (("pose", self.pose_model), ("detect", self.detect_model), ("weapon", self.weapon_model))
                    if m is not None
                )
                logger.info(f"[{self.zone}] Prestazioni: {self._fps:.1f} FPS, NPU ms medi: {npu}")

            if draw:
                self._draw_hud(display_frame, person_count, dangers, danger_held, gesture_triggered)
                with self.display_lock:
                    self.latest_display = display_frame
                    self.display_version += 1

            if getattr(config, "HEADLESS", False) and self.frame_idx % 150 == 0:
                logger.info(
                    f"[{self.zone}] Status: frame={self.frame_idx}, persone={person_count}, "
                    f"allarmi_attivi={list(detected_this_frame)}, clip_rec={self.clip.is_recording()}, fps={self._fps:.1f}"
                )

    def _draw_hud(self, display_frame, person_count, dangers, danger_held, gesture_triggered):
        hud.draw_progress_bar(display_frame, "Gesto aiuto", self.gesture.progress())
        hud.draw_panel(display_frame, [
            ("Persone", False, f"({person_count})"),
            ("Arma", bool(dangers), "(" + ", ".join(f"{t} {s:.1f}s" for t, s in danger_held.items()) + ")" if dangers else ""),
            ("Assembramento", self.crowd_start_time is not None, ""),
            ("Gesto aiuto", gesture_triggered, ""),
        ], recording=self.clip.is_recording())

        # Etichetta zona (in alto a destra della sotto-immagine) + FPS
        fh, fw = display_frame.shape[:2]
        cv2.putText(display_frame, f"{self.zone}", (fw - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"{self._fps:.0f} FPS", (fw - 120, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    def _cleanup(self):
        if self.capture is not None:
            self.capture.release()
        self.executor.shutdown(wait=False)
        self.gesture.close()
        try:
            self.detect_model.release()
            self.pose_model.release()
            if self.weapon_model is not None:
                self.weapon_model.release()
        except Exception:
            pass


def _placeholder(zone, width=1280, height=720):
    """Riquadro nero con etichetta quando una telecamera non ha ancora un frame."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, f"{zone}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "IN ATTESA SEGNALE...", (40, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def compose_grid(pipelines, target_h=None):
    """
    Compone i display delle telecamere affiancati orizzontalmente,
    normalizzando all'altezza target e separandoli con una sottile linea.
    Senza target_h si usa l'altezza nativa dei frame: nessun ingrandimento su
    CPU, la scala a schermo intero la fa già la finestra.
    """
    # Solo le zone con segnale: con una telecamera sola si vede a tutto schermo.
    # Se nessuna ha segnale, un riquadro di attesa per ogni zona.
    frames = [f for f in (p.get_display() for p in pipelines) if f is not None]
    if not frames:
        frames = [_placeholder(p.zone) for p in pipelines]
    if target_h is None:
        target_h = max(f.shape[0] for f in frames)

    tiles = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h != target_h:
            scale = target_h / h
            frame = cv2.resize(frame, (int(w * scale), target_h))
        tiles.append(frame)

    # separatore verticale
    sep = np.full((target_h, 4, 3), 60, dtype=np.uint8)
    composed = []
    for i, t in enumerate(tiles):
        if i > 0:
            composed.append(sep)
        composed.append(t)
    return cv2.hconcat(composed) if composed else _placeholder("TreeEyes")


def main():
    cameras = getattr(config, "CAMERAS", None)
    if not cameras:
        raise RuntimeError("Nessuna telecamera definita in config.CAMERAS")

    init_auto_cameras(cameras)

    # 1. Connessione broker MQTT (una connessione condivisa). Se il broker non
    #    è ancora raggiungibile (rete non pronta all'accensione) si ritenta in background.
    mqtt_client = build_mqtt_client(retry=True)

    os.makedirs("snapshots", exist_ok=True)
    os.makedirs(getattr(config, "VIDEO_CLIP_DIR", "clips"), exist_ok=True)

    # Pulizia iniziale storage
    cleanup_old_files("snapshots", getattr(config, "MAX_SNAPSHOTS_COUNT", 500))
    cleanup_old_files(getattr(config, "VIDEO_CLIP_DIR", "clips"), getattr(config, "MAX_CLIPS_COUNT", 100))
    last_cleanup_time = time.time()

    # 2. Avvio pipeline (una per telecamera, con modelli NPU dedicati)
    pipelines = [CameraPipeline(cam, mqtt_client, index=i) for i, cam in enumerate(cameras)]
    for p in pipelines:
        p.start()

    # 3. Handshake e heartbeat per zona, attivi solo mentre la sua telecamera
    #    manda frame: il backend vede online solo le zone che funzionano davvero.
    stop_event = threading.Event()
    for p in pipelines:
        start_handshake_retry(p.device_id, stop_event, ready=p.camera_online)
    heartbeat_services = [
        start_heartbeat(mqtt_client, device_id=p.device_id, is_active=p.camera_online) for p in pipelines
    ]

    window_name = getattr(config, "WINDOW_NAME", "TreeEyes - Vision Node")
    is_fullscreen = getattr(config, "WINDOW_FULLSCREEN", True)
    monitor_x = getattr(config, "WINDOW_MONITOR_X", 0)
    monitor_y = getattr(config, "WINDOW_MONITOR_Y", 0)

    if not getattr(config, "HEADLESS", False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if monitor_x != 0 or monitor_y != 0:
            cv2.moveWindow(window_name, monitor_x, monitor_y)
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logger.info(
        f"Avvio completato con {len(pipelines)} telecamere "
        f"({', '.join(c['zone'] for c in cameras)}). "
        f"HEADLESS={getattr(config, 'HEADLESS', False)}, FULLSCREEN={is_fullscreen}. "
        f"Controlli: 'q'/ESC esci, 'f' toggle fullscreen, 'r' reset allarmi."
    )

    first_shown = False
    shown_versions = None
    last_show = 0.0
    min_show_interval = 1.0 / max(1, getattr(config, "DISPLAY_MAX_FPS", 30))
    try:
        while True:
            # Pulizia periodica storage
            now = time.time()
            if now - last_cleanup_time > getattr(config, "STORAGE_CLEANUP_INTERVAL_SECONDS", 1800):
                cleanup_old_files("snapshots", getattr(config, "MAX_SNAPSHOTS_COUNT", 500))
                cleanup_old_files(getattr(config, "VIDEO_CLIP_DIR", "clips"), getattr(config, "MAX_CLIPS_COUNT", 100))
                last_cleanup_time = now

            if not getattr(config, "HEADLESS", False):
                # Ricomposizione solo se una telecamera ha un frame nuovo (e al massimo
                # a DISPLAY_MAX_FPS): prima il ciclo girava a vuoto occupando un core.
                versions = tuple(p.display_version for p in pipelines)
                if versions != shown_versions and now - last_show >= min_show_interval:
                    shown_versions = versions
                    last_show = now
                    grid = compose_grid(pipelines)
                    cv2.imshow(window_name, grid)

                if not first_shown:
                    first_shown = True
                    if monitor_x != 0 or monitor_y != 0:
                        cv2.moveWindow(window_name, monitor_x, monitor_y)
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    for p in pipelines:
                        p.reset()
                if key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                    if not is_fullscreen:
                        if monitor_x != 0 or monitor_y != 0:
                            cv2.moveWindow(window_name, monitor_x, monitor_y)
                        cv2.resizeWindow(window_name, 1600, 600)
                    logger.info(f"Fullscreen: {'ATTIVATO' if is_fullscreen else 'DISATTIVATO'}")
            else:
                time.sleep(0.05)

    finally:
        stop_event.set()
        for p in pipelines:
            p.stop()
        for p in pipelines:
            p.join(timeout=2.0)
        if not getattr(config, "HEADLESS", False):
            cv2.destroyAllWindows()
        for hb in heartbeat_services:
            if hb is not None:
                hb.stop()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
