import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from rknnlite.api import RKNNLite

import config
from events import (
    build_mqtt_client,
    publish_event,
    start_heartbeat,
    notify_telegram,
    verify_handshake,
    upload_snapshot,
)
import pose_heuristics as ph
from hand_gesture import HelpGestureDetector
import hud
import vlm_confirm
from video_clip import ClipRecorder
from logger import logger
from rknn_backend import PoseModel, DetectModel, SimpleTracker, POSE_SKELETON

POSE_RKNN_PATH = getattr(config, "POSE_RKNN_PATH", "yolov8s-pose.rknn")
DETECT_RKNN_PATH = getattr(config, "DETECT_RKNN_PATH", "rknn_test/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn")
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


def open_capture(cam_cfg):
    source = cam_cfg["camera_index"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
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
        self._init_cap()

    def _init_cap(self):
        try:
            self.cap = open_capture(self.cam_cfg)
            self.consecutive_failures = 0
        except Exception as e:
            logger.error(f"[{self.cam_cfg.get('zone', '?')}] Inizializzazione capture fallita: {e}")
            self.cap = None

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._loop, daemon=True)
        self.worker_thread.start()
        return self

    def _loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._init_cap()
                continue

            ok, frame = self.cap.read()
            if not ok:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 10:
                    logger.warning(f"[{self.cam_cfg.get('zone', '?')}] Riconnessione webcam...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(2.0)
                else:
                    time.sleep(0.02)
                continue

            self.consecutive_failures = 0
            with self.lock:
                self.frame = frame
            self.new_frame_event.set()

    def read(self, timeout=1.0):
        if self.new_frame_event.wait(timeout=timeout):
            self.new_frame_event.clear()
            with self.lock:
                if self.frame is not None:
                    return True, self.frame.copy()
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

    def __init__(self, cam_cfg, mqtt_client):
        super().__init__(daemon=True)
        self.cam_cfg = cam_cfg
        self.device_id = cam_cfg["device_id"]
        self.zone = cam_cfg.get("zone", self.device_id)
        self.mqtt_client = mqtt_client

        # Modelli NPU PROPRI della pipeline, su NPU_CORE_AUTO: il runtime RKNN
        # bilancia le inferenze delle due telecamere sui 3 core del RK3588,
        # ottenendo vero parallelismo (niente lock che serializza su un core solo).
        logger.info(f"[{self.zone}] Caricamento modelli NPU dedicati (AUTO core)...")
        self.detect_model = DetectModel(DETECT_RKNN_PATH, core_mask=RKNNLite.NPU_CORE_AUTO, conf_thresh=config.CONF_THRESHOLD)
        self.pose_model = PoseModel(POSE_RKNN_PATH, core_mask=RKNNLite.NPU_CORE_AUTO, conf_thresh=config.CONF_THRESHOLD)

        self.capture = None
        self.gesture = HelpGestureDetector()
        self.clip = ClipRecorder(device_id=self.device_id)
        self.tracker = SimpleTracker()
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Cadenze inferenza (configurabili)
        self.pose_every = max(1, getattr(config, "POSE_PROCESS_EVERY", 1))
        self.detect_every = max(1, getattr(config, "DETECT_PROCESS_EVERY", 2))
        self.hands_every = max(1, getattr(config, "HANDS_PROCESS_EVERY", 3))

        self.consecutive_counts = {}
        self.last_event_time = {}
        self.crowd_start_time = None
        self.frame_idx = 0
        self.last_detections = []
        self.last_pose = []

        self._fps = 0.0
        self._last_fps_t = time.time()

        self.latest_display = None
        self.display_lock = threading.Lock()
        self.running = False

    def reset(self):
        self.last_event_time.clear()
        self.consecutive_counts.clear()
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
        with self.display_lock:
            if self.latest_display is None:
                return None
            return self.latest_display.copy()

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        self.capture = VideoCaptureThread(self.cam_cfg).start()

        while self.running:
            ok, raw_frame = self.capture.read(timeout=1.5)
            if not ok:
                logger.warning(f"[{self.zone}] Frame non disponibile o timeout cattura.")
                continue

            now = time.time()
            self.frame_idx += 1

            # 1. Aggiorna buffer circolare clip video con il frame raw pulito
            self.clip.update(raw_frame)

            detected_this_frame = set()
            person_count = 0

            # 2. Inferenze in parallelo su thread dedicati, con cadenze indipendenti.
            #    I risultati vengono riusati nei frame in cui l'inferenza è saltata.
            futures = {}
            if self.frame_idx % self.pose_every == 0:
                futures["pose"] = self.executor.submit(self.pose_model.infer, raw_frame)
            if self.frame_idx % self.detect_every == 0:
                futures["detect"] = self.executor.submit(self.detect_model.infer, raw_frame)
            if self.frame_idx % self.hands_every == 0:
                small_raw = cv2.resize(raw_frame, (raw_frame.shape[1] // 2, raw_frame.shape[0] // 2))
                futures["hands"] = self.executor.submit(self.gesture.detect, small_raw)

            if "pose" in futures:
                self.last_pose = futures["pose"].result()
            if "detect" in futures:
                self.last_detections = futures["detect"].result()
            gesture_triggered = futures["hands"].result() if "hands" in futures else False

            pose_results = self.last_pose
            detections = self.last_detections

            # 3. Frame di rendering dedicato all'HUD
            display_frame = raw_frame.copy()

            # --- Analisi Oggetti Pericolosi ---
            for det in detections:
                class_name = det["class_name"].strip()
                confidence = det["score"]
                if class_name == "person":
                    person_count += 1
                if class_name in config.DANGER_CLASS_MAP:
                    danger_type = config.DANGER_CLASS_MAP[class_name]
                    detected_this_frame.add(danger_type)
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    self.consecutive_counts[danger_type] = self.consecutive_counts.get(danger_type, 0) + 1
                    if self.consecutive_counts[danger_type] >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                        self._fire_event(now, danger_type, confidence, raw_frame)

            # --- Analisi Pose e Caduta ---
            tracked = self.tracker.update(pose_results)

            for det in tracked:
                track_id = det["track_id"]
                x1, y1, x2, y2 = map(int, det["box"])
                xyxy = det["box"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(display_frame, f"ID {track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                keypoints = [(kp[0], kp[1]) for kp in det["keypoints"]]
                kconf = [kp[2] for kp in det["keypoints"]]
                for kx, ky, kc in det["keypoints"]:
                    if kc > 0.5:
                        cv2.circle(display_frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)

                for sk in POSE_SKELETON:
                    p1 = det["keypoints"][sk[0] - 1]
                    p2 = det["keypoints"][sk[1] - 1]
                    if p1[2] > 0.5 and p2[2] > 0.5:
                        cv2.line(display_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 128, 0), 2)

                if ph.is_fallen(xyxy, keypoints, kconf):
                    detected_this_frame.add("persona_a_terra")
                    self.consecutive_counts["persona_a_terra"] = self.consecutive_counts.get("persona_a_terra", 0) + 1

            # Innesco allarme caduta validato
            if "persona_a_terra" in detected_this_frame and self.consecutive_counts.get("persona_a_terra", 0) >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                self._fire_event(now, "persona_a_terra", 1.0, raw_frame)

            # Reset contatori per pericoli non più visibili nel frame corrente
            for danger_type in list(self.consecutive_counts.keys()):
                if danger_type not in detected_this_frame:
                    self.consecutive_counts[danger_type] = 0

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

            hud.draw_progress_bar(display_frame, "Gesto aiuto", self.gesture.progress())
            hud.draw_panel(display_frame, [
                ("Persone", False, f"({person_count})"),
                ("Caduta", "persona_a_terra" in detected_this_frame, ""),
                ("Assembramento", self.crowd_start_time is not None, ""),
                ("Gesto aiuto", gesture_triggered, ""),
            ], recording=self.clip.is_recording())

            # Etichetta zona (in alto a destra della sotto-immagine) + FPS
            fh, fw = display_frame.shape[:2]
            cv2.putText(display_frame, f"{self.zone}", (fw - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"{self._fps:.0f} FPS", (fw - 120, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            with self.display_lock:
                self.latest_display = display_frame

            if getattr(config, "HEADLESS", False) and self.frame_idx % 150 == 0:
                logger.info(
                    f"[{self.zone}] Status: frame={self.frame_idx}, persone={person_count}, "
                    f"allarmi_attivi={list(detected_this_frame)}, clip_rec={self.clip.is_recording()}, fps={self._fps:.1f}"
                )

        # cleanup pipeline
        if self.capture is not None:
            self.capture.release()
        self.executor.shutdown(wait=False)
        self.gesture.close()
        try:
            self.detect_model.release()
            self.pose_model.release()
        except Exception:
            pass


def _placeholder(zone, width=1280, height=720):
    """Riquadro nero con etichetta quando una telecamera non ha ancora un frame."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, f"{zone}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "IN ATTESA SEGNALE...", (40, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def compose_grid(pipelines, target_h=720):
    """
    Compone i display delle telecamere affiancati orizzontalmente,
    normalizzando all'altezza target e separandoli con una sottile linea.
    """
    tiles = []
    for p in pipelines:
        frame = p.get_display()
        if frame is None:
            frame = _placeholder(p.zone)
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

    # 1. Handshake di autenticazione per OGNI telecamera
    for cam in cameras:
        verify_handshake(cam["device_id"])

    # 2. Connessione broker MQTT (una connessione condivisa)
    mqtt_client = build_mqtt_client()

    # 3. Avvio Heartbeat periodico per OGNI telecamera (ognuna sul proprio topic)
    heartbeat_services = [start_heartbeat(mqtt_client, device_id=cam["device_id"]) for cam in cameras]

    os.makedirs("snapshots", exist_ok=True)
    os.makedirs(getattr(config, "VIDEO_CLIP_DIR", "clips"), exist_ok=True)

    # Pulizia iniziale storage
    cleanup_old_files("snapshots", getattr(config, "MAX_SNAPSHOTS_COUNT", 500))
    cleanup_old_files(getattr(config, "VIDEO_CLIP_DIR", "clips"), getattr(config, "MAX_CLIPS_COUNT", 100))
    last_cleanup_time = time.time()

    # 4. Avvio pipeline (una per telecamera, con modelli NPU dedicati)
    pipelines = [CameraPipeline(cam, mqtt_client) for cam in cameras]
    for p in pipelines:
        p.start()

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
    try:
        while True:
            # Pulizia periodica storage
            now = time.time()
            if now - last_cleanup_time > getattr(config, "STORAGE_CLEANUP_INTERVAL_SECONDS", 1800):
                cleanup_old_files("snapshots", getattr(config, "MAX_SNAPSHOTS_COUNT", 500))
                cleanup_old_files(getattr(config, "VIDEO_CLIP_DIR", "clips"), getattr(config, "MAX_CLIPS_COUNT", 100))
                last_cleanup_time = now

            if not getattr(config, "HEADLESS", False):
                grid = compose_grid(pipelines)
                cv2.imshow(window_name, grid)

                if not first_shown:
                    first_shown = True
                    if monitor_x != 0 or monitor_y != 0:
                        cv2.moveWindow(window_name, monitor_x, monitor_y)
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                key = cv2.waitKey(1) & 0xFF
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
