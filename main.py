import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
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
import hand_gesture as hg
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


def _process_event_async(danger_type, confidence, snapshot_path, raw_frame, mqtt_client):
    """
    Gestisce l'upload dello snapshot HTTP e l'invio dell'allarme MQTT in background,
    utilizzando il frame raw pulito (senza bounding box o artefatti HUD).
    """
    # 1. Carica snapshot JPG al backend per ottenere l'URL pubblico (frame_url)
    frame_url = upload_snapshot(raw_frame if raw_frame is not None else snapshot_path)

    description = ""
    # 2. VLM confirmation se abilitato (saltato per 'segnale_aiuto' per garantire reattività immediata)
    if config.VLM_ENABLED and danger_type != "segnale_aiuto":
        confirmed, description = vlm_confirm.confirm(snapshot_path, danger_type)
        if not confirmed:
            logger.info(f"Evento {danger_type} scartato dal VLM (falso positivo): {description}")
            return

    # 3. Pubblica allarme istantaneo via MQTT sul topic parco/<MAC>/camera con frame_url
    publish_event(
        mqtt_client=mqtt_client,
        danger_type=danger_type,
        confidence=confidence,
        frame_url=frame_url,
        description=description,
    )

    # 4. Notifica Telegram opzionale
    label = f"{danger_type} - {description}" if description else danger_type
    notify_telegram(f"⚠️ TreeEyes - {label} (Camera: {config.DEVICE_ID})")


def fire_event(now, danger_type, confidence, raw_frame, mqtt_client, last_event_time, clip):
    """
    Innesca un evento di pericolo, salvando il frame raw non alterato su disco
    e avviando clip e upload in modo thread-safe.
    """
    cooldown = getattr(config, "EVENT_COOLDOWN_MAP", {}).get(danger_type, config.EVENT_COOLDOWN_SECONDS)
    if now - last_event_time.get(danger_type, 0) < cooldown:
        return False

    last_event_time[danger_type] = now
    os.makedirs("snapshots", exist_ok=True)
    snapshot_path = f"snapshots/{danger_type}_{int(now)}.jpg"
    frame_copy = raw_frame.copy()
    cv2.imwrite(snapshot_path, frame_copy)

    # Avvia buffer di registrazione video (3-5s post-evento) con frame pulito
    clip.trigger(danger_type)

    threading.Thread(
        target=_process_event_async,
        args=(danger_type, confidence, snapshot_path, frame_copy, mqtt_client),
        daemon=True,
    ).start()
    return True


def open_capture():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la sorgente video: {config.CAMERA_INDEX}")

    # Imposta formato MJPG se disponibile (riduce il carico del bus USB)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if getattr(config, "CAMERA_WIDTH", None):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    if getattr(config, "CAMERA_HEIGHT", None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if getattr(config, "CAMERA_FPS", None):
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Telecamera configurata: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    return cap


class VideoCaptureThread:
    """
    Thread di acquisizione video non-bloccante a latenza zero.
    Svuota costantemente il buffer hardware del driver V4L2 mantenendo
    disponibile esclusivamente il frame più recente per il loop principale.
    """
    def __init__(self):
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
            self.cap = open_capture()
            self.consecutive_failures = 0
        except Exception as e:
            logger.error(f"Inizializzazione capture fallita: {e}")
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
                    logger.warning("Riconnessione webcam nel thread di acquisizione...")
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


def main():
    logger.info("Caricamento modelli RKNN...")
    detect_model = DetectModel(DETECT_RKNN_PATH, core_mask=RKNNLite.NPU_CORE_0, conf_thresh=config.CONF_THRESHOLD)
    pose_model = PoseModel(POSE_RKNN_PATH, core_mask=RKNNLite.NPU_CORE_1, conf_thresh=config.CONF_THRESHOLD)
    tracker = SimpleTracker()
    executor = ThreadPoolExecutor(max_workers=3)
    frame_idx = 0
    last_detections = []
    heartbeat_service = None

    # 1. Handshake di autenticazione e autorizzazione con il backend centrale HTTP
    verify_handshake()

    # 2. Connessione broker MQTT per allarmi in tempo reale
    mqtt_client = build_mqtt_client()
    # 3. Avvio Heartbeat periodico verso il backend TreeEyes (ogni 60s)
    heartbeat_service = start_heartbeat(mqtt_client)
    clip = ClipRecorder()

    os.makedirs("snapshots", exist_ok=True)
    os.makedirs(getattr(config, "VIDEO_CLIP_DIR", "clips"), exist_ok=True)

    # Pulizia iniziale storage
    cleanup_old_files("snapshots", getattr(config, "MAX_SNAPSHOTS_COUNT", 500))
    cleanup_old_files(getattr(config, "VIDEO_CLIP_DIR", "clips"), getattr(config, "MAX_CLIPS_COUNT", 100))
    last_cleanup_time = time.time()

    consecutive_counts = {}
    last_event_time = {}
    crowd_start_time = None

    # Avvio thread video dedicato a latenza zero
    video_stream = VideoCaptureThread().start()

    window_name = getattr(config, "WINDOW_NAME", "TreeEyes - Vision Node")
    is_fullscreen = getattr(config, "WINDOW_FULLSCREEN", True)
    monitor_x = getattr(config, "WINDOW_MONITOR_X", 0)
    monitor_y = getattr(config, "WINDOW_MONITOR_Y", 0)

    # Inizializzazione finestra OpenCV: supporta fullscreen e spostamento sul monitor desiderato
    if not getattr(config, "HEADLESS", False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if monitor_x != 0 or monitor_y != 0:
            cv2.moveWindow(window_name, monitor_x, monitor_y)
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logger.info(
        f"Avvio completato. HEADLESS={getattr(config, 'HEADLESS', False)}, "
        f"FULLSCREEN={is_fullscreen} (Monitor X={monitor_x}, Y={monitor_y}). "
        f"Controlli: 'q'/ESC esci, 'f' toggle fullscreen, 'r' reset allarmi."
    )

    try:
        while True:
            ok, raw_frame = video_stream.read(timeout=1.5)
            if not ok:
                logger.warning("Frame video non disponibile o timeout cattura.")
                continue

            now = time.time()
            frame_idx += 1

            # Pulizia periodica storage ogni STORAGE_CLEANUP_INTERVAL_SECONDS
            if now - last_cleanup_time > getattr(config, "STORAGE_CLEANUP_INTERVAL_SECONDS", 1800):
                cleanup_old_files("snapshots", getattr(config, "MAX_SNAPSHOTS_COUNT", 500))
                cleanup_old_files(getattr(config, "VIDEO_CLIP_DIR", "clips"), getattr(config, "MAX_CLIPS_COUNT", 100))
                last_cleanup_time = now

            # 1. Aggiorna buffer circolare clip video con il frame raw pulito
            clip.update(raw_frame)

            detected_this_frame = set()
            person_count = 0

            # 2. Inferenze in parallelo su thread dedicati usando il frame raw
            small_raw = cv2.resize(raw_frame, (raw_frame.shape[1] // 2, raw_frame.shape[0] // 2))

            pose_future = executor.submit(pose_model.infer, raw_frame)
            hands_future = executor.submit(hg.detect_help_gesture, small_raw)
            if frame_idx % 2 == 0:
                detect_future = executor.submit(detect_model.infer, raw_frame)
                last_detections = detect_future.result()
            detections = last_detections

            pose_results = pose_future.result()
            gesture_triggered = hands_future.result()

            # 3. Creazione frame di rendering dedicato all'HUD e visualizzazione
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
                    consecutive_counts[danger_type] = consecutive_counts.get(danger_type, 0) + 1
                    if consecutive_counts[danger_type] >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                        fire_event(now, danger_type, confidence, raw_frame, mqtt_client, last_event_time, clip)

            # --- Analisi Pose e Caduta ---
            tracked = tracker.update(pose_results)

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
                    consecutive_counts["persona_a_terra"] = consecutive_counts.get("persona_a_terra", 0) + 1

            # Innesco allarme caduta validato
            if "persona_a_terra" in detected_this_frame and consecutive_counts.get("persona_a_terra", 0) >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                fire_event(now, "persona_a_terra", 1.0, raw_frame, mqtt_client, last_event_time, clip)

            # Reset contatori per pericoli non più visibili nel frame corrente
            for danger_type in list(consecutive_counts.keys()):
                if danger_type not in detected_this_frame:
                    consecutive_counts[danger_type] = 0

            # --- Assembramento ---
            if person_count >= config.CROWD_COUNT_THRESHOLD:
                if crowd_start_time is None:
                    crowd_start_time = now
                elif now - crowd_start_time >= config.CROWD_SECONDS_THRESHOLD:
                    fire_event(now, "assembramento", 1.0, raw_frame, mqtt_client, last_event_time, clip)
            else:
                crowd_start_time = None

            # --- Skeleton + gesto aiuto ---
            if gesture_triggered:
                fire_event(now, "segnale_aiuto", 1.0, raw_frame, mqtt_client, last_event_time, clip)

            # --- HUD e visualizzatore ---
            hud.draw_progress_bar(display_frame, "Gesto aiuto", hg.gesture_progress())
            hud.draw_panel(display_frame, [
                ("Persone", False, f"({person_count})"),
                ("Caduta", "persona_a_terra" in detected_this_frame, ""),
                ("Assembramento", crowd_start_time is not None, ""),
                ("Gesto aiuto", gesture_triggered, ""),
            ], recording=clip.is_recording())

            if not getattr(config, "HEADLESS", False):
                cv2.imshow(window_name, display_frame)

                # Al primo frame renderizzato, riapplica posizione e fullscreen per garantire compatibilità X11 / Wayland
                if frame_idx == 1:
                    if monitor_x != 0 or monitor_y != 0:
                        cv2.moveWindow(window_name, monitor_x, monitor_y)
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    last_event_time.clear()
                    consecutive_counts.clear()
                    crowd_start_time = None
                    logger.info("Reset manuale cooldown/contatori (demo).")
                if key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                    if not is_fullscreen:
                        if monitor_x != 0 or monitor_y != 0:
                            cv2.moveWindow(window_name, monitor_x, monitor_y)
                        cv2.resizeWindow(window_name, 1280, 720)
                    logger.info(f"Fullscreen: {'ATTIVATO' if is_fullscreen else 'DISATTIVATO'}")
            else:
                # In modalità headless senza display, mantieni una breve pausa per cedere la CPU
                time.sleep(0.001)
                if frame_idx % 150 == 0:
                    logger.info(
                        f"Status: frame={frame_idx}, persone={person_count}, "
                        f"allarmi_attivi={list(detected_this_frame)}, clip_rec={clip.is_recording()}"
                    )

    finally:
        video_stream.release()
        if not getattr(config, "HEADLESS", False):
            cv2.destroyAllWindows()
        detect_model.release()
        pose_model.release()
        if heartbeat_service is not None:
            heartbeat_service.stop()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
