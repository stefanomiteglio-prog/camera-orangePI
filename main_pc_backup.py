import time
import os
import threading

import cv2
from ultralytics import YOLO

import config
from events import (
    build_mqtt_client,
    publish_event,
    start_heartbeat,
    notify_telegram,
    verify_handshake,
    upload_snapshot,
)
import hand_gesture as hg
import hud
import vlm_confirm
from video_clip import ClipRecorder
from logger import logger


def _process_event_async(danger_type, confidence, snapshot_path, frame, mqtt_client):
    """
    Gestisce l'upload dello snapshot HTTP e l'invio dell'allarme MQTT in background,
    senza bloccare il loop di cattura ed inferenza video.
    """
    # 1. Upload snapshot JPG al backend per ottenere l'URL pubblico (frame_url)
    frame_url = upload_snapshot(frame if frame is not None else snapshot_path)

    description = ""
    # 2. VLM confirmation se abilitato (saltato per 'segnale_aiuto' per garantire reattività immediata)
    if config.VLM_ENABLED and danger_type != "segnale_aiuto":
        confirmed, description = vlm_confirm.confirm(snapshot_path, danger_type)
        if not confirmed:
            logger.info(f"Evento {danger_type} scartato dal VLM (falso positivo): {description}")
            return

    # 3. Invio istantaneo dell'allarme sul topic MQTT con frame_url
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


def fire_event(now, danger_type, confidence, frame, mqtt_client, last_event_time, clip):
    cooldown = getattr(config, "EVENT_COOLDOWN_MAP", {}).get(danger_type, config.EVENT_COOLDOWN_SECONDS)
    if now - last_event_time.get(danger_type, 0) < cooldown:
        return False

    last_event_time[danger_type] = now
    os.makedirs("snapshots", exist_ok=True)
    snapshot_path = f"snapshots/{danger_type}_{int(now)}.jpg"
    frame_copy = frame.copy()
    cv2.imwrite(snapshot_path, frame_copy)

    # Avvia buffer di registrazione video (3-5s post-evento)
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
        raise RuntimeError("Impossibile aprire la sorgente video. Controlla CAMERA_INDEX in config.py")

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


def main():
    logger.info("Caricamento modelli...")
    model = YOLO(config.YOLO_WEIGHTS)
    pose_model = YOLO(config.YOLO_POSE_WEIGHTS)
    fire_model = YOLO(config.FIRE_WEIGHTS) if config.FIRE_ENABLED else None

    cap = open_capture()
    consecutive_failures = 0
    heartbeat_service = None

    # 1. Handshake di autenticazione e autorizzazione con il backend centrale HTTP
    verify_handshake()

    # 2. Connessione broker MQTT per allarmi in tempo reale
    mqtt_client = build_mqtt_client()
    # 3. Avvio Heartbeat periodico verso il backend TreeEyes (ogni 60s)
    heartbeat_service = start_heartbeat(mqtt_client)
    clip = ClipRecorder()

    os.makedirs("snapshots", exist_ok=True)

    consecutive_counts: dict[str, int] = {}
    last_event_time: dict[str, float] = {}
    crowd_start_time: float | None = None

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
        f"Avvio completato. FULLSCREEN={is_fullscreen} (Monitor X={monitor_x}, Y={monitor_y}). "
        f"Controlli: 'q'/ESC esci, 'f' toggle fullscreen, 'r' reset allarmi."
    )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                logger.warning(f"Frame non disponibile ({consecutive_failures})")
                if consecutive_failures >= 10:
                    logger.warning("Riconnessione webcam...")
                    cap.release()
                    time.sleep(2)
                    try:
                        cap = open_capture()
                        consecutive_failures = 0
                        logger.info("Webcam riconnessa.")
                    except RuntimeError:
                        logger.error("Riconnessione fallita, ritento tra 2s.")
                        time.sleep(2)
                else:
                    time.sleep(0.3)
                continue
            consecutive_failures = 0

            now = time.time()
            detected_this_frame = set()
            person_count = 0

            # --- Stadio 1: oggetti pericolosi ---
            results = model.predict(frame, conf=config.CONF_THRESHOLD, device=config.DEVICE, verbose=False)[0]
            for box in results.boxes:
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                if class_name == "person":
                    person_count += 1
                if class_name in config.DANGER_CLASS_MAP:
                    danger_type = config.DANGER_CLASS_MAP[class_name]
                    detected_this_frame.add(danger_type)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    consecutive_counts[danger_type] = consecutive_counts.get(danger_type, 0) + 1
                    if consecutive_counts[danger_type] >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                        fire_event(now, danger_type, confidence, frame, mqtt_client, last_event_time, clip)

            # --- Fuoco/fumo ---
            if fire_model is not None:
                fr = fire_model.predict(frame, conf=config.CONF_THRESHOLD, device=config.DEVICE, verbose=False)[0]
                for box in fr.boxes:
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    detected_this_frame.add("fuoco_fumo")
                    consecutive_counts["fuoco_fumo"] = consecutive_counts.get("fuoco_fumo", 0) + 1
                    if consecutive_counts["fuoco_fumo"] >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                        fire_event(now, "fuoco_fumo", confidence, frame, mqtt_client, last_event_time, clip)

            # --- Stadio 2: pose e tracking ---
            pose_results = pose_model.track(
                frame, conf=config.CONF_THRESHOLD, device=config.DEVICE,
                persist=True, verbose=False, tracker="botsort_reid.yaml",
            )[0]

            if pose_results.boxes is not None and pose_results.boxes.id is not None:
                for box, kp, track_id in zip(pose_results.boxes, pose_results.keypoints, pose_results.boxes.id):
                    track_id = int(track_id)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            for danger_type in list(consecutive_counts.keys()):
                if danger_type not in detected_this_frame:
                    consecutive_counts[danger_type] = 0

            # --- Assembramento ---
            if person_count >= config.CROWD_COUNT_THRESHOLD:
                if crowd_start_time is None:
                    crowd_start_time = now
                elif now - crowd_start_time >= config.CROWD_SECONDS_THRESHOLD:
                    fire_event(now, "assembramento", 1.0, frame, mqtt_client, last_event_time, clip)
            else:
                crowd_start_time = None

            # --- Skeleton + gesto aiuto ---
            hg.draw_skeleton(frame)
            gesture_triggered = hg.detect_help_gesture(frame)
            if gesture_triggered:
                fire_event(now, "segnale_aiuto", 1.0, frame, mqtt_client, last_event_time, clip)
            hud.draw_progress_bar(frame, "Gesto aiuto", hg.gesture_progress())

            # --- Clip 10s pre/post ---
            clip.update(frame)

            # --- HUD di stato (per esposizione) ---
            hud.draw_panel(frame, [
                ("Persone", False, f"({person_count})"),
                ("Assembramento", crowd_start_time is not None, ""),
                ("Gesto aiuto", gesture_triggered, ""),
            ], recording=clip.is_recording())

            if not getattr(config, "HEADLESS", False):
                cv2.imshow(window_name, frame)
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if heartbeat_service is not None:
            heartbeat_service.stop()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
