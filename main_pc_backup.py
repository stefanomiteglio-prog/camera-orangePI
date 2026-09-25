import time
import os
import threading

import cv2
from ultralytics import YOLO

import config
from events import build_mqtt_client, publish_event, notify_telegram
import pose_heuristics as ph
import hand_gesture as hg
import hud
import vlm_confirm
from video_clip import ClipRecorder
from logger import logger


def _confirm_and_publish(danger_type, confidence, snapshot_path, mqtt_client):
    confirmed, description = vlm_confirm.confirm(snapshot_path, danger_type)
    if not confirmed:
        logger.info(f"Evento {danger_type} scartato dal VLM (falso positivo): {description}")
        return
    label = f"{danger_type} - {description}" if description else danger_type
    publish_event(mqtt_client, danger_type, confidence, snapshot_path, description)
    notify_telegram(f"⚠️ TreeEyes - {label} - Parco Robinson")


def fire_event(now, danger_type, confidence, frame, mqtt_client, last_event_time, clip):
    if now - last_event_time.get(danger_type, 0) < config.EVENT_COOLDOWN_SECONDS:
        return False
    snapshot_path = f"snapshots/{danger_type}_{int(now)}.jpg"
    cv2.imwrite(snapshot_path, frame)
    last_event_time[danger_type] = now
    clip.trigger(danger_type)
    threading.Thread(
        target=_confirm_and_publish,
        args=(danger_type, confidence, snapshot_path, mqtt_client),
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

    mqtt_client = build_mqtt_client()
    clip = ClipRecorder()

    os.makedirs("snapshots", exist_ok=True)

    consecutive_counts: dict[str, int] = {}
    last_event_time: dict[str, float] = {}
    crowd_start_time: float | None = None

    logger.info("Avvio. Premi 'q' per uscire.")

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

            # --- Stadio 2: pose, tracking, caduta, rissa, vandalismo ---
            pose_results = pose_model.track(
                frame, conf=config.CONF_THRESHOLD, device=config.DEVICE,
                persist=True, verbose=False, tracker="botsort_reid.yaml",
            )[0]

            people = []
            if pose_results.boxes is not None and pose_results.boxes.id is not None:
                for box, kp, track_id in zip(pose_results.boxes, pose_results.keypoints, pose_results.boxes.id):
                    track_id = int(track_id)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                    keypoints = kp.xy[0].tolist()
                    if ph.is_fallen(xyxy, keypoints):
                        detected_this_frame.add("persona_a_terra")
                        consecutive_counts["persona_a_terra"] = consecutive_counts.get("persona_a_terra", 0) + 1

                    motion, wrist_motion = ph.update_motion(track_id, keypoints)
                    center = ph.person_center(xyxy)
                    diag = ph.person_diag(xyxy)
                    people.append((center, motion, diag))

                    wrist_ratio = wrist_motion / diag
                    if ph.in_zone(center, config.PROTECTED_ZONE):
                        cv2.putText(frame, f"vandal ratio: {wrist_ratio:.2f}", (x1, y2 + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        if wrist_ratio >= config.VANDAL_WRIST_RATIO_THRESHOLD:
                            detected_this_frame.add("vandalismo")
                            consecutive_counts["vandalismo"] = consecutive_counts.get("vandalismo", 0) + 1

            zx1, zy1, zx2, zy2 = config.PROTECTED_ZONE
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 1)

            if ph.is_fighting(people):
                detected_this_frame.add("rissa")
                consecutive_counts["rissa"] = consecutive_counts.get("rissa", 0) + 1

            for danger_type, threshold in (
                ("persona_a_terra", config.CONSECUTIVE_FRAMES_THRESHOLD),
                ("rissa", config.CONSECUTIVE_FRAMES_THRESHOLD),
                ("vandalismo", config.VANDAL_FRAMES_THRESHOLD),
            ):
                if danger_type in detected_this_frame and consecutive_counts.get(danger_type, 0) >= threshold:
                    fire_event(now, danger_type, 1.0, frame, mqtt_client, last_event_time, clip)

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
                ("Caduta", "persona_a_terra" in detected_this_frame, ""),
                ("Rissa", "rissa" in detected_this_frame, ""),
                ("Vandalismo", "vandalismo" in detected_this_frame, ""),
                ("Assembramento", crowd_start_time is not None, ""),
                ("Gesto aiuto", gesture_triggered, ""),
            ], recording=clip.is_recording())

            cv2.imshow("TreeEyes - Vision Node (debug)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                last_event_time.clear()
                consecutive_counts.clear()
                crowd_start_time = None
                logger.info("Reset manuale cooldown/contatori (demo).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
