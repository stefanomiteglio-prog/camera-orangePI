import time
import os
import threading

import cv2

import config
from events import build_mqtt_client, publish_event, notify_telegram
import pose_heuristics as ph
import hand_gesture as hg
import hud
import vlm_confirm
from video_clip import ClipRecorder
from logger import logger
from rknn_backend import PoseModel, DetectModel, SimpleTracker
from concurrent.futures import ThreadPoolExecutor

POSE_RKNN_PATH = "yolov8s-pose.rknn"
DETECT_RKNN_PATH = "rknn_test/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn"
RKNN_TARGET = "rk3588"


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


def main():
    logger.info("Caricamento modelli RKNN...")
    from rknnlite.api import RKNNLite
    detect_model = DetectModel(DETECT_RKNN_PATH, core_mask=RKNNLite.NPU_CORE_0, conf_thresh=config.CONF_THRESHOLD)
    pose_model = PoseModel(POSE_RKNN_PATH, core_mask=RKNNLite.NPU_CORE_1, conf_thresh=config.CONF_THRESHOLD)
    tracker = SimpleTracker()
    executor = ThreadPoolExecutor(max_workers=3)
    frame_idx = 0
    last_detections = []

    cap = open_capture()
    consecutive_failures = 0

    mqtt_client = build_mqtt_client()
    clip = ClipRecorder()

    os.makedirs("snapshots", exist_ok=True)

    consecutive_counts = {}
    last_event_time = {}
    crowd_start_time = None

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

            # --- Stadio 1+2+gesto in parallelo su thread separati ---
            frame_idx += 1
            small_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

            pose_future = executor.submit(pose_model.infer, frame)
            hands_future = executor.submit(hg.detect_help_gesture, small_frame)
            if frame_idx % 2 == 0:
                detect_future = executor.submit(detect_model.infer, frame)
                last_detections = detect_future.result()
            detections = last_detections

            pose_results = pose_future.result()
            gesture_triggered = hands_future.result()

            for det in detections:
                class_name = det["class_name"].strip()
                confidence = det["score"]
                if class_name == "person":
                    person_count += 1
                if class_name in config.DANGER_CLASS_MAP:
                    danger_type = config.DANGER_CLASS_MAP[class_name]
                    detected_this_frame.add(danger_type)
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    consecutive_counts[danger_type] = consecutive_counts.get(danger_type, 0) + 1
                    if consecutive_counts[danger_type] >= config.CONSECUTIVE_FRAMES_THRESHOLD:
                        fire_event(now, danger_type, confidence, frame, mqtt_client, last_event_time, clip)

            # --- Stadio 2: tracking, caduta, rissa, vandalismo (pose gia' calcolata sopra) ---
            tracked = tracker.update(pose_results)

            people = []
            for det in tracked:
                track_id = det["track_id"]
                x1, y1, x2, y2 = map(int, det["box"])
                xyxy = det["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                keypoints = [(kp[0], kp[1]) for kp in det["keypoints"]]
                kconf = [kp[2] for kp in det["keypoints"]]
                for kx, ky, kc in det["keypoints"]:
                    if kc > 0.5:
                        cv2.circle(frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)
                from rknn_backend import POSE_SKELETON
                for sk in POSE_SKELETON:
                    p1 = det["keypoints"][sk[0]-1]
                    p2 = det["keypoints"][sk[1]-1]
                    if p1[2] > 0.5 and p2[2] > 0.5:
                        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 128, 0), 2)
                if ph.is_fallen(xyxy, keypoints, kconf):
                    detected_this_frame.add("persona_a_terra")
                    consecutive_counts["persona_a_terra"] = consecutive_counts.get("persona_a_terra", 0) + 1

                motion, wrist_motion = ph.update_motion(track_id, keypoints, kconf)
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

            # --- Skeleton + gesto aiuto (mediapipe, invariato) ---
            if gesture_triggered:
                fire_event(now, "segnale_aiuto", 1.0, frame, mqtt_client, last_event_time, clip)
            hud.draw_progress_bar(frame, "Gesto aiuto", hg.gesture_progress())

            # --- Clip 10s pre/post ---
            clip.update(frame)

            # --- HUD di stato ---
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
        detect_model.release()
        pose_model.release()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
