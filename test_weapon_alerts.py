"""
Test locale dei nuovi allarmi del modello armi (weapon_threat.rknn) SENZA
telecamera e SENZA NPU: simula ciò che main.py invia al backend quando rileva
una delle classi segnalate (grenade, knife).

Per ogni classe:
  1. carica uno snapshot JPG finto      (POST /api/vision/snapshot)
  2. pubblica l'allarme MQTT            (topic parco/<device_id>/camera)
usando le stesse funzioni di events.py che usa main.py.

Backend e broker sono quelli su QUESTO PC (localhost): gli indirizzi vengono
sovrascritti solo in memoria, config.py e gli altri programmi restano invariati.

Uso:
    pip install paho-mqtt requests
    python test_weapon_alerts.py                       # tutte le classi
    python test_weapon_alerts.py --class knife         # una sola classe
    python test_weapon_alerts.py --device mini_cam_02  # altra telecamera censita (Zona B)
    python test_weapon_alerts.py --image foto.jpg      # snapshot reale al posto di quello finto
"""

import argparse
import json
import os
import sys
import tempfile
import time

import config

# --- Override SOLO in memoria: punta tutto a localhost ----------------------
LOCAL_HOST = "localhost"
config.BACKEND_HTTP_HOST = LOCAL_HOST
config.BACKEND_HTTP_URL = f"http://{LOCAL_HOST}:{config.BACKEND_HTTP_PORT}"
config.BACKEND_ME_URL = f"{config.BACKEND_HTTP_URL}/devices/me"
config.BACKEND_SNAPSHOT_URL = f"{config.BACKEND_HTTP_URL}/api/vision/snapshot"
config.BACKEND_CLIP_URL = f"{config.BACKEND_HTTP_URL}/api/vision/clip"
config.BACKEND_VIDEO_UPLOAD_URL = config.BACKEND_CLIP_URL
config.MQTT_BROKER_HOST = LOCAL_HOST
config.MQTT_CLIENT_ID = "treeeyes-weapon-test"  # diverso dal nodo reale: non lo sconnette
config.TELEGRAM_ENABLED = False

import events  # noqa: E402  (dopo l'override, così usa localhost)

# Topic su cui il backend comanda le sirene del modellino quando riceve un allarme telecamera
TOPIC_ACTUATOR_ALARM = "smartcity/actuators/alarm"

# JPEG 1x1 valido, usato se OpenCV non è installato e non viene passato --image
_DUMMY_JPG = (
    b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00"
    b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
    b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 \",#\x1c\x1c(7),01444\x1f"
    b"\x279=82<.342\xFF\xC0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00\x1f"
    b"\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02"
    b"\x03\x04\x05\x06\x07\x08\t\n\x0b\xFF\xDA\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xFF\xD9"
)


def make_snapshot(class_name, alert_type, device_id, image_path=None):
    """Ritorna ciò che va passato a events.upload_snapshot: un frame o un path."""
    if image_path:
        return image_path
    try:
        import cv2
        import numpy as np
    except ImportError:
        path = os.path.join(tempfile.gettempdir(), f"test_{alert_type}.jpg")
        with open(path, "wb") as f:
            f.write(_DUMMY_JPG)
        return path

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (200, 160), (440, 340), (0, 0, 255), 2)
    cv2.putText(frame, f"{class_name} (TEST)", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"alert_type: {alert_type}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"device: {device_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def main():
    class_map = config.WEAPON_CLASS_MAP
    parser = argparse.ArgumentParser(description="Invia al backend locale gli allarmi del modello armi.")
    # mini_cam_01 (Zona A) è tra i dispositivi di default del backend locale
    parser.add_argument("--device", default="mini_cam_01",
                        help="device_id/MAC di una telecamera censita nel backend (default: %(default)s)")
    parser.add_argument("--class", dest="classes", action="append", choices=list(class_map),
                        help="classe da simulare (ripetibile). Default: tutte")
    parser.add_argument("--confidence", type=float, default=0.87)
    parser.add_argument("--interval", type=float, default=2.0, help="secondi tra un allarme e il successivo")
    parser.add_argument("--image", help="JPG da usare come snapshot al posto di quello generato")
    args = parser.parse_args()

    classes = args.classes or list(class_map)
    print(f"Backend: {config.BACKEND_HTTP_URL} | Broker MQTT: {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}")
    print(f"Telecamera: {args.device} | Classi: {', '.join(classes)}\n")

    # 1. Handshake: la telecamera deve essere censita nel backend come 'camera'
    if not events.verify_handshake(args.device):
        print(f"\n[ERRORE] Handshake fallito per '{args.device}' (dettagli nel log sopra).\n"
              f"Controlla che il backend sia avviato su {config.BACKEND_HTTP_URL} e che il device sia\n"
              f"registrato come telecamera, oppure usa --device con una già censita (es. mini_cam_02).")
        return 1

    # 2. MQTT
    client = events.build_mqtt_client()
    if client is None:
        print(f"\n[ERRORE] Broker MQTT non raggiungibile su {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}.")
        return 1

    # Ascolta i comandi sirena: conferma che il backend ha elaborato l'allarme
    alarm_types_seen = set()

    def on_message(_client, _userdata, msg):
        try:
            alarm_type = json.loads(msg.payload.decode()).get("alarm_type")
        except Exception:
            return
        if alarm_type:
            alarm_types_seen.add(alarm_type)

    client.on_message = on_message
    time.sleep(1.0)  # attende la connessione prima di sottoscrivere
    client.subscribe(TOPIC_ACTUATOR_ALARM, qos=1)

    # 3. Un allarme per classe, come farebbe main.py
    results = []
    for i, class_name in enumerate(classes):
        alert_type = class_map[class_name]
        print(f"--- {class_name} -> alert_type '{alert_type}' ---")
        frame_url = events.upload_snapshot(
            make_snapshot(class_name, alert_type, args.device, args.image), device_id=args.device
        )
        sent = events.publish_event(
            mqtt_client=client,
            danger_type=alert_type,
            confidence=args.confidence,
            frame_url=frame_url,
            description=f"Test locale modello armi: classe {class_name}",
            device_id=args.device,
        )
        results.append((class_name, alert_type, frame_url, sent))
        if i < len(classes) - 1:
            time.sleep(args.interval)

    time.sleep(3.0)  # lascia al backend il tempo di rispondere sul topic sirene
    client.loop_stop()
    client.disconnect()

    # 4. Riepilogo
    print("\n================ RIEPILOGO ================")
    all_ok = True
    for class_name, alert_type, frame_url, sent in results:
        expected = f"TELECAMERA_{alert_type.upper()}"
        if expected in alarm_types_seen:
            backend = "elaborato dal backend (sirena modellino)"
        else:
            backend = "nessuna conferma sirena (verifica nella webapp)"
        all_ok = all_ok and sent and bool(frame_url)
        print(f"{class_name:10s} -> {alert_type:14s} | snapshot: {'OK' if frame_url else 'FALLITO'} "
              f"| MQTT: {'OK' if sent else 'FALLITO'} | {backend}")
    print("===========================================")
    print("Apri la webapp (http://localhost:5173): le pratiche di sicurezza devono mostrare\n"
          "i nuovi eventi con etichetta 'Coltello rilevato' e 'Granata rilevata'.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
