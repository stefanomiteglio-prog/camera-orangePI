import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Union, Any

try:
    import cv2
except ImportError:
    cv2 = None
import paho.mqtt.client as mqtt
import requests

import config
from logger import logger


# ============================================================
# HTTP REST BACKEND CLIENT
# ============================================================

def verify_handshake(device_id: Optional[str] = None, backend_url: Optional[str] = None) -> bool:
    """
    Verifica all'avvio che la telecamera sia registrata ed attiva a sistema.

    Endpoint: GET /devices/me (oppure GET /api/vision/me)
    Header: X-MAC-Address: <TUO_MAC_ADDRESS>
    """
    mac = device_id or config.DEVICE_ID
    base_url = backend_url or getattr(config, "BACKEND_HTTP_URL", f"http://{config.BACKEND_HTTP_HOST}:{config.BACKEND_HTTP_PORT}")
    headers = {"X-MAC-Address": mac}
    timeout = getattr(config, "HTTP_TIMEOUT_SECONDS", 10)

    # Lista degli endpoint da testare (/devices/me e fallback a /api/vision/me)
    endpoints = [
        getattr(config, "BACKEND_ME_URL", f"{base_url}/devices/me"),
        f"{base_url}/api/vision/me",
        f"{base_url}/devices/me",
    ]
    # Rimuovi duplicati preservando l'ordine
    unique_endpoints = list(dict.fromkeys(endpoints))

    last_error = None
    for url in unique_endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                zone = data.get("zone", "Zona non specificata")
                status = data.get("status", "ok")
                dev_type = data.get("type", "camera")
                logger.info(
                    f"Handshake backend RIUSCITO ({url}): "
                    f"Telecamera '{mac}' autorizzata per zona '{zone}' (status={status}, tipo={dev_type})"
                )
                return True
            elif resp.status_code == 401:
                logger.error(
                    f"Handshake fallito (HTTP 401 Unauthorized): "
                    f"Il MAC address '{mac}' non è censito nel database del backend!"
                )
                return False
            elif resp.status_code == 403:
                logger.error(
                    f"Handshake fallito (HTTP 403 Forbidden): "
                    f"Il dispositivo '{mac}' esiste ma non è configurato come 'camera'!"
                )
                return False
            elif resp.status_code in (404, 405):
                # Endpoint non trovato su questa rotta, prova il prossimo
                continue
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            break

    if last_error:
        logger.warning(
            f"Handshake backend non completato ({last_error}). "
            f"Il sistema continuerà in modalità autonoma."
        )
    return False


def upload_snapshot(frame_or_path: Any, device_id: Optional[str] = None) -> Optional[str]:
    """
    Carica il frame JPG dell'evento sul backend per renderlo visibile nella dashboard.

    Endpoint: POST /api/vision/snapshot
    Header: X-MAC-Address: <TUO_MAC_ADDRESS>
    Content-Type: multipart/form-data
    Body: file binario immagine con chiave 'file' (es. snapshot.jpg)
    Ritorna: URL pubblico della foto salvata (da inserire in 'frame_url' MQTT), oppure None in caso di errore.
    """
    mac = device_id or config.DEVICE_ID
    url = getattr(config, "BACKEND_SNAPSHOT_URL", f"{config.BACKEND_HTTP_URL}/api/vision/snapshot")
    headers = {"X-MAC-Address": mac}
    timeout = getattr(config, "HTTP_TIMEOUT_SECONDS", 10)

    try:
        if isinstance(frame_or_path, str):
            if not os.path.isfile(frame_or_path):
                logger.error(f"File snapshot non trovato per upload: {frame_or_path}")
                return None
            with open(frame_or_path, "rb") as f:
                image_bytes = f.read()
            filename = os.path.basename(frame_or_path)
        else:
            # È un frame OpenCV (numpy ndarray)
            if cv2 is None:
                logger.error("OpenCV (cv2) non disponibile per codificare il frame in memoria.")
                return None
            success, encoded = cv2.imencode(".jpg", frame_or_path)
            if not success:
                logger.error("Codifica frame JPG fallita per upload snapshot.")
                return None
            image_bytes = encoded.tobytes()
            filename = f"snapshot_{int(time.time())}.jpg"

        files = {"file": (filename, image_bytes, "image/jpeg")}
        resp = requests.post(url, headers=headers, files=files, timeout=timeout)

        if resp.status_code == 200:
            data = resp.json()
            snapshot_url = data.get("url")
            logger.info(f"Snapshot caricato con successo: {snapshot_url}")
            return snapshot_url
        else:
            logger.warning(f"Errore upload snapshot HTTP ({resp.status_code}): {resp.text[:200]}")
            return None

    except Exception as e:
        logger.warning(f"Eccezione durante l'upload dello snapshot a {url}: {e}")
        return None


def upload_video_clip(clip_path: str, reason: str = "alert", device_id: Optional[str] = None) -> Optional[dict]:
    """
    Carica la clip video MP4 registrata sul backend.

    Endpoint: POST /api/vision/clip
    Header: X-MAC-Address: <TUO_MAC_ADDRESS>
    Content-Type: multipart/form-data
    Body: file video con chiave 'file' (es. clip.mp4)
    """
    mac = device_id or config.DEVICE_ID
    url = getattr(config, "BACKEND_CLIP_URL", getattr(config, "BACKEND_VIDEO_UPLOAD_URL", f"{config.BACKEND_HTTP_URL}/api/vision/clip"))
    headers = {"X-MAC-Address": mac}
    timeout = getattr(config, "HTTP_TIMEOUT_SECONDS", 20)

    if not os.path.isfile(clip_path):
        logger.error(f"File clip video non trovato: {clip_path}")
        return None

    try:
        with open(clip_path, "rb") as fh:
            files = {"file": (os.path.basename(clip_path), fh, "video/mp4")}
            data = {
                "device_id": mac,
                "event_type": reason,
                "timestamp": int(time.time()),
            }
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)

        if resp.status_code == 200:
            result = resp.json()
            logger.info(
                f"Clip video caricata con successo: {result.get('video_url')} "
                f"(associata a evento ID: {result.get('event_id')})"
            )
            return result
        else:
            logger.warning(f"Errore upload clip video ({resp.status_code}): {resp.text[:200]}")
            return None

    except Exception as e:
        logger.error(f"Eccezione durante l'upload della clip video a {url}: {e}")
        return None


# ============================================================
# MQTT
# ============================================================

def build_mqtt_client(client_id: Optional[str] = None):
    """
    Crea e connette il client MQTT della Vision Node.

    Una sola connessione è condivisa da tutte le telecamere: ciascuna pubblica
    poi sul proprio topic (parco/<device_id>/camera). Per questo il client_id
    NON è più il device_id di una singola telecamera, ma un identificativo unico
    del nodo (config.MQTT_CLIENT_ID), evitando che due connessioni con lo stesso
    client_id si sconnettano a vicenda.
    """
    if not getattr(config, "MQTT_ENABLED", True):
        logger.warning("MQTT disabilitato in config.py (modalità offline).")
        return None

    cid = client_id or getattr(config, "MQTT_CLIENT_ID", "treeeyes-vision-node")

    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=cid,
            protocol=mqtt.MQTTv5,
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(
            client_id=cid,
            protocol=mqtt.MQTTv5,
        )

    mqtt_username = getattr(config, "MQTT_USERNAME", None)
    mqtt_password = getattr(config, "MQTT_PASSWORD", None)

    if mqtt_username and mqtt_password:
        client.username_pw_set(
            mqtt_username,
            mqtt_password,
        )

    try:
        client.connect(
            config.MQTT_BROKER_HOST,
            config.MQTT_BROKER_PORT,
            keepalive=60,
        )
        client.loop_start()

        logger.info(
            f"MQTT connesso a {config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT} (client_id={cid})"
        )
        return client

    except Exception as e:
        logger.warning(
            f"Errore connessione MQTT ({e}): broker non raggiungibile. "
            f"Il sistema continuerà in locale senza MQTT."
        )
        return None


# ============================================================
# PUBBLICAZIONE EVENTI ALLARME
# ============================================================

def publish_event(
    mqtt_client,
    danger_type: str,
    confidence: float,
    frame_url: Optional[str] = None,
    description: Optional[str] = None,
    device_id: Optional[str] = None,
):
    """
    Pubblica un allarme istantaneo via MQTT secondo la specifica.

    Il device_id è quello della telecamera che ha generato l'evento (Zona A o
    Zona B): determina sia il campo 'device_id' del payload sia il topic.

    Topic: parco/<device_id>/camera
    QoS: 1
    Payload JSON:
    {
      "device_id": "treeeyes_zona_a",
      "type": "alert",
      "val_number": 1,
      "alert_type": "segnale_aiuto",
      "confidence": 0.95,
      "frame_url": "http://<IP_SERVER>:8000/static/snapshots/...",
      "sampling_time": "2026-09-25T15:30:00Z"
    }
    """
    dev = device_id or config.DEVICE_ID
    topic = f"parco/{dev}/camera"
    sampling_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "device_id": dev,
        "type": "alert",
        "val_number": 1,
        "alert_type": danger_type,
        "confidence": round(float(confidence), 2),
        "sampling_time": sampling_time,
    }

    if frame_url:
        payload["frame_url"] = frame_url

    if description:
        payload["description"] = description

    if mqtt_client is None:
        logger.debug(f"MQTT offline: allarme {danger_type} ({dev}) non inviato al broker.")
        return False

    try:
        message = json.dumps(payload, ensure_ascii=False)
        result = mqtt_client.publish(
            topic,
            message,
            qos=1,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Errore pubblicazione MQTT su {topic} (rc={result.rc})")
            return False

        logger.info(
            f"Allarme MQTT inviato [{dev}]: {danger_type} (conf={float(confidence):.2f}, frame_url={frame_url})"
        )
        logger.debug(f"MQTT payload: {message}")
        return True

    except Exception as e:
        logger.error(f"Errore pubblicazione evento MQTT: {e}")
        return False


# ============================================================
# HEARTBEAT PERIODICO (BATTITO CARDIACO)
# ============================================================

def publish_heartbeat(mqtt_client, device_id: Optional[str] = None) -> bool:
    """
    Pubblica un battito cardiaco (heartbeat) periodico via MQTT secondo la specifica TreeEyes:

    Topic: parco/<MAC_TELECAMERA>/heartbeat
    QoS: 1
    Payload JSON:
    {
      "device_id": "<MAC_TELECAMERA>",
      "type": "heartbeat"
    }
    """
    mac = device_id or config.DEVICE_ID
    # Topic sempre derivato dal device_id della telecamera (una per zona),
    # così ogni telecamera ha il proprio heartbeat indipendente.
    topic = f"parco/{mac}/heartbeat"

    payload = {
        "device_id": mac,
        "type": "heartbeat",
    }

    if mqtt_client is None:
        logger.debug(f"MQTT offline: heartbeat non inviato al broker per {mac}.")
        return False

    try:
        message = json.dumps(payload, ensure_ascii=False)
        result = mqtt_client.publish(
            topic,
            message,
            qos=1,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Errore pubblicazione heartbeat MQTT su {topic} (rc={result.rc})")
            return False

        logger.info(f"Heartbeat MQTT inviato: topic={topic}, device_id={mac}")
        logger.debug(f"Heartbeat payload: {message}")
        return True

    except Exception as e:
        logger.error(f"Errore durante l'invio dell'heartbeat MQTT: {e}")
        return False


class HeartbeatService:
    """
    Gestore thread in background per l'invio del battito cardiaco (heartbeat) periodico.
    Evita che la telecamera venga considerata offline dal backend TreeEyes in assenza di allarmi.
    """

    def __init__(
        self,
        mqtt_client,
        interval: Optional[float] = None,
        device_id: Optional[str] = None,
    ):
        self.mqtt_client = mqtt_client
        self.interval = float(
            interval
            if interval is not None
            else getattr(config, "HEARTBEAT_INTERVAL_SECONDS", 60)
        )
        self.device_id = device_id or config.DEVICE_ID
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Avvia il thread in background per l'heartbeat periodico."""
        if not getattr(config, "MQTT_ENABLED", True) or self.mqtt_client is None:
            logger.warning("Heartbeat non avviato: MQTT disabilitato o client non connesso.")
            return self

        if self._thread is not None and self._thread.is_alive():
            logger.warning("Heartbeat thread già attivo.")
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="HeartbeatThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Heartbeat service avviato (intervallo: {self.interval}s, topic: parco/{self.device_id}/heartbeat)."
        )
        return self

    def stop(self, timeout: float = 2.0):
        """Arresta il thread dell'heartbeat in modo tempestivo e sicuro."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("Heartbeat service arrestato.")

    def _run(self):
        # Primo invio immediato per comunicare istantaneamente lo stato online
        try:
            publish_heartbeat(self.mqtt_client, self.device_id)
        except Exception as e:
            logger.error(f"Errore primo invio heartbeat: {e}")

        # Ciclo periodico: attende l'intervallo o termina subito alla chiamata di stop()
        while not self._stop_event.wait(self.interval):
            try:
                publish_heartbeat(self.mqtt_client, self.device_id)
            except Exception as e:
                logger.error(f"Errore ciclo heartbeat: {e}")


def start_heartbeat(
    mqtt_client,
    interval: Optional[float] = None,
    device_id: Optional[str] = None,
) -> HeartbeatService:
    """Helper per istanziare e avviare il servizio di heartbeat in background."""
    service = HeartbeatService(mqtt_client, interval=interval, device_id=device_id)
    return service.start()


# ============================================================
# TELEGRAM
# ============================================================

def notify_telegram(message: str):
    """
    Invia una notifica Telegram opzionale se abilitata in config.py.
    """
    telegram_enabled = getattr(config, "TELEGRAM_ENABLED", False)
    if not telegram_enabled:
        return False

    bot_token = getattr(config, "TELEGRAM_BOT_TOKEN", None)
    chat_id = getattr(config, "TELEGRAM_CHAT_ID", None)

    if not bot_token or not chat_id:
        logger.warning("Telegram abilitato ma TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}

    try:
        response = requests.post(url, data=data, timeout=10)
        if response.ok:
            logger.info("Notifica Telegram inviata con successo.")
            return True
        logger.warning(f"Errore Telegram: HTTP {response.status_code}")
    except Exception as e:
        logger.warning(f"Errore invio notifica Telegram: {e}")

    return False