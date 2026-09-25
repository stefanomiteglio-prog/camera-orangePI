import json
import time

import paho.mqtt.client as mqtt
import requests

import config
from logger import logger


# ============================================================
# MQTT
# ============================================================

def build_mqtt_client():
    """
    Crea e connette il client MQTT della Vision Node.

    Il device_id deve corrispondere al MAC address della
    telecamera registrata nel backend.
    """
    if not getattr(config, "MQTT_ENABLED", True):
        logger.warning("MQTT disabilitato in config.py (modalità offline).")
        return None

    client = mqtt.Client(
        client_id=config.DEVICE_ID,
        protocol=mqtt.MQTTv5,
    )

    # Autenticazione MQTT, se configurata
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
            f"MQTT connesso a "
            f"{config.MQTT_BROKER_HOST}:{config.MQTT_BROKER_PORT}"
        )

        logger.info(
            f"Vision Node ID: {config.DEVICE_ID}"
        )

        logger.info(
            f"MQTT topic eventi: {config.MQTT_TOPIC}"
        )
        return client

    except Exception as e:
        logger.warning(f"Errore connessione MQTT ({e}): broker non raggiungibile. Il sistema continuerà in locale senza MQTT.")
        return None


# ============================================================
# PUBBLICAZIONE EVENTI
# ============================================================

def publish_event(
    mqtt_client,
    danger_type,
    confidence,
    snapshot_path=None,
    description=None,
):
    """
    Pubblica un evento rilevato dalla telecamera.

    Il payload è compatibile con il backend TreeEyes/FastAPI.

    Formato:

    {
        "device_id": "...",
        "type": "alert",
        "val_number": 1.0,
        "alert_type": "rissa",
        "confidence": 0.91,
        "sampling_time": "...",
        "frame_url": "...",
        "description": "..."
    }

    Il backend utilizza:
      - device_id      -> MAC address della camera
      - type           -> tipo di Event
      - val_number     -> valore numerico dell'evento
      - alert_type     -> tipo specifico di pericolo
      - confidence     -> confidenza AI
      - frame_url      -> riferimento allo snapshot
    """

    # Timestamp UTC in formato ISO 8601
    sampling_time = time.strftime(
        "%Y-%m-%dT%H:%M:%S+00:00",
        time.gmtime(),
    )

    payload = {
        # Deve essere il MAC address registrato nel backend
        "device_id": config.DEVICE_ID,

        # Il backend si aspetta un Event di tipo alert
        "type": "alert",

        # Evento binario: allarme presente
        "val_number": 1.0,

        # Tipo specifico di pericolo
        # es. rissa, arma, persona_a_terra, vandalismo...
        "alert_type": danger_type,

        # Confidenza del modello AI
        "confidence": round(float(confidence), 2),

        # Momento in cui l'evento è stato rilevato
        "sampling_time": sampling_time,

        # Snapshot associato all'evento
        "frame_url": snapshot_path,

        # Descrizione eventualmente prodotta dal VLM
        "description": description,
    }

    if mqtt_client is None:
        logger.debug(f"MQTT offline: evento {danger_type} non inviato al broker.")
        return False

    try:
        message = json.dumps(
            payload,
            ensure_ascii=False,
        )

        result = mqtt_client.publish(
            config.MQTT_TOPIC,
            message,
            qos=1,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(
                f"Errore pubblicazione MQTT "
                f"(rc={result.rc})"
            )
            return False

        logger.info(
            f"Evento pubblicato: "
            f"{danger_type} "
            f"(confidence={float(confidence):.2f})"
        )

        logger.debug(
            f"MQTT payload: {message}"
        )

        return True

    except Exception as e:
        logger.error(
            f"Errore pubblicazione evento MQTT: {e}"
        )
        return False


# ============================================================
# TELEGRAM
# ============================================================

def notify_telegram(message):
    """
    Invia una notifica Telegram se Telegram è abilitato.

    Se Telegram è disabilitato, la funzione non fa nulla.

    Questa parte rimane indipendente dall'invio MQTT:
    un eventuale errore Telegram non deve impedire
    la pubblicazione dell'evento al backend.
    """

    telegram_enabled = getattr(
        config,
        "TELEGRAM_ENABLED",
        False,
    )

    if not telegram_enabled:
        return False

    bot_token = getattr(
        config,
        "TELEGRAM_BOT_TOKEN",
        None,
    )

    chat_id = getattr(
        config,
        "TELEGRAM_CHAT_ID",
        None,
    )

    if not bot_token or not chat_id:
        logger.warning(
            "Telegram abilitato ma "
            "TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID non configurati."
        )
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{bot_token}/sendMessage"
    )

    data = {
        "chat_id": chat_id,
        "text": message,
    }

    try:
        response = requests.post(
            url,
            data=data,
            timeout=10,
        )

        if response.ok:
            logger.info("Notifica Telegram inviata.")
            return True

        logger.warning(
            f"Errore Telegram: "
            f"HTTP {response.status_code}"
        )

    except Exception as e:
        logger.warning(
            f"Errore invio Telegram: {e}"
        )

    return False