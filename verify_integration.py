"""
Script di verifica e validazione end-to-end dell'integrazione telecamera IA:
1. Handshake HTTP (GET /devices/me e GET /api/vision/me) con validazione MAC, 401 e 403.
2. Upload Snapshot JPG (POST /api/vision/snapshot) e verifica disponibilità immagine.
3. Invio e ricezione allarme MQTT in tempo reale (topic parco/<MAC>/camera, QoS 1).
4. Upload Clip Video MP4 (POST /api/vision/clip) e correlazione automatica con l'evento.
5. Simulazione del workflow asincrono completo.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
import requests
import paho.mqtt.client as mqtt

import config
import events

# Imposta host locale per il test
config.BACKEND_HTTP_URL = "http://localhost:8000"
config.BACKEND_ME_URL = "http://localhost:8000/api/vision/me"
config.BACKEND_SNAPSHOT_URL = "http://localhost:8000/api/vision/snapshot"
config.BACKEND_CLIP_URL = "http://localhost:8000/api/vision/clip"
config.MQTT_BROKER_HOST = "localhost"
config.MQTT_TOPIC = f"parco/{config.DEVICE_ID}/camera"

PASS = "\033[92m[PASSED]\033[0m"
FAIL = "\033[91m[FAILED]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def test_handshake():
    print(f"\n{INFO} --- TEST 1: Handshake HTTP e Verifica Autorizzazione ---")
    
    # 1.1 Test telecamera autorizzata
    ok = events.verify_handshake(device_id="c0:74:2b:fb:00:3f")
    assert ok, "Handshake per c0:74:2b:fb:00:3f doveva riuscire"
    print(f" {PASS} 1.1 Handshake per MAC censito (c0:74:2b:fb:00:3f) -> 200 OK")

    # 1.2 Test MAC non registrato (atteso 401)
    resp_401 = requests.get(
        "http://localhost:8000/api/vision/me",
        headers={"X-MAC-Address": "mac_inesistente_99"},
        timeout=5,
    )
    assert resp_401.status_code == 401, f"Atteso 401, ricevuto {resp_401.status_code}"
    print(f" {PASS} 1.2 Handshake per MAC non registrato -> 401 Unauthorized (Dettaglio: {resp_401.json().get('detail')})")

    # 1.3 Test dispositivo non telecamera (es. cestino 14:91:82:3f:07:c5, atteso 403)
    resp_403 = requests.get(
        "http://localhost:8000/api/vision/me",
        headers={"X-MAC-Address": "14:91:82:3f:07:c5"},
        timeout=5,
    )
    assert resp_403.status_code == 403, f"Atteso 403, ricevuto {resp_403.status_code}"
    print(f" {PASS} 1.3 Handshake per device non telecamera -> 403 Forbidden (Dettaglio: {resp_403.json().get('detail')})")


def test_snapshot_upload():
    print(f"\n{INFO} --- TEST 2: Upload Snapshot JPG (Anteprima Evento) ---")
    dummy_jpg_path = "/tmp/test_gesture_frame.jpg"
    
    # 1x1 dummy JPEG valido
    with open(dummy_jpg_path, "wb") as f:
        f.write(
            b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00"
            b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
            b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 \",#\x1c\x1c(7),01444\x1f"
            b"\x279=82<.342\xFF\xC0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00\x1f"
            b"\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02"
            b"\x03\x04\x05\x06\x07\x08\t\n\x0b\xFF\xDA\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xFF\xD9"
        )

    url = events.upload_snapshot(dummy_jpg_path)
    assert url is not None, "Upload snapshot fallito!"
    assert url.startswith("http://localhost:8000/static/snapshots/"), f"URL snapshot inatteso: {url}"
    print(f" {PASS} 2.1 Snapshot caricato con successo sul backend")
    print(f"       -> frame_url restituito: {url}")

    # Verifica accessibilità pubblica del file
    get_resp = requests.get(url, timeout=5)
    assert get_resp.status_code == 200, f"File non accessibile via GET: {get_resp.status_code}"
    print(f" {PASS} 2.2 File snapshot scaricabile via HTTP GET (Status 200 OK, size={len(get_resp.content)} bytes)")

    if os.path.exists(dummy_jpg_path):
        os.remove(dummy_jpg_path)
    return url


def test_mqtt_alert(frame_url):
    print(f"\n{INFO} --- TEST 3: Invio Allarme MQTT in Tempo Reale ---")
    
    received_messages = []

    # Configura subscriber indipendente per verificare la ricezione
    try:
        sub_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="verifier_sub", protocol=mqtt.MQTTv5)
    except (AttributeError, TypeError):
        sub_client = mqtt.Client(client_id="verifier_sub", protocol=mqtt.MQTTv5)
    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        received_messages.append((msg.topic, payload))

    sub_client.on_message = on_message
    sub_client.connect("localhost", 1883, keepalive=60)
    sub_client.subscribe("parco/#", qos=1)
    sub_client.loop_start()
    time.sleep(0.4)

    # Connetti il client della telecamera
    pub_client = events.build_mqtt_client()
    assert pub_client is not None, "Connessione client MQTT telecamera fallita!"

    # Invia l'evento allarme
    ok = events.publish_event(
        mqtt_client=pub_client,
        danger_type="segnale_aiuto",
        confidence=0.96,
        frame_url=frame_url,
        description="Riconosciuta richiesta di aiuto con gesto della mano",
    )
    assert ok, "Pubblicazione allarme fallita!"
    time.sleep(0.6)

    sub_client.loop_stop()
    sub_client.disconnect()
    pub_client.loop_stop()
    pub_client.disconnect()

    assert len(received_messages) > 0, "Nessun messaggio MQTT ricevuto sul topic!"
    topic, payload = received_messages[0]

    # Validazione campi obbligatori da specifica
    assert topic == f"parco/{config.DEVICE_ID}/camera", f"Topic inatteso: {topic}"
    assert payload.get("device_id") == config.DEVICE_ID, f"device_id errato: {payload.get('device_id')}"
    assert payload.get("type") == "alert", f"type errato: {payload.get('type')}"
    assert payload.get("val_number") == 1, f"val_number errato: {payload.get('val_number')}"
    assert payload.get("alert_type") == "segnale_aiuto", f"alert_type errato: {payload.get('alert_type')}"
    assert payload.get("confidence") == 0.96, f"confidence errata: {payload.get('confidence')}"
    assert payload.get("frame_url") == frame_url, f"frame_url errato: {payload.get('frame_url')}"
    assert "sampling_time" in payload, "sampling_time mancante"

    print(f" {PASS} 3.1 Allarme ricevuto su topic: {topic} (QoS 1)")
    print(f" {PASS} 3.2 Payload JSON conforme alle specifiche:")
    print(json.dumps(payload, indent=6))


def test_mqtt_heartbeat():
    print(f"\n{INFO} --- TEST 3.B: Invio Heartbeat Periodico MQTT (Stato Online) ---")
    received_messages = []

    try:
        sub_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="verifier_sub_hb", protocol=mqtt.MQTTv5)
    except (AttributeError, TypeError):
        sub_client = mqtt.Client(client_id="verifier_sub_hb", protocol=mqtt.MQTTv5)

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        received_messages.append((msg.topic, payload))

    sub_client.on_message = on_message
    sub_client.connect("localhost", 1883, keepalive=60)
    sub_client.subscribe(f"parco/{config.DEVICE_ID}/heartbeat", qos=1)
    sub_client.loop_start()
    time.sleep(0.4)

    pub_client = events.build_mqtt_client()
    assert pub_client is not None, "Connessione client MQTT telecamera fallita!"

    ok = events.publish_heartbeat(pub_client)
    assert ok, "Pubblicazione heartbeat fallita!"
    time.sleep(0.6)

    sub_client.loop_stop()
    sub_client.disconnect()
    pub_client.loop_stop()
    pub_client.disconnect()

    assert len(received_messages) > 0, "Nessun messaggio heartbeat ricevuto sul topic!"
    topic, payload = received_messages[0]

    assert topic == f"parco/{config.DEVICE_ID}/heartbeat", f"Topic inatteso: {topic}"
    assert payload.get("device_id") == config.DEVICE_ID, f"device_id errato: {payload.get('device_id')}"
    assert payload.get("type") == "heartbeat", f"type errato: {payload.get('type')}"

    print(f" {PASS} 3.B.1 Heartbeat ricevuto su topic: {topic} (QoS 1)")
    print(f" {PASS} 3.B.2 Payload JSON conforme alle specifiche:")
    print(json.dumps(payload, indent=6))


def test_video_clip_upload():
    print(f"\n{INFO} --- TEST 4: Upload Clip Video MP4 & Correlazione Backend ---")
    dummy_clip_path = "/tmp/test_clip_help.mp4"
    
    with open(dummy_clip_path, "wb") as f:
        f.write(b"SIMULATED_MP4_VIDEO_STREAM_BYTES")

    result = events.upload_video_clip(
        clip_path=dummy_clip_path,
        reason="segnale_aiuto",
        device_id=config.DEVICE_ID,
    )
    assert result is not None, "Upload clip video fallito!"
    assert "video_url" in result, "video_url mancante nella risposta del backend"
    print(f" {PASS} 4.1 Clip MP4 caricata con successo su POST /api/vision/clip")
    print(f"       -> video_url: {result.get('video_url')}")
    print(f"       -> event_id associato: {result.get('event_id')}")

    # Verifica accessibilità del video via GET
    get_resp = requests.get(result.get("video_url"), timeout=5)
    assert get_resp.status_code == 200, f"Video non accessibile via GET: {get_resp.status_code}"
    print(f" {PASS} 4.2 File video scaricabile via HTTP GET (Status 200 OK)")

    if os.path.exists(dummy_clip_path):
        os.remove(dummy_clip_path)


def main():
    print("=" * 65)
    print(" VERIFICA SISTEMA VISION NODE -> BACKEND & MQTT BROKER")
    print("=" * 65)
    print(f"Device MAC:   {config.DEVICE_ID}")
    print(f"Backend HTTP: {config.BACKEND_HTTP_URL}")
    print(f"Broker MQTT:  {config.MQTT_BROKER_HOST}:1883")
    print("=" * 65)

    test_handshake()
    frame_url = test_snapshot_upload()
    test_mqtt_alert(frame_url)
    test_mqtt_heartbeat()
    test_video_clip_upload()

    print("\n" + "=" * 65)
    print("\033[92m TUTTI I TEST SONO STATI SUPERATI CON SUCCESSO! (100% OK) \033[0m")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
