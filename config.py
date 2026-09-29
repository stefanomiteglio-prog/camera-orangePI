"""
Configurazione del nodo IA di visione (Stadio 1: detection oggetti).

Modifica questi valori in base al tuo setup: indice webcam, endpoint MQTT,
device_id da usare per far apparire questo nodo come un sensore ESP32
nel backend TreeEyes.
"""

# --- Sorgente video -----------------------------------------------------
CAMERA_INDEX = "/dev/v4l/by-id/usb-USB_Cam_Manufacturer_HDMI_USB_Camera_88901c069d01ae53-video-index0"         # Punta a /dev/video1 (HDMI USB Camera)
# CAMERA_INDEX = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN5100-video-index0"         # 0 = prima webcam disponibile. Cambia se usi una IP cam (vedi note in main.py)

# Risoluzione acquisizione (720p per massima efficienza, alzare a 1920x1080 se serve più dettaglio)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# --- Identificazione Dispositivo (MAC Address) ----------------------------
# Deve corrispondere al MAC address censito nel backend (es. c0:74:2b:fb:00:3f o mini_cam_01)
DEVICE_ID = "c0:74:2b:fb:00:3f"

# --- Backend HTTP REST ---------------------------------------------------
# Host Backend HTTP: http://<IP_SERVER>:8000 (o porta 5173 se tramite reverse-proxy)
BACKEND_HTTP_HOST = "192.168.88.10"
BACKEND_HTTP_PORT = 8000
BACKEND_HTTP_URL = f"http://{BACKEND_HTTP_HOST}:{BACKEND_HTTP_PORT}"

# Endpoint specifici per la telecamera
BACKEND_ME_URL = f"{BACKEND_HTTP_URL}/devices/me"           # Handshake/autorizzazione (o /api/vision/me)
BACKEND_SNAPSHOT_URL = f"{BACKEND_HTTP_URL}/api/vision/snapshot" # Upload snapshot JPG anteprima
BACKEND_CLIP_URL = f"{BACKEND_HTTP_URL}/api/vision/clip"         # Upload clip MP4 registrata
BACKEND_VIDEO_UPLOAD_URL = BACKEND_CLIP_URL                 # Retrocompatibilità

# Timeout richieste HTTP in secondi
HTTP_TIMEOUT_SECONDS = 10

# --- MQTT -----------------------------------------------------------------
MQTT_ENABLED = True                  # Imposta a False per disabilitare MQTT (test offline)
MQTT_BROKER_HOST = "192.168.88.10"    # Indirizzo del broker Mosquitto
MQTT_BROKER_PORT = 1883
# Topic conforme allo standard Parco: parco/<DEVICE_ID>/camera (oppure parco/<DEVICE_ID>/events)
MQTT_TOPIC = f"parco/{DEVICE_ID}/camera"

# --- Telegram (opzionale, riusa il bot che hai gia') ----------------------
TELEGRAM_ENABLED = False           # metti True quando hai token e chat_id
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# --- Modello YOLO / RKNN ----------------------------------------------
# Percorsi modelli RKNN compilati per NPU Rockchip (RK3588 su Orange Pi 5)
POSE_RKNN_PATH = "yolov8s-pose.rknn"
DETECT_RKNN_PATH = "rknn_test/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn"
RKNN_TARGET = "rk3588"

# Modelli PyTorch (usati nel test locale o fallback PC)
YOLO_WEIGHTS = "yolov8s.pt"
YOLO_POSE_WEIGHTS = "yolov8s-pose.pt"
DEVICE = "cpu"   # "cuda" per usare GPU NVIDIA, "cpu" come fallback

# Modalità di esecuzione e visualizzazione a schermo:
HEADLESS = False               # False apre la finestra OpenCV con HUD, True opera in background (headless/service)
WINDOW_NAME = "TreeEyes - Vision Node"  # Titolo della finestra
WINDOW_FULLSCREEN = True       # True = avvia a schermo intero (fullscreen), False = finestra ridimensionabile
WINDOW_MONITOR_X = 0           # Offset X (pixel) per selezionare il monitor (es. 0 per monitor principale, 1920 per secondo monitor HDMI)
WINDOW_MONITOR_Y = 0           # Offset Y (pixel) del monitor desiderato

# Gestione e ritenzione storage locale (pulizia automatica file obsoleti su SD/eMMC)
MAX_SNAPSHOTS_COUNT = 500
MAX_CLIPS_COUNT = 100
STORAGE_CLEANUP_INTERVAL_SECONDS = 1800  # ogni 30 minuti

# Confidence minima per considerare valida una rilevazione
CONF_THRESHOLD = 0.3

# Classi COCO gia' presenti nel modello pre-addestrato che ci interessano.
# COCO include gia' "knife" (coltello) e "baseball bat" (bastone) di serie.
# Fuoco/fumo NON e' una classe COCO: richiede un modello dedicato (vedi README).
DANGER_CLASS_MAP = {
    "knife": "arma",
    "baseball bat": "arma",
}

# Numero di frame consecutivi in cui il pericolo deve essere rilevato
# prima di generare un evento, per ridurre i falsi positivi.
CONSECUTIVE_FRAMES_THRESHOLD = 5

EVENT_COOLDOWN_SECONDS = 5
EVENT_COOLDOWN_MAP = {
    "segnale_aiuto": 5,             # Cooldown 5s per gesto aiuto (specifica)
    "arma": 10,
    "persona_a_terra": 10,
    "assembramento": 30,
}

FALL_ASPECT_RATIO = 1.4
FALL_TORSO_ANGLE_DEG = 55

CROWD_COUNT_THRESHOLD = 5
CROWD_SECONDS_THRESHOLD = 60

FIRE_WEIGHTS = "fire_smoke.pt"
FIRE_ENABLED = False

AUDIO_SAMPLE_RATE = 32000
AUDIO_CHUNK_SECONDS = 2
AUDIO_TAG_CONF_THRESHOLD = 0.15
AUDIO_TAGS_MAP = {
    "Glass": "vetro_rotto",
    "Shatter": "vetro_rotto",
    "Screaming": "urla",
    "Shout": "urla",
    "Yell": "urla",
}
WHISPER_MODEL = "small"
HELP_KEYWORDS = ["aiuto", "aiutatemi", "aiutami", "help"]
AUDIO_EVENT_COOLDOWN_SECONDS = 20

HELP_GESTURE_HOLD_FRAMES = 8
HELP_GESTURE_OPEN_HOLD_FRAMES = 3
HELP_GESTURE_WINDOW_SECONDS = 4
HELP_GESTURE_TUCK_RATIO = 0.35

VIDEO_CLIP_ENABLED = True
VIDEO_CLIP_PRE_SECONDS = 5          # Buffer 5s prima dell'evento
VIDEO_CLIP_POST_SECONDS = 5         # Buffer 5s dopo l'evento (3-5s da specifiche)
VIDEO_CLIP_DIR = "clips"
VIDEO_CLIP_TRIGGER_TYPES = {"arma", "persona_a_terra", "fuoco_fumo", "segnale_aiuto", "assembramento"}

VLM_ENABLED = True
VLM_ENDPOINT = "http://localhost:11434/api/generate"
VLM_MODEL = "llava:7b"
VLM_TIMEOUT_SECONDS = 60

VLM_QUESTIONS = {
    "arma": "Nell'immagine e' visibile un'arma (coltello, bastone, oggetto usato come arma) impugnata o minacciosamente vicino a una persona?",
    "persona_a_terra": "Nell'immagine c'e' una persona sdraiata o accasciata a terra, come se fosse caduta o ferita?",
    "fuoco_fumo": "Nell'immagine e' visibile fuoco o fumo reale?",
    "assembramento": "Nell'immagine c'e' un gruppo insolitamente numeroso di persone assembrate?",
    "segnale_aiuto": "Nell'immagine una persona sta facendo un gesto con la mano che sembra una richiesta di aiuto?",
}
VLM_DEFAULT_QUESTION = "Nell'immagine e' visibile una situazione di pericolo reale?"

# Soglie gesto aiuto basate sul tempo reale (indipendenti dagli FPS)
HELP_GESTURE_OPEN_HOLD_SECONDS = 0.3
HELP_GESTURE_TUCKED_HOLD_SECONDS = 0.8
