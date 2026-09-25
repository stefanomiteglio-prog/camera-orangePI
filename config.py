"""
Configurazione del nodo IA di visione (Stadio 1: detection oggetti).

Modifica questi valori in base al tuo setup: indice webcam, endpoint MQTT,
device_id da usare per far apparire questo nodo come un sensore ESP32
nel backend TreeEyes.
"""

# --- Sorgente video -----------------------------------------------------
CAMERA_INDEX = "/dev/v4l/by-id/usb-USB_Cam_Manufacturer_HDMI_USB_Camera_88901c069d01ae53-video-index0"         # Punta a /dev/video1 (HDMI USB Camera)
# CAMERA_INDEX = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN5100-video-index0"         # 0 = prima webcam disponibile. Cambia se usi una IP cam (vedi note in main.py)

# --- MQTT -----------------------------------------------------------------
MQTT_BROKER_HOST = "10.88.91.27"     # indirizzo del tuo broker Mosquitto
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "treeeyes/parco-robinson/vision/events"
DEVICE_ID = "vision-node-01"       # identifica questo nodo come faresti con un ESP32

# --- Telegram (opzionale, riusa il bot che hai gia') ----------------------
TELEGRAM_ENABLED = False           # metti True quando hai token e chat_id
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# --- Modello YOLO -----------------------------------------------------
# yolov8n.pt = modello COCO pre-addestrato "nano", leggero e veloce.
# Sulla tua RTX 5050 puoi tranquillamente salire a yolov8s.pt o yolov8m.pt
# per maggiore accuratezza mantenendo real-time.
YOLO_WEIGHTS = "yolov8s.pt"
YOLO_POSE_WEIGHTS = "yolov8s-pose.pt"
DEVICE = "cpu"   # "cuda" per usare la RTX 5050, "cpu" come fallback

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
EVENT_COOLDOWN_SECONDS = 30

FALL_ASPECT_RATIO = 1.4
FALL_TORSO_ANGLE_DEG = 55
FIGHT_DISTANCE_PX = 150
FIGHT_MOTION_RATIO_THRESHOLD = 0.35
MOTION_SMOOTHING_FRAMES = 5

PROTECTED_ZONE = (200, 150, 450, 400)
VANDAL_WRIST_RATIO_THRESHOLD = 0.35
VANDAL_FRAMES_THRESHOLD = 25

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
VIDEO_CLIP_PRE_SECONDS = 10
VIDEO_CLIP_POST_SECONDS = 10
VIDEO_CLIP_DIR = "clips"
VIDEO_CLIP_TRIGGER_TYPES = {"rissa", "arma", "persona_a_terra", "vandalismo", "fuoco_fumo", "segnale_aiuto"}
BACKEND_VIDEO_UPLOAD_URL = "http://10.88.91.27:8000/api/vision/clip"

VLM_ENABLED = True
VLM_ENDPOINT = "http://localhost:11434/api/generate"
VLM_MODEL = "llava:7b"
VLM_TIMEOUT_SECONDS = 60

VLM_QUESTIONS = {
    "arma": "Nell'immagine e' visibile un'arma (coltello, bastone, oggetto usato come arma) impugnata o minacciosamente vicino a una persona?",
    "persona_a_terra": "Nell'immagine c'e' una persona sdraiata o accasciata a terra, come se fosse caduta o ferita?",
    "rissa": "Nell'immagine due o piu' persone si stanno picchiando o aggredendo fisicamente?",
    "vandalismo": "Nell'immagine una persona sta danneggiando, colpendo o manomettendo un oggetto del parco (panchina, cestino)?",
    "fuoco_fumo": "Nell'immagine e' visibile fuoco o fumo reale?",
    "assembramento": "Nell'immagine c'e' un gruppo insolitamente numeroso di persone assembrate?",
    "segnale_aiuto": "Nell'immagine una persona sta facendo un gesto con la mano che sembra una richiesta di aiuto?",
}
VLM_DEFAULT_QUESTION = "Nell'immagine e' visibile una situazione di pericolo reale?"

# Soglie gesto aiuto basate sul tempo reale (indipendenti dagli FPS)
HELP_GESTURE_OPEN_HOLD_SECONDS = 0.3
HELP_GESTURE_TUCKED_HOLD_SECONDS = 0.8
