"""
Configurazione del nodo IA di visione (Stadio 1: detection oggetti).

Modifica questi valori in base al tuo setup: indice webcam, endpoint MQTT,
device_id da usare per far apparire questo nodo come un sensore ESP32
nel backend TreeEyes.
"""

# --- Sorgente video: risoluzione/fps di default per ogni telecamera -----
# Valori usati da una voce di CAMERAS quando non specifica width/height/fps.
# Con DUE telecamere in parallelo si privilegia il framerate: 640x480 alleggerisce
# bus USB, resize e MediaPipe (i modelli NPU lavorano comunque a 640 in input,
# quindi la detection perde poco). Per più dettaglio: 800x600 o 1280x720.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# --- Cadenza inferenze (1 = ogni frame, 2 = un frame sì e uno no, ...) ----
# Alzare questi valori aumenta gli FPS a scapito della reattività del rilevatore.
POSE_PROCESS_EVERY = 1     # stima pose/scheletro
DETECT_PROCESS_EVERY = 2   # detection oggetti COCO (persone/bastone/coltello)
WEAPON_PROCESS_EVERY = 2   # detection armi (weapon_threat.rknn), sfalsata di un frame rispetto a COCO
HANDS_PROCESS_EVERY = 3    # gesto aiuto (MediaPipe, il più costoso su CPU)

# Complessità del modello MediaPipe Hands: 0 = lite (veloce), 1 = full (preciso)
HANDS_MODEL_COMPLEXITY = 0

# --- Telecamere multiple (Zona A / Zona B) -------------------------------
# Ogni telecamera invia al backend con il PROPRIO 'device_id' (stringa libera,
# NON più un unico MAC). Registra questi device_id nella lista dispositivi del
# server, uno per zona.
#
# 'camera_index' è il path stabile V4L2 by-id: identifica fisicamente la
# telecamera indipendentemente dall'ordine di enumerazione USB all'avvio.
# (Su Windows/PC per un test locale puoi mettere un intero: 0, 1, ...)
CAMERAS = [
    {
        "device_id": "treeeyes_zona_a",
        "zone": "Zona A",
        "camera_index": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-video-index0",
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
    },
    {
        "device_id": "treeeyes_zona_b",
        "zone": "Zona B",
        "camera_index": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN5100-video-index0",
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
    },
]

# --- Retrocompatibilità (single-camera: main_pc_backup.py, test_camera.py) -
# Alias sulla prima telecamera, così i vecchi script a singola telecamera
# continuano a funzionare senza modifiche.
CAMERA_INDEX = CAMERAS[0]["camera_index"]

# device_id di fallback, usato SOLO se una funzione viene chiamata senza un
# device_id esplicito. Con il multi-camera ogni pipeline passa sempre il proprio.
DEVICE_ID = CAMERAS[0]["device_id"]

# client_id univoco della connessione MQTT (una sola connessione condivisa da
# entrambe le telecamere, che pubblicano ciascuna sul proprio topic).
MQTT_CLIENT_ID = "treeeyes-vision-node"

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
# Topic heartbeat periodico per segnalare stato online: parco/<DEVICE_ID>/heartbeat
MQTT_HEARTBEAT_TOPIC = f"parco/{DEVICE_ID}/heartbeat"
# Intervallo invio heartbeat in secondi (specifica TreeEyes: ogni 60 secondi)
HEARTBEAT_INTERVAL_SECONDS = 60

# --- Telegram (opzionale, riusa il bot che hai gia') ----------------------
TELEGRAM_ENABLED = False           # metti True quando hai token e chat_id
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# --- Modello YOLO / RKNN ----------------------------------------------
# Percorsi modelli RKNN compilati per NPU Rockchip (RK3588 su Orange Pi 5)
POSE_RKNN_PATH = "yolov8s-pose.rknn"
DETECT_RKNN_PATH = "rknn_test/rknn_model_zoo/examples/yolov8/model/yolov8n.rknn"
RKNN_TARGET = "rk3588"

# Modello dedicato armi/minacce (presente solo sull'Orange Pi, nella cartella
# del progetto). Se il file manca il nodo parte comunque, senza questo modello.
WEAPON_ENABLED = True
WEAPON_RKNN_PATH = "weapon_threat.rknn"
# Ordine delle classi = ordine di addestramento del modello: NON cambiarlo.
WEAPON_CLASSES = ["Gun", "explosion", "grenade", "knife"]
# Soglia dedicata: più alta di CONF_THRESHOLD perché il modello tende ai falsi
# positivi a bassa confidenza (nel test a 0.10 riconosce "qualsiasi oggetto").
WEAPON_CONF_THRESHOLD = 0.35
# Classe del modello -> alert_type inviato al backend. Le classi non elencate
# qui ("Gun") vengono ignorate: nessun riquadro e nessuna segnalazione.
WEAPON_CLASS_MAP = {
    "explosion": "esplosione",
    "grenade": "granata",
    "knife": "coltello",
}

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
# "knife" usa lo stesso alert_type del modello armi (WEAPON_CLASS_MAP), così i
# due modelli condividono contatore e cooldown e non generano allarmi doppi.
DANGER_CLASS_MAP = {
    "knife": "coltello",
    "baseball bat": "arma",
}

# Numero di frame consecutivi in cui il pericolo deve essere rilevato
# prima di generare un evento, per ridurre i falsi positivi.
CONSECUTIVE_FRAMES_THRESHOLD = 5   # usato solo da main_pc_backup.py

# Conferma a tempo (main.py): arma, coltello, granata ed esplosione
# devono restare visibili per questi secondi prima di inviare la segnalazione.
# Indipendente dagli FPS.
DANGER_CONFIRM_SECONDS = 3.0
# Il detector "sfarfalla": buchi più brevi di questa durata non azzerano il
# conteggio dei secondi. Oltre, l'arma è considerata sparita e si riparte da zero.
DANGER_CONFIRM_GAP_SECONDS = 0.5

EVENT_COOLDOWN_SECONDS = 5
EVENT_COOLDOWN_MAP = {
    "segnale_aiuto": 5,             # Cooldown 5s per gesto aiuto (specifica)
    "arma": 10,
    "coltello": 10,
    "granata": 10,
    "esplosione": 10,
    "assembramento": 30,
}

# --- Presenza (accensione luci di zona) -----------------------------------
# Quando la telecamera vede almeno una persona pubblica parco/<device_id>/presence:
# il backend accende le luci della stessa zona (non è un allarme, non va in webapp).
PRESENCE_ENABLED = True
# Frame consecutivi con almeno una persona prima di segnalare la presenza
PRESENCE_MIN_FRAMES = 3
# Finché la persona resta in zona il messaggio viene ripetuto con questo intervallo,
# per tenere accese le luci: deve restare SOTTO la durata di accensione impostata
# sul backend (setting "presence_light_seconds", default 30s).
PRESENCE_REPUBLISH_SECONDS = 20

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
VIDEO_CLIP_TRIGGER_TYPES = {"arma", "coltello", "granata", "esplosione", "fuoco_fumo", "segnale_aiuto", "assembramento"}

VLM_ENABLED = True
VLM_ENDPOINT = "http://localhost:11434/api/generate"
VLM_MODEL = "llava:7b"
VLM_TIMEOUT_SECONDS = 60

VLM_QUESTIONS = {
    "arma": "Nell'immagine e' visibile un'arma (coltello, bastone, oggetto usato come arma) impugnata o minacciosamente vicino a una persona?",
    "coltello": "Nell'immagine e' visibile un coltello o una lama impugnata o minacciosamente vicino a una persona?",
    "granata": "Nell'immagine e' visibile una granata o un ordigno esplosivo?",
    "esplosione": "Nell'immagine e' visibile un'esplosione reale (fiammata, palla di fuoco, nube di detriti)?",
    "fuoco_fumo": "Nell'immagine e' visibile fuoco o fumo reale?",
    "assembramento": "Nell'immagine c'e' un gruppo insolitamente numeroso di persone assembrate?",
    "segnale_aiuto": "Nell'immagine una persona sta facendo un gesto con la mano che sembra una richiesta di aiuto?",
}
VLM_DEFAULT_QUESTION = "Nell'immagine e' visibile una situazione di pericolo reale?"

# Soglie gesto aiuto basate sul tempo reale (indipendenti dagli FPS)
HELP_GESTURE_OPEN_HOLD_SECONDS = 0.3
HELP_GESTURE_TUCKED_HOLD_SECONDS = 0.8
