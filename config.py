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

# Controlli V4L2 applicati all'apertura con v4l2-ctl (sudo apt install v4l-utils).
# In poca luce le webcam UVC allungano l'esposizione e calano gli FPS: il
# coltello in movimento diventa una scia che nessun modello riconosce.
# Con la "priorità esposizione" disattivata il framerate resta fisso.
# I nomi cambiano tra kernel/telecamere (elenco: v4l2-ctl -d <dev> -L): quelli
# non supportati vengono ignorati. Per un'esposizione manuale breve aggiungi
# ad es. "auto_exposure": 1, "exposure_time_absolute": 150 (vecchi kernel:
# "exposure_auto": 1, "exposure_absolute": 150) e alza "gain".
CAMERA_V4L2_CONTROLS = {
    "exposure_auto_priority": 0,        # kernel vecchi
    "exposure_dynamic_framerate": 0,    # kernel >= 5.x
}

# --- Cadenza inferenze (1 = ogni frame, 2 = un frame sì e uno no, ...) ----
# Alzare questi valori aumenta gli FPS a scapito della reattività del rilevatore.
POSE_PROCESS_EVERY = 1     # stima pose/scheletro
DETECT_PROCESS_EVERY = 2   # detection oggetti COCO (persone/bastone/coltello)
WEAPON_PROCESS_EVERY = 2   # detection armi (weapon_threat.rknn), sfalsata di un frame rispetto a COCO
HANDS_PROCESS_EVERY = 3    # gesto aiuto (MediaPipe, il più costoso su CPU)
# MediaPipe gira sulla CPU: senza persone viste dal modello pose (NPU) è inutile
# cercare mani, e si salta del tutto.
HANDS_ONLY_WITH_PERSON = True

# Complessità del modello MediaPipe Hands: 0 = lite (veloce), 1 = full (preciso)
HANDS_MODEL_COMPLEXITY = 0

# --- Telecamere multiple (Zona A / Zona B) -------------------------------
# Ogni telecamera invia al backend con il PROPRIO 'device_id' (stringa libera,
# NON più un unico MAC). Registra questi device_id nella lista dispositivi del
# server, uno per zona.
#
# 'camera_index' può essere:
#   - "auto": la zona prende la prima telecamera USB libera (qualunque modello),
#     in ordine di zona; con una sola telecamera collegata si accende solo la
#     prima zona. Con due telecamere l'abbinamento zona/telecamera segue l'ordine
#     alfabetico dei nomi in /dev/v4l/by-id (stabile finché sono le stesse).
#   - un path V4L2 by-id: fissa quella telecamera fisica a quella zona,
#     indipendentemente dall'ordine di enumerazione USB (es.
#     "/dev/v4l/by-id/usb-USB_Cam_Manufacturer_HDMI_USB_Camera_88901c069d01ae53-video-index0").
#     Un path fisso non viene mai preso da una zona "auto".
#   - un intero (0, 1, ...) per un test locale su Windows/PC.
# Una zona si annuncia al backend (handshake + heartbeat) solo mentre la sua
# telecamera manda frame: senza telecamera la zona risulta offline.
CAMERAS = [
    {
        "device_id": "treeeyes_zona_a",
        "zone": "Zona A",
        "camera_index": "auto",
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
    },
    {
        "device_id": "treeeyes_zona_b",
        "zone": "Zona B",
        "camera_index": "auto",
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
    },
]

# --- Retrocompatibilità (single-camera: main_pc_backup.py, test_camera.py) -
# Alias sulla prima telecamera, così i vecchi script a singola telecamera
# continuano a funzionare senza modifiche. "auto" diventa la prima telecamera
# USB collegata (non /dev/video0: sull'Orange Pi è l'ingresso HDMI della scheda).
CAMERA_INDEX = CAMERAS[0]["camera_index"]
if CAMERA_INDEX == "auto":
    import glob as _glob
    _usb_cams = sorted(_glob.glob("/dev/v4l/by-id/*-video-index0"))
    CAMERA_INDEX = _usb_cams[0] if _usb_cams else 0

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

# Se il backend non risponde all'handshake (es. rete non ancora pronta
# all'accensione) si ritenta ogni N secondi finché non risponde.
HANDSHAKE_RETRY_SECONDS = 30

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

# Dimensione di input (larghezza, altezza) dei tre modelli NPU: deve essere quella
# con cui sono stati compilati i .rknn. I modelli attuali (export Ultralytics FP16
# e yolov8n del model zoo) sono 640x640. Dopo la conversione INT8 con
# export_onnx_split.py + convert_rknn_int8.py (default 640x480, vedi README) usa
# (640, 480): il frame della telecamera entra senza bordo, circa -25% di calcolo.
# I tre modelli vanno convertiti insieme, alla stessa dimensione.
NPU_INPUT_SIZE = (640, 640)

# Core NPU del RK3588 (0, 1, 2) assegnato a ogni modello, una voce per telecamera
# nello stesso ordine di CAMERAS. "auto" lascia la scelta al runtime RKNN, che
# però può mettere due inferenze sullo stesso core lasciandone un altro fermo.
# Layout di default: la pose (yolov8s, il modello più pesante, ogni frame) ha un
# core per telecamera; detect COCO e armi (più leggeri, a frame alterni, mai in
# contemporanea nella stessa pipeline) condividono il core centrale.
# Per verificare il bilanciamento guarda i tempi "NPU ms" nel log di stato.
NPU_CORES = [
    {"pose": 0, "detect": 1, "weapon": 1},
    {"pose": 2, "detect": 1, "weapon": 1},
]

# Modello dedicato armi/minacce (presente solo sull'Orange Pi, nella cartella
# del progetto). Se il file manca il nodo parte comunque, senza questo modello.
WEAPON_ENABLED = True
WEAPON_RKNN_PATH = "weapon_threat.rknn"
# Ordine delle classi = ordine di addestramento del modello: NON cambiarlo.
WEAPON_CLASSES = ["Gun", "explosion", "grenade", "knife"]
# Nuovo modello addestrato con train_weapon_model.py (vedi README, "Modello
# coltelli"): dopo export_onnx_split.py + convert_rknn_int8.py usa
#   WEAPON_RKNN_PATH = "weapon_knife_split.rknn"
#   WEAPON_CLASSES = ["knife", "gun"]
#   WEAPON_CLASS_THRESHOLDS = {"knife": 0.40, "gun": 0.50}
# (e NPU_INPUT_SIZE = (640, 480) insieme agli altri modelli INT8).

# Soglia di rilevamento PER CLASSE del modello armi. Solo le classi elencate
# vengono considerate: "explosion" e "grenade" sono escluse (inutili in questo
# scenario, generavano solo falsi positivi e "rubavano" celle al coltello).
# Riferimento: un coltello da cucina a ~1 m dà tipicamente 0.4-0.8; la vecchia
# soglia 0.90 scartava quasi tutti i rilevamenti veri. I falsi positivi isolati
# li elimina la conferma a finestra mobile (DANGER_WINDOW_*), non la soglia.
WEAPON_CLASS_THRESHOLDS = {
    "knife": 0.40,
    "Gun": 0.50,
}
# Classe del modello -> alert_type inviato al backend
WEAPON_CLASS_MAP = {
    "Gun": "arma_da_fuoco",
    "gun": "arma_da_fuoco",
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
DISPLAY_MAX_FPS = 30           # Ricomposizione finestra al massimo a questo ritmo, e solo con frame nuovi (libera CPU)

# Gestione e ritenzione storage locale (pulizia automatica file obsoleti su SD/eMMC)
MAX_SNAPSHOTS_COUNT = 500
MAX_CLIPS_COUNT = 100
STORAGE_CLEANUP_INTERVAL_SECONDS = 1800  # ogni 30 minuti

# Confidence minima per considerare valida una rilevazione
CONF_THRESHOLD = 0.3

# Confidenza minima di qualunque allarme pubblicato via MQTT (deve essere
# SUPERIORE al valore): ultima rete di sicurezza in events.publish_event.
# Le soglie vere sono per classe (WEAPON_CLASS_THRESHOLDS, COCO_DANGER_THRESHOLDS).
# Per le armi la confidenza pubblicata è la MEDIA degli score nella finestra di
# conferma. Era 0.90: bloccava di fatto ogni rilevamento di coltello.
ALERT_MIN_CONF = 0.30

# Classi COCO gia' presenti nel modello pre-addestrato che ci interessano.
# COCO include gia' "knife" (coltello) e "baseball bat" (bastone) di serie.
# Fuoco/fumo NON e' una classe COCO: richiede un modello dedicato (vedi README).
# "knife" usa lo stesso alert_type del modello armi (WEAPON_CLASS_MAP), così i
# due modelli condividono conferma e cooldown e non generano allarmi doppi.
# Il "knife" COCO è addestrato proprio su coltelli da cucina: a distanza
# ravvicinata (<= 1-2 m) è un buon secondo parere rispetto al modello armi.
DANGER_CLASS_MAP = {
    "knife": "coltello",
    "baseball bat": "arma",
}
# Soglia per classe COCO (solo le classi elencate generano allarmi)
COCO_DANGER_THRESHOLDS = {
    "knife": 0.35,
    "baseball bat": 0.45,
}

# Oggetto vicino a un polso di una persona (keypoint del modello pose): la
# soglia di quella rilevazione viene abbassata di DANGER_HAND_BONUS. Un coltello
# in mano è molto più credibile di uno "visto" sullo sfondo.
DANGER_HAND_BONUS = 0.10
# Distanza massima polso-oggetto, in frazione dell'altezza della persona
DANGER_HAND_MARGIN = 0.25
# True = conta SOLO oggetti vicini a una mano. Riduce molto i falsi positivi,
# ma richiede che il modello pose veda i polsi: con riprese a ~1 m che
# inquadrano solo busto e mani la persona può non essere rilevata, quindi di
# default è False. Attivalo se il campo inquadra le persone per intero.
DANGER_REQUIRE_HAND = False

# Numero di frame consecutivi in cui il pericolo deve essere rilevato
# prima di generare un evento, per ridurre i falsi positivi.
CONSECUTIVE_FRAMES_THRESHOLD = 5   # usato solo da main_pc_backup.py

# Conferma armi a finestra mobile (main.py + danger_logic.py), indipendente
# dagli FPS. Nella finestra degli ultimi DANGER_WINDOW_SECONDS l'arma deve:
#   - essere vista per almeno DANGER_CONFIRM_SECONDS (dal primo all'ultimo avvistamento),
#   - in almeno DANGER_MIN_HITS frame,
#   - e nella frazione DANGER_MIN_RATIO dei frame dal primo avvistamento.
# Così una lama che sparisce per qualche frame (di taglio, mano che ruota) non
# azzera la conferma, mentre i falsi positivi isolati non arrivano mai a scattare.
# A 17-20 FPS: 1 s di conferma = circa 9-10 frame positivi su ~18-20.
DANGER_WINDOW_SECONDS = 2.0
DANGER_CONFIRM_SECONDS = 1.0
DANGER_MIN_HITS = 6
DANGER_MIN_RATIO = 0.5

# --- Raccolta campioni per il riaddestramento (sample_collector.py) -------
# Salva frame + etichette YOLO pre-compilate quando un modello vede un'arma
# candidata (anche sotto soglia): dopo una revisione manuale diventano dataset
# per train_weapon_model.py (prepare_weapon_dataset.py --extra weapon_samples).
WEAPON_SAMPLES_ENABLED = True
WEAPON_SAMPLES_DIR = "weapon_samples"
WEAPON_SAMPLES_MIN_CONF = 0.25
WEAPON_SAMPLES_INTERVAL_SECONDS = 2.0    # al massimo un frame ogni N secondi per telecamera
WEAPON_SAMPLES_MAX_COUNT = 3000          # i più vecchi vengono eliminati

EVENT_COOLDOWN_SECONDS = 5
EVENT_COOLDOWN_MAP = {
    "segnale_aiuto": 5,             # Cooldown 5s per gesto aiuto (specifica)
    "arma": 10,
    "arma_da_fuoco": 10,
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
VIDEO_CLIP_TRIGGER_TYPES = {"arma", "arma_da_fuoco", "coltello", "granata", "esplosione", "fuoco_fumo", "segnale_aiuto", "assembramento"}

# Conferma VLM (Ollama) prima di pubblicare. DISATTIVATA sull'Orange Pi:
# llava:7b sulla CPU del RK3588 impiega da decine di secondi a minuti, va in
# timeout (fail-open = allarme comunque, ma in ritardo) oppure, guardando il
# frame intero dove il coltello è piccolo, scarta allarmi veri. Per usarla
# punta VLM_ENDPOINT a un Ollama sul server con GPU (es. http://192.168.88.10:11434/...).
VLM_ENABLED = False
VLM_ENDPOINT = "http://localhost:11434/api/generate"
VLM_MODEL = "llava:7b"
VLM_TIMEOUT_SECONDS = 60

VLM_QUESTIONS = {
    "arma": "Nell'immagine e' visibile un'arma (coltello, bastone, oggetto usato come arma) impugnata o minacciosamente vicino a una persona?",
    "arma_da_fuoco": "Nell'immagine e' visibile un'arma da fuoco (pistola, fucile) impugnata o vicino a una persona?",
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
