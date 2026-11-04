# TreeEyes - Nodo IA di visione (Stadio 1+2)

Prototipo del nodo di visione locale: rileva **armi** (coltelli, bastoni) usando YOLOv8
pre-addestrato su COCO, e pubblica gli eventi via MQTT nello stesso "spirito" degli
ESP32, così il backend può generare automaticamente una pratica.

## Setup

```bash
cd treeeyes_vision
python -m venv venv
source venv/bin/activate   # su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

La prima esecuzione scarica automaticamente i pesi `yolov8n.pt` (~6MB).

Modifica `config.py`:
- `MQTT_BROKER_HOST` / `MQTT_BROKER_PORT` → il tuo broker Mosquitto
- `CAMERAS` → lista delle telecamere (vedi sotto "Due telecamere")
- `DEVICE` → lascia `"cuda"` per usare la RTX 5050 (molto più veloce), `"cpu"` come fallback
- `TELEGRAM_ENABLED = True` + token/chat_id se vuoi le notifiche subito attive

## Due telecamere (Zona A / Zona B)

Il nodo gira con **due telecamere in contemporanea**, definite nella lista
`CAMERAS` in `config.py`. Ogni telecamera invia al backend con il **proprio
`device_id`** (stringa libera, non più un MAC unico), da registrare nella lista
dispositivi del server:

```python
CAMERAS = [
    {"device_id": "treeeyes_zona_a", "zone": "Zona A",
     "camera_index": "/dev/v4l/by-id/usb-..._USB_2.0_Camera-video-index0", ...},
    {"device_id": "treeeyes_zona_b", "zone": "Zona B",
     "camera_index": "/dev/v4l/by-id/usb-..._USB_2.0_Camera_SN5100-video-index0", ...},
]
```

- `camera_index` usa il path **V4L2 by-id**: l'associazione zona→telecamera
  resta stabile anche se cambia l'ordine di enumerazione USB.
- Ogni telecamera pubblica su topic distinti: `parco/<device_id>/camera` e
  `parco/<device_id>/heartbeat`, con snapshot/clip a nome del proprio `device_id`.
- Ogni pipeline ha i propri modelli NPU (pose, detect, armi), ciascuno fissato
  su un core del RK3588 tramite `NPU_CORES` in `config.py`: la pose (la più
  pesante) ha un core per telecamera, detect e armi condividono il terzo.
  Ogni 300 frame il log riporta FPS e tempi NPU medi per modello, per
  verificare il bilanciamento.
- La finestra mostra le **due telecamere affiancate**, ciascuna con HUD, etichetta
  di zona e indicatore FPS.

Poi:
```bash
python main.py
```

Vedrai una finestra con il video e i box rossi disegnati sopra gli oggetti pericolosi
rilevati, utile per calibrare `CONF_THRESHOLD` e `CONSECUTIVE_FRAMES_THRESHOLD`.

## Cosa copre già questo Stadio 1

- ✅ Coltelli e bastoni ("knife", "baseball bat" sono già classi COCO, nessun training necessario)
- ✅ Logica anti-falsi-positivi: un pericolo deve persistere per N frame prima di scattare
- ✅ Cooldown per non spammare eventi/notifiche
- ✅ Pubblicazione MQTT compatibile con lo schema dei tuoi sensori ESP32
- ✅ Notifica Telegram immediata (riusa il bot che hai già)

## Stadio 2 (fatto)

- ✅ Tracking di sessione (ByteTrack, ID temporaneo per persona)
## Stadio 3 (fatto)

- ✅ Assembramento sospetto: conteggio persone sopra soglia per N secondi

## Fuoco/fumo

Disabilitato di default (`FIRE_ENABLED = False`). Per attivarlo:
1. Procurati un modello YOLO fire/smoke già addestrato (es. da Roboflow Universe, cerca "fire smoke detection yolov8")
2. Salvalo come `fire_smoke.pt` nella cartella
3. Metti `FIRE_ENABLED = True` in config.py

## Modulo audio (audio_node.py)

Script separato, gira in parallelo a `main.py`. Rileva:
- Vetro rotto, urla (PANNs, classificazione suoni pre-addestrata)
- Richieste di aiuto (Whisper, trascrizione + keyword spotting su "aiuto")

Uso:
```bash
python audio_node.py
```

Pubblica eventi sullo stesso topic MQTT/schema del nodo video.

## Adattare il payload MQTT

In `events.py`, la funzione `publish_event` costruisce un JSON con
`device_id`, `event_type`, `confidence`, `timestamp`, `snapshot`. Se il tuo
backend FastAPI si aspetta nomi di campo diversi per creare la pratica,
modifica quella funzione — è l'unico punto da toccare.

## Heartbeat periodico (Keep-alive TreeEyes)

Per evitare che la telecamera venga marcata come offline dal backend in assenza di allarmi, un thread dedicato in background (`HeartbeatService` in `events.py`) invia periodicamente un battito cardiaco:
- **Frequenza:** ogni 60 secondi (immediato all'avvio + periodico)
- **Topic MQTT:** `parco/<MAC_TELECAMERA>/heartbeat` (QoS 1)
- **Payload JSON:**
  ```json
  {
    "device_id": "<MAC_TELECAMERA>",
    "type": "heartbeat"
  }
  ```
Con due telecamere viene avviato **un heartbeat per ciascun `device_id`**, sui
rispettivi topic `parco/<device_id>/heartbeat`.

Parametri configurabili in `config.py`:
- `HEARTBEAT_INTERVAL_SECONDS = 60`

## Presenza (accensione luci di zona)

Quando una telecamera vede almeno una persona (detector COCO o modello pose) per
`PRESENCE_MIN_FRAMES` frame consecutivi, pubblica (`publish_presence` in `events.py`):
- **Topic MQTT:** `parco/<device_id>/presence` (QoS 1)
- **Payload JSON:**
  ```json
  {
    "device_id": "treeeyes_zona_a",
    "type": "presence",
    "presence": true,
    "person_count": 2,
    "sampling_time": "2026-09-25T15:30:00Z"
  }
  ```
Il backend accende le luci della **stessa zona** del `device_id` (campo zona del
dispositivo: deve contenere "Zona A" / "Zona B"). Non è un allarme: nessuno
snapshot/clip, nessun evento o pratica in webapp. Finché la persona resta in zona
il messaggio viene ripetuto ogni `PRESENCE_REPUBLISH_SECONDS` per tenere accese le luci.

Parametri configurabili in `config.py`:
- `PRESENCE_ENABLED = True`
- `PRESENCE_MIN_FRAMES = 3`
- `PRESENCE_REPUBLISH_SECONDS = 20` (deve restare sotto `presence_light_seconds` del backend, default 30)

## Skeleton completo e segnale di aiuto

- ✅ Skeleton corpo intero (MediaPipe Pose, 33 punti, stile "Palantir")
- ✅ Segnale di aiuto con le mani: palmo aperto → entro 4 secondi → pugno con
  pollice piegato dentro il palmo, tenuto per alcuni frame (gesto internazionale
  "Signal for Help")

Nota: MediaPipe Hands rileva max 2 mani senza associarle a un track_id specifico
(va bene con una persona sola in scena; con più persone contemporanee andrebbe
associato il gesto alla persona più vicina tramite le coordinate).

## Precisione migliorata

- Gesto aiuto: soglie normalizzate sulla diagonale della mano (invariante alla distanza), richiede palmo aperto stabile prima del pugno

## HUD di stato e Finestra a Schermo Intero

Pannello in alto a sinistra mostra live lo stato di ogni rilevatore (verde=ok,
rosso=attivo), il conteggio persone, e una barra di progresso per il gesto
aiuto. Indicatore REC in alto a destra quando sta registrando una clip.

### Gestione Finestra e Monitor (config.py):
- `WINDOW_FULLSCREEN = True`: avvia il programma direttamente a schermo intero.
- `WINDOW_MONITOR_X` / `WINDOW_MONITOR_Y`: seleziona su quale monitor proiettare la finestra (es. `WINDOW_MONITOR_X = 1920` per il secondo monitor HDMI).
- Tasti interattivi a runtime:
  - `f`: Attiva/disattiva schermo intero al volo (toggle fullscreen).
  - `q` / `ESC`: Chiude l'applicazione.
  - `r`: Reset manuale dei contatori e cooldown degli eventi.

## Clip video 5s prima + 5s dopo

Ogni evento rilevante (arma, fuoco/fumo, segnale aiuto, assembramento)
salva automaticamente una clip `.mp4` con buffer circolare pre-evento e post-evento,
poi la carica sul backend via HTTP POST a `BACKEND_CLIP_URL`.
Le clip restano anche salvate in `clips/`.

## Robustezza per l'esposizione

- Log eventi su file in `logs/treeeyes.log` (oltre alla console)
- Riconnessione automatica della webcam se si disconnette a metà demo
- Tasto `r` durante l'esecuzione: reset manuale di cooldown/contatori, utile per
  ripartire puliti tra una prova e l'altra davanti alla commissione

## Conferma "tipo Google Lens" (VLM locale)

Quando un'euristica rileva un pericolo, prima di notificare/creare la pratica,
la foto viene mandata a un modello di visione locale (via Ollama) che risponde
in linguaggio naturale se il pericolo è reale e lo descrive. Filtra i falsi
positivi delle euristiche e arricchisce la notifica con una descrizione umana.
Gira in background, non rallenta il video.

Setup:
1. Installa [Ollama](https://ollama.com/download) (Windows/Mac/Linux)
2. `ollama pull llava:7b` (circa 4.7GB, gira bene sulla tua RTX 5050)
3. Lascia Ollama avviato in background (di default ascolta su `localhost:11434`)

Se Ollama non è raggiungibile, il sistema passa gli eventi comunque
(fail-open, non blocca la demo) e lo segnala nei log.

Se `llava:7b` è troppo lento in demo, prova un modello più leggero: `ollama pull moondream`
e imposta `VLM_MODEL = "moondream"` in config.py.

## Note per il test in laboratorio

- L'assembramento richiede `CROWD_COUNT_THRESHOLD` persone per `CROWD_SECONDS_THRESHOLD` secondi
- `audio_node.py` scarica al primo avvio i pesi PANNs/Whisper (qualche centinaio di MB)

## Modelli NPU in INT8 (velocità)

In produzione pose e armi girano in INT8 a head separata, 640x640:
`yolov8s-pose_split640.rknn` (13 MB, ~40 ms per inferenza contro ~140 dell'FP16)
e `weapon_threat_split640.rknn` (4,1 MB, ~20 ms contro ~70). Lo yolov8n del
model zoo è già INT8 a 640x640 e resta com'è. I vecchi `yolov8s-pose.rknn` e
`weapon_threat.rknn` (export standard di Ultralytics, FP16) funzionano ancora.

L'export standard non si può quantizzare così com'è: mette coordinate in pixel e
confidenze nello stesso tensore e in INT8 le confidenze perderebbero precisione.
Per questo la conversione passa da un export **a head separata**
(`export_onnx_split.py`) e la decodifica la fa `rknn_backend.py` su CPU.
`rknn_backend.py` riconosce da solo il formato.

**Risoluzione: 640x640, non 640x480.** I modelli sono addestrati su immagini
640x640 con bordo grigio. A 640x480 (frame senza bordo) il modello armi perde le
esplosioni anche senza quantizzazione: confidenza massima sugli snapshot 0,69 in
FP16 640x640, 0,45 in FP16 640x480, 0,31 in INT8 640x480. In INT8 640x640 il
confronto con l'FP16 su 31 immagini reali è 23 rilevazioni su 23, esplosione 0,70.

1. **Export ONNX** (qualsiasi PC con `pip install ultralytics onnx`, anche Windows;
   default 640x640):
   ```bash
   python export_onnx_split.py yolov8s-pose.pt --output yolov8s-pose_split640.onnx
   python export_onnx_split.py weapon_threat.pt --output weapon_threat_split640.onnx
   ```
   Lo script stampa le classi del modello: per le armi l'ordine deve coincidere
   con `WEAPON_CLASSES` in `config.py`.
2. **Immagini di calibrazione**: le cartelle `snapshots/` e `clips/` dell'Orange Pi
   (150-300 immagini reali, varie per luce e presenza di persone).
3. **Conversione INT8** (Linux x86_64/aarch64 o WSL2, `pip install rknn-toolkit2 opencv-python`;
   sull'Orange Pi in un venv separato con rknn-toolkit2 della stessa versione di
   `librknnrt`, es. 2.3.2):
   ```bash
   python convert_rknn_int8.py yolov8s-pose_split640.onnx --calib snapshots clips
   python convert_rknn_int8.py weapon_threat_split640.onnx --calib snapshots clips
   ```
   Con `--analyze` confronta float e INT8 strato per strato. Sull'Orange Pi la pose
   richiede ~5 minuti con 150 immagini (`--max-images 150`).
4. **Sull'Orange Pi**: copia i `*_split640.rknn` nella cartella del progetto e
   aggiorna `POSE_RKNN_PATH` e `WEAPON_RKNN_PATH` in `config.py`;
   `NPU_INPUT_SIZE` resta `(640, 640)`.
5. **Verifica**: nel log `Prestazioni` pose ~40-50 ms e armi ~20-35 ms (con due
   telecamere sui core condivisi); `sudo cat /sys/kernel/debug/rknpu/load` mostra
   il carico per core. Controlla che armi e scheletri vengano ancora riconosciuti:
   se l'INT8 perde precisione su un modello, rifai la calibrazione con più immagini
   o usa `--analyze`.

### Cadenze e modalità riposo

Il ciclo di ogni telecamera aspetta i modelli del frame prima di passare al
successivo. Per contenere il carico (`config.py`):

- `PROCESS_MAX_FPS = 20`: frame elaborati al secondo per telecamera; le webcam USB
  ignorano `CAMERA_FPS` sotto 30, quindi i frame in eccesso vengono scartati.
- `POSE_PROCESS_EVERY = 2`, `DETECT_PROCESS_EVERY = 3`, `WEAPON_PROCESS_EVERY = 3`:
  nei frame saltati si riusano gli ultimi risultati.
- Modalità riposo: senza persone per `IDLE_AFTER_SECONDS` (5 s) i tre modelli NPU
  girano a turno ogni `IDLE_PROCESS_EVERY` frame (6); il log segnala
  "Modalità riposo attiva/disattivata" e la riga `Prestazioni` mostra "(riposo)".

Gli FPS del log e dell'HUD sono la media degli intervalli tra frame elaborati.
