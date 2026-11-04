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

**Sull'Orange Pi è disattivata** (`VLM_ENABLED = False`): llava:7b sulla CPU del
RK3588 va in timeout o, guardando il frame intero, scarta coltelli veri. Per
usarla punta `VLM_ENDPOINT` a un Ollama sul server con GPU.

Se `llava:7b` è troppo lento in demo, prova un modello più leggero: `ollama pull moondream`
e imposta `VLM_MODEL = "moondream"` in config.py.

## Note per il test in laboratorio

- L'assembramento richiede `CROWD_COUNT_THRESHOLD` persone per `CROWD_SECONDS_THRESHOLD` secondi
- `audio_node.py` scarica al primo avvio i pesi PANNs/Whisper (qualche centinaio di MB)

## Modelli NPU in INT8 (velocità)

I modelli `yolov8s-pose.rknn` (26 MB) e `weapon_threat.rknn` (7,9 MB) nascono
dall'export standard di Ultralytics, che **non quantizza**: girano in FP16,
circa 3 volte più lenti dell'INT8 (la pose occupa ~110 ms per frame su un core).
Lo yolov8n del model zoo è già INT8.

L'export standard non si può quantizzare così com'è: mette coordinate in pixel e
confidenze nello stesso tensore e in INT8 le confidenze perderebbero precisione.
Per questo la conversione passa da un export **a head separata**
(`export_onnx_split.py`) e la decodifica la fa `rknn_backend.py` su CPU.
`rknn_backend.py` riconosce da solo il formato: i `.rknn` attuali continuano a funzionare.

1. **Export ONNX** (qualsiasi PC con `pip install ultralytics onnx`, anche Windows):
   ```bash
   python export_onnx_split.py yolov8s-pose.pt
   python export_onnx_split.py weapon_threat.pt
   python export_onnx_split.py yolov8n.pt
   ```
   Default input 640x480 (come le telecamere: nessun bordo, ~25% di calcolo in meno).
   Lo script stampa le classi del modello: per le armi l'ordine deve coincidere
   con `WEAPON_CLASSES` in `config.py`.
2. **Immagini di calibrazione**: copia dall'Orange Pi le cartelle `snapshots/` e `clips/`
   (100-300 immagini reali, varie per luce e presenza di persone).
3. **Conversione INT8** (Linux x86_64/aarch64 o WSL2, `pip install rknn-toolkit2 opencv-python`):
   ```bash
   python convert_rknn_int8.py yolov8s-pose_split.onnx --calib snapshots clips
   python convert_rknn_int8.py weapon_threat_split.onnx --calib snapshots clips
   python convert_rknn_int8.py yolov8n_split.onnx --calib snapshots clips
   ```
   Con `--analyze` confronta float e INT8 strato per strato.
4. **Sull'Orange Pi**: copia i tre `*_split.rknn`, aggiorna `POSE_RKNN_PATH`,
   `WEAPON_RKNN_PATH`, `DETECT_RKNN_PATH` e imposta `NPU_INPUT_SIZE = (640, 480)`
   in `config.py`. I tre modelli vanno sostituiti insieme (stessa dimensione di input).
5. **Verifica**: nel log `Prestazioni` la pose dovrebbe scendere a circa 30-40 ms
   (da ~110); `sudo cat /sys/kernel/debug/rknpu/load` mostra il carico per core.
   Controlla che armi e scheletri vengano ancora riconosciuti: se l'INT8 perde
   precisione su un modello, rifai la calibrazione con più immagini o usa `--analyze`.

## Rilevamento coltelli e armi

Due modelli lavorano insieme e sommano le prove nello stesso `alert_type`
(`coltello`): il modello armi dedicato (`WEAPON_RKNN_PATH`) e la classe `knife`
del detector COCO, addestrata proprio sui coltelli da cucina e quindi affidabile
a distanza ravvicinata (≤ 1-2 m).

Come decide (`main.py` + `danger_logic.py`):

1. **Soglia per classe** (`WEAPON_CLASS_THRESHOLDS`, `COCO_DANGER_THRESHOLDS`):
   coltello 0.40 (modello armi) e 0.35 (COCO). Le classi non elencate, come
   `explosion` e `grenade`, sono escluse già prima dell'argmax.
2. **Bonus mano**: se il box è vicino a un polso rilevato dal modello pose, la
   soglia scende di `DANGER_HAND_BONUS` (0.10). Con `DANGER_REQUIRE_HAND = True`
   contano solo gli oggetti in mano (consigliato quando le persone sono
   inquadrate per intero).
3. **Conferma a finestra mobile**: nelle ultime `DANGER_WINDOW_SECONDS` (2 s)
   il coltello deve essere visto per almeno `DANGER_CONFIRM_SECONDS` (1 s), in
   almeno `DANGER_MIN_HITS` frame e nel `DANGER_MIN_RATIO` (50%) dei frame. Una
   lama che sparisce per qualche frame non azzera la conferma, mentre un falso
   positivo isolato non arriva mai a far scattare l'allarme. La confidenza
   inviata via MQTT è la media della finestra.
4. `ALERT_MIN_CONF` (0.30) resta solo come rete di sicurezza in `publish_event`.

L'HUD mostra "Arma: coltello 65%", cioè quanto manca alla conferma. Sul box
compare "mano" quando l'oggetto è vicino a un polso.

**Taratura**: se i coltelli veri non scattano, abbassa di 0.05 la soglia della
classe. Se scattano falsi allarmi, alzala oppure attiva `DANGER_REQUIRE_HAND`.

**Telecamere**: `CAMERA_V4L2_CONTROLS` disattiva la "priorità esposizione"
(framerate fisso anche con poca luce, meno sfocatura). Serve `sudo apt install v4l-utils`.

### Raccolta campioni (per il riaddestramento)

Con `WEAPON_SAMPLES_ENABLED = True` ogni candidato arma con score ≥ 0.25 (al
massimo un frame ogni 2 s per telecamera) viene salvato in `weapon_samples/`,
con le etichette YOLO già compilate. Prima del training rivedili con CVAT,
Roboflow o labelImg: correggi i box e **svuota il .txt dei falsi positivi**
(diventano esempi negativi, molto utili).

## Modello coltelli: dataset e addestramento

`weapon_threat.rknn` (YOLOv8n, classi Gun/explosion/grenade/knife) è addestrato
su immagini web generiche. Il nuovo modello `weapon_knife` (YOLOv8s, classi
`knife`, `gun`) si addestra sul PC con GPU:

```bash
pip install -r requirements-train.txt      # dopo PyTorch con CUDA
```

**1. Dataset** (`prepare_weapon_dataset.py`): unisce le fonti e uniforma le classi.

| Fonte | Cosa porta | Come |
|---|---|---|
| COCO 2017 | coltelli da cucina in contesto reale | `--coco 4000` (FiftyOne, senza account) |
| Open Images V7 | Kitchen knife, Knife, Dagger, Handgun | `--openimages 3000` (FiftyOne, senza account) |
| COCO hard negative | forchette, cucchiai, forbici, telefoni… senza coltelli | `--negatives 2000` |
| [OD-WeaponDetection](https://github.com/ari-dasci/OD-WeaponDetection), Knife_detection | ~2000 immagini di coltelli da sorveglianza (CC-BY-4.0) | clona il repo, poi `--voc <cartella>` |
| [Roboflow: knife-detection](https://universe.roboflow.com/ai-0jtbr/knife-detection-hgvy2/dataset/2) | ~9000 immagini, export YOLOv8 | `--roboflow ai-0jtbr/knife-detection-hgvy2/2` + `ROBOFLOW_API_KEY` |
| I vostri campioni | stesse telecamere, luce e inquadratura | `--extra weapon_samples` (dopo la revisione) |

Altri dataset Roboflow Universe (es. [knife](https://universe.roboflow.com/main-maege/knife-gmyes),
[Knife_hand_holding](https://universe.roboflow.com/testing-y9g4b/knife_hand_holding)) si
aggiungono con `--roboflow workspace/progetto/versione`: la versione si legge
sulla pagina del dataset. Le classi composte come "holding-knife" vengono
scartate in automatico; forza la mappatura con `--map "Nome=knife"`.

Il repo OD-WeaponDetection pesa circa 2.2 GB. `--voc` cerca gli .xml in tutte le
sottocartelle: se il nome della cartella del subset coltelli è diverso, adatta il path.

```bash
git clone --depth 1 https://github.com/ari-dasci/OD-WeaponDetection datasets/_downloads/OD-WeaponDetection
python prepare_weapon_dataset.py --out datasets/weapon_knife --clean \
    --coco 4000 --openimages 3000 --negatives 2000 \
    --voc datasets/_downloads/OD-WeaponDetection/Knife_detection \
    --roboflow ai-0jtbr/knife-detection-hgvy2/2 \
    --extra weapon_samples
```

**2. Training** (crea `weapon_knife.pt` e `weapon_knife_split.onnx`):
```bash
python train_weapon_model.py --data datasets/weapon_knife/data.yaml
```
Stampa precision e recall per classe. `runs/weapon/weapon_knife/BoxF1_curve.png`
indica la confidenza ottimale da usare in `WEAPON_CLASS_THRESHOLDS`.

**3. Conversione RKNN** (Linux/WSL2). La calibrazione deve contenere coltelli:
```bash
python convert_rknn_int8.py weapon_knife_split.onnx --calib datasets/weapon_knife/images/val snapshots clips
```
L'algoritmo di default è `mmse`. Se l'INT8 perde precisione (`--analyze`),
`--fp16` crea un modello non quantizzato.

**4. Orange Pi**: copia `weapon_knife_split.rknn` e in `config.py` imposta
`WEAPON_RKNN_PATH = "weapon_knife_split.rknn"`, `WEAPON_CLASSES = ["knife", "gun"]`,
`WEAPON_CLASS_THRESHOLDS = {"knife": 0.40, "gun": 0.50}`. L'input è 640x480 come
gli altri modelli split: `NPU_INPUT_SIZE = (640, 480)` vale per tutti e tre.

Test della logica di conferma (senza NPU): `python test_danger_logic.py`.
