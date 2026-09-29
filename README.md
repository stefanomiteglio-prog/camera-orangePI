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
- `DEVICE` → lascia `"cuda"` per usare la RTX 5050 (molto più veloce), `"cpu"` come fallback
- `TELEGRAM_ENABLED = True` + token/chat_id se vuoi le notifiche subito attive

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
- ✅ Persona a terra (angolo del busto e aspect ratio)

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

## Skeleton completo e segnale di aiuto

- ✅ Skeleton corpo intero (MediaPipe Pose, 33 punti, stile "Palantir")
- ✅ Segnale di aiuto con le mani: palmo aperto → entro 4 secondi → pugno con
  pollice piegato dentro il palmo, tenuto per alcuni frame (gesto internazionale
  "Signal for Help")

Nota: MediaPipe Hands rileva max 2 mani senza associarle a un track_id specifico
(va bene con una persona sola in scena; con più persone contemporanee andrebbe
associato il gesto alla persona più vicina tramite le coordinate).

## Precisione migliorata

- Caduta: non solo aspect ratio del box, ma angolo del busto (spalle-anche) — più robusto
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

Ogni evento rilevante (arma, caduta, fuoco/fumo, segnale aiuto, assembramento)
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
