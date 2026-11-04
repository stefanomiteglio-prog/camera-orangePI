# Modifiche: rilevamento coltelli/armi

Scenario: coltelli da cucina, riprese a ≤ 1 m (persona inquadrata a busto),
telecamere a 17-20 FPS. Prima quasi nessun coltello veniva segnalato.

## Perché non funzionava

- `ALERT_MIN_CONF = 0.90` filtrava tutto: un coltello reale dà circa 0.4-0.8.
  Lo `WEAPON_CONF_THRESHOLD = 0.35` non veniva mai usato.
- Serviva vedere l'arma per **3 s di fila** sopra 0.90: due frame persi (lama di
  taglio, mano che ruota) azzeravano il conteggio.
- Le classi `explosion` e `grenade` del modello armi potevano nascondere il coltello.
- La conferma VLM (llava:7b) sull'Orange Pi andava in timeout o scartava
  coltelli veri.

## Cosa è cambiato

| File | Modifica |
|---|---|
| `config.py` | Soglie per classe (`WEAPON_CLASS_THRESHOLDS`, `COCO_DANGER_THRESHOLDS`), conferma a finestra mobile (`DANGER_WINDOW_*`), bonus mano (`DANGER_HAND_*`), `ALERT_MIN_CONF` 0.30, VLM disattivato, controlli V4L2, raccolta campioni |
| `danger_logic.py` (nuovo) | Conferma a finestra mobile e vicinanza oggetto-polso |
| `main.py` | Nuova valutazione pericoli (soglie per classe, bonus mano, finestra mobile), HUD con % di conferma, `v4l2-ctl` all'apertura delle telecamere |
| `rknn_backend.py` | Modello armi con solo le classi attive, NMS per classe |
| `events.py` | Soglia minima di default 0.30 |
| `sample_collector.py` (nuovo) | Salva frame con etichette YOLO per il riaddestramento |
| `prepare_weapon_dataset.py` (nuovo) | Dataset unificato da COCO, Open Images, OD-WeaponDetection, Roboflow e campioni propri |
| `train_weapon_model.py` (nuovo) | Addestramento YOLOv8s coltelli/pistole + export ONNX per la NPU |
| `convert_rknn_int8.py` | Quantizzazione `mmse` di default, opzione `--fp16` |
| `test_danger_logic.py` (nuovo) | Test della logica di conferma (`python test_danger_logic.py`) |
| `requirements-train.txt` (nuovo) | Dipendenze per dataset e training (PC con GPU) |
| `README.md` | Sezioni "Rilevamento coltelli e armi" e "Modello coltelli: dataset e addestramento" |
| `.gitignore`, `.gitattributes` | Esclusi dataset/campioni/run/modelli; fine riga LF per tutti |

## Prossimi passi

1. Sull'Orange Pi: `sudo apt install v4l-utils`, avvio di `main.py`, prova con un
   coltello. Soglie da tarare in `config.py` (vedi README).
2. Sul PC con GPU: dataset e training del nuovo modello (vedi README), poi
   conversione RKNN e cambio di `WEAPON_RKNN_PATH`/`WEAPON_CLASSES`.
3. Rivedere i campioni in `weapon_samples/` e aggiungerli al dataset.

I modelli `.rknn`/`.pt` non sono nel repository (vedi `.gitignore`): vanno
copiati a mano sull'Orange Pi.
