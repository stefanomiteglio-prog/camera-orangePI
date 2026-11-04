"""
Riconoscimento del gesto di aiuto ("Signal for Help"): stesso metodo del
vecchio progetto Telecamera (riconoscimento_mano_definitivo.py), che lì
funzionava bene.

  - MediaPipe Hands, una mano, confidenza 0.7, modello completo;
  - immagine specchiata come in fase di registrazione del dataset: la
    specchiatura è applicata alle coordinate (x -> -x), equivalente a
    cv2.flip(frame, 1) senza copiare l'immagine;
  - per ogni frame 21 landmark come (x - x_polso, y - y_polso);
  - sequenza degli ultimi 15 frame con la mano -> RandomForest (630 feature);
  - allarme quando la predizione "aiuto" esce per più di 3 volte di fila.

Il modello viene da train_gesture_classifier.py (dataset gesture_dataset/dataset_seq.csv).
"""
import os
import pickle
import time
from collections import deque

import numpy as np

import config


def landmarks_to_frame(landmarks, mirror=True):
    """Landmark MediaPipe -> 42 valori (x - x_polso, y - y_polso), come nel dataset."""
    pts = np.array([(p.x, p.y) for p in landmarks], dtype=np.float32)
    pts -= pts[0]
    if mirror:
        pts[:, 0] = -pts[:, 0]
    return pts.reshape(-1)


class HelpGestureDetector:
    """
    Stato e istanza MediaPipe Hands PROPRI: ogni telecamera deve avere il suo
    detector (sequenza di landmark e grafo MediaPipe non sono condivisibili).
    """

    def __init__(self, model_path=None):
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        conf = getattr(config, "HELP_GESTURE_HAND_CONFIDENCE", 0.7)
        self.hands = self._mp_hands.Hands(
            max_num_hands=1,
            model_complexity=getattr(config, "HANDS_MODEL_COMPLEXITY", 1),
            min_detection_confidence=conf,
            min_tracking_confidence=conf,
        )
        path = model_path or getattr(config, "HELP_GESTURE_MODEL_PATH", "gesture_model.pkl")
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.model.n_jobs = 1   # predizione singola: i thread costano più del calcolo
        self._aiuto_idx = list(self.model.classes_).index("aiuto")
        self.seq_len = data.get("sequence_length", getattr(config, "HELP_GESTURE_SEQUENCE_LENGTH", 15))

        self.mirror = getattr(config, "HELP_GESTURE_MIRROR", True)
        self.needed = getattr(config, "HELP_GESTURE_CONSECUTIVE", 4)
        self.reset_s = getattr(config, "HELP_GESTURE_RESET_SECONDS", 2.0)

        self._seq = deque(maxlen=self.seq_len)
        self._last_hand_t = 0.0
        self._probs = []            # probabilità "aiuto" delle predizioni consecutive
        self.last_confidence = 0.0  # media delle probabilità che hanno generato l'ultimo allarme

    def _reset(self):
        self._seq.clear()
        self._probs = []

    def progress(self) -> float:
        return min(1.0, len(self._probs) / self.needed) if self.needed > 0 else 0.0

    def idle(self):
        """Nessuna persona in scena: si azzera la sequenza senza eseguire MediaPipe."""
        self._reset()

    def detect(self, frame, draw=True) -> bool:
        import cv2

        now = time.time()
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            # Come nel vecchio progetto la sequenza sopravvive a brevi perdite
            # della mano; dopo reset_s senza mano riparte da zero.
            if now - self._last_hand_t > self.reset_s:
                self._reset()
            return False

        hand = result.multi_hand_landmarks[0]
        if draw:
            self._mp_draw.draw_landmarks(frame, hand, self._mp_hands.HAND_CONNECTIONS)
        if now - self._last_hand_t > self.reset_s:
            self._reset()
        self._last_hand_t = now
        self._seq.append(landmarks_to_frame(hand.landmark, self.mirror))
        if len(self._seq) < self.seq_len:
            return False

        probs = self.model.predict_proba(np.concatenate(self._seq).reshape(1, -1))[0]
        if int(np.argmax(probs)) != self._aiuto_idx:
            self._probs = []
            return False
        self._probs.append(float(probs[self._aiuto_idx]))
        if len(self._probs) >= self.needed:
            self.last_confidence = sum(self._probs) / len(self._probs)
            self._reset()
            return True
        return False

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass


# Compatibilità con codice legacy a singola telecamera (main_pc_backup.py)
_default_detector = None


def _get_default() -> HelpGestureDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = HelpGestureDetector()
    return _default_detector


def detect_help_gesture(frame) -> bool:
    return _get_default().detect(frame)


def gesture_progress() -> float:
    return _get_default().progress()
