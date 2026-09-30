import math
import time

import cv2
import mediapipe as mp

import config

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def _hand_diag(landmarks) -> float:
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0


def _is_open_palm(landmarks, diag) -> bool:
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    extended = 0
    for tip, mcp in zip(tips, mcps):
        dist = math.hypot(landmarks[tip].x - landmarks[0].x, landmarks[tip].y - landmarks[0].y)
        mcp_dist = math.hypot(landmarks[mcp].x - landmarks[0].x, landmarks[mcp].y - landmarks[0].y)
        if dist > mcp_dist * 1.15:
            extended += 1
    return extended >= 3


def _is_thumb_tucked_fist(landmarks, diag) -> bool:
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    folded = 0
    for tip, mcp in zip(tips, mcps):
        dist = math.hypot(landmarks[tip].x - landmarks[0].x, landmarks[tip].y - landmarks[0].y)
        mcp_dist = math.hypot(landmarks[mcp].x - landmarks[0].x, landmarks[mcp].y - landmarks[0].y)
        if dist < mcp_dist * 1.05:
            folded += 1

    thumb_tip = landmarks[4]
    palm = landmarks[9]
    thumb_dist_ratio = math.hypot(thumb_tip.x - palm.x, thumb_tip.y - palm.y) / diag

    return folded >= 3 and thumb_dist_ratio < config.HELP_GESTURE_TUCK_RATIO


class HelpGestureDetector:
    """
    Rilevatore del "Signal for Help" con stato e istanza MediaPipe Hands PROPRI.

    Ogni telecamera deve avere il suo detector: lo stato del gesto (palmo aperto →
    pugno col pollice ripiegato) e il grafo MediaPipe non sono condivisibili tra
    più stream senza falsare i rilevamenti.
    """

    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._state = {"open_since": None, "open_seen_at": None, "tucked_since": None}

    def progress(self) -> float:
        if self._state["tucked_since"] is None:
            return 0.0
        elapsed = time.time() - self._state["tucked_since"]
        return min(1.0, elapsed / config.HELP_GESTURE_TUCKED_HOLD_SECONDS)

    def detect(self, frame) -> bool:
        """
        Rileva il "Signal for Help": palmo aperto seguito, entro pochi secondi,
        da un pugno chiuso col pollice ripiegato dentro il palmo.
        Soglie basate sul tempo reale (secondi), non su conteggio frame,
        così funzionano a qualsiasi framerate della pipeline.
        Disegna gli skeleton delle mani sul frame passato (in-place).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            self._state["tucked_since"] = None
            self._state["open_since"] = None
            return False

        triggered = False
        now = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            diag = _hand_diag(lm)

            if _is_open_palm(lm, diag):
                if self._state["open_since"] is None:
                    self._state["open_since"] = now
                if now - self._state["open_since"] >= config.HELP_GESTURE_OPEN_HOLD_SECONDS:
                    self._state["open_seen_at"] = now
                self._state["tucked_since"] = None

            elif _is_thumb_tucked_fist(lm, diag):
                self._state["open_since"] = None
                open_at = self._state["open_seen_at"]
                if open_at is not None and now - open_at <= config.HELP_GESTURE_WINDOW_SECONDS:
                    if self._state["tucked_since"] is None:
                        self._state["tucked_since"] = now
                    if now - self._state["tucked_since"] >= config.HELP_GESTURE_TUCKED_HOLD_SECONDS:
                        triggered = True
                        self._state["open_seen_at"] = None
                        self._state["tucked_since"] = None
            else:
                self._state["open_since"] = None
                self._state["tucked_since"] = None

        return triggered

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass


# ============================================================
# Compatibilità con codice legacy (singola telecamera):
# main_pc_backup.py e altri usano le funzioni a livello di modulo.
# ============================================================
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
