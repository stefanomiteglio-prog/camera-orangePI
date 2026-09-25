import math
import time

import cv2
import mediapipe as mp

import config

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

_gesture_state = {"open_since": None, "open_seen_at": None, "tucked_since": None}


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


def gesture_progress() -> float:
    if _gesture_state["tucked_since"] is None:
        return 0.0
    elapsed = time.time() - _gesture_state["tucked_since"]
    return min(1.0, elapsed / config.HELP_GESTURE_TUCKED_HOLD_SECONDS)


def detect_help_gesture(frame) -> bool:
    """
    Rileva il "Signal for Help": palmo aperto seguito, entro pochi secondi,
    da un pugno chiuso col pollice ripiegato dentro il palmo.
    Soglie basate sul tempo reale (secondi), non su conteggio frame,
    cosi' funzionano a qualsiasi framerate della pipeline.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        _gesture_state["tucked_since"] = None
        _gesture_state["open_since"] = None
        return False

    triggered = False
    now = time.time()

    for hand_landmarks in result.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = hand_landmarks.landmark
        diag = _hand_diag(lm)

        if _is_open_palm(lm, diag):
            if _gesture_state["open_since"] is None:
                _gesture_state["open_since"] = now
            if now - _gesture_state["open_since"] >= config.HELP_GESTURE_OPEN_HOLD_SECONDS:
                _gesture_state["open_seen_at"] = now
            _gesture_state["tucked_since"] = None

        elif _is_thumb_tucked_fist(lm, diag):
            _gesture_state["open_since"] = None
            open_at = _gesture_state["open_seen_at"]
            if open_at is not None and now - open_at <= config.HELP_GESTURE_WINDOW_SECONDS:
                if _gesture_state["tucked_since"] is None:
                    _gesture_state["tucked_since"] = now
                if now - _gesture_state["tucked_since"] >= config.HELP_GESTURE_TUCKED_HOLD_SECONDS:
                    triggered = True
                    _gesture_state["open_seen_at"] = None
                    _gesture_state["tucked_since"] = None
        else:
            _gesture_state["open_since"] = None
            _gesture_state["tucked_since"] = None

    return triggered
