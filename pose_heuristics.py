import math
from typing import Dict
from collections import deque

import config

prev_keypoints: Dict[int, tuple] = {}
motion_history: Dict[int, deque] = {}
wrist_history: Dict[int, deque] = {}

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
L_WRIST, R_WRIST = 9, 10


def is_fallen(box_xyxy, keypoints=None, kconf=None) -> bool:
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    if h == 0:
        return False

    have_good_keypoints = (
        keypoints and len(keypoints) > R_HIP and kconf is not None
        and kconf[L_SHOULDER] >= 0.5 and kconf[R_SHOULDER] >= 0.5
        and kconf[L_HIP] >= 0.5 and kconf[R_HIP] >= 0.5
    )

    if have_good_keypoints:
        sx = (keypoints[L_SHOULDER][0] + keypoints[R_SHOULDER][0]) / 2
        sy = (keypoints[L_SHOULDER][1] + keypoints[R_SHOULDER][1]) / 2
        hx = (keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2
        hy = (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2
        dx, dy = hx - sx, hy - sy
        angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
        return angle >= config.FALL_TORSO_ANGLE_DEG

    # Fallback: niente keypoints affidabili, usa solo l aspect ratio del box
    return (w / h) >= config.FALL_ASPECT_RATIO


def update_motion(track_id: int, keypoints, kconf=None) -> tuple:
    """Ritorna (motion_full_body, motion_polsi), entrambi smoothati.
    Ignora i keypoints con bassa confidenza per ridurre il rumore."""
    global prev_keypoints
    prev = prev_keypoints.get(track_id)

    raw_full = 0.0
    raw_wrist = 0.0
    valid_count = 0
    if prev is not None and len(prev) == len(keypoints):
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(prev, keypoints)):
            if kconf is not None and kconf[i] < 0.5:
                continue
            d = math.hypot(x2 - x1, y2 - y1)
            raw_full += d
            valid_count += 1
            if i in (L_WRIST, R_WRIST):
                raw_wrist += d

    prev_keypoints[track_id] = keypoints

    hist = motion_history.setdefault(track_id, deque(maxlen=config.MOTION_SMOOTHING_FRAMES))
    hist.append(raw_full)
    whist = wrist_history.setdefault(track_id, deque(maxlen=config.MOTION_SMOOTHING_FRAMES))
    whist.append(raw_wrist)

    return sum(hist) / len(hist), sum(whist) / len(whist)


def person_diag(box_xyxy) -> float:
    x1, y1, x2, y2 = box_xyxy
    return math.hypot(x2 - x1, y2 - y1) or 1.0


def person_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_fighting(people: list) -> bool:
    """people: lista di (center, motion, diag)"""
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            c1, m1, d1 = people[i]
            c2, m2, d2 = people[j]
            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            ratio1, ratio2 = m1 / d1, m2 / d2
            if dist <= config.FIGHT_DISTANCE_PX and ratio1 >= config.FIGHT_MOTION_RATIO_THRESHOLD and ratio2 >= config.FIGHT_MOTION_RATIO_THRESHOLD:
                return True
    return False


def in_zone(center, zone) -> bool:
    x, y = center
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= x <= zx2 and zy1 <= y <= zy2
