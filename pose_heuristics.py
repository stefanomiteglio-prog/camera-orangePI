import math

import config

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


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


