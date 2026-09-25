import cv2

GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)
WHITE = (255, 255, 255)


def draw_panel(frame, rows, recording=False):
    h, w = frame.shape[:2]
    panel_h = 24 * len(rows) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 22
    for label, active, extra in rows:
        color = RED if active else GREEN
        text = f"{label}: {'ATTIVO' if active else 'ok'}"
        if extra:
            text += f" {extra}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 24

    if recording:
        cv2.circle(frame, (w - 25, 25), 8, RED, -1)
        cv2.putText(frame, "REC", (w - 70, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)


def draw_progress_bar(frame, label, progress, pos=(10, None)):
    h, w = frame.shape[:2]
    y = h - 40
    x = 10
    bar_w = 220
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * progress), y + 14), YELLOW, -1)
