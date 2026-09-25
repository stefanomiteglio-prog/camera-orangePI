import os
import time
import threading
from collections import deque

import cv2
import requests

import config
from logger import logger


class ClipRecorder:
    def __init__(self):
        self.buffer = deque()
        self.recording_frames = None
        self.recording_until = None
        self.active_reason = None

    def is_recording(self) -> bool:
        return self.recording_until is not None

    def update(self, frame):
        now = time.time()
        self.buffer.append((now, frame.copy()))
        while self.buffer and now - self.buffer[0][0] > config.VIDEO_CLIP_PRE_SECONDS:
            self.buffer.popleft()

        if self.recording_until is not None:
            self.recording_frames.append((now, frame.copy()))
            if now >= self.recording_until:
                self._finalize()

    def trigger(self, danger_type: str):
        if not config.VIDEO_CLIP_ENABLED or danger_type not in config.VIDEO_CLIP_TRIGGER_TYPES:
            return
        if self.recording_until is not None:
            return
        self.active_reason = danger_type
        self.recording_frames = list(self.buffer)
        self.recording_until = time.time() + config.VIDEO_CLIP_POST_SECONDS

    def _finalize(self):
        frames = self.recording_frames
        reason = self.active_reason
        self.recording_frames = None
        self.recording_until = None
        self.active_reason = None
        threading.Thread(target=self._write_and_upload, args=(frames, reason), daemon=True).start()

    def _write_and_upload(self, frames, reason):
        if not frames:
            return
        os.makedirs(config.VIDEO_CLIP_DIR, exist_ok=True)
        path = os.path.join(config.VIDEO_CLIP_DIR, f"{reason}_{int(time.time())}.mp4")

        h, w = frames[0][1].shape[:2]
        duration = frames[-1][0] - frames[0][0]
        fps = len(frames) / duration if duration > 0 else 10.0
        fps = max(1.0, min(fps, 30.0))

        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, f in frames:
            writer.write(f)
        writer.release()
        logger.info(f"Clip salvata {path} ({len(frames)} frame, ~{fps:.1f} fps, {duration:.1f}s)")

        try:
            with open(path, "rb") as fh:
                requests.post(
                    config.BACKEND_VIDEO_UPLOAD_URL,
                    files={"video": fh},
                    data={
                        "device_id": config.DEVICE_ID,
                        "event_type": reason,
                        "timestamp": int(time.time()),
                    },
                    timeout=15,
                )
            logger.info("Clip caricata sul backend.")
        except Exception as e:
            logger.error(f"Errore upload clip: {e}")
