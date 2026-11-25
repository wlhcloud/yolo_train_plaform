"""One-frame-per-task MJPEG buffers; slow viewers never block inference."""

import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def preview_font():
    for path in [
        "C:/Windows/Fonts/msyh.ttc",
        "static/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, 17)
    return ImageFont.load_default()


def render_preview(
    frame, tracks, regions, names, counts, second, direction, as_array=False
):
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width)
    image = cv2.resize(frame, (round(width * scale), round(height * scale)))
    h, w = image.shape[:2]
    labels = []
    for role, region in regions.items():
        color = (100, 220, 60) if role == "主路" else (40, 200, 255)
        polygon = np.rint(np.asarray(region["polygon"]) * [w, h]).astype(np.int32)
        cv2.polylines(image, [polygon], True, color, 2)
        labels.append((int(polygon[0][0]), max(0, int(polygon[0][1])), role))
        if region.get("mode") == "line" and len(region.get("line", [])) == 2:
            a, b = np.rint(np.asarray(region["line"]) * [w, h]).astype(np.int32)
            cv2.line(image, tuple(a), tuple(b), color, 2)
    for track in tracks:
        x1, y1, x2, y2, tid, confidence, category, *_ = track
        a, b = (
            (round(x1 * scale), round(y1 * scale)),
            (round(x2 * scale), round(y2 * scale)),
        )
        cv2.rectangle(image, a, b, (60, 230, 80), 2)
        labels.append(
            (
                max(0, a[0]),
                max(32, a[1] - 22),
                f"{names[int(category)]} ID:{int(tid)} {confidence:.2f}",
            )
        )
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = preview_font()
    for x, y, text in labels:
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle(box, fill=(15, 35, 20))
        draw.text((x, y), text, font=font, fill=(120, 255, 150))
    draw.rectangle((0, 0, w, 30), fill=(20, 30, 40))
    hour = int(second // 3600) % 24
    minute = int(second // 60) % 60
    seconds = int(second) % 60
    caption = f"{direction} {hour:02}:{minute:02}:{seconds:02}  当前机位：主路 {counts.get('主路', 0)} / 辅路 {counts.get('辅路', 0)}"
    draw.text((8, 4), caption, font=font, fill="white")
    output = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    if as_array:
        return output
    ok, jpeg = cv2.imencode(
        ".jpg",
        output,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    if not ok:
        raise ValueError("实时画面编码失败")
    return jpeg.tobytes()


def waiting_frame():
    image = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(
        image,
        "Waiting for inference...",
        (90, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (220, 230, 240),
        2,
    )
    return cv2.imencode(".jpg", image)[1].tobytes()


class LiveFrames:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = waiting_frame()
        self.sequence = 0
        self.viewers = 0
        self.done = False
        self.last_render = 0.0

    def reset(self):
        with self.condition:
            self.jpeg = waiting_frame()
            self.sequence += 1
            self.done = False
            self.last_render = 0.0
            self.condition.notify_all()

    def wants_frame(self):
        with self.condition:
            if not self.viewers or time.monotonic() - self.last_render < 0.125:
                return False
            self.last_render = time.monotonic()
            return True

    def publish(self, jpeg):
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.done = True
            self.condition.notify_all()

    def stream(self):
        last = -1
        with self.condition:
            self.viewers += 1
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.sequence != last or self.done, timeout=3
                    )
                    jpeg, last, done = self.jpeg, self.sequence, self.done
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                if done:
                    break
        finally:
            with self.condition:
                self.viewers -= 1


_buffers = {}
_buffers_lock = threading.Lock()


def frames_for(task_id):
    with _buffers_lock:
        if task_id not in _buffers:
            _buffers[task_id] = LiveFrames()
        return _buffers[task_id]
