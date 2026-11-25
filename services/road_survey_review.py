"""Independent, bounded review clips. Output timestamps follow recording time."""

import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from services import road_survey_service as survey
from services.road_survey_live import preview_font, render_preview


def review_window(stream, start, duration=300):
    start = survey.finite_number(start, 0, 7 * 86400)
    duration = survey.finite_number(duration, 1, 300)
    files = stream["files"]
    first, last = files[0]["start"], files[-1]["end"]
    if not first <= start < last:
        raise ValueError("抽查起始时间须在该机位的录像时间范围内")
    end = min(start + duration, last)
    if not any(f["start"] < end and f["end"] > start for f in files):
        raise ValueError("所选时段没有录像")
    return start, end


class ReviewWriter:
    """Resample onto a fixed 10fps recording-time axis; gaps stay visible."""

    def __init__(self, path, start, end, width, height, fps=10):
        self.start, self.end, self.fps = start, end, fps
        self.width, self.height = width, height
        self.count = 0
        self.total = math.ceil((end - start) * fps - 1e-7)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_until(self, end, image, inclusive=False):
        end = min(end, self.end)
        prepared = None
        while self.count < self.total:
            timestamp = self.start + self.count / self.fps
            if timestamp > end + 1e-7 or (not inclusive and timestamp >= end - 1e-7):
                break
            if prepared is None:
                prepared = cv2.resize(image, (self.width, self.height)).tobytes()
            if self.process.poll() is not None:
                raise RuntimeError("回放视频编码器提前退出")
            self.process.stdin.write(prepared)
            self.count += 1

    def close(self, success):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if not success and self.process.poll() is None:
            self.process.kill()
        try:
            code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("回放视频编码超时")
        if success and (code != 0 or self.count != self.total):
            raise RuntimeError("回放视频未完整生成")


def gap_frame(width, height):
    image = Image.new("RGB", (width, height), (25, 32, 42))
    ImageDraw.Draw(image).text(
        (30, height // 2),
        "此时段无录像 · 停录间隔保留在回放时间轴中",
        font=preview_font(),
        fill="white",
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def decode_segment(item, start, end, check_cancel):
    """FFmpeg performs accurate post-decode seeking for MTS, not keyframe-only cuts."""
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            item["path"],
            "-ss",
            str(start - item["start"]),
            "-t",
            str(end - start),
            "-vf",
            "setpts=PTS-STARTPTS",
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    size = item["width"] * item["height"] * 3
    count = 0
    try:
        while True:
            check_cancel()
            raw = process.stdout.read(size)
            if not raw:
                break
            if len(raw) != size:
                raise RuntimeError("抽查视频解码帧不完整")
            timestamp = start + count / item["fps"]
            if timestamp >= end - 1e-7:
                break
            count += 1
            yield (
                timestamp,
                np.frombuffer(raw, np.uint8)
                .reshape(item["height"], item["width"], 3)
                .copy(),
            )
        if count == 0 or count / item["fps"] < end - start - max(0.12, 2 / item["fps"]):
            raise ValueError(f"所选片段未完整解码：{item['path']}")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def generate_review(
    model,
    config,
    stream,
    source,
    start,
    end,
    output_path,
    progress=lambda value: None,
    check_cancel=lambda: None,
):
    from boxmot import ByteTrack

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.mp4")
    first = stream["files"][0]
    width = min(960, first["width"])
    width -= width % 2
    height = max(2, round(first["height"] * width / first["width"] / 2) * 2)
    writer = ReviewWriter(temporary, start, end, width, height)
    blank = gap_frame(width, height)
    counter = survey.RegionCounter(source["regions"], config["class_mapping"])
    counts = {role: {} for role in survey.ROAD_TYPES}
    events, coverage = [], []
    warmup = max(stream["files"][0]["start"], start - 10)
    tracker, previous_end = None, None
    success = False
    try:
        for item in stream["files"]:
            begin, finish = max(warmup, item["start"]), min(end, item["end"])
            if finish <= begin:
                continue
            check_cancel()
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("视频文件已变化，请重新扫描后再抽查")
            if tracker is None or begin - previous_end > 0.1:
                tracker = ByteTrack(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=max(1, round(item["fps"])),
                )
                counter.previous.clear()
                counter.counted.clear()
            writer.write_until(max(start, begin), blank)
            image_size, batch_size = survey.inference_settings(config, item)
            batch_state = {"batch_size": batch_size}
            batch, times, last_image = [], [], None

            def infer():
                nonlocal last_image, batch_size
                check_cancel()
                from services.road_survey_scheduler import predict_with_backoff

                results = predict_with_backoff(model, batch, image_size, batch_state)
                batch_size = batch_state["batch_size"]
                for result, timestamp in zip(results, times):
                    tracks = tracker.update(
                        result.boxes.data.detach().cpu().numpy(), result.orig_img
                    )
                    for event in counter.consume(
                        tracks, item["width"], item["height"], timestamp, model.names
                    ):
                        if start <= event["second"] < end:
                            events.append({**event, "file": item["path"]})
                            labels = counts[event["road_type"]]
                            labels[event["class_name"]] = (
                                labels.get(event["class_name"], 0) + 1
                            )
                    if timestamp >= start:
                        # Render only output frames, while detecting/tracking every source frame.
                        next_time = start + writer.count / writer.fps
                        if next_time <= timestamp + 1e-7:
                            totals = {
                                role: sum(labels.values())
                                for role, labels in counts.items()
                            }
                            last_image = render_preview(
                                result.orig_img,
                                tracks,
                                source["regions"],
                                model.names,
                                totals,
                                timestamp,
                                stream["direction"],
                                as_array=True,
                            )
                            writer.write_until(timestamp, last_image, inclusive=True)
                progress(
                    {
                        "phase": "running",
                        "seconds": round(max(0, times[-1] - start), 1),
                        "duration": end - start,
                        "counts": counts,
                    }
                )

            for timestamp, image in decode_segment(item, begin, finish, check_cancel):
                batch.append(image)
                times.append(timestamp)
                if len(batch) >= batch_size:
                    infer()
                    batch = []
                    times = []
            if batch:
                infer()
            if finish > start:
                writer.write_until(
                    finish, last_image if last_image is not None else blank
                )
                coverage.append((max(start, begin), finish))
            previous_end = finish
        writer.write_until(end, blank)
        check_cancel()
        success = True
    finally:
        writer.close(success)
        if not success:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output_path)
    audit_path = output_path.with_suffix(".json")
    audit_path.write_text(
        json.dumps(
            {
                "start": start,
                "end": end,
                "counts": counts,
                "events": events,
                "config": config,
                "stream": stream,
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "video_path": str(output_path),
        "audit_path": str(audit_path),
        "duration": end - start,
        "coverage_seconds": survey.covered_seconds(coverage, start, end),
        "start": start,
        "end": end,
        "counts": counts,
        "frames": writer.count,
        "fps": writer.fps,
    }
