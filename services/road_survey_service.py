"""Road surveys: explicit timelines, camera-specific regions and auditable crossings.

No filename is interpreted as a timestamp. A stream is a single fixed camera,
with its cards explicitly ordered. Frame checkpoints preserve tracking and counts.
"""

import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mts", ".m2ts", ".mp4", ".mov", ".avi", ".mkv", ".ts"}
ROAD_TYPES = ("主路", "辅路")
EXPORT_VERSION = 3
CATEGORIES = (
    "小客车",
    "出租车",
    "单机公交车",
    "铰接公交车",
    "大货车",
    "小货车",
    "大客车非公交",
    "摩托车",
    "电动自行车",
    "普通自行车",
    "外卖电动自行车",
    "电动三轮车",
    "快递货运三轮车",
)
EXCLUDE = "不纳入模板"


class Cancelled(Exception):
    pass


def natural_key(value):
    text = str(value).replace("第一张", "第1张").replace("第二张", "第2张")
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", text)]


def matches_road(config, road_root, road_name):
    """Use directory identity; support saved tasks created before road_root existed."""
    if not road_root:
        return config.get("road_name") == road_name
    root = os.path.normcase(os.path.abspath(road_root))
    if config.get("road_root"):
        return os.path.normcase(os.path.abspath(config["road_root"])) == root
    folders = [
        card.get("folder", "")
        for direction in config.get("directions", [])
        for stream in direction.get("streams", [])
        for card in stream.get("cards", [])
    ]
    if folders and all(folders):
        try:
            return all(
                os.path.commonpath([root, os.path.normcase(os.path.abspath(folder))])
                == root
                for folder in folders
            )
        except ValueError:
            return False
    return config.get("road_name") == road_name


def finite_number(value, minimum=0, maximum=float("inf")):
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("数值超出允许范围")
    return number


def validate_config(config, require_regions=False):
    if not isinstance(config, dict):
        raise ValueError("调查配置必须是对象")
    if not str(config.get("road_name", "")).strip():
        raise ValueError("请填写道路名称")
    if str(config.get("inference_resolution", "640")) not in {
        "640",
        "960",
        "1280",
        "original",
    }:
        raise ValueError("请选择有效的推理分辨率")
    batch = config.get("inference_batch_size", 0)
    if isinstance(batch, bool) or not isinstance(batch, int) or not 0 <= batch <= 64:
        raise ValueError("推理批量必须是0至64的整数，0表示使用服务器默认")
    try:
        date.fromisoformat(config.get("survey_date", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("请填写有效的调查日期") from error
    directions = config.get("directions", [])
    if not 1 <= len(directions) <= 2:
        raise ValueError("每条道路配置一至两个方向")
    names = [d.get("name", "").strip() for d in directions]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("方向名称不能为空或重复")
    for direction in directions:
        if not direction.get("streams"):
            raise ValueError(f"{direction['name']} 请添加机位序列")
        for stream in direction["streams"]:
            try:
                datetime.fromisoformat(
                    config["survey_date"] + "T" + stream.get("start_time", "")
                )
            except (TypeError, ValueError) as error:
                raise ValueError("请填写有效的首段拍摄起始时间") from error
            if not stream.get("cards"):
                raise ValueError("每个机位至少需要一个卡目录")
            for card in stream["cards"]:
                if not str(card.get("folder", "")).strip():
                    raise ValueError("请填写卡目录")
                finite_number(card.get("gap_seconds", 0), 0, 86400)
            regions = stream.get("regions", {})
            if require_regions and not regions:
                raise ValueError(
                    f"{direction['name']} / {stream.get('name', '')} 尚未配置主路或辅路区域"
                )
            for role, region in regions.items():
                if role not in ROAD_TYPES:
                    raise ValueError("区域只支持主路和辅路")
                polygon = region.get("polygon", [])
                line, arrow = region.get("line", []), region.get("arrow", [])
                mode = region.get("mode", "region")
                if mode not in {"region", "line"}:
                    raise ValueError("无效的计数方式")
                if len(polygon) < 3:
                    raise ValueError(f"{role} 需要至少三个顶点的区域多边形")
                if mode == "line" and (len(line) != 2 or len(arrow) != 2):
                    raise ValueError(f"{role} 的过线计数需要计数线和通行箭头")
                for point in polygon + line + arrow:
                    if len(point) != 2:
                        raise ValueError("无效坐标")
                    for value in point:
                        finite_number(value, 0, 1)
                contour = np.asarray(polygon, np.float32)
                if abs(cv2.contourArea(contour)) < 0.0001:
                    raise ValueError(f"{role} 区域面积过小")
                for i in range(len(polygon)):
                    for j in range(i + 2, len(polygon)):
                        if i == 0 and j == len(polygon) - 1:
                            continue
                        if crossing_point(
                            polygon[i],
                            polygon[(i + 1) % len(polygon)],
                            [polygon[j], polygon[(j + 1) % len(polygon)]],
                        ):
                            raise ValueError(f"{role} 区域边界不能自交")
                if mode != "line":
                    continue
                a, b = map(np.asarray, line)
                u, v = map(np.asarray, arrow)
                if np.linalg.norm(b - a) < 0.005 or np.linalg.norm(v - u) < 0.005:
                    raise ValueError("计数线或通行箭头过短")
                if abs(cross(b - a, v - u)) < 0.00001:
                    raise ValueError("通行箭头不能与计数线平行")
            if len(regions) == 2:
                masks = []
                for role in ROAD_TYPES:
                    mask = np.zeros((1024, 1024), np.uint8)
                    polygon = np.rint(
                        np.asarray(regions[role]["polygon"]) * 1023
                    ).astype(np.int32)
                    cv2.fillPoly(mask, [polygon], 1)
                    masks.append(cv2.erode(mask, np.ones((3, 3), np.uint8)))
                if np.any(masks[0] & masks[1]):
                    raise ValueError("主路和辅路区域不能重叠")
    mapping = config.get("class_mapping", {})
    if any(value not in CATEGORIES + (EXCLUDE, "") for value in mapping.values()):
        raise ValueError("无效的车型映射")
    return config


def probe_video(path):
    """Use decoded-frame rate, not MTS interlaced field rate (50 vs 25)."""
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        stderr=subprocess.PIPE,
        timeout=30,
    )
    data = json.loads(output)
    stream = data["streams"][0]
    n, d = stream["avg_frame_rate"].split("/")
    fps = float(n) / float(d)
    duration = float(stream.get("duration") or data["format"]["duration"])
    if (
        not math.isfinite(fps)
        or not 0 < fps <= 240
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"视频时长或帧率无效：{path}")
    return {
        "fps": fps,
        "duration": duration,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def inference_settings(config, metadata):
    from services.road_survey_scheduler import automatic_batch, integer_setting

    resolution = str(config.get("inference_resolution", "640"))
    if resolution == "original":
        size = (
            math.ceil(metadata["height"] / 32) * 32,
            math.ceil(metadata["width"] / 32) * 32,
        )
        fallback = 1
    else:
        size = int(resolution)
        fallback = {640: 16, 960: 4, 1280: 2}[size]
    requested = config.get("inference_batch_size", 0) or integer_setting(
        "ROAD_SURVEY_BATCH_SIZE", 0, 0, 64
    )
    return size, requested or automatic_batch(size, fallback)


def config_signature(config):
    # Regions and model affect results, but not the timeline preview.
    timeline = {
        "survey_date": config["survey_date"],
        "directions": [
            {
                "name": d["name"],
                "streams": [
                    {
                        "name": s.get("name", ""),
                        "start_time": s["start_time"],
                        "cards": s["cards"],
                    }
                    for s in d["streams"]
                ],
            }
            for d in config["directions"]
        ],
    }
    return hashlib.sha256(
        json.dumps(timeline, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def scan_manifest(config, progress=lambda value: None, check_cancel=lambda: None):
    validate_config(config)
    streams, seen = [], set()
    midnight = datetime.fromisoformat(config["survey_date"])
    for di, direction in enumerate(config["directions"]):
        for si, source in enumerate(direction["streams"]):
            cursor = (
                datetime.fromisoformat(
                    config["survey_date"] + "T" + source["start_time"]
                )
                - midnight
            ).total_seconds()
            files = []
            for ci, card in enumerate(source["cards"]):
                folder = Path(card["folder"]).expanduser().resolve()
                if not folder.is_dir():
                    raise ValueError(f"目录不存在：{folder}")
                videos = sorted(
                    (
                        p
                        for p in folder.rglob("*")
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                    ),
                    key=lambda p: natural_key(p.relative_to(folder)),
                )
                if not videos:
                    raise ValueError(f"目录没有视频：{folder}")
                cursor += finite_number(card.get("gap_seconds", 0), 0, 86400)
                for path in videos:
                    check_cancel()
                    resolved = os.path.normcase(str(path.resolve()))
                    if resolved in seen:
                        raise ValueError(
                            f"同一视频被重复加入，请检查卡目录是否包含彼此：{path}"
                        )
                    seen.add(resolved)
                    metadata = probe_video(path)
                    stat = path.stat()
                    item = {
                        "path": str(path),
                        "name": path.name,
                        "card": ci + 1,
                        "card_label": card_directory_label(card["folder"]),
                        "start": cursor,
                        "end": cursor + metadata["duration"],
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        **metadata,
                    }
                    files.append(item)
                    cursor = item["end"]
                    progress(
                        {
                            "phase": "scanning",
                            "files": len(seen),
                            "current_file": str(path),
                        }
                    )
            streams.append(
                {
                    "key": f"{di}:{si}",
                    "direction_index": di,
                    "stream_index": si,
                    "direction": direction["name"],
                    "name": source.get("name", "机位"),
                    "files": files,
                }
            )
    return {
        "signature": config_signature(config),
        "streams": streams,
        "file_count": len(seen),
        "duration": sum(f["duration"] for s in streams for f in s["files"]),
    }


def verify_manifest(config, manifest):
    if not manifest or manifest.get("signature") != config_signature(config):
        raise ValueError("视频配置已变化，请重新扫描并确认时间表")
    for stream in manifest["streams"]:
        for item in stream["files"]:
            try:
                stat = Path(item["path"]).stat()
            except OSError as error:
                raise ValueError(
                    f"视频文件不可读，请检查磁盘并重新扫描：{item['path']}"
                ) from error
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"视频文件已变化，请重新扫描：{item['path']}")


def validate_camera_coverage(config, manifest):
    """Two simultaneous cameras must not both contribute the same road section."""
    for index, stream in enumerate(manifest["streams"]):
        source = config["directions"][stream["direction_index"]]["streams"][
            stream["stream_index"]
        ]
        for other in manifest["streams"][:index]:
            if stream["direction"] != other["direction"]:
                continue
            other_source = config["directions"][other["direction_index"]]["streams"][
                other["stream_index"]
            ]
            common = set(source.get("regions", {})) & set(
                other_source.get("regions", {})
            )
            if common and any(
                min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0.1
                for a in stream["files"]
                for b in other["files"]
            ):
                raise ValueError(
                    f"{stream['direction']} 的多个机位在相同时段重复覆盖 {'、'.join(sorted(common))}，请调整区域或起始时间，避免重复计数"
                )


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def crossing_point(previous, current, line):
    p, q = np.asarray(previous), np.asarray(current)
    a, b = map(np.asarray, line)
    motion, edge = q - p, b - a
    denominator = cross(motion, edge)
    if abs(denominator) < 1e-9:
        return None
    t, u = cross(a - p, edge) / denominator, cross(a - p, motion) / denominator
    if 0 < t <= 1 and 0 <= u <= 1:
        return p + t * motion, t
    return None


class RegionCounter:
    """Default: count each ID once inside each region. Line crossing is optional."""

    def __init__(self, regions, class_mapping):
        self.regions = regions
        self.mapping = class_mapping
        self.previous = {}
        self.counted = set()
        self.excluded = 0
        self.label_counts = {role: {} for role in ROAD_TYPES}

    def consume(self, tracks, width, height, timestamp, names):
        events = []
        for track in tracks:
            x1, y1, x2, y2, tid, confidence, category, *_ = track
            tid, category = int(tid), int(category)
            point = np.array([(x1 + x2) / (2 * width), y2 / height])
            previous = self.previous.get(tid)
            self.previous[tid] = (point, timestamp)
            for role, region in self.regions.items():
                if (role, tid) in self.counted:
                    continue
                location, event_time = point, timestamp
                if region.get("mode", "region") == "line":
                    if previous is None:
                        continue
                    old_point, old_time = previous
                    if timestamp - old_time > 2:
                        continue
                    crossing = crossing_point(old_point, point, region["line"])
                    if crossing is None:
                        continue
                    location, fraction = crossing
                    arrow = np.asarray(region["arrow"][1]) - np.asarray(
                        region["arrow"][0]
                    )
                    if np.dot(point - old_point, arrow) <= 0:
                        continue
                    event_time = old_time + fraction * (timestamp - old_time)
                if (
                    cv2.pointPolygonTest(
                        np.asarray(region["polygon"], np.float32),
                        tuple(map(float, location)),
                        False,
                    )
                    < 0
                ):
                    continue
                class_name = names[category]
                mapped = self.mapping[class_name]
                self.counted.add((role, tid))
                labels = self.label_counts[role]
                labels[class_name] = labels.get(class_name, 0) + 1
                if mapped == EXCLUDE:
                    self.excluded += 1
                events.append(
                    {
                        "track_id": tid,
                        "road_type": role,
                        "class_name": class_name,
                        "category": mapped,
                        "confidence": float(confidence),
                        "second": event_time,
                        "count_mode": region.get("mode", "region"),
                    }
                )
                break
        # Keep only currently relevant positions; counted IDs remain for stream-wide deduplication.
        if len(self.previous) > 2000:
            self.previous = {
                key: value
                for key, value in self.previous.items()
                if timestamp - value[1] <= 2
            }
        return events


def run_stream(
    model,
    config,
    stream,
    source,
    audit_path,
    progress=lambda value: None,
    check_cancel=lambda: None,
    preview=None,
    checkpoint_path=None,
    resume=False,
):
    from boxmot import ByteTrack

    counter = RegionCounter(source["regions"], config["class_mapping"])
    tracker = ByteTrack(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=max(1, round(stream["files"][0]["fps"])),
    )
    buckets, coverage = {}, []
    checkpoint, restored = None, None
    if checkpoint_path is not None:
        from services.road_survey_checkpoint import (
            StreamCheckpoint,
            reserve_track_ids,
            signature,
        )

        reserve_track_ids()
        checkpoint = StreamCheckpoint(
            checkpoint_path, signature(config, stream, source, model)
        )
        restored = checkpoint.load() if resume else None
        if not restored:
            checkpoint.path.unlink(missing_ok=True)
    start_file, start_frame = 0, 0
    if restored:
        tracker, counter = restored["tracker"], restored["counter"]
        buckets, coverage = restored["buckets"], restored["coverage"]
        start_file, start_frame = restored["file_index"], restored["next_frame"]
        reserve_track_ids(restored["track_id_highwater"])
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    if restored and (
        not Path(audit_path).is_file()
        or Path(audit_path).stat().st_size < restored["audit_offset"]
    ):
        raise ValueError("计数明细与帧断点不一致，请重新运行")
    with open(audit_path, "r+" if restored else "w", encoding="utf-8") as audit:
        if restored:
            audit.seek(restored["audit_offset"])
            audit.truncate()
        last_saved = [0.0]

        def save_position(file_index, next_frame, force=False):
            if checkpoint and (force or time.monotonic() - last_saved[0] >= 5):
                from boxmot.trackers.bytetrack.basetrack import BaseTrack

                checkpoint.save(
                    {
                        "tracker": tracker,
                        "counter": counter,
                        "buckets": buckets,
                        "coverage": coverage,
                        "file_index": file_index,
                        "next_frame": next_frame,
                        "track_id_highwater": BaseTrack._count,
                    },
                    audit,
                )
                last_saved[0] = time.monotonic()

        def safe_cancel(file_index, next_frame):
            try:
                check_cancel()
            except Cancelled:
                save_position(file_index, next_frame, force=True)
                raise

        save_position(start_file, start_frame, force=True)

        for fi, item in enumerate(stream["files"]):
            if fi < start_file:
                continue
            safe_cancel(fi, start_frame if fi == start_file else 0)
            image_size, batch_size = inference_settings(config, item)
            batch_state = {"batch_size": batch_size}
            # A deliberate time gap breaks continuity. Also reset positions before seeking a new recording.
            if (
                fi
                and not (fi == start_file and start_frame)
                and item["start"] - stream["files"][fi - 1]["end"] > 0.1
            ):
                tracker = ByteTrack(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=max(1, round(item["fps"])),
                )
                counter.previous.clear()
                counter.counted.clear()
            cap = cv2.VideoCapture(item["path"])
            if not cap.isOpened():
                raise ValueError(f"无法打开视频：{item['path']}")
            count, frames = start_frame if fi == start_file else 0, []

            def infer_batch():
                nonlocal count, batch_size
                safe_cancel(fi, count)
                from services.road_survey_scheduler import predict_with_backoff

                results = predict_with_backoff(model, frames, image_size, batch_state)
                batch_size = batch_state["batch_size"]
                for result in results:
                    # Update the tracker even on empty frames so stale tracks age out.
                    boxes = result.boxes.data.detach().cpu().numpy()
                    tracks = tracker.update(boxes, result.orig_img)
                    second = item["start"] + count / item["fps"]
                    events = counter.consume(
                        tracks, item["width"], item["height"], second, model.names
                    )
                    for event in events:
                        if event["second"] >= item["end"] + 0.1:
                            continue
                        audit.write(
                            json.dumps(
                                {
                                    **event,
                                    "file": item["path"],
                                    "direction": stream["direction"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        if event["category"] != EXCLUDE:
                            slot = str(int(event["second"] // 300) * 300)
                            bucket = buckets.setdefault(
                                event["road_type"], {}
                            ).setdefault(slot, {})
                            bucket[event["category"]] = (
                                bucket.get(event["category"], 0) + 1
                            )
                    count += 1
                save_position(fi, count)
                road_counts = {
                    role: sum(counter.label_counts[role].values())
                    for role in ROAD_TYPES
                }
                class_counts = {
                    role: {
                        category: sum(
                            values.get(category, 0)
                            for values in buckets.get(role, {}).values()
                        )
                        for category in CATEGORIES
                    }
                    for role in ROAD_TYPES
                }
                if preview is not None:
                    preview(
                        result.orig_img,
                        tracks,
                        source["regions"],
                        model.names,
                        road_counts,
                        second,
                        stream["direction"],
                    )
                progress(
                    {
                        "phase": "running",
                        "stream": stream["key"],
                        "file_index": fi,
                        "current_file": item["path"],
                        "frame": count,
                        "batch_size": batch_size,
                        "inference_resolution": f"{image_size[1]}×{image_size[0]}"
                        if isinstance(image_size, tuple)
                        else str(image_size),
                        "direction": stream["direction"],
                        "stream_name": stream.get("name", "机位"),
                        "road_counts": road_counts,
                        "class_counts": class_counts,
                        "label_counts": {
                            role: dict(values)
                            for role, values in counter.label_counts.items()
                        },
                        "estimated_frames": round(item["fps"] * item["duration"]),
                        "counts": sum(
                            sum(v.values())
                            for slots in buckets.values()
                            for v in slots.values()
                        ),
                    }
                )

            try:
                if count:
                    # MTS random seeking is not frame-exact. Decode/discard earlier frames;
                    # saved tracking state means these frames are never inferred or counted again.
                    progress(
                        {
                            "phase": "resuming",
                            "stream": stream["key"],
                            "current_file": item["path"],
                            "frame": count,
                            "estimated_frames": round(item["fps"] * item["duration"]),
                        }
                    )
                    for skipped in range(count):
                        if skipped % 100 == 0:
                            safe_cancel(fi, count)
                        if not cap.grab():
                            raise ValueError("无法定位已保存的帧断点，请检查源视频")
                while True:
                    safe_cancel(fi, count)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                    if len(frames) >= batch_size:
                        infer_batch()
                        frames = []
                if frames:
                    infer_batch()
                if count == 0 or count / item["fps"] < item["duration"] - 2:
                    raise ValueError(
                        f"视频未完整解码，已停止以免把缺失当作零流量：{item['path']}"
                    )
            finally:
                cap.release()
            audit.flush()
            for role in source["regions"]:
                coverage.append(
                    {"road_type": role, "start": item["start"], "end": item["end"]}
                )
            save_position(fi + 1, 0, force=True)
    return {
        "key": stream["key"],
        "direction": stream["direction"],
        "buckets": buckets,
        "coverage": coverage,
        "excluded": counter.excluded,
        "audit_path": str(audit_path),
    }


def covered_seconds(intervals, start, end):
    pieces = sorted(
        (max(start, a), min(end, b)) for a, b in intervals if a < end and b > start
    )
    total, cursor = 0.0, start
    for a, b in pieces:
        if b > cursor:
            total += b - max(cursor, a)
            cursor = b
    return total


def safe_name(value):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")[:80] or "调查"


def card_directory_label(folder):
    name = re.split(r"[/\\]", str(folder).rstrip("/\\"))[-1]
    match = re.fullmatch(
        r"(?:第)?([1-9]|[1-5][0-9]|6[0-4]|[一二三四五六七八九十])(?:张)?(?:卡)?", name
    )
    if not match:
        match = re.fullmatch(
            r"卡([1-9]|[1-5][0-9]|6[0-4]|[一二三四五六七八九十])", name
        )
    if not match:
        return ""
    number = match[1]
    chinese = {str(i): text for i, text in enumerate("一二三四五六七八九十", 1)}
    return chinese.get(number, number) + "卡"


def recording_card_label(item, source=None):
    # Resolve legacy manifests using the saved source directory, not its list position.
    if source is not None:
        index = int(item.get("card", 0)) - 1
        cards = source.get("cards", [])
        if 0 <= index < len(cards):
            return card_directory_label(cards[index]["folder"])
    if "card_label" in item:
        return item["card_label"]
    if item.get("path"):
        parts = re.split(r"[/\\]", item["path"])
        parent = parts[-2] if len(parts) > 1 else ""
        return card_directory_label(parent)
    return ""


def location_with_cards(road_name, files, start, end, source=None):
    """Label a report interval with its actual recording-card sources."""
    labels = list(
        dict.fromkeys(
            recording_card_label(item, source)
            for item in sorted(files, key=lambda f: f["start"])
            if item["start"] < end and item["end"] > start
        )
    )
    labels = [label for label in labels if label]
    return str(road_name) + ("（" + "、".join(labels) + "）" if labels else "")


def report_source_files(config, manifest, direction, role):
    files = []
    for stream in manifest["streams"]:
        if stream.get("direction") != direction:
            continue
        roles = stream.get("road_types")
        source = stream.get("card_source")
        if "direction_index" in stream and "stream_index" in stream:
            source = config["directions"][stream["direction_index"]]["streams"][
                stream["stream_index"]
            ]
            if roles is None:
                roles = source.get("regions", {})
        if roles is not None and role not in roles:
            continue
        files.extend(
            {**item, "card_label": recording_card_label(item, source)}
            for item in stream["files"]
        )
    return files


def export_workbooks(config, manifest, completed, output_dir):
    """Template's 21 columns and merged headers, without its historic sample values."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_files = [f for stream in manifest["streams"] for f in stream["files"]]
    start = math.floor(min(f["start"] for f in all_files) / 300) * 300
    end = math.ceil(max(f["end"] for f in all_files) / 300) * 300
    midnight = datetime.fromisoformat(config["survey_date"])
    exports = []
    for direction in config["directions"]:
        for role in ROAD_TYPES:
            source_files = report_source_files(
                config, manifest, direction["name"], role
            )
            results = [
                r for r in completed.values() if r["direction"] == direction["name"]
            ]
            intervals = [
                (c["start"], c["end"])
                for r in results
                for c in r["coverage"]
                if c["road_type"] == role
            ]
            wb = Workbook()
            ws = wb.active
            ws.title = "5分钟"
            headers = [
                "编号",
                "5分钟编号",
                "调查日期",
                "地点编号",
                "地点名称",
                "调查道路",
                "运行方向",
                "时段",
                "小客车",
                "出租车",
                "公交车",
                "",
                "大货车",
                "小货车",
                "大客车\n(非公交)",
                "摩托车",
                "自行车",
                "",
                "",
                "三轮车",
                "",
            ]
            ws.append(headers)
            ws.append(
                [None] * 9
                + ["出租车", "单机", "铰接"]
                + [None] * 4
                + ["电动", "普通", "外卖电动自行车", "电动", "快递货运"]
            )
            for col in list(range(1, 11)) + list(range(13, 17)):
                ws.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)
            for a, b in [(11, 12), (17, 19), (20, 21)]:
                ws.merge_cells(start_row=1, start_column=a, end_row=1, end_column=b)
            partial = 0
            for index, slot in enumerate(range(start, end, 300), 1):
                timestamp = midnight + timedelta(seconds=slot)
                finish = midnight + timedelta(seconds=slot + 300)
                values = {
                    category: sum(
                        r["buckets"].get(role, {}).get(str(slot), {}).get(category, 0)
                        for r in results
                    )
                    for category in CATEGORIES
                }
                coverage = covered_seconds(intervals, slot, slot + 300)
                complete = coverage >= 299.95
                partial += int(not complete)
                ws.append(
                    [
                        index,
                        index,
                        int(timestamp.strftime("%Y%m%d")),
                        str(config.get("spot_code", "")),
                        location_with_cards(
                            config["road_name"], source_files, slot, slot + 300
                        ),
                        role,
                        direction["name"],
                        timestamp.strftime("%H:%M") + "-" + finish.strftime("%H:%M"),
                        *[values[c] if coverage > 0 else None for c in CATEGORIES],
                    ]
                )
                if not complete:
                    ws.cell(index + 2, 8).comment = Comment(
                        f"不完整时段：有效录像覆盖 {coverage:.2f}/300 秒。数值仅为已覆盖部分的观测计数，不代表完整5分钟；未覆盖留空。",
                        "道路调查",
                    )
                    for cell in ws[index + 2]:
                        cell.fill = PatternFill("solid", fgColor="FFF2CC")
            thin = Side(style="thin", color="CBD5E1")
            for row in ws:
                for cell in row:
                    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                    cell.alignment = Alignment(
                        horizontal="center", vertical="center", wrap_text=True
                    )
                    if cell.row <= 2:
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill("solid", fgColor="D9E1F2")
            for col in range(1, 22):
                ws.column_dimensions[get_column_letter(col)].width = (
                    10 if col >= 9 else [8, 12, 13, 12, 26, 12, 12, 18][col - 1]
                )
            ws.row_dimensions[1].height = 32
            ws.freeze_panes = "I3"
            ws.print_title_rows = "1:2"
            ws.sheet_properties.pageSetUpPr.fitToPage = True
            ws.page_setup.orientation = "landscape"
            ws.page_setup.paperSize = ws.PAPERSIZE_A3
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
            ws.print_options.horizontalCentered = True
            filename = (
                safe_name(
                    f"{config['road_name']}_{direction['name']}_{role}_{config['survey_date']}"
                )
                + ".xlsx"
            )
            path = output_dir / filename
            temp = path.with_suffix(".tmp.xlsx")
            wb.save(temp)
            os.replace(temp, path)
            exports.append(
                {
                    "name": filename,
                    "path": str(path),
                    "direction": direction["name"],
                    "road_type": role,
                    "partial_slots": partial,
                    "total": sum(
                        sum(r["buckets"].get(role, {}).get(str(slot), {}).values())
                        for r in results
                        for slot in range(start, end, 300)
                    ),
                }
            )
    return exports


def discover_roads(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("道路根目录不存在")
    roads = []
    for road in sorted(root.iterdir(), key=lambda p: natural_key(p.name)):
        if not road.is_dir() or road.name.startswith(("$", ".", "System Volume")):
            continue
        directions = {}
        for folder in sorted(road.iterdir(), key=lambda p: natural_key(p.name)):
            if not folder.is_dir():
                continue
            match = re.search(r"北向南|南向北|东向西|西向东", folder.name)
            if not match:
                continue
            cards = sorted(
                [
                    p
                    for p in folder.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ],
                key=lambda p: natural_key(p.name),
            )
            if any(
                p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                for p in folder.iterdir()
            ):
                cards = [folder]
            cards = [
                p
                for p in cards
                if any(
                    f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
                    for f in p.rglob("*")
                )
            ]
            if not cards:
                continue
            time = re.search(r"(\d{1,2})[:：](\d{2})", folder.name)
            stream = {
                "name": folder.name,
                "start_time": f"{int(time[1]):02d}:{time[2]}:00"
                if time
                else "07:00:00",
                "cards": [{"folder": str(p), "gap_seconds": 0} for p in cards],
                "regions": {},
            }
            directions.setdefault(match[0], {"name": match[0], "streams": []})[
                "streams"
            ].append(stream)
        if directions:
            roads.append(
                {
                    "road_name": road.name,
                    "directions": list(directions.values()),
                    "root": str(road),
                }
            )
    return roads
