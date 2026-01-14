from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from boxmot import ByteTrack


class PlotEngine:
    def __init__(
        self,
        model,
        labels_target_map,
        area_annotations,
        region_label_map,
        font_path,
    ):
        self.model = model
        self.labels_target_map = labels_target_map
        self.area_annotations = area_annotations
        self.region_label_map = region_label_map
        self.font_path = font_path

        self.tracker = ByteTrack(
            track_thresh=0.5,  # 高置信度阈值
            track_buffer=30,  # 跟踪器缓冲区大小（帧数）
            match_thresh=0.8,  # 匹配阈值
            frame_rate=25,  # 视频帧率
        )
        # 每个 track_id 只统计一次
        self.counted_ids = set()
        # 每个区域内每个 track_id 只统计一次
        self.counted_ids_per_region = defaultdict(set)
        self.reset_stats()

    def reset_stats(self):
        self.total_count = 0
        self.class_counts = {}
        self.region_stats = defaultdict(lambda: {"total": 0, "class_counts": {}})
        self.counted_ids.clear()
        self.counted_ids_per_region.clear()

    def _area_statistics(self, img_w, img_h, cx, cy, class_name, track_id):
        if not self.area_annotations:
            return

        for ann in self.area_annotations:
            poly = [(int(p["x"] * img_w), int(p["y"] * img_h)) for p in ann["points"]]

            inside = (
                cv2.pointPolygonTest(np.array(poly, np.int32), (cx, cy), False) >= 0
            )
            if not inside:
                continue

            region_name = self.region_label_map.get(ann["label_id"], "unknown")

            # 避免重复统计同一个 track_id
            if track_id not in self.counted_ids_per_region[region_name]:
                self.region_stats[region_name]["total"] += 1
                self.region_stats[region_name]["class_counts"].setdefault(class_name, 0)
                self.region_stats[region_name]["class_counts"][class_name] += 1
                self.counted_ids_per_region[region_name].add(track_id)

    def _area_painting(self, img_w, img_h, frame):
        if not self.area_annotations:
            return
        for ann in self.area_annotations:
            poly = [(int(p["x"] * img_w), int(p["y"] * img_h)) for p in ann["points"]]
            cv2.polylines(frame, [np.array(poly, np.int32)], True, (255, 0, 0), 2)

    def plot_frame(self, result, frame):
        img_h, img_w = frame.shape[:2]

        # 画区域
        self._area_painting(img_w, img_h, frame)

        if not hasattr(result, "boxes") or result.boxes is None:
            return frame

        detections = []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = map(float, box)  # 保持浮点数
            # 创建6维检测结果: [x1, y1, x2, y2, conf, cls]
            detections.append([x1, y1, x2, y2, conf, cls])

        if len(detections) > 0:
            detections_np = np.array(detections, dtype=np.float32)
        else:
            detections_np = np.zeros((0, 6), dtype=np.float32)
        # 更新跟踪器，得到带 track_id 的结果
        tracks = self.tracker.update(detections_np, frame)

        for track in tracks:
            # ByteTrack返回的跟踪结果格式: [x1, y1, x2, y2, track_id, score, class]
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            bbox = [x1, y1, x2, y2]
            class_id = int(track[-1])
            class_name = self.model.names[int(class_id)]
            class_name = self.labels_target_map.get(class_name, class_name)

            # 全局统计，每个 track_id 只统计一次
            if track_id is not None and track_id not in self.counted_ids:
                self.total_count += 1
                self.class_counts.setdefault(class_name, 0)
                self.class_counts[class_name] += 1
                self.counted_ids.add(track_id)

            # 区域统计
            if track_id is not None:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                self._area_statistics(img_w, img_h, cx, cy, class_name, track_id)

            label = f"{class_name} {conf:.2f}"

            # 中文字体绘制
            try:
                font = ImageFont.truetype(self.font_path, 14)
                text_w, text_h = font.getbbox(label)[2:4]
            except Exception:
                font = None
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

            # 绘制框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(
                frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1
            )

            if font:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1, y1 - text_h - 2), label, font=font, fill=(0, 0, 0))
                frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

        return frame
