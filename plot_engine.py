import os
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
        # 预加载中文字体
        self.font = None
        if font_path:
            try:
                self.font = ImageFont.truetype(font_path, 20)
            except Exception as e:
                print(f"加载字体失败: {e}")
        # 中文标签缓存
        self._text_cache = {}
        # 检查框
        self.detections = []

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
        # 跟踪轨迹摘要：track_id -> {class_name, first_area, last_area, first_frame, last_frame, is_approaching, area_ratio}
        # 用于"仅统计由远到近的物体"功能；first_frame/last_frame 用于交通量调查按时间段分桶
        self.track_summary = {}
        # 全局帧计数器：track_frame 每被调用一次代表处理了一帧（批次间串行，帧号与视频内帧序一致）
        self.frame_idx = 0
        self.reset_stats()

    def reset_stats(self):
        self.total_count = 0
        self.class_counts = {}
        self.region_stats = defaultdict(lambda: {"total": 0, "class_counts": {}})
        self.counted_ids.clear()
        self.counted_ids_per_region.clear()
        self.track_summary.clear()
        self.frame_idx = 0

    def finalize_track_summary(self, area_ratio_threshold=None):
        """
        推理结束时计算每个 track 的轨迹方向（由远到近）。
        当末帧包围盒面积 / 首帧包围盒面积 >= 阈值 时，认为该物体在靠近相机（approaching）。
        默认阈值从环境变量 APPROACHING_AREA_RATIO 读取，默认 1.5。
        返回带 is_approaching / area_ratio 字段的 track_summary。
        """
        if area_ratio_threshold is None:
            try:
                area_ratio_threshold = float(os.environ.get("APPROACHING_AREA_RATIO", "1.5"))
            except (TypeError, ValueError):
                area_ratio_threshold = 1.5
        for _tid, info in self.track_summary.items():
            first = float(info.get("first_area", 0.0))
            last = float(info.get("last_area", 0.0))
            if first > 0:
                ratio = last / first
                info["is_approaching"] = bool(ratio >= area_ratio_threshold)
                info["area_ratio"] = round(ratio, 3)
            else:
                info["is_approaching"] = False
                info["area_ratio"] = 0.0
        return self.track_summary

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

    def track_frame(self, result):
        """仅执行跟踪，返回跟踪结果（串行操作）

        优化：避免 Python list 循环、避免额外 numpy 数组分配
        """
        # 帧计数器必须先于一切早退分支自增，保证帧号与视频内真实帧序一致
        frame_no = self.frame_idx
        self.frame_idx += 1

        if len(result.boxes) == 0:
            return []

        boxes_data = result.boxes.data
        if boxes_data.is_cuda:
            boxes_np = boxes_data.cpu().numpy()
        else:
            boxes_np = boxes_data.numpy()

        if len(boxes_np) == 0:
            return []

        # 直接传 numpy 数组给 ByteTrack（避免 Python list 重建）
        # 第二个参数 orig_img 是可选的，ByteTrack 用它做 motion estimation 时才需要
        tracks = self.tracker.update(boxes_np, result.orig_img)
        n_names = len(self.model.names)
        track_results = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, class_id, _ = track
            track_id = int(track_id)
            class_id = int(class_id)
            if class_id < 0 or class_id >= n_names:
                continue
            class_name = self.model.names[class_id]
            # 记录轨迹首末包围盒面积（用于由远到近判断）
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            ts = self.track_summary.get(track_id)
            if ts is None:
                self.track_summary[track_id] = {
                    "class_name": class_name,
                    "first_area": area,
                    "last_area": area,
                    "first_frame": frame_no,
                    "last_frame": frame_no,
                }
            else:
                ts["last_area"] = area
                ts["last_frame"] = frame_no
            # 统计（去重）
            if track_id not in self.counted_ids:
                self.total_count += 1
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                self.counted_ids.add(track_id)
            track_results.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'class_id': class_id,
                'class_name': class_name,
                'track_id': track_id,
            })
        return track_results

    def draw_frame(self, frame, track_results):
        """仅执行绘制，返回绘制后的帧（可并行操作）

        优化：去掉 PIL 中文字体渲染（每次 alpha 混合非常慢），改用 cv2.putText。
        中文会显示为方块/问号，但绘制速度提升 5-10 倍。如必须中文可后续用 cv2 + freetype。

        修复：NVDEC 路径下 boxes 是 640x640 模型输入尺度，frame 是原图（1920x1080），
        需要按 frame/640 缩放坐标；软解路径下 boxes 已是 frame 原始尺度，无需缩放。
        """
        if not track_results:
            return frame

        h, w = frame.shape[:2]
        # 自适应判定：frame 不是 640x640 时，按模型输入尺度 640 还原坐标
        if h == 640 and w == 640:
            scale_x, scale_y = 1.0, 1.0
        else:
            scale_x = w / 640.0
            scale_y = h / 640.0

        for track in track_results:
            x1, y1, x2, y2 = track['box']
            conf = track['conf']
            class_name = track['class_name']
            track_id = track['track_id']

            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            # 完整标签（中文用 cv2 会乱码，但绘制速度极快）
            full_label = f"{class_name} {conf:.2f} ID:{track_id}"
            (label_w, label_h), baseline = cv2.getTextSize(
                full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 标签背景 + 文字（单次 putText 调用）
            label_y = max(y1 - 4, label_h + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - label_h - baseline),
                (x1 + label_w, label_y),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame,
                full_label,
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def _get_class_text_image(self, text):
        """缓存渲染中文类别名到 BGRA 图片"""
        if text in self._text_cache:
            return self._text_cache[text]
        if self.font is None:
            return None
        
        try:
            dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
            bbox = dummy_draw.textbbox((0, 0), text, font=self.font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w <= 0 or h <= 0:
                return None
            
            img = Image.new('RGBA', (w, h), (0, 255, 0, 255))
            draw = ImageDraw.Draw(img)
            draw.text((-bbox[0], -bbox[1]), text, font=self.font, fill=(255, 255, 255, 255))
            bgra = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
            self._text_cache[text] = bgra
            return bgra
        except Exception as e:
            print(f"渲染文字失败: {e}")
            return None

    def plot_frame(self, result, frame):
        """完整流程：跟踪 + 绘制（串行操作，用于单帧处理）"""
        track_results = self.track_frame(result)
        return self.draw_frame(frame, track_results)
