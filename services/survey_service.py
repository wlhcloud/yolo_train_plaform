"""
交通量调查统计表服务

针对如下磁盘目录结构做批量视频推理并自动生成「交通量调查统计表」Excel：

    根目录/
      ├─ 调查点名称（如 "和平里东街 -路口南50米处 12-25 XT"）
      │    ├─ 调查道路（如 "和平里东街"）
      │    │    ├─ 运行方向（如 "北向南"）
      │    │    │    ├─ 6：00-6：30.MTS
      │    │    │    └─ 6：30-7：00.MTS
      │    │    └─ 南向北/
      ...

规则：
- 视频文件名中包含时间段（兼容全角冒号"："与多种连字符），按时间正序识别；
- 目标首次被跟踪到的时间 = 视频起始时间 + 首帧号 / fps，按 5 分钟时段分桶；
- 分类映射：模型英文类别 -> 数据库 label.target 中文类别（由调用方传入）；
- 每个三级子目录（调查点/道路/方向）生成一个 Excel，保存在该子目录下。

既被 Web 端（routes.py 的调查推理接口）调用，也被命令行脚本 export_survey.py 调用，
不依赖 Flask 上下文。
"""
import os
import re
import subprocess
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

SLOT_MINUTES = 5  # Excel 报表时段粒度（分钟）
MINUTE_SLOT = 1  # 数据库存储粒度：每分钟一条
DAY_MINUTES = 24 * 60
OTHER_LABEL = "其他"

VIDEO_EXTS = {
    ".mts", ".m2ts", ".mt2s", ".mp4", ".avi", ".mov",
    ".mkv", ".ts", ".wmv", ".flv", ".webm", ".mpg", ".mpeg",
}

# 匹配 "6：00-6：30" / "6:00~6:30" / "06-00 至 07-00" 之类的时段写法
_TIME_RANGE_RE = re.compile(
    r"(\d{1,2})\s*[:：]\s*(\d{1,2})\s*[-－—~～至]\s*(\d{1,2})\s*[:：]\s*(\d{1,2})"
)


def parse_video_time_range(filename):
    """从视频文件名解析时间段。

    返回 (start_minute, end_minute)：一天内的分钟数（end<=start 时按跨天 +24h 处理）。
    解析失败返回 None，调用方应记录告警并按 00:00 起算。
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    m = _TIME_RANGE_RE.search(stem)
    if not m:
        return None
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    if not (0 <= h1 < 24 and 0 <= h2 < 48 and 0 <= m1 < 60 and 0 <= m2 < 60):
        return None
    start = h1 * 60 + m1
    end = h2 * 60 + m2
    if end <= start:
        end += DAY_MINUTES
    return start, end


def collect_survey_videos(root):
    """扫描根目录，返回按「调查点/道路/方向」分组的视频清单。

    分组不依赖相对根目录的深度：取视频所在文件夹向上三层目录名作为
    (调查点, 道路, 方向)。因此无论用户填的是调查点的上层目录、调查点本身，
    还是道路/方向层级，只要包含视频都能正确分组。

    返回 { (调查点, 道路, 方向): {"folder": 方向层文件夹绝对路径, "videos": [按时间正序的文件名]} }
    每组内按文件名时段升序排列。
    """
    root = os.path.abspath(root)
    groups = {}
    for base, dirs, files in os.walk(root):
        videos = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        if not videos:
            continue
        # 向上取三层目录名：当前文件夹=方向，父=道路，祖父=调查点（不足三层时留空）
        folder = os.path.abspath(base)
        direction = os.path.basename(folder)
        road = os.path.basename(os.path.dirname(folder))
        spot = os.path.basename(os.path.dirname(os.path.dirname(folder)))
        # 去掉 Windows 盘符根目录这类空名
        key = (spot or "未命名调查点", road or "未知道路", direction or "未知方向")
        g = groups.setdefault(key, {"folder": folder, "videos": []})
        g["videos"].extend(videos)

    for g in groups.values():
        g["videos"].sort(key=lambda f: parse_video_time_range(f) or (0, 0))
    return groups


def count_videos_by_depth(root):
    """诊断用：统计根目录下各相对深度（0=根目录本身,1,2,3,...）的视频文件数。"""
    root = os.path.abspath(root)
    depth_counts = {}
    total = 0
    for base, dirs, files in os.walk(root):
        rel = os.path.relpath(base, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        n = sum(1 for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
        if n:
            depth_counts[depth] = depth_counts.get(depth, 0) + n
            total += n
    return total, depth_counts


def _probe_fps(video_path):
    """优先 ffprobe 获取真实帧率（cv2 读 .MTS 时 fps 可能不准），失败返回 None。"""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stderr=subprocess.STDOUT,
            timeout=15,
        ).decode().strip()
        if "/" in out:
            num, den = out.split("/")
            if float(den) > 0:
                return float(num) / float(den)
        elif out:
            return float(out)
    except Exception:
        pass
    return None


def infer_video_for_survey(model, plot_engine, video_path, batch_size=16, frame_cb=None):
    """对单个视频做「批量检测 + 跟踪」推理（不绘制、不产出视频）。

    plot_engine 需由调用方按视频新建（内部会 reset_stats）。
    frame_cb(done_frames, total_frames) 用于进度上报。
    返回 (fps, track_summary)：track_summary 每条含
    class_name / first_area / last_area / first_frame / last_frame / is_approaching。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = _probe_fps(video_path)
    if not fps or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not fps or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    plot_engine.reset_stats()
    done = 0
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) >= batch_size:
                _survey_predict_track(model, plot_engine, frames)
                done += len(frames)
                frames = []
                if frame_cb and total_frames > 0:
                    frame_cb(min(done, total_frames), total_frames)
        if frames:
            _survey_predict_track(model, plot_engine, frames)
            done += len(frames)
            if frame_cb and total_frames > 0:
                frame_cb(min(done, total_frames), total_frames)
    finally:
        cap.release()

    plot_engine.finalize_track_summary()
    return fps, dict(plot_engine.track_summary)


def _survey_predict_track(model, plot_engine, frames):
    """一批帧：模型推理 + 逐帧跟踪（ultralytics 会自动把框映射回原图尺度）。"""
    results = model.predict(frames, verbose=False)
    for r in results:
        plot_engine.track_frame(r)


def summarize_to_minutes(track_summary, fps, video_start_min, labels_map,
                        only_approaching=False):
    """把单视频的 track 汇总到「每分钟」桶（数据库存储粒度）。

    返回 {minute_of_day: {中文类别: 数量}}。
    video_start_min：一天内的分钟数（来自文件名，如 6:00 -> 360）。
    """
    minutes = {}
    if not fps or fps <= 0:
        fps = 25.0
    for info in track_summary.values():
        if only_approaching and not info.get("is_approaching"):
            continue
        first_frame = int(info.get("first_frame", 0))
        minute = int(video_start_min + (first_frame / fps) / 60.0) % DAY_MINUTES
        cat = labels_map.get(info.get("class_name"), OTHER_LABEL)
        bucket = minutes.setdefault(minute, {})
        bucket[cat] = bucket.get(cat, 0) + 1
    return minutes


def aggregate_minutes_to_slots(minutes, slot_minutes=SLOT_MINUTES):
    """把每分钟统计聚合为 N 分钟时段（供 Excel 使用）。

    返回 [(slot_start_minute, {类别: 数量}), ...] 按 slot 升序。
    """
    slots = {}
    for minute, counts in minutes.items():
        slot = (minute // slot_minutes) * slot_minutes
        bucket = slots.setdefault(slot, {})
        for k, v in counts.items():
            bucket[k] = bucket.get(k, 0) + v
    return sorted(slots.items())


def _slot_label(slot):
    end = (slot + SLOT_MINUTES) % DAY_MINUTES
    return f"{slot // 60:02d}:{slot % 60:02d}-{end // 60:02d}:{end % 60:02d}"


def build_survey_excel(out_path, meta, slot_rows, categories):
    """生成交通量调查统计表 Excel。

    meta: {survey_date, spot_code, spot_name, road, direction}
    slot_rows: [(slot_minute, {类别: 数量}), ...] 已按时间升序
    categories: 中文类别列（有序）
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "交通量调查"

    headers = ["调查日期", "地点编号", "地点名称", "调查道路", "运行方向", "时段"] \
        + list(categories) + ["合计"]
    ncols = len(headers)

    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")

    # 标题行
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    title = ws.cell(row=1, column=1, value="交通量调查统计表")
    title.font = Font(bold=True, size=14)
    title.alignment = center

    # 表头行
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=2, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    # 数据行
    totals = {c: 0 for c in categories}
    grand = 0
    row = 3
    for slot, counts in slot_rows:
        row_total = 0
        vals = [
            meta.get("survey_date", ""),
            meta.get("spot_code", ""),
            meta.get("spot_name", ""),
            meta.get("road", ""),
            meta.get("direction", ""),
            _slot_label(slot),
        ]
        for c in categories:
            v = int(counts.get(c, 0))
            vals.append(v)
            totals[c] += v
            row_total += v
        vals.append(row_total)
        grand += row_total
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border = border
            cell.alignment = center
        row += 1

    # 合计行
    vals = ["合计", "", "", "", "", ""] + [totals[c] for c in categories] + [grand]
    for c, v in enumerate(vals, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = Font(bold=True)
        cell.border = border
        cell.alignment = center

    # 列宽与冻结
    widths = [13, 10, 30, 16, 10, 14] + [9] * len(categories) + [9]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A3"

    wb.save(out_path)


def run_survey(root, model, labels_map, categories,
               only_approaching=False, progress_cb=None, survey_date=None):
    """扫描 root 三级目录并逐视频推理，每个三级目录生成一个 Excel。

    参数：
      root: 视频根目录（调查点一级目录的父目录）
      model: 已加载的 YOLO 模型
      labels_map: {模型英文类别: 中文类别}（来自 label 表 name->target）
      categories: Excel 的中文类别列（有序，来自 label 表 target 去重）
      only_approaching: 是否仅统计由远到近的轨迹
      progress_cb(done_videos, total_videos, current_file, frame_done, frame_total)
      survey_date: 调查日期字符串，默认今天

    返回 [ {excel_path, spot, road, direction, video_count, track_count}, ... ]
    """
    # 延迟导入避免与上层模块形成环
    from plot_engine import PlotEngine

    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise ValueError(f"根目录不存在: {root}")

    groups = collect_survey_videos(root)
    if not groups:
        total, depth_counts = count_videos_by_depth(root)
        if total == 0:
            raise ValueError(
                f"目录 {root} 下未找到任何视频文件。"
                f"支持格式：{', '.join(sorted(VIDEO_EXTS))}。"
                f"请确认填写的是服务端可访问的目录，且目录内存在视频文件"
            )
        raise ValueError(
            f"目录 {root} 找到 {total} 个视频文件（按相对深度分布：{depth_counts}），"
            f"但未能分组。请确认目录结构为「调查点/道路/方向」或其子层级"
        )

    survey_date = survey_date or datetime.now().strftime("%Y-%m-%d")
    total_videos = sum(len(g["videos"]) for g in groups.values())
    done_videos = 0
    results = []

    for (spot, road, direction), g in groups.items():
        minutes = {}  # {minute_of_day: {类别: 数量}} —— 数据库每分钟粒度
        starts, ends = [], []
        track_count = 0

        for fname in g["videos"]:
            video_path = os.path.join(g["folder"], fname)
            tr = parse_video_time_range(fname)
            if tr:
                starts.append(tr[0])
                ends.append(tr[1])
                video_start = tr[0]
            else:
                video_start = 0  # 文件名无时段：按 00:00 起算，仅告警不中断

            plot_engine = PlotEngine(
                model=model,
                labels_target_map=labels_map,
                area_annotations=None,
                region_label_map={},
                font_path=None,
            )

            def _frame_cb(frame_done, frame_total, _f=fname):
                if progress_cb:
                    progress_cb(done_videos, total_videos, _f, frame_done, frame_total)

            fps, tracks = infer_video_for_survey(
                model, plot_engine, video_path, frame_cb=_frame_cb
            )
            video_minutes = summarize_to_minutes(
                tracks, fps, video_start, labels_map, only_approaching
            )
            for minute, counts in video_minutes.items():
                bucket = minutes.setdefault(minute, {})
                for k, v in counts.items():
                    bucket[k] = bucket.get(k, 0) + v
            track_count += len(tracks)
            done_videos += 1
            if progress_cb:
                progress_cb(done_videos, total_videos, fname, 0, 0)

        # 每分钟明细（有序，供数据库存储）
        minute_rows = sorted(minutes.items())

        # Excel 时段行范围：该文件夹内所有视频时段的并集（每 5 分钟一行，空行也保留）
        if starts:
            lo = (min(starts) // SLOT_MINUTES) * SLOT_MINUTES
            hi = max(ends)
            slot_keys = [s % DAY_MINUTES for s in range(lo, hi, SLOT_MINUTES)]
        else:
            slot_keys = []
        # 先把分钟数据聚合成 5 分钟桶
        slot_counts = dict(aggregate_minutes_to_slots(minutes, SLOT_MINUTES))
        slot_keys = sorted(set(slot_keys) | set(slot_counts.keys()))
        slot_rows = [(k, slot_counts.get(k, {})) for k in slot_keys]

        # 出现了未映射类别时补充"其他"列
        used = {c for _, counts in slot_rows for c in counts}
        final_categories = list(categories)
        if used and not set(used).issubset(set(categories)):
            final_categories = final_categories + [
                c for c in sorted(used) if c not in categories
            ]

        out_name = f"交通量调查统计表_{survey_date}.xlsx"
        out_path = os.path.join(g["folder"], out_name)
        meta = {
            "survey_date": survey_date,
            "spot_code": "",  # 地点编号留空
            "spot_name": spot,
            "road": road,
            "direction": direction,
        }
        build_survey_excel(out_path, meta, slot_rows, final_categories)

        results.append({
            "excel_path": out_path,
            "spot": spot,
            "road": road,
            "direction": direction,
            "video_count": len(g["videos"]),
            "track_count": track_count,
            "survey_date": survey_date,
            "minute_rows": minute_rows,  # [(minute_of_day, {类别: 数量}), ...]
        })

    return results
