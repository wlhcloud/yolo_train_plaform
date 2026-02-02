import os
import posixpath
import subprocess
import uuid

import cv2

from PIL import Image as PILImage


def extract_first_frame(source_url, save_dir):
    """
    从视频文件或 RTSP 中抽取第一帧
    """
    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source_url}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("读取第一帧失败")

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, frame)
    return save_path


def start_ffmpeg_rtsp_push(width, height, fps, push_url):
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",  # 改为info，便于调试（生产可改回error）
        "-y",  # 强制覆盖输出，避免卡住
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",  # 从标准输入读取帧
        "-an",  # 禁用音频
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",  # 最快编码速度，优先实时性
        "-tune",
        "zerolatency",  # 零延迟，适配实时推流
        "-g",
        str(fps * 2),  # 关键帧间隔（2秒1个关键帧）
        "-bufsize",
        "512k",  # 缓冲区大小，提升稳定性
        "-rtsp_transport",
        "tcp",  # 强制TCP传输（UDP易丢包）
        "-f",
        "rtsp",
        "-rtsp_flags",
        "listen",  # 被动模式（部分RTSP服务器需要）
        push_url,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def start_ffmpeg_video_save(width, height, fps, output_path):
    """
    保存视频

    :param width: 说明
    :param height: 说明
    :param fps: 说明
    :param output_path: 说明
    """
    ffmpeg_cmd = [
        "ffmpeg",
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
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    return subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def get_file_info(file_path, file_type):
    """
    获取文件基本信息（大小、尺寸）
    :param file_path: 文件完整路径
    :param file_type: 文件类型（image/video）
    :return: 包含size_bytes, width, height的字典
    """
    info = {"size_bytes": os.path.getsize(file_path), "width": None, "height": None}

    try:
        if file_type == "image":
            img = PILImage.open(file_path)
            info["width"], info["height"] = img.size
        elif file_type == "video":
            # 对于视频，需要先提取第一帧再获取尺寸
            cover_path = extract_first_frame(file_path, os.path.dirname(file_path))
            img = PILImage.open(cover_path)
            info["width"], info["height"] = img.size
    except Exception as e:
        print(f"获取文件{file_path}信息失败: {e}")

    return info


def get_relative_path(full_path):
    """
    将完整路径转换为相对于static目录的URL路径（统一使用正斜杠）
    :param full_path: 完整文件路径
    :return: 相对URL路径
    """
    relative_path = os.path.relpath(full_path, "static")
    return posixpath.join(*relative_path.split(os.sep))


def get_project_upload_path(project_id, filename=None):
    """
    获取项目上传文件的完整路径
    :param project_id: 项目ID
    :param filename: 文件名（可选）
    :return: 完整的文件/目录路径
    """
    base_path = os.path.join("static/uploads", str(project_id))
    if filename:
        return os.path.join(base_path, filename)
    return base_path


def get_app_root():
    """获取应用根目录路径"""
    return os.getcwd()
