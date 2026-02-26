import base64
import io
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


def save_base64_image(base64_data, save_path):
    """
    保存Base64编码的图片数据到文件
    :param base64_data: Base64编码的图片数据
    :param save_path: 保存路径
    :return: 是否保存成功
    """
    try:
        import base64

        # 解码Base64数据
        image_data = base64.b64decode(base64_data)

        # 写入文件
        with open(save_path, "wb") as f:
            f.write(image_data)

        return True
    except Exception as e:
        print(f"保存Base64图片失败: {e}")
        return False


def save_yolobase_model(model_file):
    """
    保存上传的模型文件到uploaded_models目录

    Args:
        model_file (FileStorage): 上传的模型文件

    Returns:
        str: 保存的模型文件路径
    """
    if not model_file or not model_file.filename:
        raise ValueError("未提供模型文件")

    filename = model_file.filename
    # 确保文件名安全
    from werkzeug.utils import secure_filename

    filename = secure_filename(filename)

    # 确保文件扩展名为.pt
    if not filename.endswith(".pt"):
        filename += ".pt" if "." not in filename else ""

    uploaded_models_dir = os.path.join("projects", "yolo_base")
    os.makedirs(uploaded_models_dir, exist_ok=True)

    model_path = os.path.join(uploaded_models_dir, filename)
    model_file.save(model_path)

    # 确保文件已正确保存
    if not os.path.exists(model_path):
        raise ValueError("模型文件保存失败")

    return model_path


class CustomImageFile:
    """
    自定义文件对象类，完全适配原inference_image方法的需求：
    1. 有filename属性
    2. 支持save()方法保存到指定路径
    """

    def __init__(self, image_bytes, filename):
        self.image_bytes = image_bytes  # 图片二进制数据
        self.filename = filename  # 文件名（带后缀）
        self._file = io.BytesIO(image_bytes)  # 二进制文件流

        try:
            with PILImage.open(self._file) as img:
                self.image_size = (img.width, img.height)
            self._file.seek(0)
        except Exception as e:
            raise ValueError(f"解析图片尺寸失败：{str(e)}")

    def save(self, save_path):
        with open(save_path, "wb") as f:
            f.write(self.image_bytes)

    def close(self):
        self._file.close()


def base64_to_custom_image_file(base64_str, filename="inference_input.jpg"):
    """
    将base64字符串转换为CustomImageFile对象（适配原inference_image方法）
    :param base64_str: 原始base64编码的图片字符串（可带data:image前缀）
    :param filename: 自定义文件名（含后缀，如xxx.jpg/xxx.png）
    :return: CustomImageFile对象
    """
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)

        PILImage.open(io.BytesIO(image_bytes)).verify()

        custom_file = CustomImageFile(image_bytes, filename)
        return custom_file
    except Exception as e:
        raise ValueError(f"base64转换为文件对象失败：{str(e)}")
