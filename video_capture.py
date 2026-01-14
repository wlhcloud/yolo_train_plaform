import os
import tempfile
import cv2
import time
import threading
from datetime import datetime
from PIL import Image as PILImage
from models import db, Image
from project_dir_manager import ProjectDirManager


class VideoCapture:
    """视频文件截图管理器，支持常见视频格式（MP4/AVI/MKV等）"""

    def __init__(self):
        # 维护视频截图线程状态，key为视频任务ID，value为任务详情
        self.video_threads = {}

    def capture_video_images(self, project_id, video_file_content, interval, max_count):
        """
        从视频文件流中截取图片并保存到项目目录（不落地原视频，仅保存截图）
        Args:
            project_id (int): 项目ID
            video_file_content : 视频文件流
            interval (int): 截图间隔（秒）
            max_count (int): 最大截图数量
        """
        if not video_file_content:
            return {"success": False, "message": "视频文件不能为空"}
        if interval < 1 or interval > 3600:
            return {"success": False, "message": "截图间隔必须在1-3600秒之间"}
        if max_count < 1 or max_count > 10000:
            return {"success": False, "message": "最大截图数量必须在1-10000之间"}

        thread_id = f"video_{project_id}"
        self.video_threads[thread_id] = {
            "running": True,
            "captured_count": 0,
            "max_count": max_count,
            "error": None,
        }

        temp_video_path = None
        try:
            project_upload_dir = ProjectDirManager.ensure_project_upload_dir(project_id)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_path = temp_file.name
            temp_file.write(video_file_content)
            temp_file.close()

            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                self.video_threads[thread_id][
                    "error"
                ] = "无法打开视频文件，格式不支持或文件损坏"
                return

            fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率（每秒帧数）
            if fps <= 0:  # 兼容部分视频帧率获取失败的场景
                fps = 25
                self.video_threads[thread_id][
                    "warning"
                ] = "无法获取视频帧率，默认使用25帧/秒"
            frame_interval = int(fps * interval)  # 转换为「帧间隔」（每多少帧截取一张）
            current_frame = 0

            # 步骤5：循环截取视频帧（参考RTSP的循环逻辑，保持状态更新一致性）
            while (
                self.video_threads[thread_id]["running"]
                and self.video_threads[thread_id]["captured_count"] < max_count
            ):

                # 检查线程是否被终止（支持手动停止截图）
                if not self.video_threads[thread_id]["running"]:
                    break

                # 读取当前视频帧
                ret, frame = cap.read()
                if not ret:
                    self.video_threads[thread_id][
                        "warning"
                    ] = f"视频已读取完毕，仅截取到{self.video_threads[thread_id]['captured_count']}张图片"
                    break

                # 按计算的帧间隔截取图片
                if current_frame % frame_interval == 0:
                    # 生成唯一截图文件名
                    timestamp = int(datetime.now().timestamp() * 1000)
                    filename = f"{timestamp}_video.jpg"
                    file_path = os.path.join(project_upload_dir, filename)

                    save_success = cv2.imwrite(
                        file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if save_success:
                        with PILImage.open(file_path) as img:
                            width, height = img.size

                        # 计算相对路径和POSIX路径
                        relative_path = ProjectDirManager.get_relative_path(file_path)
                        posix_path = ProjectDirManager.get_posix_path(relative_path)
                        posix_path = posix_path.replace("static/", "", 1)

                        # 保存截图信息到数据库
                        from app import create_app

                        application = create_app()
                        with application.app_context():
                            try:
                                image_record = Image(
                                    filename=filename,
                                    original_filename=f"video_capture_{timestamp}.jpg",
                                    path=posix_path,
                                    project_id=project_id,
                                    width=width,
                                    height=height,
                                )
                                db.session.add(image_record)
                                db.session.commit()

                                # 更新截图计数
                                self.video_threads[thread_id]["captured_count"] += 1
                            except Exception as db_e:
                                db.session.rollback()
                                self.video_threads[thread_id][
                                    "error"
                                ] = f"数据库保存失败: {str(db_e)}"
                    else:
                        self.video_threads[thread_id][
                            "error"
                        ] = f"截图保存失败，路径：{file_path}"

                current_frame += 1

                time.sleep(0.001)

        except Exception as e:
            self.video_threads[thread_id]["error"] = f"视频截图过程中出错: {str(e)}"
        finally:
            if "cap" in locals() and cap.isOpened():
                cap.release()
                cv2.destroyAllWindows()

            # 删除临时视频文件，避免残留
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass

            self.video_threads[thread_id]["running"] = False

    def stop_video_capture(self, thread_id):
        """
        停止指定视频截图任务

        Args:
            thread_id (str): 视频任务ID（由 start_video_capture 返回或 get_video_status 查询）
        """
        if thread_id in self.video_threads:
            # 仅更新运行状态，截图线程会在下一次循环中终止
            self.video_threads[thread_id]["running"] = False

    def get_video_status(self, project_id):
        """
        获取指定视频截图任务状态

        Args:
            project_id (int): 项目id

        Returns:
            dict: 视频截图任务状态（包含运行状态、已截图数量、错误信息等）
        """
        # 返回任务状态，无此任务则返回空字典
        thread_id = f"video_{project_id}"
        return self.video_threads.get(thread_id, {})

    def clean_expired_threads(self, expired_seconds=3600):
        """
        清理过期的视频任务状态（可选扩展，参考原类的资源管理思想）
        避免 self.video_threads 过大，占用内存

        Args:
            expired_seconds (int): 过期时间（秒），默认1小时
        """
        current_time = datetime.now().timestamp()
        expired_thread_ids = []

        for thread_id, thread_info in self.video_threads.items():
            # 任务已停止且运行结束时间超过过期时间
            if not thread_info.get("running", False):
                task_end_time = thread_info.get("end_time", current_time)
                if current_time - task_end_time > expired_seconds:
                    expired_thread_ids.append(thread_id)

        # 批量删除过期任务
        for thread_id in expired_thread_ids:
            del self.video_threads[thread_id]
