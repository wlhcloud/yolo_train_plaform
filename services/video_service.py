import threading
from video_capture import video_capture
from models import Project


class VideoService:
    """视频服务"""

    def start_video_capture(
        self,
        project_id,
        video_file_content,
        interval,
        max_count,
        start_time=None,
        end_time=None,
    ):
        """启动视频截图"""
        try:
            project = Project.query.get_or_404(project_id)

            if not video_file_content:
                return {"success": False, "message": "视频流不能为空"}

            # 启动视频截图线程
            def video_thread():
                video_capture.capture_video_images(
                    project_id,
                    video_file_content,
                    interval,
                    max_count,
                    start_time,
                    end_time,
                )

            thread = threading.Thread(target=video_thread)
            thread.daemon = True
            thread.start()

            return {"success": True, "message": "视频截图已启动"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def stop_video_capture(self, project_id):
        """停止RTSP截图"""
        try:
            video_capture.stop_video_capture(project_id)
            return {"success": True, "message": "视频截图已停止"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_video_status(self, project_id):
        """获取RTSP截图状态"""
        try:
            status = video_capture.get_video_status(project_id)
            return {"success": True, "status": status}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_video_time_capture(
        self, project_id, video_file_content, video_current_time
    ):
        """根据时间截图"""
        try:
            current_time = video_capture.get_video_time_capture(
                project_id, video_file_content, video_current_time
            )
            return {"success": True, "current_time": current_time}
        except Exception as e:
            return {"success": False, "message": str(e)}


video_service = VideoService()
