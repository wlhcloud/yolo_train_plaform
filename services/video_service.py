import threading
from flask import current_app
from app import create_app
from models import Project


class VideoService:
    """视频服务"""

    def start_video_capture(self, project_id, video_file_content, interval, max_count):
        """启动视频截图"""
        try:
            project = Project.query.get_or_404(project_id)

            if not video_file_content:
                return {"success": False, "message": "视频流不能为空"}

            # 启动RTSP截图线程
            application = create_app()

            def rtsp_thread():
                with application.app_context():
                    camera_capture = application.video_capture
                    camera_capture.capture_video_images(
                        project_id, video_file_content, interval, max_count
                    )

            thread = threading.Thread(target=rtsp_thread)
            thread.daemon = True
            thread.start()

            return {"success": True, "message": "视频截图已启动"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def stop_video_capture(self, project_id):
        """停止RTSP截图"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.video_capture
                camera_capture.stop_video_capture(project_id)

            return {"success": True, "message": "视频截图已停止"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_video_status(self, project_id):
        """获取RTSP截图状态"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.video_capture
                status = camera_capture.get_video_status(project_id)

            return {"success": True, "status": status}
        except Exception as e:
            return {"success": False, "message": str(e)}


video_service = VideoService()
