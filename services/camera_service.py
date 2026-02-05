from datetime import datetime
import os
import threading

import cv2
from camera_capture import camera_capture
from models import Image, Project, db
from loguru import logger as log
from managers.rtsp_threadpool_manager import rtsp_manager
from project_dir_manager import ProjectDirManager
from utils import save_base64_image
from PIL import Image as PILImage


class CameraService:
    """摄像头服务"""

    def start_rtsp_capture(self, project_id, rtsp_url, interval, max_count):
        """启动RTSP截图"""
        try:
            project = Project.query.get_or_404(project_id)

            if not rtsp_url:
                return {"success": False, "message": "RTSP地址不能为空"}

            # 启动RTSP截图线程
            def rtsp_thread():
                camera_capture.capture_rtsp_images(
                    project_id, rtsp_url, interval, max_count
                )

            thread = threading.Thread(target=rtsp_thread)
            thread.daemon = True
            thread.start()

            return {"success": True, "message": "RTSP截图已启动"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def stop_rtsp_capture(self, project_id):
        """停止RTSP截图"""
        try:
            camera_capture.stop_rtsp_capture(project_id)

            return {"success": True, "message": "RTSP截图已停止"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_rtsp_status(self, project_id):
        """获取RTSP截图状态"""
        try:
            status = camera_capture.get_rtsp_status(project_id)

            return {"success": True, "status": status}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def discover_onvif_devices(self):
        """发现ONVIF设备"""
        try:
            devices = camera_capture.discover_onvif_devices()

            return {"success": True, "devices": devices}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_onvif_profiles(self, device_ip, device_port, username, password):
        """获取ONVIF设备的配置文件列表"""
        try:
            if not username or not password:
                return {"success": False, "message": "用户名和密码不能为空"}

            profiles = camera_capture.get_onvif_profiles(
                device_ip, device_port, username, password
            )

            return {"success": True, "profiles": profiles}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def start_onvif_capture(
        self,
        project_id,
        device_ip,
        device_port,
        username,
        password,
        profile_token,
        interval,
        max_count,
    ):
        """启动ONVIF截图"""
        try:
            project = Project.query.get_or_404(project_id)

            if not all([device_ip, device_port, username, password, profile_token]):
                return {"success": False, "message": "设备信息不完整"}

            # 启动ONVIF截图线程（使用改进版本）
            def onvif_thread():
                # 使用改进的截图方法
                camera_capture.capture_onvif_images_improved(
                    project_id,
                    device_ip,
                    device_port,
                    username,
                    password,
                    profile_token,
                    interval,
                    max_count,
                )

            thread = threading.Thread(target=onvif_thread)
            thread.daemon = True
            thread.start()

            return {"success": True, "message": "ONVIF截图已启动"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def stop_onvif_capture(self, project_id, profile_token):
        """停止ONVIF截图"""
        try:
            camera_capture.stop_onvif_capture(project_id, profile_token)

            return {"success": True, "message": "ONVIF截图已停止"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_onvif_status(self, project_id, profile_token):
        """获取ONVIF截图状态"""
        try:
            status = camera_capture.get_onvif_status(project_id, profile_token)

            return {"success": True, "status": status}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def preview_rtsp_stream(self, project_id, rtsp_url):
        """预览RTSP流"""
        try:
            if not rtsp_url:
                return {"success": False, "message": "RTSP地址不能为空"}

            log.info(f"开始预览RTSP-----{project_id}:{rtsp_url}")
            task_id = project_id
            result = rtsp_manager.add_rtsp_task(
                task_id=task_id,
                rtsp_url=rtsp_url,
                is_inference=False,
            )

            return {"success": True, "message": "RTSP预览已启动", "data": result}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def save_rtsp_manual_capture(self, project_id, screenshot_data):
        """保存RTSP手动截图"""
        project_upload_dir = ProjectDirManager.ensure_project_upload_dir(project_id)
        # 生成文件名
        timestamp = int(datetime.now().timestamp() * 1000)
        filename = f"{timestamp}_rtsp.jpg"
        file_path = os.path.join(project_upload_dir, filename)

        # 保存base64图片数据到文件
        success = save_base64_image(screenshot_data, file_path)
        if not success:
            return {"success": False, "message": "保存截图失败"}
        # 获取图片尺寸
        img = PILImage.open(file_path)
        width, height = img.size

        # 计算相对路径和POSIX路径
        relative_path = ProjectDirManager.get_relative_path(file_path)
        posix_path = ProjectDirManager.get_posix_path(relative_path)
        posix_path = posix_path.replace("static/", "", 1)

        # 保存到数据库
        image = Image(
            filename=filename,
            original_filename=f"rtsp_capture_{timestamp}.jpg",
            path=posix_path,
            project_id=project_id,
            width=width,
            height=height,
        )
        db.session.add(image)
        db.session.commit()
        return {
            "success": True,
            "message": "截图保存成功",
            "data": {"image_id": image.id},
        }

    def stop_rtsp_preview(self, project_id):
        """停止RTSP预览"""
        try:
            log.info(f"停止预览RTSP-----{project_id}")
            task_id = project_id
            rtsp_manager.stop_rtsp_task(task_id=task_id)

            return {"success": True, "message": "RTSP预览已停止"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_rtsp_preview_status(self, project_id):
        """获取RTSP预览状态"""
        try:
            task_status = rtsp_manager.get_task_status(task_id=project_id)
            if not task_status:
                return {"success": False, "message": "无预览任务"}
            return {"success": True, "status": task_status}
        except Exception as e:
            return {"success": False, "message": str(e)}


camera_service = CameraService()
