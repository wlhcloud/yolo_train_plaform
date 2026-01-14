import threading
from flask import current_app
from app import create_app
from models import Project


class CameraService:
    """摄像头服务"""
    
    def start_rtsp_capture(self, project_id, rtsp_url, interval, max_count):
        """启动RTSP截图"""
        try:
            project = Project.query.get_or_404(project_id)
            
            if not rtsp_url:
                return {'success': False, 'message': 'RTSP地址不能为空'}
            
            # 启动RTSP截图线程
            application = create_app()
            
            def rtsp_thread():
                with application.app_context():
                    camera_capture = application.camera_capture
                    camera_capture.capture_rtsp_images(project_id, rtsp_url, interval, max_count)
            
            thread = threading.Thread(target=rtsp_thread)
            thread.daemon = True
            thread.start()
            
            return {'success': True, 'message': 'RTSP截图已启动'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def stop_rtsp_capture(self, project_id):
        """停止RTSP截图"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.camera_capture
                camera_capture.stop_rtsp_capture(project_id)
            
            return {'success': True, 'message': 'RTSP截图已停止'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def get_rtsp_status(self, project_id):
        """获取RTSP截图状态"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.camera_capture
                status = camera_capture.get_rtsp_status(project_id)
            
            return {'success': True, 'status': status}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def discover_onvif_devices(self):
        """发现ONVIF设备"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.camera_capture
                devices = camera_capture.discover_onvif_devices()
            
            return {'success': True, 'devices': devices}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def get_onvif_profiles(self, device_ip, device_port, username, password):
        """获取ONVIF设备的配置文件列表"""
        try:
            if not username or not password:
                return {'success': False, 'message': '用户名和密码不能为空'}
            
            application = create_app()
            with application.app_context():
                camera_capture = application.camera_capture
                profiles = camera_capture.get_onvif_profiles(device_ip, device_port, username, password)
            
            return {'success': True, 'profiles': profiles}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def start_onvif_capture(self, project_id, device_ip, device_port, username, password, profile_token, interval, max_count):
        """启动ONVIF截图"""
        try:
            project = Project.query.get_or_404(project_id)
            
            if not all([device_ip, device_port, username, password, profile_token]):
                return {'success': False, 'message': '设备信息不完整'}
            
            # 启动ONVIF截图线程（使用改进版本）
            application = create_app()
            
            def onvif_thread():
                with application.app_context():
                    camera_capture = application.camera_capture
                    # 使用改进的截图方法
                    camera_capture.capture_onvif_images_improved(
                        project_id, device_ip, device_port, username, password,
                        profile_token, interval, max_count)
            
            thread = threading.Thread(target=onvif_thread)
            thread.daemon = True
            thread.start()
            
            return {'success': True, 'message': 'ONVIF截图已启动'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def stop_onvif_capture(self, project_id, profile_token):
        """停止ONVIF截图"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.camera_capture
                camera_capture.stop_onvif_capture(project_id, profile_token)
            
            return {'success': True, 'message': 'ONVIF截图已停止'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def get_onvif_status(self, project_id, profile_token):
        """获取ONVIF截图状态"""
        try:
            application = create_app()
            with application.app_context():
                camera_capture = application.camera_capture
                status = camera_capture.get_onvif_status(project_id, profile_token)
            
            return {'success': True, 'status': status}
        except Exception as e:
            return {'success': False, 'message': str(e)}