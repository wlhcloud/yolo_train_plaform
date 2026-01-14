from concurrent.futures import ThreadPoolExecutor
import mimetypes
import os
import sys
import threading
from flask import Flask
from dotenv import load_dotenv
import pytz
from datetime import datetime
from sqlalchemy import text
import locale
from sqlalchemy.orm import scoped_session, sessionmaker
from PIL import ImageFont

# 将当前目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()
project_task_locks = {}
# 线程池（全局复用，控制最大并发数）
MAX_WORKERS = 5
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
# 媒体服务器路径
MEDIA_SERVER = "http://127.0.0.1:8080/live/"  # 推流服务器地址
RTSP_SERVER = "rtsp://127.0.0.1:8554/live/"  # 媒体服务地址
# 全局变量：后台线程用的scoped_session工厂
backend_session_factory = None

# 中文字体
font_path = "static/fonts/truetype/wqy/wqy-zenhei.ttc"


def create_app():
    app = Flask(__name__)
    print("Python 默认编码:", sys.getdefaultencoding())
    print("文件系统编码:", sys.getfilesystemencoding())
    print("系统首选编码:", locale.getpreferredencoding())
    # 配置
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"options": "-c client_encoding=utf8"}
    }
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    print(">>> DATABASE_URL (repr):", repr(os.environ.get("DATABASE_URL")))
    print(">>> DATABASE_URL (raw) :", os.environ.get("DATABASE_URL"))
    # 设置时区为东八区（北京时间）
    app.config["TIMEZONE"] = "Asia/Shanghai"

    # 确保必要的目录存在
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/datasets", exist_ok=True)
    os.makedirs("static/models", exist_ok=True)

    # 注册数据库
    from models import db

    db.init_app(app)

    # 创建表
    with app.app_context():
        # 确保 schema 存在
        db.session.execute(text("CREATE SCHEMA IF NOT EXISTS yolov8_platform"))
        db.session.commit()
        db.create_all()

        global backend_session_factory
        backend_session_factory = scoped_session(
            sessionmaker(bind=db.engine),  # 绑定已初始化的db.engine
            scopefunc=threading.get_ident,  # 按线程ID隔离session
        )
        # 把工厂也挂载到app实例上，方便其他模块调用
        app.backend_session_factory = backend_session_factory

    # 注册蓝图
    # 动态导入routes模块
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "routes", os.path.join(os.path.dirname(os.path.abspath(__file__)), "routes.py")
    )
    routes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routes)
    app.register_blueprint(routes.main)

    # 初始化摄像头捕获管理器
    from camera_capture import CameraCapture
    from video_capture import VideoCapture

    app.camera_capture = CameraCapture()
    app.video_capture = VideoCapture()

    # 添加时区处理的模板过滤器
    @app.template_filter("beijing_time")
    def beijing_time_filter(dt):
        if dt:
            # 将UTC时间转换为北京时间
            utc = pytz.timezone("UTC")
            beijing = pytz.timezone("Asia/Shanghai")
            utc_time = utc.localize(dt)
            beijing_time = utc_time.astimezone(beijing)
            return beijing_time.strftime("%Y-%m-%d %H:%M:%S")
        return "未知"

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5500)
