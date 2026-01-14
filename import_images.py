import os
from datetime import datetime
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ================== 配置 ==================
# 修改为你自己的数据库 URI（示例为 SQLite）
DATABASE_URL = "sqlite:///F:/work/mavenPorjects_MY_AI/YOLOv8Web/instance/yolov8_platform.db"  # 例如：sqlite:///instance/app.db

# 要导入的项目 ID（必须存在于 project 表中）
PROJECT_ID = 5

# 图片文件夹路径（绝对路径）
IMAGE_FOLDER = "F:\\work\\mavenPorjects_MY_AI\\YOLOv8Web\\static\\datasets\\5\\images\\train"

# 图片在 Web 应用中的相对路径前缀（用于 path 字段）
# 例如：如果你把图片复制到 static/uploads/，则 path = f"uploads/{filename}"
WEB_PATH_PREFIX = "uploads/5"

# 支持的图片扩展名
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# ================== 数据模型 ==================
Base = declarative_base()

class Project(Base):
    __tablename__ = 'project'
    id = Column(Integer, primary_key=True)

class ImageModel(Base):
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False)
    dataset_type = Column(String(20))
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

# ================== 主程序 ==================
def main():
    # 创建数据库引擎和会话
    engine = create_engine(DATABASE_URL, echo=False)  # echo=True 可查看 SQL 日志
    Base.metadata.create_all(engine)  # 确保表存在（不会覆盖已有表）
    Session = sessionmaker(bind=engine)
    session = Session()

    # 检查 project 是否存在
    project = session.query(Project).filter_by(id=PROJECT_ID).first()
    if not project:
        print(f"错误：project_id={PROJECT_ID} 不存在！")
        return

    # 遍历图片文件夹
    for root, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            original_path = os.path.join(root, file)
            try:
                with Image.open(original_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"跳过无效图片 {file}: {e}")
                continue

            # 生成唯一文件名（避免冲突）
            safe_filename = os.path.basename(original_path).replace(" ", "_")
            # 如果你希望重命名，可以加上时间戳或 UUID，例如：
            # safe_filename = str(uuid.uuid4()) + ext

            # 构造 Web 路径（用于前端访问）
            web_path = os.path.join(WEB_PATH_PREFIX, safe_filename).replace("\\", "/")
            filename = file  # 与 original_filename 相同
            web_path = f"{WEB_PATH_PREFIX}/{filename}".replace("\\", "/")

            # 创建 ImageModel 实例
            image_record = ImageModel(
                filename=safe_filename,
                original_filename=file,
                path=web_path,
                project_id=PROJECT_ID,
                dataset_type=None,  # 初始未分配
                width=width,
                height=height,
                created_at=datetime.utcnow()
            )

            session.add(image_record)
            print(f"已添加: {file} ({width}x{height})")

    # 提交事务
    try:
        session.commit()
        print("\n✅ 所有图片已成功导入数据库！")
    except Exception as e:
        session.rollback()
        print(f"\n❌ 导入失败: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    main()

    #运行程序
    # python ./import_images.py