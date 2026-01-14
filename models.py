import time
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Project(db.Model):
    __tablename__ = "project"
    __table_args__ = {"schema": "yolov8_platform"}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    ai_assist_enabled = db.Column(db.Boolean, default=False)  # AI辅助标注是否开启

    # 关联关系
    # images = db.relationship('Image', backref='project', lazy=True, cascade='all, delete-orphan')
    # labels = db.relationship('Label', backref='project', lazy=True, cascade='all, delete-orphan')
    export_records = db.relationship(
        "ExportRecord",
        back_populates="project",
        lazy=True,
        cascade="all, delete-orphan",
    )


class Image(db.Model):
    __tablename__ = "image"
    __table_args__ = {"schema": "yolov8_platform"}
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    original_filename = db.Column(db.String(100), nullable=False)
    path = db.Column(db.String(200), nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    dataset_type = db.Column(
        db.String(20), default="unassigned"
    )  # train, val, test, unassigned

    # 外键
    project_id = db.Column(
        db.Integer, db.ForeignKey("yolov8_platform.project.id"), nullable=False
    )

    # 关联关系
    annotations = db.relationship(
        "Annotation", backref="image", lazy=True, cascade="all, delete-orphan"
    )


class Label(db.Model):
    __tablename__ = "label"
    __table_args__ = {"schema": "yolov8_platform"}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    color = db.Column(db.String(7), default="#0066ff")  # HEX颜色代码
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    target = db.Column(db.String(50), nullable=True)

    # 外键
    project_id = db.Column(
        db.Integer, db.ForeignKey("yolov8_platform.project.id"), nullable=False
    )

    # 关联关系
    annotations = db.relationship(
        "Annotation", backref="yolov8_platform.label", lazy=True
    )


class Annotation(db.Model):
    __tablename__ = "annotation"
    __table_args__ = {"schema": "yolov8_platform"}
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(
        db.Integer, db.ForeignKey("yolov8_platform.image.id"), nullable=False
    )
    label_id = db.Column(
        db.Integer, db.ForeignKey("yolov8_platform.label.id"), nullable=False
    )
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)

    # 关系 (移除与backref冲突的定义)


# 导出记录模型
class ExportRecord(db.Model):
    __tablename__ = "export_record"
    __table_args__ = {"schema": "yolov8_platform"}
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(
        db.Integer, db.ForeignKey("yolov8_platform.project.id"), nullable=False
    )
    format = db.Column(db.String(50), nullable=False)  # 导出格式 (onnx, torchscript等)
    path = db.Column(db.String(500), nullable=False)  # 导出文件路径
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关系
    project = db.relationship("Project", back_populates="export_records")


# LLM配置模型
class LLMConfig(db.Model):
    __tablename__ = "llm_config"
    __table_args__ = {"schema": "yolov8_platform"}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # 配置名称
    base_url = db.Column(db.String(500), nullable=False)  # API基础URL
    api_key = db.Column(db.String(200), nullable=False)  # API密钥
    model = db.Column(db.String(100), nullable=False)  # 模型名称
    is_active = db.Column(db.Boolean, default=False)  # 是否为当前激活配置
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


# 更新项目模型以包含导出记录关系


# 更新项目模型以包含导出记录关系
# 标注任务主表（对应任务管理核心信息）
class AnnotationTask(db.Model):
    __tablename__ = "task_annotation"
    __table_args__ = {
        "schema": "yolov8_platform",
        "comment": "AI模型训练平台-标注任务主表，管理标注任务的基础信息和状态管控",
    }

    id = db.Column(db.Integer, primary_key=True, comment="标注任务唯一标识，自增主键")
    project_id = db.Column(
        db.Integer, nullable=False, comment="关联项目ID，对应项目表主键"
    )
    start_time = db.Column(db.Date, nullable=False, comment="任务开始日期")
    end_time = db.Column(db.Date, nullable=False, comment="任务结束日期")
    principal = db.Column(db.String(50), nullable=False, comment="任务负责人姓名")
    total_count = db.Column(
        db.Integer, default=0, nullable=False, comment="任务分配的总图片数量"
    )
    is_submitted = db.Column(
        db.SmallInteger,
        default=0,
        nullable=False,
        comment="任务提交状态：0-未提交，1-已提交",
    )
    completed_count = db.Column(
        db.Integer, default=0, nullable=False, comment="任务已完成标注的图片数量"
    )
    created_at = db.Column(
        db.DateTime, default=datetime.utcnow, comment="任务创建时间，自动生成当前时间戳"
    )
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="任务更新时间，修改时自动刷新",
    )

    task_items = db.relationship(
        "AnnotationTaskItem",
        backref="annotation_task",
        lazy=True,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        db.ForeignKeyConstraint(
            ["project_id"],
            ["yolov8_platform.project.id"],
            ondelete="RESTRICT",
            name="fk_task_annotation_project",
        ),
        db.CheckConstraint("start_time <= end_time", name="ck_task_time_range"),
        db.CheckConstraint("is_submitted IN (0, 1)", name="ck_task_submitted_status"),
        {
            "schema": "yolov8_platform",
            "comment": "AI模型训练平台-标注任务主表，管理标注任务的基础信息和状态管控",
        },
    )

    def __repr__(self):
        return f"<AnnotationTask(id={self.id}, principal={self.principal}, is_submitted={self.is_submitted})>"


# 标注任务明细表（子表，记录每个任务下图片的标注状态）
class AnnotationTaskItem(db.Model):
    __tablename__ = "task_annotation_item"
    __table_args__ = {
        "schema": "yolov8_platform",
        "comment": "AI模型训练平台-标注任务明细表，记录每个任务下图片的标注进度和状态",
    }

    id = db.Column(
        db.Integer, primary_key=True, comment="标注任务明细唯一标识，自增主键"
    )
    task_id = db.Column(db.Integer, nullable=False, comment="关联标注任务主表ID")
    image_id = db.Column(db.Integer, nullable=False, comment="关联待标注图片ID")
    annotate_status = db.Column(
        db.SmallInteger,
        default=0,
        nullable=False,
        comment="图片标注状态：0-未标注，1-已标注",
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow, comment="明细创建时间")
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="明细更新时间，状态变更时自动刷新",
    )

    # 约束定义
    __table_args__ = (
        db.ForeignKeyConstraint(
            ["task_id"],
            ["yolov8_platform.task_annotation.id"],
            ondelete="CASCADE",
            name="fk_task_item_task",
        ),
        db.ForeignKeyConstraint(
            ["image_id"],
            ["yolov8_platform.image.id"],
            ondelete="RESTRICT",
            name="fk_task_item_image",
        ),
        db.CheckConstraint("annotate_status IN (0, 1)", name="ck_item_annotate_status"),
        db.UniqueConstraint("task_id", "image_id", name="uk_task_image_unique"),
        {
            "schema": "yolov8_platform",
            "comment": "AI模型训练平台-标注任务明细表，记录每个任务下图片的标注进度和状态",
        },
    )

    def __repr__(self):
        return f"<AnnotationTaskItem(id={self.id}, task_id={self.task_id}, image_id={self.image_id}, annotate_status={self.annotate_status})>"


class ReasoningMaterial(db.Model):
    """
    推理素材Model（对应yolov8_platform.reasoning_materials表）
    存储图片、视频、RTSP三种类型的推理素材及相关信息
    """

    # 指定Schema和表名（对应PostgreSQL的yolov8_platform.reasoning_materials）
    __table_args__ = {
        "schema": "yolov8_platform",  # 对应数据库中的schema
        "extend_existing": True,  # 避免重复定义模型的报错
    }
    __tablename__ = "reasoning_materials"

    # 字段定义（与优化后的表结构一一对应）
    id = db.Column(
        db.BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="唯一标识，自增主键",
    )
    material_type = db.Column(
        db.SmallInteger,
        nullable=False,
        comment="素材类型：1=图片，2=视频，3=RTSP（网络流）",
    )
    project_id = db.Column(
        db.BigInteger, nullable=False, comment="关联项目ID，对应projects表id"
    )
    filename = db.Column(
        db.String(100),
        nullable=False,
        comment="素材名称（含文件后缀，如test.jpg、stream.rtsp）",
    )
    source_type = db.Column(
        db.String(100),
        nullable=True,
        comment="资源类型名称",
    )
    path = db.Column(
        db.String(1000),
        nullable=False,
        comment="素材存储路径/网络地址（本地路径或RTSP地址）",
    )
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    size_bytes = db.Column(db.Integer)
    cover_image = db.Column(
        db.String(1000),
        nullable=False,
        comment="第一帧图片存储路径",
    )
    start_time = db.Column(
        db.DateTime,
        nullable=True,
        default=None,
        comment="开始推理时间",
    )
    end_time = db.Column(
        db.DateTime,
        nullable=True,
        default=None,
        comment="结束推理时间",
    )
    status = db.Column(
        db.SmallInteger,
        nullable=False,
        default=0,
        comment="素材状态：0=未开始，1=推理中，2=已推理",
    )

    config = db.Column(
        db.JSON, comment="素材配置信息（JSON格式），含分类、区域框选等配置"
    )
    result = db.Column(
        db.JSON, comment="推理结果信息（JSON格式），含模型输出、分类映射等结果"
    )
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="创建时间，默认当前系统时间",
    )
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="修改时间，默认当前系统时间，支持触发器自动更新",
    )

    def __repr__(self):
        """
        模型打印格式，便于调试
        """
        return f"<ReasoningMaterial(id={self.id}, name='{self.name}', material_type={self.material_type}, status={self.status})>"
