# FuturoGen - YOLOv8 Web Platform

FuturoGen是一个基于YOLOv8的目标检测Web平台，提供从数据管理、图像标注、模型训练到模型导出的一站式解决方案。

## 功能介绍

### 1. 项目管理
- 创建、查看、删除项目
- 项目概览显示统计信息和进度

### 2. 图像管理
- 批量上传图像（支持ZIP压缩包）
- 拖拽上传
- 图像分页浏览
- 删除单张图像

### 3. 图像标注
- 在线图像标注工具
- 支持矩形框标注
- 实时保存标注结果
- 标签管理

### 4. 数据集划分
- 自动划分训练集、验证集和测试集
- 可视化数据集分布

### 5. 模型训练
- 基于YOLOv8的目标检测模型训练
- 实时查看训练状态和日志
- 训练过程可视化

### 6. 模型导出
- 支持导出为ONNX、TorchScript等格式
- 导出历史记录管理
- 模型文件下载

## 技术架构

### 前端技术
- HTML5/CSS3/JavaScript
- Bootstrap 5 UI框架
- Jinja2 模板引擎
- Chart.js 图表库

### 后端技术
- Python 3.8+
- Flask Web框架
- SQLAlchemy ORM
- SQLite 数据库

### 核心库
- Ultralytics YOLOv8 - 目标检测模型
- OpenCV - 图像处理
- Pillow - 图像操作
- PyYAML - 配置文件处理

### 目录结构
```
yolotrain/
├── app.py              # 应用入口
├── routes.py           # 路由定义
├── models.py           # 数据模型
├── requirements.txt    # 依赖包列表
├── .env                # 环境配置
├── static/             # 静态资源
│   ├── uploads/        # 上传文件
│   ├── datasets/       # 数据集
│   ├── models/         # 模型文件
│   └── css/js/images/  # 前端资源
├── templates/          # HTML模板
│   ├── base.html       # 基础模板
│   ├── index.html      # 首页
│   ├── project_detail.html  # 项目详情
│   ├── images.html     # 图像管理
│   ├── annotate.html   # 图像标注
│   ├── dataset.html    # 数据集划分
│   ├── train.html      # 模型训练
│   └── export.html     # 模型导出
└── instance/           # 数据库文件
    └── yolov8_platform.db
```

## 安装指南

### 系统要求
- Python 3.8 或更高版本
- pip 包管理器
- 至少4GB内存（推荐8GB以上）
- 至少10GB磁盘空间

### 安装步骤

1. 克隆项目代码：
```bash
git clone <repository-url>
cd yolotrain
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
复制 [.env.example](file:///Users/boyan/yolotrain/.env.example) 文件为 [.env](file:///Users/boyan/yolotrain/.env) 并根据需要修改配置：
```bash
cp .env.example .env
```

5. 初始化数据库：
```bash
python -c "from app import create_app; from models import db; app = create_app(); with app.app_context(): db.create_all()"
```

6. 启动应用：
```bash
python app.py
```

7. 访问应用：
打开浏览器访问 `http://localhost:5005`

## 使用指南

### 创建项目
1. 点击首页的"创建项目"按钮
2. 输入项目名称和描述
3. 点击"创建"按钮

### 上传图像
1. 进入项目详情页面
2. 点击"图片管理"标签
3. 选择图像文件或ZIP压缩包
4. 点击"上传"按钮

### 图像标注
1. 在项目详情页面点击"图片标注"标签
2. 选择要标注的图像
3. 在图像上拖拽创建标注框
4. 选择或创建标签
5. 标注自动保存

### 数据集划分
1. 点击"数据集划分"标签
2. 查看数据集分布情况
3. 系统会自动划分训练集、验证集和测试集

### 模型训练
1. 点击"模型训练"标签
2. 配置训练参数
3. 点击"开始训练"按钮
4. 实时查看训练进度和日志

### 模型导出
1. 训练完成后点击"模型导出"标签
2. 选择导出格式（ONNX、TorchScript等）
3. 点击"导出模型"按钮
4. 导出完成后可下载模型文件

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

### 开发环境设置
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

### 代码规范
- 遵循PEP8代码规范
- 添加适当的注释
- 编写清晰的提交信息

## 许可证

本项目采用MIT许可证，详情请见 [LICENSE](file:///Users/boyan/yolotrain/LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至项目维护者