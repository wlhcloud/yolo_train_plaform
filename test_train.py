import os
import sys
from ultralytics import YOLO

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 检查data.yaml文件是否存在
data_yaml_path = 'static/datasets/1/data.yaml'
if not os.path.exists(data_yaml_path):
    print(f"数据集配置文件不存在: {data_yaml_path}")
    sys.exit(1)

print("开始加载YOLOv8模型...")
try:
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit(1)

print("开始训练...")
try:
    # 训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=1,  # 只训练1个epoch用于测试
        imgsz=640,
        project='static/datasets/1',
        name='train_results'
    )
    print("训练完成!")
except Exception as e:
    print(f"训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)