import os
import math
from flask import current_app
from models import db, Image, Label, Annotation


class AnnotationService:
    """图片标注服务"""

    def get_labels(self, project_id):
        """获取项目标签"""
        labels = Label.query.filter_by(project_id=project_id).all()
        labels_data = [
            {"id": label.id, "name": label.name, "color": label.color}
            for label in labels
        ]
        return labels_data

    def create_label(self, project_id, name, color="#0066ff"):
        """创建标签"""
        label = Label(name=name, color=color, project_id=project_id)
        db.session.add(label)
        db.session.commit()

        return {"id": label.id, "name": label.name, "color": label.color}

    def delete_label(self, label_id):
        """删除标签"""
        label = Label.query.get_or_404(label_id)
        db.session.delete(label)
        db.session.commit()
        return {"success": True}

    def update_label(self, label_id, name, color, target):
        """更新标签"""
        label = Label.query.get_or_404(label_id)

        if not name:
            return {"success": False, "message": "标签名称不能为空"}

        label.name = name
        label.color = color

        db.session.commit()

        return {
            "success": True,
            "id": label.id,
            "name": label.name,
            "color": label.color,
        }

    def get_annotations(self, image_id):
        """获取图片标注"""
        annotations = Annotation.query.filter_by(image_id=image_id).all()
        annotations_data = [
            {
                "id": ann.id,
                "label_id": ann.label_id,
                "x": ann.x,
                "y": ann.y,
                "width": ann.width,
                "height": ann.height,
            }
            for ann in annotations
        ]
        return annotations_data

    def save_annotations(self, image_id, annotations):
        """保存图片标注"""
        # 删除现有的标注
        Annotation.query.filter_by(image_id=image_id).delete()

        # 添加新的标注
        for ann in annotations:
            # 确保坐标值是有效数字且在有效范围内
            try:
                x = float(ann["x"])
                y = float(ann["y"])
                width = float(ann["width"])
                height = float(ann["height"])

                # 确保值在有效范围内
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # 只有当值有效时才保存
                if not (
                    math.isnan(x)
                    or math.isnan(y)
                    or math.isnan(width)
                    or math.isnan(height)
                ):
                    annotation = Annotation(
                        image_id=image_id,
                        label_id=int(ann["label_id"]),
                        x=x,  # 归一化中心点x坐标 (0-1)
                        y=y,  # 归一化中心点y坐标 (0-1)
                        width=width,  # 归一化宽度 (0-1)
                        height=height,  # 归一化高度 (0-1)
                    )
                    db.session.add(annotation)
            except (ValueError, TypeError) as e:
                # 忽略无效的标注数据
                print(f"Skipping invalid annotation data: {ann}, error: {e}")
                continue

        db.session.commit()

        # 生成YOLO格式的标注文件
        image = Image.query.get_or_404(image_id)
        labels = Label.query.filter_by(project_id=image.project_id).all()
        label_map = {label.id: idx for idx, label in enumerate(labels)}

        # 创建标注文件路径
        txt_path = os.path.splitext(image.path)[0] + ".txt"

        # 确保目录存在
        txt_dir = os.path.dirname(
            os.path.join(current_app.root_path, "static", txt_path)
        )
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        # 写入YOLO格式标注文件
        with open(os.path.join(current_app.root_path, "static", txt_path), "w") as f:
            for ann in annotations:
                try:
                    if ann["label_id"] in label_map:
                        label_idx = label_map[ann["label_id"]]
                        # YOLO格式: class_id x_center y_center width height (都是相对于图片尺寸的归一化值)
                        # 确保值在有效范围内
                        x = float(ann["x"])
                        y = float(ann["y"])
                        width = float(ann["width"])
                        height = float(ann["height"])

                        x = max(0, min(1, x))
                        y = max(0, min(1, y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # 只有当值有效时才写入文件
                        if not (
                            math.isnan(x)
                            or math.isnan(y)
                            or math.isnan(width)
                            or math.isnan(height)
                        ):
                            f.write(f"{label_idx} {x} {y} {width} {height}\n")
                except (ValueError, TypeError) as e:
                    # 忽略无效的标注数据
                    print(
                        f"Skipping invalid annotation data for YOLO file: {ann}, error: {e}"
                    )
                    continue

        return {"success": True}
