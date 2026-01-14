import os
import random
import shutil
import yaml
from flask import current_app
from models import db, Project, Image, Label


class DatasetService:
    """数据集管理服务"""
    
    def organize_dataset_directories(self, project_id):
        """组织数据集目录结构，将图片和标注文件放到对应目录中"""
        with current_app.app_context():
            project = Project.query.get(project_id)
            if not project:
                return
            
            # 创建数据集目录结构
            project_dir = os.path.join(current_app.root_path, 'static/datasets', str(project_id))
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)
            
            os.makedirs(project_dir, exist_ok=True)
            
            # 创建images和labels子目录
            for dataset_type in ['train', 'val', 'test']:
                img_dir = os.path.join(project_dir, 'images', dataset_type)
                label_dir = os.path.join(project_dir, 'labels', dataset_type)
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(label_dir, exist_ok=True)
            
            # 获取所有已分配的图片
            assigned_images = Image.query.filter(
                Image.project_id == project_id,
                Image.dataset_type.in_(['train', 'val', 'test'])
            ).all()
            
            # 复制图片和标注文件到对应目录
            for img in assigned_images:
                # 源文件路径
                src_img_path = os.path.join(current_app.root_path, 'static', img.path)
                src_label_path = os.path.splitext(src_img_path)[0] + '.txt'
                
                # 目标文件路径
                dst_img_path = os.path.join(project_dir, 'images', img.dataset_type, img.filename)
                dst_label_path = os.path.join(project_dir, 'labels', img.dataset_type, os.path.splitext(img.filename)[0] + '.txt')
                
                # 复制图片文件
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                
                # 复制标注文件
                if os.path.exists(src_label_path):
                    shutil.copy(src_label_path, dst_label_path)
            
            # 创建data.yaml文件
            labels = Label.query.filter_by(project_id=project_id).all()
            names = [label.name for label in labels]
            
            data_yaml = {
                'path': project_dir,
                'train': os.path.join('images', 'train'),
                'val': os.path.join('images', 'val'),
                'test': os.path.join('images', 'test'),
                'nc': len(names),
                'names': names
            }
            
            with open(os.path.join(project_dir, 'data.yaml'), 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    def update_dataset(self, project_id, image_ids, dataset_type):
        """更新数据集类型"""
        # 更新数据集类型
        Image.query.filter(Image.id.in_(image_ids)).update(
            {Image.dataset_type: dataset_type}, 
            synchronize_session=False
        )
        db.session.commit()
        
        # 重新组织数据集目录结构
        self.organize_dataset_directories(project_id)
        
        return {'success': True}
    
    def auto_assign_dataset(self, project_id):
        """自动分配数据集类型"""
        # 获取所有已标注的图片
        annotated_images = Image.query.filter(
            Image.project_id == project_id,
            Image.annotations.any()
        ).all()
        
        # 如果没有已标注的图片，返回错误
        if not annotated_images:
            return {
                'success': False,
                'message': '没有找到已标注的图片'
            }, 400
        
        # 打乱图片顺序以确保随机性
        random.shuffle(annotated_images)
        
        # 按照最佳实践比例划分数据集: 70% 训练集, 20% 验证集, 10% 测试集
        total_count = len(annotated_images)
        train_count = int(total_count * 0.7)
        val_count = int(total_count * 0.2)
        # 剩余的为测试集
        
        # 分配数据集类型
        for i, image in enumerate(annotated_images):
            if i < train_count:
                image.dataset_type = 'train'
            elif i < train_count + val_count:
                image.dataset_type = 'val'
            else:
                image.dataset_type = 'test'
        
        # 提交更改到数据库
        db.session.commit()
        
        # 立即组织数据集目录结构
        self.organize_dataset_directories(project_id)
        
        # 返回划分结果
        return {
            'success': True,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': total_count - train_count - val_count
        }