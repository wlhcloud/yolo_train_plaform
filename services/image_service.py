import os
import zipfile
import posixpath
from datetime import datetime
from flask import current_app
from PIL import Image as PILImage
from models import db, Image


class ImageService:
    """图片服务，处理图片上传和管理相关逻辑"""
    
    def upload_zip_images(self, project_id, zip_file):
        """上传并解压ZIP文件中的图片"""
        # 保存ZIP文件
        zip_path = os.path.join('static/uploads', f"{project_id}_{zip_file.filename}")
        zip_file.save(zip_path)
        
        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extract_path = os.path.join('static/uploads', str(project_id))
            os.makedirs(extract_path, exist_ok=True)
            zip_ref.extractall(extract_path)
        
        # 删除ZIP文件
        os.remove(zip_path)
        
        # 遍历解压后的文件并添加到数据库
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, 'static')
                    # 使用posixpath处理URL路径，确保在Windows上也使用正斜杠
                    relative_path = posixpath.join(*relative_path.split(os.sep))
                    img = PILImage.open(file_path)
                    width, height = img.size
                    
                    image = Image(
                        filename=file,
                        original_filename=file,
                        path=relative_path,
                        project_id=project_id,
                        width=width,
                        height=height
                    )
                    db.session.add(image)
        
        db.session.commit()
        return {'success': True, 'message': 'ZIP文件上传并解压成功'}
    
    @staticmethod
    def delete_images(image_ids):
        """
        批量删除图片
        
        Args:
            image_ids: 要删除的图片ID列表
            
        Returns:
            tuple: (成功删除的数量, 错误信息列表)
        """
        deleted_count = 0
        errors = []
        
        for image_id in image_ids:
            try:
                image = Image.query.get(image_id)
                if not image:
                    errors.append(f"图片ID {image_id} 不存在")
                    continue
                
                # 获取图片文件路径
                image_path = os.path.join(current_app.root_path, image.path)
                
                # 删除数据库记录
                db.session.delete(image)
                
                # 删除图片文件
                if os.path.exists(image_path):
                    os.remove(image_path)
                    
                # 删除对应的YOLO格式标注文件（.txt文件）
                txt_file_path = os.path.splitext(image_path)[0] + '.txt'
                if os.path.exists(txt_file_path):
                    os.remove(txt_file_path)
                    
                deleted_count += 1
            except Exception as e:
                errors.append(f"删除图片ID {image_id} 时出错: {str(e)}")
        
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            errors.append(f"数据库提交失败: {str(e)}")
            deleted_count = 0
            
        return deleted_count, errors
    
    @staticmethod
    def delete_unannotated_images(project_id):
        """
        删除项目中所有未标注的图片
        
        Args:
            project_id: 项目ID
            
        Returns:
            tuple: (成功删除的数量, 错误信息列表)
        """
        # 查询所有未标注的图片
        unannotated_images = Image.query.filter(
            Image.project_id == project_id,
            ~Image.annotations.any()  # 没有关联的标注
        ).all()
        
        image_ids = [image.id for image in unannotated_images]
        return ImageService.delete_images(image_ids)