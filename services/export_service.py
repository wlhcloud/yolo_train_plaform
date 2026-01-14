import os
import shutil
from flask import current_app, url_for
from models import db, ExportRecord
from ultralytics import YOLO


class ExportService:
    """模型导出服务"""
    
    def export_model(self, project_id, format):
        """导出模型"""
        try:
            # 检查原始模型文件是否存在
            model_path = os.path.join(current_app.root_path, 'static/models', str(project_id), 'train', 'weights', 'best.pt')
            if not os.path.exists(model_path):
                return {'success': False, 'message': '模型文件不存在，请先训练模型'}
            
            # 确定导出目录和文件路径
            export_dir = os.path.join(current_app.root_path, 'static/models', str(project_id), 'export')
            os.makedirs(export_dir, exist_ok=True)
            
            export_filepath = ''
            source_filepath = ''  # 已存在的导出文件路径
            
            if format == 'onnx':
                export_filepath = os.path.join(export_dir, 'model.onnx')
                # 检查是否已存在导出的ONNX文件
                existing_onnx = os.path.join(current_app.root_path, 'static/models', str(project_id), 'train', 'weights', 'best.onnx')
                if os.path.exists(existing_onnx):
                    source_filepath = existing_onnx
            elif format == 'torchscript':
                export_filepath = os.path.join(export_dir, 'model.torchscript')
            else:
                return {'success': False, 'message': f'不支持的导出格式: {format}'}
            
            # 如果有已存在的导出文件，直接复制
            if source_filepath and os.path.exists(source_filepath):
                shutil.copy2(source_filepath, export_filepath)
            # 否则重新导出
            elif not os.path.exists(export_filepath):
                # 加载模型
                model = YOLO(model_path)
                
                # 使用正确的参数进行导出
                if format == 'onnx':
                    # 对于ONNX导出，使用正确的参数
                    model.export(format='onnx', project=export_dir, name='model')
                elif format == 'torchscript':
                    # 对于TorchScript导出，使用正确的参数
                    model.export(format='torchscript', project=export_dir, name='model')
            
            # 保存导出记录到数据库
            export_record = ExportRecord(
                project_id=project_id,
                format=format,
                path=export_filepath
            )
            db.session.add(export_record)
            db.session.commit()
            
            # 生成相对于static目录的路径，用于下载
            relative_export_path = os.path.relpath(export_filepath, current_app.root_path)
            
            return {
                'success': True, 
                'message': '导出成功', 
                'path': export_filepath,
                'download_url': url_for('main.download_file', filename=relative_export_path)
            }
        except Exception as e:
            import traceback
            error_msg = f'导出失败: {str(e)}'
            print(f"导出错误详情: {error_msg}")
            traceback.print_exc()
            return {'success': False, 'message': error_msg}