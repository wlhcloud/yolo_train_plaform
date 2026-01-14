import os
import threading
import traceback
from datetime import datetime
from flask import current_app
from models import db, Project
from ultralytics import YOLO
import torch
import shutil


class TrainingService:
    """模型训练服务"""
    
    def __init__(self):
        # 全局变量用于存储训练状态和训练进程
        self.training_status = {}
        self.training_processes = {}
    
    def train_model(self, project_id, epochs=20, model_arch='yolov8n.pt', img_size=640, batch_size=16, use_gpu=True):
        """训练YOLO模型，包含详细的状态更新、日志记录和错误处理"""
        print(f"训练函数被调用，项目ID: {project_id}")
        
        try:
            print(f"开始训练项目 {project_id}")
            
            # 在线程中创建应用实例
            from app import create_app
            application = create_app()
            
            with application.app_context():
                print(f"进入应用上下文，项目ID: {project_id}")
                
                # 初始化训练状态和日志
                if project_id not in self.training_status:
                    self.training_status[project_id] = {
                        'status': 'preparing',
                        'message': '准备训练数据...',
                        'progress': 0,
                        'log': ''
                    }
                else:
                    self.training_status[project_id].update({
                        'status': 'preparing',
                        'message': '准备训练数据...',
                        'progress': 0,
                        'log': ''
                    })
                
                def update_log(message):
                    """更新训练日志"""
                    print(f"[训练日志 {project_id}] {message}")
                    self.training_status[project_id]['log'] += message + '\n'
                
                update_log(f"开始准备训练数据，项目ID: {project_id}")
                
                # 获取项目信息
                project = Project.query.get(project_id)
                if not project:
                    error_msg = "项目不存在"
                    update_log(error_msg)
                    raise Exception(error_msg)
                
                update_log(f"获取项目信息成功，项目名称: {project.name}")
                
                # 检查是否应该停止训练
                if self.training_status.get(project_id, {}).get('stop_requested'):
                    log_msg = '训练已停止'
                    self.training_status[project_id] = {
                        'status': 'stopped',
                        'message': log_msg,
                        'progress': 0,
                        'log': self.training_status[project_id].get('log', '') + log_msg + '\n'
                    }
                    update_log(log_msg)
                    return
                
                # 检查数据集目录是否存在
                project_dir = os.path.join(application.root_path, 'static/datasets', str(project_id))
                data_yaml_path = os.path.join(project_dir, 'data.yaml')
                
                update_log(f"项目目录: {project_dir}")
                update_log(f"数据配置文件路径: {data_yaml_path}")
                
                update_log("检查数据集配置文件...")
                
                from services.dataset_service import DatasetService
                dataset_service = DatasetService()
                
                if not os.path.exists(data_yaml_path):
                    # 如果数据集目录不存在，重新组织一次
                    log_msg = '数据集配置文件不存在，正在重新组织数据集...'
                    self.training_status[project_id].update({
                        'message': '正在组织数据集...',
                        'progress': 5
                    })
                    update_log(log_msg)
                    dataset_service.organize_dataset_directories(project_id)
                
                # 检查是否应该停止训练
                if self.training_status.get(project_id, {}).get('stop_requested'):
                    log_msg = '训练已停止'
                    self.training_status[project_id] = {
                        'status': 'stopped',
                        'message': log_msg,
                        'progress': 0,
                        'log': self.training_status[project_id].get('log', '') + log_msg + '\n'
                    }
                    update_log(log_msg)
                    return
                
                # 检查data.yaml文件是否存在
                if not os.path.exists(data_yaml_path):
                    error_msg = "数据集配置文件不存在"
                    update_log(error_msg)
                    raise Exception(error_msg)
                
                # 更新状态：开始加载模型
                self.training_status[project_id].update({
                    'message': '加载预训练模型...',
                    'progress': 10
                })
                update_log("加载预训练YOLOv8模型...")
                
                # 开始训练
                model_path = model_arch
                update_log(f"尝试加载预训练模型: {model_path}")
                model = YOLO(model_path)  # 加载预训练模型
                update_log(f"预训练模型加载成功! 模型路径: {model_path}")
                
                # 保存模型引用以便可能的停止操作
                self.training_processes[project_id] = model
                
                # 更新状态：开始训练
                self.training_status[project_id].update({
                    'status': 'training',
                    'message': '正在训练模型...',
                    'progress': 15
                })
                update_log(f"开始训练模型，共{epochs}个epochs...")
                
                # 训练模型
                update_log(f"开始训练模型，配置: 数据文件={data_yaml_path}, epochs={epochs}, 图像尺寸={img_size}x{img_size}, 批次大小={batch_size}")
                
                # 确定训练设备
                if use_gpu and torch.cuda.is_available():
                    device = 0  # 使用GPU
                    update_log("使用GPU进行训练")
                else:
                    device = 'cpu'  # 使用CPU
                    update_log("使用CPU进行训练")
                
                # 添加训练回调函数来监控进度
                def on_train_epoch_end(trainer):
                    """训练epoch结束时的回调函数"""
                    if project_id in self.training_status:
                        current_epoch = trainer.epoch + 1
                        total_epochs = trainer.epochs
                        progress = int((current_epoch / total_epochs) * 80) + 15  # 15-95%的进度范围
                        
                        # 检查是否应该停止训练
                        if self.training_status.get(project_id, {}).get('stop_requested'):
                            trainer.stop = True
                            return
                        
                        self.training_status[project_id].update({
                            'progress': progress,
                            'message': f'训练中... Epoch {current_epoch}/{total_epochs}',
                            'current_epoch': current_epoch,
                            'total_epochs': total_epochs
                        })
                        update_log(f"完成 Epoch {current_epoch}/{total_epochs}, 进度: {progress}%")
                
                # 使用add_callback方法添加回调函数
                model.add_callback('on_train_epoch_end', on_train_epoch_end)
                
                results = model.train(
                    data=data_yaml_path,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,  # 将batch_size改为batch
                    project=project_dir,
                    name='train_results',
                    exist_ok=True,  # 允许覆盖现有结果
                    device=device  # 正确设置设备
                )
                update_log("模型训练完成!")
                update_log(f"训练结果保存路径: {os.path.join(project_dir, 'train_results')}")
                
                # 检查是否应该停止训练
                if self.training_status.get(project_id, {}).get('stop_requested'):
                    log_msg = '训练已停止'
                    self.training_status[project_id] = {
                        'status': 'stopped',
                        'message': log_msg,
                        'progress': 0,
                        'log': self.training_status[project_id].get('log', '') + log_msg + '\n'
                    }
                    update_log(log_msg)
                    return
                
                # 更新训练状态 - 训练完成
                self.training_status[project_id].update({
                    'status': 'completed',
                    'message': '训练完成，正在保存结果...',
                    'progress': 90
                })
                update_log("训练完成，正在保存结果...")
                
                # 保存最佳模型
                best_model_path = os.path.join(project_dir, 'train_results', 'weights', 'best.pt')
                update_log(f"检查最佳模型文件是否存在: {best_model_path}")
                
                if os.path.exists(best_model_path):
                    update_log(f"找到最佳模型文件，开始复制到保存目录: {best_model_path}")
                    
                    # 将最佳模型复制到模型存储目录
                    model_save_dir = os.path.join(current_app.root_path, 'static/models', str(project_id), 'train', 'weights')
                    os.makedirs(model_save_dir, exist_ok=True)
                    shutil.copy(best_model_path, os.path.join(model_save_dir, 'best.pt'))
                    
                    update_log(f"模型文件已成功复制到保存目录: {model_save_dir}")
                    
                    # 更新项目信息
                    project.model_path = os.path.join('models', str(project_id), 'train', 'weights', 'best.pt')
                    project.last_trained = datetime.now()
                    db.session.commit()
                else:
                    error_msg = "未找到训练完成的最佳模型文件"
                    update_log(error_msg)
                    raise Exception(error_msg)
                
                # 更新训练状态 - 完成
                self.training_status[project_id].update({
                    'status': 'completed',
                    'message': '模型训练完成并已保存',
                    'progress': 100
                })
                update_log("模型训练完成并已保存")
                update_log(f"训练流程结束，最终状态: {self.training_status[project_id]}")
                
        except Exception as e:
            # 在异常处理中也需要使用application实例
            error_msg = f'训练出错: {str(e)}'
            print(error_msg)
            traceback.print_exc()
            
            try:
                # 创建新的应用实例
                from app import create_app
                application = create_app()
                with application.app_context():
                    # 更新训练状态 - 错误
                    log_msg = f'训练出错: {str(e)}'
                    self.training_status[project_id].update({
                        'status': 'error',
                        'message': log_msg,
                        'progress': 0,
                        'error_details': str(e),
                        'traceback': traceback.format_exc(),
                        'log': self.training_status[project_id].get('log', '') + log_msg + '\n' + traceback.format_exc()
                    })
                    update_log(log_msg)
                    update_log(traceback.format_exc())
                    update_log(f"错误发生时的项目ID: {project_id}")
                    
                    # 记录错误到项目信息
                    project = Project.query.get_or_404(project_id)
                    if project:
                        project.last_error = str(e)
                        db.session.commit()
            except Exception as inner_e:
                # 如果连应用上下文都获取不到，至少在内存中记录错误
                print(f'在异常处理中获取应用上下文失败: {str(inner_e)}')
                self.training_status[project_id].update({
                    'status': 'error',
                    'message': f'严重错误: {str(e)}',
                    'progress': 0,
                    'error_details': str(e),
                    'traceback': traceback.format_exc(),
                    'log': self.training_status[project_id].get('log', '') + f'严重错误: {str(e)}\n' + traceback.format_exc()
                })
        finally:
            # 清理训练进程引用
            if project_id in self.training_processes:
                del self.training_processes[project_id]
    
    def start_training(self, project_id, epochs=20, model_arch='yolov8n.pt', img_size=640, batch_size=16, use_gpu=True):
        """启动模型训练"""
        try:
            # 检查是否已有训练在进行
            if project_id in self.training_status and self.training_status[project_id]['status'] in ['preparing', 'training']:
                return {'success': False, 'message': '训练已在进行中'}
            
            # 重置训练状态
            self.training_status[project_id] = {
                'status': 'preparing',
                'message': '准备训练数据...',
                'progress': 0,
                'log': '',
                'stop_requested': False
            }
            
            # 在后台线程中启动训练
            training_thread = threading.Thread(
                target=self.train_model, 
                args=(project_id, epochs, model_arch, img_size, batch_size, use_gpu)
            )
            training_thread.daemon = True
            training_thread.start()
            
            return {'success': True, 'message': '训练已启动'}
        except Exception as e:
            return {'success': False, 'message': f'启动训练失败: {str(e)}'}
    
    def stop_training(self, project_id):
        """停止模型训练"""
        print(f"收到停止训练请求，项目ID: {project_id}")
        
        # 设置停止请求标志
        if project_id in self.training_status:
            self.training_status[project_id]['stop_requested'] = True
            self.training_status[project_id]['status'] = 'stopping'
            self.training_status[project_id]['message'] = '正在停止训练...'
            print("设置停止请求标志")
            
            # 尝试停止训练进程（如果可能）
            if project_id in self.training_processes:
                # 注意：ultralytics库没有直接的停止方法，我们只能设置标志位
                # 实际的停止将在训练的下一个检查点发生
                pass
            
            return {'success': True, 'message': '停止请求已发送'}
        else:
            print("没有找到训练状态")
            return {'success': False, 'message': '没有正在进行的训练'}
    
    def get_training_status(self, project_id):
        """获取训练状态"""
        print(f"收到训练状态查询请求，项目ID: {project_id}")
        status = self.training_status.get(project_id, {
            'status': 'idle',
            'message': '等待开始',
            'progress': 0
        })
        print(f"返回训练状态: {status}")
        return status