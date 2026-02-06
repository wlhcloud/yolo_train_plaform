import os
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from flask import current_app
import tempfile
import shutil
from plot_engine import PlotEngine
from project_dir_manager import ProjectDirManager
from models import Label
from PIL import Image, ImageDraw, ImageFont
from app import font_path
from utils import start_ffmpeg_video_save
from managers.rtsp_threadpool_manager import rtsp_manager
from loguru import logger as log


class InferenceManager:
    """模型推理管理器，用于处理各种推理任务"""

    def __init__(self, project_id, labels: List[Label] = None, material=None):
        self.project_id = project_id
        self.project_model_dir = ProjectDirManager.get_project_model_dir(project_id)
        self.project_dataset_dir = ProjectDirManager.get_project_dataset_dir(project_id)

        self.labels_target_map = {}
        if labels is not None:
            self.labels_target_map = {
                label.name: label.target for label in labels if label.target
            }

        ProjectDirManager.ensure_project_model_dir(project_id)

        self.material_config = None
        self.material = material
        if material is not None:
            self.material_config = material.config
        self.region_label_map = {}
        self.region_stats = {}
        self.area_annotations = None
        # 初始化区域信息
        self.init_arealabel(self.material_config)

    def init_arealabel(self, material_config):
        if material_config is not None:
            self.region_label_map = {
                l["id"]: l["name"] for l in material_config["labels"]
            }

            for ann in material_config["annotations"]:
                region_name = self.region_label_map.get(ann["label_id"], "unknown")
                if region_name not in self.region_stats:
                    self.region_stats[region_name] = {"total": 0, "class_counts": {}}

            self.area_annotations = material_config["annotations"]

    def get_system_model_path(self, model_name):
        """
        获取系统训练模型的路径

        Args:
            model_name (str): 模型文件名 (如 'best.pt', 'last.pt')

        Returns:
            str: 模型文件的完整路径
        """
        train_results_dir = os.path.join(
            self.project_dataset_dir, "train_results", "weights"
        )
        model_path = os.path.join(train_results_dir, model_name)

        if os.path.exists(model_path):
            return model_path
        return None

    def save_uploaded_model(self, model_file):
        """
        保存上传的模型文件到uploaded_models目录

        Args:
            model_file (FileStorage): 上传的模型文件

        Returns:
            str: 保存的模型文件路径
        """
        if not model_file or not model_file.filename:
            raise ValueError("未提供模型文件")

        filename = model_file.filename
        # 确保文件名安全
        from werkzeug.utils import secure_filename

        filename = secure_filename(filename)

        # 确保文件扩展名为.pt
        if not filename.endswith(".pt"):
            filename += ".pt" if "." not in filename else ""

        # 使用uploaded_models目录，与RTSP功能统一
        uploaded_models_dir = os.path.join(
            "projects", str(self.project_id), "uploaded_models"
        )
        os.makedirs(uploaded_models_dir, exist_ok=True)

        model_path = os.path.join(uploaded_models_dir, filename)
        model_file.save(model_path)

        # 确保文件已正确保存
        if not os.path.exists(model_path):
            raise ValueError("模型文件保存失败")

        return model_path

    def get_existing_models(self):
        """
        获取项目模型目录中已存在的所有.pt模型文件

        Returns:
            list: 已存在的模型文件名列表
        """
        existing_models = []

        if os.path.exists(self.project_model_dir):
            for filename in os.listdir(self.project_model_dir):
                if filename.endswith(".pt") and os.path.isfile(
                    os.path.join(self.project_model_dir, filename)
                ):
                    existing_models.append(filename)

        return sorted(existing_models)

    def get_uploaded_models(self):
        """
        获取项目已上传模型目录中的所有.pt模型文件

        Returns:
            list: 已上传的模型文件名列表
        """
        uploaded_models = []
        uploaded_models_dir = os.path.join(
            "projects", str(self.project_id), "uploaded_models"
        )

        if os.path.exists(uploaded_models_dir):
            for filename in os.listdir(uploaded_models_dir):
                if filename.endswith(".pt") and os.path.isfile(
                    os.path.join(uploaded_models_dir, filename)
                ):
                    uploaded_models.append(filename)

        return sorted(uploaded_models)

    def load_model(
        self, model_type, system_model=None, model_file=None, existing_model=None
    ):
        """
        加载模型

        Args:
            model_type (str): 模型类型 ('system', 'upload', 'uploaded', 或 'existing')
            system_model (str): 系统模型名称
            model_file (FileStorage或str): 上传的模型文件或已上传的模型文件名
            existing_model (str): 已存在的模型文件名

        Returns:
            YOLO: 加载的模型对象

        Raises:
            ValueError: 当模型无法加载时抛出异常
        """
        model_path = None

        if model_type == "system":
            if not system_model:
                raise ValueError("未指定系统模型")

            model_path = self.get_system_model_path(system_model)
            if not model_path:
                raise ValueError(f"系统模型 {system_model} 不存在")
        elif model_type == "upload":
            if not model_file or not model_file.filename:
                raise ValueError("未提供上传的模型文件")

            model_path = self.save_uploaded_model(model_file)
        elif model_type == "uploaded":
            # 处理已上传的模型文件
            if not model_file:
                raise ValueError("未指定已上传的模型文件")

            uploaded_models_dir = os.path.join(
                "projects", str(self.project_id), "uploaded_models"
            )
            model_path = os.path.join(uploaded_models_dir, model_file)
            if not os.path.exists(model_path):
                raise ValueError(f"已上传模型 {model_file} 不存在")
        elif model_type == "existing":
            if not existing_model:
                raise ValueError("未指定已存在模型")

            model_path = os.path.join(self.project_model_dir, existing_model)
            if not os.path.exists(model_path):
                raise ValueError(f"已存在模型 {existing_model} 不存在")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        try:
            # 为了兼容PyTorch 2.6+，需要设置weights_only=False
            import torch

            # 临时设置torch.load的默认参数
            original_load = torch.load
            torch.load = lambda *args, **kwargs: (
                original_load(*args, **kwargs, weights_only=False)
                if "weights_only" not in kwargs
                else original_load(*args, **kwargs)
            )

            model = YOLO(model_path)

            # 恢复原始的torch.load函数
            torch.load = original_load

            return model
        except Exception as e:
            raise ValueError(f"模型加载失败: {str(e)}")

    def inference_image(self, model, image_file):
        temp_dir = tempfile.mkdtemp()
        try:
            # 保存上传的图片
            image_path = os.path.join(temp_dir, image_file.filename)
            image_file.save(image_path)
            plot_engine = PlotEngine(
                model=model,
                labels_target_map=self.labels_target_map,
                area_annotations=self.area_annotations,
                region_label_map=self.region_label_map,
                font_path=font_path,
            )
            # 推理
            result = model.predict(image_path)[0]

            # 读取原图
            image = cv2.imread(image_path)

            # 重置统计
            plot_engine.reset_stats()

            # 绘制 + 统计
            plotted_image = plot_engine.plot_frame(result, image)

            # 保存结果图
            filename = os.path.basename(image_path)
            result_filename = f"inference_result_{os.path.splitext(filename)[0]}.jpg"

            temp_path = os.path.join(temp_dir, result_filename)
            cv2.imwrite(temp_path, plotted_image)

            static_dir = os.path.join(
                current_app.root_path,
                "static",
                "inference_results",
                str(self.project_id),
            )
            os.makedirs(static_dir, exist_ok=True)

            final_path = os.path.join(static_dir, result_filename)
            shutil.copy2(temp_path, final_path)

            # 返回（直接用 PlotEngine 的统计）
            return {
                "type": "image",
                "image_url": f"/static/inference_results/{self.project_id}/{result_filename}",
                "class_counts": plot_engine.class_counts,
                "total_count": plot_engine.total_count,
                "region_stats": plot_engine.region_stats,
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def inference_video(self, model, video_file):
        log.info(f"开始推理视频----{self.material.id}:{video_path}")
        temp_dir = tempfile.mkdtemp()
        try:
            # 保存上传的视频
            video_filename = video_file.filename
            video_path = os.path.join(temp_dir, video_filename)
            video_file.save(video_path)

            plot_engine = PlotEngine(
                model=model,
                labels_target_map=self.labels_target_map,
                area_annotations=self.area_annotations,
                region_label_map=self.region_label_map,
                font_path=font_path,
            )
            cap = cv2.VideoCapture(video_path)
            video_filename = os.path.basename(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_filename = f"result_{os.path.splitext(video_filename)[0]}.mp4"
            output_path = os.path.join(temp_dir, output_filename)

            process = start_ffmpeg_video_save(
                width=width, height=height, fps=fps, output_path=output_path
            )

            # 统一重置统计
            plot_engine.reset_stats()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = model(frame, verbose=False)[0]
                    frame = plot_engine.plot_frame(result, frame)

                    process.stdin.write(frame.tobytes())

            finally:
                cap.release()
                process.stdin.close()
                process.wait()

            static_dir = os.path.join(
                current_app.root_path,
                "static",
                "inference_results",
                str(self.project_id),
            )
            os.makedirs(static_dir, exist_ok=True)

            final_path = os.path.join(static_dir, output_filename)
            shutil.copy2(output_path, final_path)

            return {
                "type": "video",
                "video_url": f"/static/inference_results/{self.project_id}/{output_filename}",
                "class_counts": plot_engine.class_counts,
                "total_count": plot_engine.total_count,
                "region_stats": plot_engine.region_stats,
            }
        except Exception as e:
            print(str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def check_system_models(self):
        """
        检查系统训练的模型是否存在

        Returns:
            dict: 包含best.pt和last.pt模型存在状态的字典
        """
        train_results_dir = os.path.join(
            self.project_dataset_dir, "train_results", "weights"
        )

        return {
            "best_model_exists": os.path.exists(
                os.path.join(train_results_dir, "best.pt")
            ),
            "last_model_exists": os.path.exists(
                os.path.join(train_results_dir, "last.pt")
            ),
        }

    def inference_rtsp(self, model, rtsp_url):
        temp_dir = tempfile.mkdtemp()

        plot_engine = PlotEngine(
            model=model,
            labels_target_map=self.labels_target_map,
            area_annotations=self.area_annotations,
            region_label_map=self.region_label_map,
            font_path=font_path,
        )

        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                return {
                    "type": "rtsp",
                    "error_message": "无法打开RTSP流",
                    "processed_frames": 0,
                    "fps": 0,
                }

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25

            # 读取第一帧确定尺寸
            ret, frame = cap.read()
            if not ret:
                return {
                    "type": "rtsp",
                    "error_message": "无法读取RTSP首帧",
                    "processed_frames": 0,
                    "fps": fps,
                }

            height, width = frame.shape[:2]

            output_filename = f"rtsp_inference_{self.project_id}.mp4"
            output_path = os.path.join(temp_dir, output_filename)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 统一重置统计
            plot_engine.reset_stats()

            max_frames = 300
            processed_frames = 0

            while processed_frames < max_frames:
                result = model(frame)[0]

                # 核心
                frame = plot_engine.plot_frame(result, frame)

                out.write(frame)
                processed_frames += 1

                ret, frame = cap.read()
                if not ret:
                    break

            cap.release()
            out.release()

            inference_results_dir = ProjectDirManager.get_project_inference_results_dir(
                self.project_id
            )
            final_path = os.path.join(inference_results_dir, output_filename)
            shutil.move(output_path, final_path)

            relative_path = ProjectDirManager.get_relative_path(final_path)
            video_url = "/" + ProjectDirManager.get_posix_path(relative_path)

            return {
                "type": "rtsp",
                "video_url": video_url,
                "processed_frames": processed_frames,
                "fps": fps,
                "class_counts": plot_engine.class_counts,
                "total_count": plot_engine.total_count,
                "region_stats": plot_engine.region_stats,
                "rtsp_url": rtsp_url,
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def inference_rtsp_realtime(self, model, rtsp_url):
        log.info(f"开始推理RSTP-----{self.material.filename}:{rtsp_url}")
        task_id = self.material.id
        result = rtsp_manager.add_rtsp_task(
            task_id=task_id,
            model=model,
            rtsp_url=rtsp_url,
            material=self.material,
            labels_target_map=self.labels_target_map,
            area_annotations=self.area_annotations,
            region_label_map=self.region_label_map,
        )
        # if result["status"] == "running":
        # status = rtsp_manager.get_task_status(task_id)
        return result
