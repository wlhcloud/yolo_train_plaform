import os
import subprocess
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

    def _try_get_engine_path(self, model_path):
        """
        尝试获取对应模型的 TensorRT engine 路径。
        - 仅在 CUDA 可用、且原模型是 .pt 时尝试。
        - 若 engine 不存在则现场 export（首次较慢，之后秒加载）。
        - 任何失败都安全回退 None（让调用方继续用 .pt）。
        - **A100 + torch 2.12.0+cu130** 必须有 tensorrt-cu13 才能导出。
        """
        try:
            if not model_path or not model_path.endswith(".pt"):
                return None
            import os
            import torch
            if not torch.cuda.is_available():
                log.info("未检测到 CUDA，跳过 TensorRT 导出")
                return None

            # 提前探测 tensorrt 是否可用（避免 export 时 127 退出码）
            try:
                import tensorrt  # noqa: F401
            except Exception as e:
                log.warning(f"未安装 tensorrt（{e}），跳过 TensorRT 导出，直接使用 .pt")
                return None

            engine_path = model_path[:-3] + ".engine"
            if os.path.exists(engine_path):
                # 检查 engine 是否比 .pt 新（pt 重新训练过则重新导出）
                if os.path.getmtime(engine_path) >= os.path.getmtime(model_path):
                    return engine_path
                log.info(f"engine 过期（pt 较新），将重新导出: {engine_path}")

            log.info(f"🔧 首次加载，正在导出 TensorRT engine（耗时约 30-90 秒）: {engine_path}")
            # 临时用 ulWrap 加载 .pt 触发 export
            from ultralytics import YOLO
            export_model = YOLO(model_path)
            # 显式跳过 onnx 中间步骤（不引入 onnx 依赖），直接 TensorRT
            exported_path = export_model.export(
                format="engine",
                half=True,
                imgsz=640,
                device=0,
                simplify=False,
                verbose=False,
            )
            # ultralytics 会把 engine 写在 .pt 同目录，文件名加 _engine 后缀
            if exported_path and os.path.exists(exported_path):
                if os.path.abspath(exported_path) != os.path.abspath(engine_path):
                    import shutil
                    shutil.copy2(exported_path, engine_path)
                    log.info(f"engine 已重命名: {engine_path}")
                log.info(f"✅ TensorRT engine 导出完成: {engine_path}")
                return engine_path
            return None
        except Exception as e:
            log.warning(f"TensorRT engine 导出失败，回退到 .pt: {e}")
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

    def _auto_batch_size(self, model):
        """根据可用显存自动选择 batch_size（适应多任务共卡场景）

        经验值：YOLOv8 640 fp16 下，每张图片推理 + NMS 中间 feature map 约占用：
          - 640x640x3 uint8 输入：1.2MB
          - 中间 feature map（多尺度）：约 50-200MB（模型越大越多）
          - 临时 NMS 缓冲区：10-50MB
        因此每 batch 大致 200MB-500MB 不等。
        """
        import torch
        try:
            if not torch.cuda.is_available():
                log.warning("CUDA 不可用，batch_size 退回 16")
                return 16

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            used_gb = total_gb - free_gb
            log.info(
                f"GPU 显存：总计 {total_gb:.2f}GB，已用 {used_gb:.2f}GB，"
                f"空闲 {free_gb:.2f}GB"
            )

            # 留 1GB buffer 给推理过程中的临时分配
            usable_gb = max(0.0, free_gb - 1.0)

            # 估算每张图的显存占用
            try:
                param_size_gb = sum(
                    p.numel() * p.element_size() for p in model.model.parameters()
                ) / 1024**3
            except Exception:
                param_size_gb = 0.05  # 默认 YOLOv8n 量级

            # 经验公式：单 batch ≈ 0.15GB + 模型参数 2 倍的临时开销
            per_batch_gb = max(0.2, 0.15 + param_size_gb * 2.0)

            batch = int(usable_gb / per_batch_gb)
            # 限制范围 [8, 256]
            batch = max(8, min(batch, 1024))

            log.info(
                f"📊 自动 batch_size={batch}（可用 {usable_gb:.2f}GB / "
                f"每 batch 约 {per_batch_gb:.2f}GB）"
            )
            return batch
        except Exception as e:
            log.warning(f"自动 batch_size 估算失败，使用默认 32: {e}")
            return 32

    def _get_video_fps(self, video_path):
        """使用 ffprobe 获取视频真实帧率，避免 OpenCV 读取 MTS 时出错"""
        try:
            import subprocess
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1', video_path
            ]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10).decode('utf-8').strip()
            for line in output.splitlines():
                if 'r_frame_rate=' in line:
                    rate_str = line.split('=')[1].strip()
                    num, denom = map(int, rate_str.split('/'))
                    fps = num / denom
                    log.info(f"ffprobe 获取帧率: {fps}")
                    return fps
        except Exception as e:
            log.warning(f"ffprobe 获取帧率失败: {e}")
        return None

    def _get_video_fps_from_probe(self, video_path):
        """用 ffprobe 解析 avg_frame_rate，失败时返回 25"""
        try:
            out = subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate',
                '-of', 'default=noprint_wrappers=1', video_path
            ], stderr=subprocess.STDOUT, timeout=10).decode().strip()
            for line in out.splitlines():
                if 'avg_frame_rate=' in line:
                    s = line.split('=')[1].strip()
                    if '/' in s:
                        n, d = map(int, s.split('/'))
                        if d > 0:
                            return n / d
                    else:
                        return float(s)
        except Exception:
            pass
        return None

    def _get_video_resolution(self, video_path):
        """用 ffprobe 获取视频宽高"""
        try:
            out = subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'default=noprint_wrappers=1', video_path
            ], stderr=subprocess.STDOUT, timeout=10).decode().strip()
            w, h = 0, 0
            for line in out.splitlines():
                if line.startswith('width='):
                    w = int(line.split('=')[1].strip())
                elif line.startswith('height='):
                    h = int(line.split('=')[1].strip())
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return 0, 0

    def load_model(
        self,
        model_type=None,
        system_model=None,
        model_file=None,
        existing_model=None,
        model_path=None,
    ):
        """
        加载模型

        Args:
            model_type (str): 模型类型 ('system', 'upload', 'uploaded', 或 'existing')
            system_model (str): 系统模型名称
            model_file (FileStorage或str): 上传的模型文件或已上传的模型文件名
            existing_model (str): 已存在的模型文件名
            model_path(str)：已经上传的模型路径

        Returns:
            YOLO: 加载的模型对象

        Raises:
            ValueError: 当模型无法加载时抛出异常
        """
        if not model_path:
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

            # 检测GPU可用性并选择合适的设备
            if torch.cuda.is_available():
                device = 'cuda'
                log.info(f"GPU可用，使用CUDA设备: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                log.warning("GPU不可用，使用CPU进行推理（速度较慢）")

            # 优先加载 TensorRT engine（仅 .pt 模型适用；engine 已是 GPU 优化版）
            engine_path = self._try_get_engine_path(model_path)
            if engine_path is not None:
                log.info(f"🚀 使用 TensorRT engine: {engine_path}")
                # TensorRT engine 不能再次 .half()/.to(device)（内部已固定）
                model = YOLO(engine_path)
            else:
                model = YOLO(model_path).to(device)
                log.info(f"模型已加载到设备: {device}")
                try:
                    import torch
                    param_device = next(model.model.parameters()).device
                    log.info(f"模型参数实际所在设备: {param_device}")
                    if device == 'cuda':
                        model = model.half()
                        log.info("模型已切换为 FP16 半精度")
                except Exception as e:
                    log.warning(f"无法获取模型参数设备或切换半精度: {e}")
            # 恢复原始的torch.load函数
            torch.load = original_load

            # 初始化 predictor（强制创建 GPU predictor，避免后续为 None）
            try:
                import numpy as np
                _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                log.info(f"predictor 已初始化，设备: {getattr(model.predictor, 'device', 'unknown')}")
            except Exception as e:
                log.warning(f"模型预热失败: {e}")

            return model
        except Exception as e:
            raise ValueError(f"模型加载失败: {str(e)}")

    def inference_image(self, model, image_input):
        temp_dir = tempfile.mkdtemp()
        try:
            # 判断传入的是文件对象还是路径字符串
            if isinstance(image_input, str):
                # 如果是路径字符串，直接使用
                image_path = image_input
                log.info(f"图片推理 - 使用路径: {image_path}")
            else:
                # 如果是文件对象，保存到临时目录
                image_path = os.path.join(temp_dir, image_input.filename)
                image_input.save(image_path)
                log.info(f"图片推理 - 保存上传文件到: {image_path}")
            
            # 验证图片文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            # 验证文件大小
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError(f"图片文件为空: {image_path}")
            log.info(f"图片文件大小: {file_size} bytes")

            plot_engine = PlotEngine(
                model=model,
                labels_target_map=self.labels_target_map,
                area_annotations=self.area_annotations,
                region_label_map=self.region_label_map,
                font_path=font_path,
            )
            
            # 推理
            log.info(f"开始模型推理: {image_path}")
            result = model.predict(image_path)[0]
            log.info(f"推理完成，检测到 {len(result.boxes)} 个目标")

            # 读取原图
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"无法读取图片文件: {image_path}")
            log.info(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")

            # 重置统计
            plot_engine.reset_stats()

            # 绘制 + 统计
            log.info("开始绘制结果")
            plotted_image = plot_engine.plot_frame(result, image)

            # 保存结果图
            filename = os.path.basename(image_path)
            result_filename = f"inference_result_{os.path.splitext(filename)[0]}.jpg"

            temp_path = os.path.join(temp_dir, result_filename)
            cv2.imwrite(temp_path, plotted_image)
            log.info(f"临时结果保存到: {temp_path}")

            static_dir = os.path.join(
                current_app.root_path,
                "static",
                "inference_results",
                str(self.project_id),
            )
            os.makedirs(static_dir, exist_ok=True)

            final_path = os.path.join(static_dir, result_filename)
            shutil.copy2(temp_path, final_path)
            log.info(f"最终结果保存到: {final_path}")

            # 将 detections 中的 numpy 类型转换为 Python 原生类型，以便 JSON 序列化
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # 返回（直接用 PlotEngine 的统计）
            return {
                "type": "image",
                "image_url": f"/static/inference_results/{self.project_id}/{result_filename}",
                "class_counts": convert_numpy_types(plot_engine.class_counts),
                "total_count": int(plot_engine.total_count),
                "region_stats": convert_numpy_types(plot_engine.region_stats),
                "detections": convert_numpy_types(plot_engine.detections),
            }

        except Exception as e:
            log.error(f"图片推理失败: {str(e)}", exc_info=True)
            raise

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_video_batch(self, model, plot_engine, batch, batch_idx, encode_process, draw_executor, preprocess_executor, enable_draw=True):
        """处理一个视频 batch：预处理、推理、跟踪、绘制、写入编码器"""
        import time
        import torch

        log.info(f"批量推理第 {batch_idx} 批: {len(batch)} 帧")
        if batch_idx == 0 or batch_idx % 10 == 0:
            try:
                log.info(f"推理前模型设备: {next(model.model.parameters()).device}")
                log.info(f"推理前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            except Exception as e:
                log.warning(f"无法获取GPU信息: {e}")

        # 手动预处理：多线程并行 resize
        t_pre_start = time.time()

        def preprocess_one(frame):
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        preprocessed = list(preprocess_executor.map(preprocess_one, batch))
        t_pre_end = time.time()

        # 转为 FP16 GPU tensor 并推理
        t_infer_start = time.time()
        device = next(model.model.parameters()).device

        # 关键优化：先把 uint8 张量搬运到 GPU，再在 GPU 上做 permute/normalize/half
        # 避免在 CPU 上做 stack + float32 转换（256 帧 1.2MB*256=300MB+），大幅减少 CPU 端工作和 H2D 传输量
        with torch.no_grad():
            # 使用 pin_memory 加速 H2D 传输，async overlap CPU/GPU
            batch_tensor = torch.stack([
                torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
                for img in preprocessed
            ])
            if device.type == "cuda":
                batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True).half()
            else:
                batch_tensor = batch_tensor.to(device=device, dtype=torch.float32)
            batch_tensor.div_(255.0)

            results = model.predict(batch_tensor, verbose=False)
        t_infer_end = time.time()

        if batch_idx == 0 or batch_idx % 10 == 0:
            try:
                log.info(f"results类型: {type(results)}, len: {len(results)}, 第一个结果类型: {type(results[0]) if hasattr(results, '__getitem__') else 'N/A'}")
                box_device = results[0].boxes.data.device if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0 else 'no boxes'
                log.info(f"预处理耗时: {t_pre_end - t_pre_start:.3f} 秒, 推理+NMS耗时: {t_infer_end - t_infer_start:.3f} 秒, 结果数量: {len(results)}, 结果设备: {box_device}, 推理后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            except Exception as e:
                log.warning(f"无法获取GPU信息: {e}")

        # 关键优化：把"每帧串行跟踪 + 多线程绘制"改成"批量跟踪 + 批量绘制"
        # 旧方案：256 帧 × (track+draw 串行) = 大量 Python 循环 + GIL 切换
        # 新方案：
        #   1) 一次性把所有 frames 的 boxes/conf/cls 拿到 CPU（1 次 D2H 而非 256 次）
        #   2) 串行 ByteTrack 更新（必须保持时序）
        #   3) 用 multiprocessing 或 GPU 端绘制
        t_track_start = time.time()

        # 一次性把所有 results 的关键数据搬到 CPU（合并 D2H，减少同步）
        # boxes_data: (B, N, 6)  其中 N 是该帧最大检测数
        all_boxes = []
        all_orig_imgs = []
        for result in results:
            if len(result.boxes) > 0:
                all_boxes.append(result.boxes.data)
            else:
                all_boxes.append(None)
            all_orig_imgs.append(result.orig_img)

        # 串行跟踪（必须保持帧之间连续性）
        track_results_list = []
        for i, result in enumerate(results):
            track_results = plot_engine.track_frame(result)
            track_results_list.append(track_results)

        t_track_end = time.time()

        # 批量绘制：使用线程池并行处理，但每个任务只处理少量帧
        # 避免 PIL 中文渲染的 GIL 问题：把绘制改成 cv2 + 缓存字体
        t_draw_start = time.time()
        if enable_draw:
            futures = []
            for i, track_results in enumerate(track_results_list):
                future = draw_executor.submit(plot_engine.draw_frame, batch[i], track_results)
                futures.append(future)

            # 按顺序写入视频帧
            try:
                for future in futures:
                    processed_frame = future.result()
                    encode_process.stdin.write(processed_frame.tobytes())
            except BrokenPipeError:
                log.error("写入编码器时管道断裂")
                raise RuntimeError("ffmpeg 编码器管道断裂")
        else:
            # 跳过绘制：直接将原始帧写入编码器（保留 track 统计，但不做 cv2 绘制）
            try:
                for frame in batch:
                    encode_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                log.error("写入编码器时管道断裂")
                raise RuntimeError("ffmpeg 编码器管道断裂")

        t_draw_end = time.time()

        # 详细耗时分析：跟踪 vs 绘制（找出瓶颈）
        if batch_idx == 0 or batch_idx % 5 == 0:
            try:
                total = (t_pre_end - t_pre_start) + (t_infer_end - t_infer_start) + \
                        (t_track_end - t_track_start) + (t_draw_end - t_draw_start)
                log.info(
                    f"⏱️ batch{batch_idx} 总:{total:.3f}s | "
                    f"预处理:{t_pre_end - t_pre_start:.3f}s | "
                    f"推理:{t_infer_end - t_infer_start:.3f}s | "
                    f"跟踪:{t_track_end - t_track_start:.3f}s | "
                    f"绘制:{t_draw_end - t_draw_start:.3f}s"
                )
            except Exception:
                pass

    def convert_llm_detections(self, model, detections, iamge_file):
        """
        将模型输出的检测结果 [[x1, y1, x2, y2, conf, cls]] 转换为指定格式的字典列表
        :param model: 模型
        :param detections: 模型原始检测结果，格式如 [[383.16, 16.6, 721.28, 447.42, 0.854, 0.0]]
        :param iamge_file: 识别图片
        :return:
        """

        # 2. 初始化结果列表
        converted_results = []
        img_width, img_height = iamge_file.image_size

        # 3. 遍历检测结果（兼容单条/多条检测框）
        # 先处理外层列表，确保每一行是一个检测框
        for det in detections:
            if isinstance(det, list) and len(det) >= 6:
                x1, y1, x2, y2, conf, cls_id = det[:6]

                # 4. 坐标归一化（转换为0-1之间的相对值）
                # 计算中心点x/y（归一化）
                x = (x1 + x2) / 2 / img_width  # 中心点x坐标归一化
                y = (y1 + y2) / 2 / img_height  # 中心点y坐标归一化
                # 计算宽度/高度（归一化）
                width = (x2 - x1) / img_width  # 检测框宽度归一化
                height = (y2 - y1) / img_height  # 检测框高度归一化

                # 5. 获取中英文标签（默认未知类别）
                Label_name = model.names[int(cls_id)]

                # 7. 构造中文标签的检测项（和英文项坐标/置信度完全一致）
                item = {
                    "label": Label_name,
                    "confidence": round(float(conf), 6),
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "width": round(width, 6),
                    "height": round(height, 6),
                }

                # 8. 添加到结果列表
                converted_results.append(item)

        return converted_results

    def inference_video(self, model, video_input, enable_draw=True):
        import time
        start_time = time.time()
        log.info(f"开始推理视频----")
        temp_dir = tempfile.mkdtemp()
        
        frame_count = 0  # 记录处理的帧数
        
        try:
            # 判断传入的是文件对象还是路径字符串
            if isinstance(video_input, str):
                # 如果是路径字符串，直接使用
                video_path = video_input
                video_filename = os.path.basename(video_path)
            else:
                # 如果是文件对象，保存到临时目录
                video_filename = video_input.filename
                video_path = os.path.join(temp_dir, video_filename)
                video_input.save(video_path)

            plot_engine = PlotEngine(
                model=model,
                labels_target_map=self.labels_target_map,
                area_annotations=self.area_annotations,
                region_label_map=self.region_label_map,
                font_path=font_path,
            )

            # 优先用 ffmpeg NVDEC GPU 硬解（避免 OpenCV 软解码吃 CPU）
            # 探测视频编码格式，选择合适的 cuvid 解码器
            cap = None
            decode_process = None
            bytes_per_frame = None
            use_ffmpeg_decode = False

            def _probe_video_codec(path):
                try:
                    out = subprocess.check_output([
                        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=codec_name',
                        '-of', 'default=noprint_wrappers=1', path
                    ], stderr=subprocess.STDOUT, timeout=10).decode().strip()
                    for line in out.splitlines():
                        if 'codec_name=' in line:
                            return line.split('=')[1].strip()
                except Exception:
                    return None
                return None

            codec = _probe_video_codec(video_path)
            # codec → cuvid 解码器映射
            codec_map = {
                'h264': 'h264_cuvid',
                'hevc': 'hevc_cuvid',
                'h265': 'hevc_cuvid',
                'mpeg2video': 'mpeg2_cuvid',
                'mpeg4': 'mpeg4_cuvid',
                'vp8': 'vp8_cuvid',
                'vp9': 'vp9_cuvid',
            }
            cuvid_decoder = codec_map.get(codec)

            if cuvid_decoder:
                try:
                    ffmpeg_decode_cmd = [
                        'ffmpeg', '-y', '-err_detect', 'ignore_err',
                        '-c:v', cuvid_decoder,
                        '-i', video_path,
                        '-f', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-'
                    ]
                    decode_process = subprocess.Popen(
                        ffmpeg_decode_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    import time as _t
                    _t.sleep(1)
                    if decode_process.poll() is None:
                        use_ffmpeg_decode = True
                        log.info(f"✅ ffmpeg NVDEC 解码器（{cuvid_decoder}）启动成功")
                    else:
                        decode_process = None
                        log.warning(f"ffmpeg NVDEC {cuvid_decoder} 启动失败，回退 OpenCV")
                except Exception as e:
                    log.warning(f"ffmpeg NVDEC 启动异常: {e}")
                    decode_process = None

            if not use_ffmpeg_decode:
                # 回退到 OpenCV（多线程软解码）
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'threads;8'
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_N_THREADS, 8)
                log.info("视频解码：使用 OpenCV 软解码（多线程）")

            # 优先用 ffprobe 获取真实帧率，避免 OpenCV 读取 MTS 时 fps 错误导致输出时长异常
            fps = self._get_video_fps(video_path)
            if fps is None:
                # 用 ffprobe 兜底获取 fps
                fps = self._get_video_fps_from_probe(video_path) or 25
            if use_ffmpeg_decode:
                # ffmpeg 路径：从原始视频获取宽高（用 ffprobe）
                w, h = self._get_video_resolution(video_path)
                width, height = w, h
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"视频参数: {width}x{height}, {fps}fps")
            bytes_per_frame = width * height * 3

            output_filename = f"result_{os.path.splitext(video_filename)[0]}.mp4"
            output_path = os.path.join(temp_dir, output_filename)

            process = start_ffmpeg_video_save(
                width=width, height=height, fps=fps, output_path=output_path
            )

            # 统一重置统计
            plot_engine.reset_stats()

            # 批量推理参数 - 根据可用显存自适应
            batch_size = self._auto_batch_size(model)
            frame_buffer = []
            
            # 使用多线程并行处理绘制和预处理
            from concurrent.futures import ThreadPoolExecutor
            import queue
            import threading
            draw_executor = ThreadPoolExecutor(max_workers=8)
            preprocess_executor = ThreadPoolExecutor(max_workers=8)
            
            # 简单流水线：主线程读帧，工作线程处理批次
            batch_queue = queue.Queue(maxsize=2)
            done_event = threading.Event()
            error_queue = queue.Queue()

            def batch_worker():
                try:
                    batch_idx = 0
                    while True:
                        batch = batch_queue.get()
                        if batch is None:
                            break
                        self._process_video_batch(model, plot_engine, batch, batch_idx, process, draw_executor, preprocess_executor, enable_draw)
                        batch_idx += 1
                except Exception as e:
                    import traceback
                    log.error(f"批次处理异常: {e}\n{traceback.format_exc()}")
                    error_queue.put(e)
                finally:
                    done_event.set()

            worker_thread = threading.Thread(target=batch_worker, daemon=True)
            worker_thread.start()

            def enqueue_batch(batch):
                while True:
                    if not error_queue.empty():
                        raise error_queue.get()
                    if done_event.is_set():
                        raise RuntimeError("视频推理工作线程已退出")
                    try:
                        batch_queue.put(batch, timeout=0.5)
                        return
                    except queue.Full:
                        continue

            try:
                if use_ffmpeg_decode:
                    # ffmpeg NVDEC 路径：从 stdout 读 raw frame
                    while True:
                        if process.poll() is not None:
                            log.error(f"ffmpeg 编码器意外退出，退出码: {process.returncode}")
                            raise RuntimeError(f"ffmpeg 编码器意外退出，退出码: {process.returncode}")
                        raw = decode_process.stdout.read(bytes_per_frame)
                        if not raw or len(raw) < bytes_per_frame:
                            break
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                        frame_buffer.append(frame)
                        if len(frame_buffer) >= batch_size:
                            frame_count += len(frame_buffer)
                            enqueue_batch(frame_buffer)
                            frame_buffer = []
                else:
                    # OpenCV 软解码路径
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if process.poll() is not None:
                            log.error(f"ffmpeg 编码器意外退出，退出码: {process.returncode}")
                            raise RuntimeError(f"ffmpeg 编码器意外退出，退出码: {process.returncode}")
                        frame_buffer.append(frame)
                        if len(frame_buffer) >= batch_size:
                            frame_count += batch_size
                            enqueue_batch(frame_buffer)
                            frame_buffer = []

                # 处理剩余的帧
                if frame_buffer:
                    frame_count += len(frame_buffer)
                    enqueue_batch(frame_buffer)
                    frame_buffer = []

                # 发送结束信号并等待工作线程
                enqueue_batch(None)
                worker_thread.join()
                if not error_queue.empty():
                    raise error_queue.get()
                if not done_event.is_set():
                    log.warning("工作线程未正常结束")

            finally:
                draw_executor.shutdown(wait=True)
                preprocess_executor.shutdown(wait=True)
                if cap is not None:
                    cap.release()
                if decode_process is not None:
                    try:
                        decode_process.stdout.close()
                    except Exception:
                        pass
                    try:
                        decode_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        decode_process.kill()
                        decode_process.wait()
                try:
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
                log.info(f"ffmpeg 进程结束，退出码: {process.returncode}")

            static_dir = os.path.join(
                current_app.root_path,
                "static",
                "inference_results",
                str(self.project_id),
            )
            os.makedirs(static_dir, exist_ok=True)
            final_path = os.path.join(static_dir, output_filename)
            log.info(f"视频推理完成，正在保存到{final_path}")
            shutil.copy2(output_path, final_path)
            log.info(f"视频保存完成，输出路径：{final_path}")
            
            # 计算总用时
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_frame = total_time / frame_count if frame_count > 0 else 0
            
            log.info(f"视频推理统计:")
            log.info(f"  - 总帧数: {frame_count}")
            log.info(f"  - 总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
            log.info(f"  - 平均每帧用时: {avg_time_per_frame:.4f} 秒")
            log.info(f"  - 推理速度: {frame_count/total_time:.2f} FPS")

            # 计算每条跟踪轨迹的方向（由远到近）
            plot_engine.finalize_track_summary()

            return {
                "type": "video",
                "video_url": f"/static/inference_results/{self.project_id}/{output_filename}",
                "class_counts": plot_engine.class_counts,
                "total_count": plot_engine.total_count,
                "region_stats": plot_engine.region_stats,
                "track_summary": plot_engine.track_summary,
                "inference_stats": {
                    "total_frames": frame_count,
                    "total_time_seconds": round(total_time, 2),
                    "total_time_minutes": round(total_time / 60, 2),
                    "avg_time_per_frame": round(avg_time_per_frame, 4),
                    "inference_fps": round(frame_count / total_time, 2) if total_time > 0 else 0,
                },
            }
        except Exception as e:
            log.error(f"视频推理失败: {str(e)}", exc_info=True)
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def inference_video_nvdec(self, model, video_input, enable_draw=True):
        """
        使用 NVDEC GPU 硬件解码进行视频推理（极速版）
        
        Args:
            model: YOLO 模型
            video_input: 视频文件路径或文件对象
            
        Returns:
            dict: 推理结果
        """
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，使用兼容 CPU 的视频推理流程")
        import time
        start_time = time.time()
        log.info(f"开始推理视频（NVDEC GPU 解码）----")
        temp_dir = tempfile.mkdtemp()
        
        frame_count = 0
        decode_time = 0
        
        try:
            # 判断传入的是文件对象还是路径字符串
            if isinstance(video_input, str):
                video_path = video_input
                video_filename = os.path.basename(video_path)
            else:
                video_filename = video_input.filename
                video_path = os.path.join(temp_dir, video_filename)
                video_input.save(video_path)

            # 使用 OpenCV 获取视频信息
            fps = self._get_video_fps(video_path)
            cap_info = cv2.VideoCapture(video_path)
            if fps is None:
                fps = cap_info.get(cv2.CAP_PROP_FPS)
            width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_info.release()
            
            log.info(f"视频信息: {width}x{height}, {fps}fps, 共 {total_frames} 帧")

            plot_engine = PlotEngine(
                model=model,
                labels_target_map=self.labels_target_map,
                area_annotations=self.area_annotations,
                region_label_map=self.region_label_map,
                font_path=font_path,
            )
            plot_engine.reset_stats()

            # 构建 ffmpeg NVDEC 解码命令
            # MTS 文件通常是 MPEG-TS 容器，需要根据视频编码选择解码器
            # 先尝试 h264_cuvid（最常见），失败则尝试 mpeg2_cuvid
            
            # 方案1：使用 h264_cuvid 解码器（适用于 H.264 编码的 MTS）
            ffmpeg_decode_cmd = [
                'ffmpeg', '-y',
                '-c:v', 'h264_cuvid',        # 显式指定 NVDEC H.264 解码器
                '-i', video_path,             # 输入文件
                '-f', 'rawvideo',             # 输出原始视频
                '-pix_fmt', 'bgr24',          # OpenCV 使用的 BGR 格式
                '-'
            ]
            
            log.info("启动 ffmpeg NVDEC GPU 解码器（h264_cuvid）...")
            log.info(f"ffmpeg 命令: {' '.join(ffmpeg_decode_cmd)}")
            decode_process = subprocess.Popen(
                ffmpeg_decode_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL  # 忽略 stderr，避免缓冲区满导致 ffmpeg 阻塞
            )
            
            # 检查 ffmpeg 进程是否正常启动
            import time
            time.sleep(2)  # 等待进程启动
            if decode_process.poll() is not None:
                stderr = decode_process.stderr.read().decode('utf-8', errors='ignore')
                log.warning(f"ffmpeg h264_cuvid 解码器启动失败")
                log.warning(f"错误信息: {stderr[:300]}")
                
                # 方案2：尝试 mpeg2_cuvid 解码器（适用于 MPEG-2 编码的 MTS）
                log.info("尝试使用 mpeg2_cuvid 解码器...")
                ffmpeg_decode_cmd_mpeg2 = [
                    'ffmpeg', '-y',
                    '-c:v', 'mpeg2_cuvid',     # NVDEC MPEG-2 解码器
                    '-i', video_path,
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-'
                ]
                log.info(f"ffmpeg 命令: {' '.join(ffmpeg_decode_cmd_mpeg2)}")
                decode_process = subprocess.Popen(
                    ffmpeg_decode_cmd_mpeg2,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                time.sleep(2)
                if decode_process.poll() is not None:
                    stderr = decode_process.stderr.read().decode('utf-8', errors='ignore')
                    log.warning(f"ffmpeg mpeg2_cuvid 解码器启动失败")
                    log.warning(f"错误信息: {stderr[:300]}")
                    log.warning("将回退到 OpenCV 解码 + GPU 推理方法")
                    raise RuntimeError(f"ffmpeg NVDEC 不支持此视频编码，请使用 OpenCV 方法")
            
            log.info("ffmpeg NVDEC 解码器启动成功，开始读取帧...")

            # 创建视频编码器
            output_filename = f"result_{os.path.splitext(video_filename)[0]}.mp4"
            output_path = os.path.join(temp_dir, output_filename)
            
            encode_process = start_ffmpeg_video_save(
                width=width, height=height, fps=fps, output_path=output_path
            )

            # 流水线并行参数 - 根据可用显存自适应
            batch_size = self._auto_batch_size(model)
            bytes_per_frame = width * height * 3
            
            # 多线程池
            from concurrent.futures import ThreadPoolExecutor
            import queue
            import threading
            
            # 流水线队列
            decode_queue = queue.Queue(maxsize=3)  # 解码后的批次
            
            # 线程池
            pipeline_executor = ThreadPoolExecutor(max_workers=3)
            draw_executor = ThreadPoolExecutor(max_workers=8)
            preprocess_executor = ThreadPoolExecutor(max_workers=8)
            
            def decode_worker():
                """解码线程：阻塞读取帧，稳定可靠"""
                log.info("解码线程已启动")
                batch_idx = 0
                frame_buffer = []
                nonlocal frame_count

                while True:
                    raw = decode_process.stdout.read(bytes_per_frame)
                    if not raw or len(raw) < bytes_per_frame:
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    frame_buffer.append(frame.copy())
                    frame_count += 1

                    if frame_count % 500 == 0:
                        log.info(f"NVDEC 已解码 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")

                    if len(frame_buffer) >= batch_size:
                        batch_idx += 1
                        try:
                            decode_queue.put((batch_idx, frame_buffer), timeout=10)
                        except queue.Full:
                            log.warning("decode_queue 已满，解码线程退出")
                            return
                        frame_buffer = []

                # 处理剩余帧
                if frame_buffer:
                    batch_idx += 1
                    decode_queue.put((batch_idx, frame_buffer))

                # 发送终止信号
                decode_queue.put(None)
                log.info(f"解码线程结束，共解码 {frame_count} 帧")
            
            def infer_worker():
                """推理线程：批量推理 + 跟踪 + 绘制 + 写入"""
                while True:
                    try:
                        item = decode_queue.get(timeout=30)
                    except queue.Empty:
                        log.error("NVDEC 30 秒无输出，可能卡住，强制终止解码进程")
                        try:
                            decode_process.kill()
                        except Exception:
                            pass
                        raise RuntimeError("NVDEC 解码超时")
                    if item is None:
                        return
                    batch_idx, frame_buffer = item
                    self._process_video_batch(model, plot_engine, frame_buffer, batch_idx, encode_process, draw_executor, preprocess_executor, enable_draw)
            
            # 启动流水线
            try:
                decode_future = pipeline_executor.submit(decode_worker)
                infer_future = pipeline_executor.submit(infer_worker)
                
                # 等待所有线程完成
                decode_future.result()
                infer_future.result()
                
            finally:
                pipeline_executor.shutdown(wait=True)
                draw_executor.shutdown(wait=True)
                preprocess_executor.shutdown(wait=True)
                
                # 关闭 stdout 并终止解码进程
                try:
                    decode_process.stdout.close()
                except Exception:
                    pass
                
                # 等待进程退出（最多10秒）
                try:
                    decode_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning("ffmpeg 解码器超时，强制终止")
                    decode_process.kill()
                    decode_process.wait()
                
                log.info(f"ffmpeg NVDEC 解码结束, 解码退出码: {decode_process.returncode}")
                
                # 关闭编码进程
                try:
                    if encode_process.stdin and not encode_process.stdin.closed:
                        encode_process.stdin.close()
                    encode_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        encode_process.kill()
                        encode_process.wait()
                    except Exception:
                        pass
                except Exception:
                    try:
                        encode_process.kill()
                    except Exception:
                        pass
                
                log.info(f"ffmpeg 编码结束, 编码退出码: {encode_process.returncode}")

            # 保存结果
            static_dir = os.path.join(
                current_app.root_path,
                "static",
                "inference_results",
                str(self.project_id),
            )
            os.makedirs(static_dir, exist_ok=True)
            final_path = os.path.join(static_dir, output_filename)
            log.info(f"视频推理完成，正在保存到 {final_path}")
            shutil.copy2(output_path, final_path)
            log.info(f"视频保存完成，输出路径：{final_path}")
            
            # 计算统计
            end_time = time.time()
            total_time = end_time - start_time
            inference_fps = frame_count / total_time if total_time > 0 else 0
            
            log.info(f"NVDEC GPU 解码视频推理统计:")
            log.info(f"  - 总帧数: {frame_count}")
            log.info(f"  - 总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
            log.info(f"  - 推理速度: {inference_fps:.2f} FPS")
            log.info(f"  - 检测目标数: {plot_engine.total_count}")

            # 计算每条跟踪轨迹的方向（由远到近）
            plot_engine.finalize_track_summary()

            return {
                "type": "video",
                "video_url": f"/static/inference_results/{self.project_id}/{output_filename}",
                "class_counts": plot_engine.class_counts,
                "total_count": plot_engine.total_count,
                "region_stats": plot_engine.region_stats,
                "track_summary": plot_engine.track_summary,
                "inference_stats": {
                    "total_frames": frame_count,
                    "total_time_seconds": round(total_time, 2),
                    "total_time_minutes": round(total_time / 60, 2),
                    "inference_fps": round(inference_fps, 2),
                    "decode_method": "NVDEC GPU",
                },
            }
            
        except Exception as e:
            log.error(f"NVDEC 视频推理失败: {str(e)}", exc_info=True)
            raise
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

            # 计算每条跟踪轨迹的方向（由远到近）
            plot_engine.finalize_track_summary()

            return {
                "type": "rtsp",
                "video_url": video_url,
                "processed_frames": processed_frames,
                "fps": fps,
                "class_counts": plot_engine.class_counts,
                "total_count": plot_engine.total_count,
                "region_stats": plot_engine.region_stats,
                "track_summary": plot_engine.track_summary,
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
