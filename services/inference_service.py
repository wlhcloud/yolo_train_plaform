from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import threading

from inference_manager import InferenceManager
from app import RTSP_SERVER, create_app, executor, project_task_locks
from loguru import logger as log
from models import ReasoningMaterial, db


class InferenceService:
    def __init__(self):
        # 模型缓存：{model_key: (model, inference_manager)}
        self._model_cache = {}
        self._cache_lock = threading.Lock()

    def _get_model_key(self, model_obj):
        """生成模型唯一标识"""
        model_type = model_obj["model_type"]
        if model_type == "uploaded":
            return f"uploaded_{model_obj['uploaded_model']}"
        elif model_type == "existing":
            return f"existing_{model_obj['existing_model']}"
        elif model_type == "system":
            return f"system_{model_obj['system_model']}"
        return None

    def _get_or_load_model(self, project_id, model_obj, labels, material):
        """获取或加载模型（带缓存）"""
        model_key = self._get_model_key(model_obj)
        
        with self._cache_lock:
            if model_key in self._model_cache:
                log.info(f"复用已加载模型: {model_key}")
                model, base_manager = self._model_cache[model_key]
                # 为当前素材创建新的 InferenceManager（共享模型）
                new_manager = InferenceManager(project_id, labels, material)
                return model, new_manager
        
        # 首次加载模型
        log.info(f"首次加载模型: {model_key}")
        app = create_app()
        with app.app_context():
            inference_manager = InferenceManager(project_id, labels, material)
            model_type = model_obj["model_type"]
            uploaded_model = model_obj["uploaded_model"]
            existing_model = model_obj["existing_model"]
            system_model = model_obj["system_model"]
            model_file = model_obj["model_file"]
            
            if model_type == "uploaded":
                model = inference_manager.load_model("uploaded", model_file=uploaded_model)
            else:
                model = inference_manager.load_model(model_type, system_model, model_file, existing_model)

            with self._cache_lock:
                # 双检：并发场景下可能别的线程已抢先加载完同一个模型，直接复用
                if model_key in self._model_cache:
                    log.info(f"并发加载冲突，复用已加载模型: {model_key}")
                    model, base_manager = self._model_cache[model_key]
                    new_manager = InferenceManager(project_id, labels, material)
                    return model, new_manager
                self._model_cache[model_key] = (model, inference_manager)

            return model, inference_manager

    def process_material_task(self, inference_manager: InferenceManager, model, material):
        """处理单个素材的推理任务"""
        material_id = material.id
        material_type = material.material_type  # 1=图片，2=视频，3=RTSP
        path = os.path.join("static", material.path)
        success = False
        result = None
        error = None

        try:
            material_obj = db.session.get(ReasoningMaterial, material_id)
            if not material_obj:
                error = f"素材 {material_id} 不存在"
                raise ValueError(error)

            # 更新状态：推理中
            material_obj.status = 1
            material_obj.start_time = datetime.now()
            db.session.commit()
            log.info(f"素材 {material_id} 开始推理，状态更新为推理中")

            if material_type == 1:
                image_path = path
                if not image_path:
                    error = f"素材 {material_id}：图片文件路径为空"
                elif not os.path.exists(image_path):
                    error = f"素材 {material_id}：图片文件不存在（路径：{image_path}）"
                else:
                    result = inference_manager.inference_image(model, image_path)
                    success = True

            elif material_type == 2:  # 视频素材
                video_path = path
                if not video_path:
                    error = f"素材 {material_id}：视频文件路径为空"
                elif not os.path.exists(video_path):
                    error = f"素材 {material_id}：视频文件不存在（路径：{video_path}）"
                else:
                    # 是否绘制检测框：默认关闭（跳过绘制可提速约 1 倍）
                    # 需要绘制时设置环境变量 ENABLE_VIDEO_DRAW=true
                    enable_draw = os.environ.get(
                        "ENABLE_VIDEO_DRAW", "false"
                    ).lower() in ("true", "1", "yes", "on")

                    # 优先尝试 NVDEC GPU 硬件解码，失败则回退到 OpenCV 软解码
                    try:
                        result = inference_manager.inference_video_nvdec(model, video_path, enable_draw=enable_draw)
                        log.info(f"NVDEC 方法推理视频完成，素材ID：{material_id}，enable_draw={enable_draw}")
                    except Exception as e:
                        log.warning(f"NVDEC 推理失败，回退到 OpenCV 软解码: {e}")
                        result = inference_manager.inference_video(model, video_path, enable_draw=enable_draw)
                        log.info(f"OpenCV 软解码推理视频完成，素材ID：{material_id}，enable_draw={enable_draw}")
                    success = True

            elif material_type == 3:  # RTSP素材
                rtsp_addr = material.path
                if not rtsp_addr:
                    error = f"素材 {material_id}：RTSP地址为空"
                elif not rtsp_addr.startswith("rtsp://"):
                    error = f"素材 {material_id}：RTSP地址格式错误"
                else:
                    result = inference_manager.inference_rtsp_realtime(model, rtsp_addr)
                    success = True

            else:
                error = f"素材 {material_id}：不支持的素材类型 {material_type}"

            result_obj = {
                "material_id": material_id,
                "material_type": material_type,
                "success": success,
                "result": result if result else None,
                "error": error,
            }

            # 更新状态：已推理
            material_obj = db.session.get(ReasoningMaterial, material_id)
            if material_obj.material_type != 3:
                material_obj.status = 2
            material_obj.end_time = datetime.now()
            material_obj.result = result_obj
            db.session.commit()
            log.info(f"素材 {material_id} 推理完成，状态更新为已推理")
            return result_obj

        except Exception as e:
            error = f"素材 {material_id} 推理异常：{str(e)}"
            log.error(error)
            db.session.rollback()

            material_obj = db.session.get(ReasoningMaterial, material_id)
            if material_obj:
                material_obj.status = 0  # 失败后回到未开始状态
                material_obj.end_time = datetime.now()
                material_obj.result = {"error": error, "success": False}
                db.session.commit()
                log.info(f"素材 {material_id} 推理失败，状态重置为未开始")

            return {"material_id": material_id, "success": False, "error": error}

        finally:
            db.session.remove()

    def process_material_task_thread_safe(self, project_id, model_obj, mat):
        """
        线程安全的素材处理（共享模型实例）
        """
        stop_event = project_task_locks[project_id]["stop_event"]
        if stop_event.is_set():
            log.info(f"项目 {project_id} 被停止，终止提交任务")
            return {"material_id": mat.id, "success": False, "error": "任务已被停止"}

        labels = model_obj["labels"]
        app = create_app()
        
        with app.app_context():
            try:
                # 获取或加载模型（带缓存）
                model, inference_manager = self._get_or_load_model(
                    project_id, model_obj, labels, mat
                )
                
                if stop_event.is_set():
                    log.info(f"项目 {project_id} 被停止，终止提交任务")
                    return {
                        "material_id": mat.id,
                        "success": False,
                        "error": "任务已被停止",
                    }
                
                result = self.process_material_task(inference_manager, model, mat)
                return result
                
            except Exception as e:
                log.error(f"素材 {mat.id} 处理异常：{str(e)}")
                return {"material_id": mat.id, "success": False, "error": str(e)}

    def background_inference_controller(self, project_id, materials, modeel_obj):
        """后台推理总控：并行处理所有素材、实时更新DB、完成后释放锁"""
        project_lock = threading.Lock()

        priority_materials = []  # 图片+视频
        rtsp_materials = []  # RTSP

        for mat in materials:
            if mat.material_type in [1, 2]:
                priority_materials.append(mat)
            elif mat.material_type == 3:
                rtsp_materials.append(mat)

        # 并行度可配置：环境变量 MAX_PARALLEL_MATERIALS，默认 3
        # 注意：并发路数不是越大越好——GPU 算力是瓶颈，2-4 路收益最大，超过 5 路基本无收益
        try:
            max_parallel = int(os.environ.get("MAX_PARALLEL_MATERIALS", "3"))
        except ValueError:
            max_parallel = 3
        max_parallel = max(1, min(max_parallel, 20))

        all_materials = priority_materials + rtsp_materials
        log.info(
            f"项目 {project_id} 开始处理素材，共 {len(all_materials)} 个，"
            f"并行度 {max_parallel}（{len(priority_materials)}个图片/视频 + {len(rtsp_materials)}个RTSP）"
        )
        try:
            # 注意：不能复用全局 executor（其唯一 worker 正在跑本总控，会死锁）
            # 这里新建专用线程池，用完即销毁
            from concurrent.futures import ThreadPoolExecutor, as_completed

            all_results = []
            completed_count = 0
            with ThreadPoolExecutor(
                max_workers=max_parallel, thread_name_prefix=f"mat-{project_id}"
            ) as mat_executor:
                future_map = {}
                for mat in all_materials:
                    if project_id not in project_task_locks:
                        break
                    if project_task_locks[project_id]["stop_event"].is_set():
                        log.info(f"项目 {project_id} 被停止，终止提交任务")
                        break
                    future = mat_executor.submit(
                        self.process_material_task_thread_safe, project_id, modeel_obj, mat
                    )
                    future_map[future] = mat.id

                for future in as_completed(future_map):
                    mat_id = future_map[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        log.error(f"素材 {mat_id} 提交执行失败: {e}")
                        result = {"material_id": mat_id, "success": False, "error": str(e)}
                    all_results.append(result)

                    # 更新进度（加锁，多线程并发写）
                    if project_id in project_task_locks:
                        with project_task_locks[project_id]["counter_lock"]:
                            project_task_locks[project_id]["completed_materials"] += 1

            log.info(
                f"项目 {project_id} 所有素材推理完成（选干ID：{project_id}），总计{len(all_results)}个"
            )
            with project_lock:
                if project_id in project_task_locks:
                    project_task_locks[project_id]["is_running"] = False
                    del project_task_locks[project_id]

        except Exception as e:
            log.error(f"项目 {project_id} 后台推理总控异常：{str(e)}")
            with project_lock:
                if project_id in project_task_locks:
                    project_task_locks[project_id]["is_running"] = False
                    del project_task_locks[project_id]
        finally:
            # 清理模型缓存
            with self._cache_lock:
                self._model_cache.clear()
                log.info(f"项目 {project_id} 模型缓存已清理")


inference_service = InferenceService()
