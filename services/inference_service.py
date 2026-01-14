from concurrent.futures import FIRST_COMPLETED, as_completed, wait
import os
import threading

from inference_manager import InferenceManager
from app import RTSP_SERVER, create_app, executor, project_task_locks
from loguru import logger as log
from models import ReasoningMaterial, db


class InferenceService:
    def __init__(self):
        pass

    def process_material_task(
        self, inference_manager: InferenceManager, model, material
    ):
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

            material_obj.status = 1  # 推理中
            db.session.commit()

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
                    result = inference_manager.inference_video(model, video_path)
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

            material_obj = db.session.get(ReasoningMaterial, material_id)
            if material_obj.material_type != 3:
                material_obj.status = 2
            material_obj.result = result_obj
            db.session.commit()

            return result_obj

        except Exception as e:
            error = f"素材 {material_id} 推理异常：{str(e)}"
            db.session.rollback()

            material_obj = db.session.get(ReasoningMaterial, material_id)
            if material_obj:
                material_obj.status = 2
                material_obj.result = {"error": error, "success": False}
                db.session.commit()

            return {"material_id": material_id, "success": False, "error": error}

        finally:
            db.session.remove()

    def process_material_task_thread_safe(self, project_id, model_obj, mat):
        """
        每个线程创建独立模型

        :param self: 说明
        :param project_id: 说明
        :param labels: 说明
        :param model_obj: 说明
        :param mat: 说明
        """
        stop_event = project_task_locks[project_id]["stop_event"]
        if stop_event.is_set():
            log.info(f"项目 {project_id} 被停止，终止提交任务")
            return {"material_id": mat.id, "success": False, "error": "任务已被停止"}

        model_type = model_obj["model_type"]
        uploaded_model = model_obj["uploaded_model"]
        existing_model = model_obj["existing_model"]
        system_model = model_obj["system_model"]
        model_file = model_obj["model_file"]
        labels = model_obj["labels"]
        app = create_app()
        with app.app_context():
            inference_manager = InferenceManager(project_id, labels, mat)

            # 加载模型
            if model_type == "uploaded":
                model = inference_manager.load_model(
                    "uploaded", model_file=uploaded_model
                )
            else:
                model = inference_manager.load_model(
                    model_type, system_model, model_file, existing_model
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

    def background_inference_controller(self, project_id, materials, modeel_obj):
        """后台推理总控：处理所有素材、实时更新DB、完成后释放锁"""
        project_lock = threading.Lock()
        all_results = []

        priority_materials = []  # 图片+视频
        rtsp_materials = []  # RTSP

        for mat in materials:
            if mat.material_type in [1, 2]:
                priority_materials.append(mat)
            elif mat.material_type == 3:
                rtsp_materials.append(mat)

        try:

            futures = set()

            for mat in priority_materials + rtsp_materials:
                if project_task_locks[project_id]["stop_event"].is_set():
                    log.info(f"项目 {project_id} 被停止，终止提交任务")
                    break

                futures.add(
                    executor.submit(
                        self.process_material_task_thread_safe,
                        project_id,
                        modeel_obj,
                        mat,
                    )
                )

                if len(futures) >= executor._max_workers:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        all_results.append(f.result())
                        project_task_locks[project_id]["completed_materials"] += 1

            for f in as_completed(futures):
                all_results.append(f.result())
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


inference_service = InferenceService()
