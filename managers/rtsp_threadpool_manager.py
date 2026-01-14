import cv2
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional, List

from models import ReasoningMaterial, db
from plot_engine import PlotEngine
from utils import start_ffmpeg_rtsp_push
from app import RTSP_SERVER, MEDIA_SERVER, create_app, font_path
from loguru import logger as log
from services.reasoning_material_service import reasoning_material_service as rm_service

# 线程池配置（根据CPU/内存调整）
MAX_WORKERS = 8  # 线程池最大线程数（建议：CPU核心数*2，或根据RTSP路数定）
FRAME_QUEUE_SIZE = 10  # 每路RTSP的帧队列大小（避免内存溢出）
RECONNECT_INTERVAL = 3  # RTSP断连重试间隔（秒）
DB_UPDATE_INTERVAL = 5  # 数据库同步频率


class RTSPTask:
    """单路RTSP流的处理任务（生产者+消费者）"""

    def __init__(
        self,
        task_id: str,
        model,
        rtsp_url: str,
        material,
        labels_target_map,
        area_annotations,
        region_label_map,
    ):
        self.task_id = task_id
        self.model = model
        self.rtsp_url = rtsp_url
        self.material = material

        self.labels_target_map = labels_target_map
        self.area_annotations = area_annotations
        self.region_label_map = region_label_map

        self.stop_flag = threading.Event()
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.status = "idle"
        self.error_msg = ""
        self.fps = 25
        self.class_counts = {}
        self.total_count = 0
        self.region_stats = {}

    def _producer(self):
        """生产者：读取RTSP帧"""
        cap = None
        self.status = "running"

        while not self.stop_flag.is_set():
            # 初始化/重连视频流
            if cap is None or not cap.isOpened():
                self.status = "reconnecting"
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    self.error_msg = f"无法连接RTSP：{self.rtsp_url}"
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                # 获取流参数
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 25

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                self.error_msg = f"RTSP帧读取失败：{self.rtsp_url}"
                cap.release()
                cap = None
                time.sleep(RECONNECT_INTERVAL)
                continue

            # 队列满时丢弃旧帧（保证实时性）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        # 资源释放
        if cap:
            cap.release()
        self.status = "stopped"

    def _consumer(self):
        """消费者：推理+绘图+推流"""
        plot_engine = None
        ffmpeg = None
        push_url = f"{RTSP_SERVER}{self.material.id}"

        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                height, width = frame.shape[:2]

                if plot_engine is None:
                    plot_engine = PlotEngine(
                        model=self.model,
                        labels_target_map=self.labels_target_map,
                        area_annotations=self.area_annotations,
                        region_label_map=self.region_label_map,
                        font_path=font_path,
                    )
                if ffmpeg is None:
                    ffmpeg = start_ffmpeg_rtsp_push(width, height, self.fps, push_url)

                # 推理+绘图
                result = self.model(frame, verbose=False)[0]
                frame = plot_engine.plot_frame(result, frame)

                # 推流
                ffmpeg.stdin.write(frame.tobytes())

                # 更新统计
                self.class_counts = plot_engine.class_counts
                self.total_count = plot_engine.total_count
                self.region_stats = plot_engine.region_stats
                self.error_msg = ""

            except queue.Empty:
                continue
            except Exception as e:
                self.error_msg = f"消费任务异常：{str(e)}"
                time.sleep(0.5)

        # 资源释放
        if ffmpeg:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        self.status = "stopped"

    def run(self):
        """启动当前RTSP任务的生产者和消费者"""
        # 重置状态
        self.stop_flag.clear()
        self.error_msg = ""

        # 启动生产者和消费者
        producer = threading.Thread(target=self._producer, daemon=True)
        consumer = threading.Thread(target=self._consumer, daemon=True)
        producer.start()
        consumer.start()

        while not self.stop_flag.is_set():
            time.sleep(0.1)

        producer.join(timeout=2)
        consumer.join(timeout=2)

    def stop(self):
        """停止当前RTSP任务"""
        self.stop_flag.set()
        self.status = "stopping"


class RTSPThreadPoolManager:
    """RTSP流的线程池管理器"""

    def __init__(self, app=None):
        self.app = None
        self.executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS, thread_name_prefix="rtsp_worker"
        )
        # 管理所有任务：key=task_id, value=(Future, RTSPTask)
        self.tasks: Dict[str, (Future, RTSPTask)] = {}
        self.lock = threading.RLock()  # 线程安全的任务字典锁

        # 定时更新数据库相关属性
        self.db_update_stop_flag = threading.Event()
        self.db_update_thread = threading.Thread(
            target=self._update_db_periodically,
            daemon=True,
            name="db_update_worker",
        )
        # 启动定时更新线程
        self.db_update_thread.start()

    def _update_db_periodically(self):
        """定时更新数据库：每5秒执行一次"""
        while not self.db_update_stop_flag.is_set():
            try:
                # 等待5秒
                self.db_update_stop_flag.wait(DB_UPDATE_INTERVAL)
                if self.db_update_stop_flag.is_set():
                    break

                if self.app is None:
                    continue

                # 加锁遍历所有任务，保证线程安全
                with self.app.app_context():
                    with self.lock:
                        if not self.tasks:
                            continue
                        log.info(f"需要同步RTSP结果的任务数：{len(self.tasks)}")
                        for task_id, (future, task) in self.tasks.items():
                            try:
                                if task.status not in ["running", "reconnecting"]:
                                    continue
                                task_status = self.get_task_status(task_id=task_id)

                                rm_service.update_material(
                                    task.material.id, task_status, 1
                                )
                            except Exception as e:
                                db.session.rollback()
                                log.error(
                                    f"更新素材 {task.material.id} 数据库失败：{str(e)}",
                                    exc_info=True,
                                )

            except Exception as e:
                log.error(f"定时更新数据库主循环异常：{str(e)}", exc_info=True)
                time.sleep(1)

    def add_rtsp_task(
        self,
        task_id: str,
        model,
        rtsp_url: str,
        material,
        labels_target_map,
        area_annotations,
        region_label_map,
    ) -> Dict:
        with self.lock:
            log.info(f"添加RTSP推理任务：{material.filename}-{material.path}")
            task_id = str(task_id)
            if task_id in self.tasks:
                return {"error": f"任务{task_id}已存在", "status": "failed"}
            task = RTSPTask(
                task_id,
                model,
                rtsp_url,
                material,
                labels_target_map,
                area_annotations,
                region_label_map,
            )
            future = self.executor.submit(task.run)
            self.tasks[task_id] = (future, task)
            mu38_url = f"{MEDIA_SERVER}/{material.id}/hls.m3u8"
            return {
                "type": "rtsp",
                "status": task.status,
                "error_msg": task.error_msg,
                "fps": task.fps,
                "task_id": task_id,
                "mu38_url": mu38_url,
                "class_counts": task.class_counts,
                "total_count": task.total_count,
                "region_stats": task.region_stats,
            }

    def stop_rtsp_task(self, task_id: str) -> Dict:
        """停止指定RTSP任务"""
        with self.lock:
            task_id = str(task_id)
            if task_id not in self.tasks:
                return {"error": f"任务{task_id}不存在", "status": "failed"}

            future, task = self.tasks[task_id]
            result = self.get_task_status(task_id=task_id)
            try:
                # 停止之前更新数据
                rm_service.update_material(task.material.id, result, 2)
                task.stop()
                future.result(timeout=3)
            except TimeoutError:
                pass

            del self.tasks[task_id]
            return {"task_id": task_id, "status": "stopped"}

    def stop_all_tasks(self) -> Dict:
        """停止所有RTSP任务"""
        with self.lock:
            stopped_tasks = []
            failed_tasks = []

            for task_id, (future, task) in self.tasks.items():
                try:
                    result = self.get_task_status(task_id=task_id)

                    # 停止后更新数据库状态
                    rm_service.update_material(task.material.id, result, 2)

                    task.stop()
                    future.result(timeout=3)
                    stopped_tasks.append(task_id)
                except Exception:
                    failed_tasks.append(task_id)

            self.tasks.clear()
            return {
                "stopped_tasks": stopped_tasks,
                "failed_tasks": failed_tasks,
                "total_stopped": len(stopped_tasks),
            }

    def get_task_status(self, task_id: Optional[str] = None) -> Dict:
        """获取任务状态（单任务/所有任务）"""
        with self.lock:
            if task_id:
                if task_id not in self.tasks:
                    return {"error": f"任务{task_id}不存在"}
                future, task = self.tasks[task_id]
                mu38_url = f"{MEDIA_SERVER}/{task_id}/hls.m3u8"
                return {
                    "task_id": task_id,
                    "status": task.status,
                    "error_msg": task.error_msg,
                    "fps": task.fps,
                    "mu38_url": mu38_url,
                    "class_counts": task.class_counts,
                    "total_count": task.total_count,
                    "region_stats": task.region_stats,
                    "is_alive": not future.done(),
                }
            else:
                # 返回所有任务状态
                all_status = {}
                for task_id, (future, task) in self.tasks.items():
                    all_status[task_id] = {
                        "status": task.status,
                        "error_msg": task.error_msg,
                        "fps": task.fps,
                        "is_alive": not future.done(),
                    }
                return {"total_tasks": len(self.tasks), "tasks": all_status}

    def set_app(self, app):
        self.app = app

    def shutdown(self):
        """关闭线程池"""
        self.db_update_stop_flag.set()
        self.db_update_thread.join(timeout=3)  # 等待定时线程退出

        self.stop_all_tasks()
        self.executor.shutdown(wait=True, cancel_futures=True)
        print("RTSP线程池已关闭")


rtsp_manager = RTSPThreadPoolManager()
