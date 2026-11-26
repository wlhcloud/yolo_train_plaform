"""Process-local road compute lanes with a separate filesystem-scan pool."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext


def integer_setting(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 至 {maximum} 之间")
    return value


def compute_workers():
    return road_workers() * camera_workers()


def road_workers():
    return integer_setting("ROAD_SURVEY_WORKERS", 1, 1, 8)


def camera_workers():
    return integer_setting("ROAD_SURVEY_CAMERA_WORKERS", 1, 1, 8)


def cuda_lane():
    import torch

    if torch.cuda.is_available():
        return torch.cuda.stream(torch.cuda.Stream())
    return nullcontext()


class RoadSurveyScheduler:
    def __init__(
        self,
        workers=None,
        scan_workers=2,
        lane_factory=cuda_lane,
        roads=None,
        cameras=None,
    ):
        self.road_workers = road_workers() if roads is None else roads
        self.camera_workers = camera_workers() if cameras is None else cameras
        self.workers = (
            self.road_workers * self.camera_workers if workers is None else workers
        )
        self.lane_factory = lane_factory
        self.compute = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="road-gpu"
        )
        self.scans = ThreadPoolExecutor(
            max_workers=scan_workers, thread_name_prefix="road-scan"
        )
        self.control = ThreadPoolExecutor(
            max_workers=self.road_workers, thread_name_prefix="road-control"
        )
        self.lock = threading.Lock()
        self.active = {"compute": 0, "scan": 0, "control": 0}
        self.queued = {"compute": 0, "scan": 0, "control": 0}

    def submit(self, kind, function, *args):
        if kind not in self.active:
            raise ValueError("未知队列类型")
        with self.lock:
            self.queued[kind] += 1

        def execute():
            with self.lock:
                self.queued[kind] -= 1
                self.active[kind] += 1
            try:
                with self.lane_factory() if kind == "compute" else nullcontext():
                    return function(*args)
            finally:
                with self.lock:
                    self.active[kind] -= 1

        try:
            future = {
                "compute": self.compute,
                "scan": self.scans,
                "control": self.control,
            }[kind].submit(execute)
        except Exception:
            with self.lock:
                self.queued[kind] -= 1
            raise

        def cancelled(done):
            if done.cancelled():
                with self.lock:
                    self.queued[kind] -= 1

        future.add_done_callback(cancelled)
        return future

    def snapshot(self):
        with self.lock:
            return {
                "workers": self.workers,
                "execution": os.environ.get("ROAD_SURVEY_EXECUTION", "thread"),
                "process_batch_cap": integer_setting(
                    "ROAD_SURVEY_PROCESS_BATCH_CAP", 32, 1, 64
                ),
                "road_workers": self.road_workers,
                "camera_workers": self.camera_workers,
                "roads_running": self.active["control"],
                "roads_queued": self.queued["control"],
                "running": self.active["compute"],
                "queued": self.queued["compute"],
                "scanning": self.active["scan"],
                "scan_queued": self.queued["scan"],
            }

    def shutdown(self):
        self.control.shutdown(wait=True)
        self.compute.shutdown(wait=True)
        self.scans.shutdown(wait=True)


_scheduler = None
_scheduler_lock = threading.Lock()


def get_scheduler():
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = RoadSurveyScheduler()
        return _scheduler


def automatic_batch(size, fallback, memory=None, workers=None):
    """Conservative estimate; actual CUDA OOM is handled by predict_with_backoff."""
    workers = compute_workers() if workers is None else workers
    if memory is None:
        import torch

        if not torch.cuda.is_available():
            return fallback
        free, total = torch.cuda.mem_get_info()
        memory = (total / 1024**3, free / 1024**3)
    total, free = memory
    if total <= 12:
        return fallback
    height, width = size if isinstance(size, tuple) else (size, size)
    per_frame = 0.18 * height * width / (640 * 640)
    budget = min(total, free) * 0.65 / workers
    return max(1, min(64, int(budget / per_frame)))


def predict_with_backoff(model, frames, image_size, state):
    """Retry detection only, before tracking/counting any frame, preserving order."""
    results = []
    offset = 0
    while offset < len(frames):
        count = min(state["batch_size"], len(frames) - offset)
        try:
            predicted = model.predict(
                frames[offset : offset + count], imgsz=image_size, verbose=False
            )
        except RuntimeError as error:
            import torch

            if not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            if count <= 1:
                raise RuntimeError(
                    "显存不足，单帧也无法处理，请降低分辨率或并行路数"
                ) from error
            state["batch_size"] = max(1, count // 2)
            torch.cuda.empty_cache()
            continue
        results.extend(predicted)
        offset += count
    return results
