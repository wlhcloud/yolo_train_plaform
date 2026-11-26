"""One spawned process per camera, preserving road scheduling and frame checkpoints."""

import contextlib
import io
import multiprocessing
import os
import queue
import signal
import sys
import time
import traceback


def arm_parent_exit(parent_pid):
    """On Linux, an unexpected web-process exit must not leave an orphan writer."""
    if sys.platform.startswith("linux"):
        import ctypes

        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(
                ctypes.get_errno(), "Cannot bind camera lifetime to its parent"
            )
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)


def put_latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def camera_entry(payload, updates, result, pictures, cancel, preview_requested):
    """Child owns its model/CUDA/tracker; only metadata and optional JPEGs cross IPC."""
    arm_parent_exit(payload.get("parent_pid", os.getppid()))
    os.chdir(payload["root"])
    from services import road_survey_service as service
    from services.road_survey_scheduler import integer_setting

    os.environ["ROAD_SURVEY_BATCH_CAP"] = str(
        integer_setting("ROAD_SURVEY_PROCESS_BATCH_CAP", 32, 1, 64)
    )

    def check_cancel():
        if cancel.is_set():
            raise service.Cancelled("机位已停止，帧断点已保留。")

    last_preview = [0.0]

    def preview(*args):
        if preview_requested.is_set() and time.monotonic() - last_preview[0] >= 0.125:
            from services.road_survey_live import render_preview

            try:
                put_latest(pictures, render_preview(*args))
                last_preview[0] = time.monotonic()
            except Exception:  # noqa: BLE001 -- preview cannot interrupt counting
                # A preview failure must never interrupt formal counting.
                traceback.print_exc()

    try:
        check_cancel()
        put_latest(updates, {"phase": "loading", "worker_pid": os.getpid()})
        from app import create_app

        # create_app prints database configuration; keep it out of child stdout/logs.
        with contextlib.redirect_stdout(io.StringIO()):
            app = create_app()
        with app.app_context():
            from inference_manager import InferenceManager

            check_cancel()
            config = payload["config"]
            model = InferenceManager(payload["project_id"]).load_model(
                "system", config.get("system_model", "best.pt"), prefer_engine=False
            )
            check_cancel()
            missing = [
                name
                for name in model.names.values()
                if not config.get("class_mapping", {}).get(name)
            ]
            if missing:
                raise ValueError("请明确车型映射：" + "、".join(missing))
            value = service.run_stream(
                model,
                config,
                payload["stream"],
                payload["source"],
                payload["audit_path"],
                lambda value: put_latest(updates, {**value, "worker_pid": os.getpid()}),
                check_cancel,
                preview=preview,
                checkpoint_path=payload["checkpoint_path"],
                resume=payload["resume"],
            )
        result.put(("finished", value))
    except service.Cancelled as error:
        result.put(("stopped", str(error)))
    except Exception as error:  # noqa: BLE001 -- transport the failure to the coordinator
        result.put(
            ("failed", {"message": str(error), "traceback": traceback.format_exc()})
        )


def collect_camera(
    process,
    updates,
    results,
    pictures,
    cancel,
    preview_requested,
    stop,
    report,
    live=None,
):
    """Drain IPC while joining so a child's queue feeder cannot deadlock on exit."""
    response = None
    while process.is_alive():
        if stop.is_set():
            cancel.set()
        if live is not None:
            with live.condition:
                if live.viewers:
                    preview_requested.set()
                else:
                    preview_requested.clear()
        while True:
            try:
                report(updates.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                jpeg = pictures.get_nowait()
            except queue.Empty:
                break
            if live is not None:
                live.publish(jpeg)
        if response is None:
            try:
                response = results.get_nowait()
            except queue.Empty:
                pass
        process.join(timeout=0.03)
    process.join()
    if response is None:
        try:
            response = results.get(timeout=1)
        except queue.Empty as error:
            raise RuntimeError(
                f"机位进程异常退出（退出码{process.exitcode}），可从最近断点恢复"
            ) from error
    status, value = response
    if status == "stopped":
        from services.road_survey_service import Cancelled

        raise Cancelled(value)
    if status != "finished":
        raise RuntimeError(value["message"] + "\n" + value["traceback"])
    # Final authoritative counts must not depend on sampled progress updates.
    report(
        {
            "counts": sum(
                sum(bucket.values())
                for role in value["buckets"].values()
                for bucket in role.values()
            ),
            "label_counts": value.get("label_counts", {}),
        }
    )
    return value


def run_camera_process(
    app, project_id, config, stream, source, prefix, stop, resume, report, live=None
):
    context = multiprocessing.get_context("spawn")
    updates, results, pictures = context.Queue(8), context.Queue(1), context.Queue(1)
    cancel, preview_requested = context.Event(), context.Event()
    payload = {
        "root": app.root_path,
        "parent_pid": os.getpid(),
        "project_id": project_id,
        "config": config,
        "stream": stream,
        "source": source,
        "audit_path": str(prefix.with_suffix(".jsonl")),
        "checkpoint_path": str(prefix.with_suffix(".checkpoint")),
        "resume": resume,
    }
    process = context.Process(
        target=camera_entry,
        args=(payload, updates, results, pictures, cancel, preview_requested),
        name="road-camera-" + stream["key"].replace(":", "-"),
    )
    try:
        process.start()
        return collect_camera(
            process,
            updates,
            results,
            pictures,
            cancel,
            preview_requested,
            stop,
            report,
            live,
        )
    finally:
        if process.pid is not None:
            # Do not terminate a writer between audit flush and atomic checkpoint replacement.
            cancel.set()
            if process.is_alive():
                while process.is_alive():
                    for channel in (updates, results, pictures):
                        try:
                            while True:
                                channel.get_nowait()
                        except queue.Empty:
                            pass
                    process.join(timeout=0.1)
            process.close()
        for channel in (updates, results, pictures):
            channel.close()
            channel.join_thread()
