"""Coordinate independent cameras without occupying a GPU lane while waiting."""

import queue
from concurrent.futures import FIRST_COMPLETED, CancelledError, wait

from services.road_survey_service import Cancelled


def run_cameras(scheduler, streams, run, progress, completed, stop):
    """Callbacks run on the coordinator thread; GPU workers never share its ORM session."""
    updates = queue.SimpleQueue()
    waiting = iter(streams)
    pending = {}
    exhausted = False
    failure = None

    def fill():
        nonlocal exhausted
        while (
            not exhausted
            and not stop.is_set()
            and len(pending) < scheduler.camera_workers
        ):
            stream = next(waiting, None)
            if stream is None:
                exhausted = True
                break
            key = stream["key"]
            pending[
                scheduler.submit(
                    "compute", run, stream, lambda value, k=key: updates.put((k, value))
                )
            ] = key

    try:
        fill()
        while pending:
            if stop.is_set():
                for future in pending:
                    future.cancel()
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            # cancel() may finish before an executor thread marks CANCELLED_AND_NOTIFIED.
            # Such queued cameras must not wait for another road's long-running job.
            done |= {future for future in pending if future.done()}
            while not updates.empty():
                key, value = updates.get()
                if failure is None:
                    progress(key, value)
            for future in done:
                key = pending.pop(future)
                try:
                    value = future.result()
                except (Cancelled, CancelledError):
                    stop.set()
                except Exception as error:  # noqa: BLE001 -- re-raised after sibling checkpoints finish
                    failure = failure or error
                    stop.set()
                else:
                    completed(key, value)
            fill()
    except Exception as error:  # noqa: BLE001 -- preserve callback/submission failures until workers stop
        failure = error
        stop.set()
    finally:
        # Never mark the road stopped while a camera is still writing its checkpoint.
        for future in pending:
            future.cancel()
        running = [future for future in pending if not future.cancelled()]
        if running:
            wait(running)
    if failure is not None:
        raise failure
    if stop.is_set():
        raise Cancelled(
            "该道路机位已停止；正在运行的机位已保存帧断点，已完成机位保留。"
        )


def aggregate_progress(states):
    values = list(states.values())
    latest = {
        "phase": "running",
        "streams": dict(states),
        "stream_count": len(values),
        "completed_streams": sum(v.get("phase") == "finished" for v in values),
        "running_streams": sum(
            v.get("phase") in {"running", "resuming", "loading"} for v in values
        ),
        "queued_streams": sum(v.get("phase") == "queued" for v in values),
        "counts": sum(v.get("counts", 0) for v in values),
    }
    return latest
