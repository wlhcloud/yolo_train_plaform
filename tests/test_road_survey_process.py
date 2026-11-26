import queue
import threading
import unittest
from unittest.mock import patch

from services.road_survey_live import LiveFrames
from services.road_survey_process import collect_camera, put_latest
from services.road_survey_service import Cancelled, inference_settings


class FakeProcess:
    exitcode = 0

    def __init__(self):
        self.checks = 0

    def is_alive(self):
        self.checks += 1
        return self.checks == 1

    def join(self, timeout=None):
        pass


class ProcessCameraTest(unittest.TestCase):
    def test_ipc_preserves_progress_preview_and_authoritative_final_counts(self):
        updates, results, pictures = queue.Queue(), queue.Queue(), queue.Queue()
        updates.put({"phase": "running", "frame": 64, "counts": 2})
        value = {
            "buckets": {"主路": {"0": {"小客车": 3}}},
            "label_counts": {"主路": {"小客车": 3}},
        }
        results.put(("finished", value))
        pictures.put(b"latest-jpeg")
        live = LiveFrames()
        live.viewers = 1
        cancel, preview, stop = threading.Event(), threading.Event(), threading.Event()
        observed = []
        actual = collect_camera(
            FakeProcess(),
            updates,
            results,
            pictures,
            cancel,
            preview,
            stop,
            observed.append,
            live,
        )
        self.assertEqual(actual, value)
        self.assertEqual(observed[0]["frame"], 64)
        self.assertEqual(observed[-1]["counts"], 3)
        self.assertEqual(live.jpeg, b"latest-jpeg")
        self.assertTrue(preview.is_set())

    def test_stop_event_is_forwarded_and_child_failures_propagate(self):
        stop = threading.Event()
        stop.set()
        cancel = threading.Event()
        results = queue.Queue()
        results.put(("stopped", "checkpoint saved"))
        with self.assertRaisesRegex(Cancelled, "checkpoint saved"):
            collect_camera(
                FakeProcess(),
                queue.Queue(),
                results,
                queue.Queue(),
                cancel,
                threading.Event(),
                stop,
                lambda value: None,
            )
        self.assertTrue(cancel.is_set())
        results.put(
            ("failed", {"message": "bad input", "traceback": "child traceback"})
        )
        with self.assertRaisesRegex(RuntimeError, "bad input"):
            collect_camera(
                FakeProcess(),
                queue.Queue(),
                results,
                queue.Queue(),
                threading.Event(),
                threading.Event(),
                threading.Event(),
                lambda value: None,
            )

    def test_latest_queue_is_bounded_and_batch_cap_does_not_change_saved_config(self):
        channel = queue.Queue(maxsize=1)
        put_latest(channel, "old")
        put_latest(channel, "latest")
        self.assertEqual(channel.get_nowait(), "latest")
        config = {"inference_resolution": "1280", "inference_batch_size": 64}
        with patch.dict("os.environ", {"ROAD_SURVEY_BATCH_CAP": "32"}):
            self.assertEqual(
                inference_settings(config, {"width": 1920, "height": 1080}), (1280, 32)
            )
        self.assertEqual(config["inference_batch_size"], 64)


if __name__ == "__main__":
    unittest.main()
