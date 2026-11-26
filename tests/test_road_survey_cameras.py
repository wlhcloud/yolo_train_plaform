import threading
import time
import unittest
from contextlib import nullcontext

from services.road_survey_cameras import aggregate_progress, run_cameras
from services.road_survey_scheduler import RoadSurveyScheduler
from services.road_survey_service import Cancelled


class CameraConcurrencyTest(unittest.TestCase):
    def test_stopping_queued_camera_does_not_wait_for_another_road(self):
        scheduler = RoadSurveyScheduler(
            workers=1, roads=2, cameras=1, lane_factory=nullcontext
        )
        release = threading.Event()
        stop = threading.Event()
        try:
            occupying = scheduler.submit("compute", release.wait, 5)
            future = scheduler.submit(
                "control",
                run_cameras,
                scheduler,
                [{"key": "queued"}],
                lambda *args: self.fail("Cancelled camera started"),
                lambda *args: None,
                lambda *args: None,
                stop,
            )
            deadline = time.monotonic() + 2
            while scheduler.snapshot()["queued"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(scheduler.snapshot()["queued"], 1)
            stop.set()
            with self.assertRaises(Cancelled):
                future.result(timeout=1)
            self.assertFalse(occupying.done())
        finally:
            stop.set()
            release.set()
            scheduler.shutdown()

    def test_separate_road_and_camera_limits_and_stop_isolation(self):
        scheduler = RoadSurveyScheduler(roads=2, cameras=2, lane_factory=nullcontext)
        started = {name: [threading.Event() for _ in range(3)] for name in "ABC"}
        stops = {name: threading.Event() for name in "ABC"}
        release = threading.Event()
        callbacks = []

        def road(name):
            owner = threading.get_ident()

            def camera(stream, report):
                index = int(stream["key"])
                started[name][index].set()
                report({"phase": "running", "frame": 10})
                while not release.wait(0.01):
                    if stops[name].is_set():
                        raise Cancelled("checkpoint saved")
                return name + stream["key"]

            def report(key, value):
                self.assertEqual(threading.get_ident(), owner)
                callbacks.append((name, key))

            result = {}
            run_cameras(
                scheduler,
                [{"key": str(i)} for i in range(3)],
                camera,
                report,
                lambda key, value: result.update({key: value}),
                stops[name],
            )
            return result

        try:
            first = scheduler.submit("control", road, "A")
            second = scheduler.submit("control", road, "B")
            for name in "AB":
                for index in range(2):
                    self.assertTrue(started[name][index].wait(3))
                self.assertFalse(started[name][2].is_set())
            third = scheduler.submit("control", road, "C")
            runtime = scheduler.snapshot()
            self.assertEqual(runtime["workers"], 4)
            self.assertEqual(runtime["roads_running"], 2)
            self.assertEqual(runtime["roads_queued"], 1)
            self.assertEqual(runtime["running"], 4)
            self.assertFalse(started["C"][0].is_set())
            stops["A"].set()
            with self.assertRaises(Cancelled):
                first.result(timeout=3)
            self.assertTrue(started["C"][0].wait(3))
            self.assertFalse(second.done())
            self.assertFalse(stops["B"].is_set())
            release.set()
            self.assertEqual(len(second.result(timeout=3)), 3)
            self.assertEqual(len(third.result(timeout=3)), 3)
            self.assertTrue(callbacks)
        finally:
            release.set()
            for stop in stops.values():
                stop.set()
            scheduler.shutdown()

    def test_single_gpu_slot_does_not_deadlock_waiting_coordinator(self):
        scheduler = RoadSurveyScheduler(roads=1, cameras=1, lane_factory=nullcontext)
        completed = {}
        try:

            def road():
                run_cameras(
                    scheduler,
                    [{"key": "0"}, {"key": "1"}],
                    lambda stream, report: stream["key"],
                    lambda *args: None,
                    lambda key, result: completed.update({key: result}),
                    threading.Event(),
                )

            scheduler.submit("control", road).result(timeout=3)
            self.assertEqual(completed, {"0": "0", "1": "1"})
        finally:
            scheduler.shutdown()

    def test_camera_failure_waits_for_sibling_checkpoint_before_returning(self):
        scheduler = RoadSurveyScheduler(roads=1, cameras=2, lane_factory=nullcontext)
        stop = threading.Event()
        started = threading.Barrier(2)
        saved = threading.Event()

        def camera(stream, report):
            started.wait(timeout=3)
            if stream["key"] == "bad":
                raise ValueError("broken video")
            self.assertTrue(stop.wait(3))
            time.sleep(0.03)
            saved.set()
            raise Cancelled("checkpoint saved")

        try:
            with self.assertRaisesRegex(ValueError, "broken video"):
                run_cameras(
                    scheduler,
                    [{"key": "bad"}, {"key": "other"}],
                    camera,
                    lambda *args: None,
                    lambda *args: None,
                    stop,
                )
            self.assertTrue(saved.is_set())
            self.assertEqual(scheduler.snapshot()["running"], 0)
        finally:
            stop.set()
            scheduler.shutdown()

    def test_progress_is_aggregated_without_losing_camera_counts(self):
        value = aggregate_progress(
            {
                "a": {"phase": "finished", "counts": 8},
                "b": {"phase": "running", "counts": 3},
                "c": {"phase": "queued"},
            }
        )
        self.assertEqual(value["counts"], 11)
        self.assertEqual(value["completed_streams"], 1)
        self.assertEqual(value["running_streams"], 1)
        self.assertEqual(value["queued_streams"], 1)


if __name__ == "__main__":
    unittest.main()
