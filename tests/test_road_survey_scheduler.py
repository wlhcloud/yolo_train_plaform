import threading
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from services.road_survey_scheduler import (
    RoadSurveyScheduler,
    automatic_batch,
    predict_with_backoff,
)


class SchedulerTest(unittest.TestCase):
    def test_manual_batch_and_validation(self):
        from services.road_survey_service import inference_settings

        size, batch = inference_settings(
            {"inference_resolution": "original", "inference_batch_size": 4},
            {"width": 1920, "height": 1080},
        )
        self.assertEqual((size, batch), ((1088, 1920), 4))

    def test_model_load_failure_restores_torch_loader(self):
        import torch

        from inference_manager import InferenceManager

        original = torch.load
        manager = InferenceManager.__new__(InferenceManager)
        with (
            patch(
                "inference_manager.YOLO", side_effect=RuntimeError("corrupt weights")
            ),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "corrupt weights"):
                manager.load_model(model_path="fake-model.pt", prefer_engine=False)
        self.assertIs(torch.load, original)

    def test_two_lanes_run_together_extra_job_queues_and_scan_is_independent(self):
        scheduler = RoadSurveyScheduler(workers=2, lane_factory=nullcontext)
        release = threading.Event()
        started = [threading.Event(), threading.Event()]

        def job(i):
            started[i].set()
            release.wait(5)
            return i

        try:
            first = scheduler.submit("compute", job, 0)
            second = scheduler.submit("compute", job, 1)
            self.assertTrue(started[0].wait(2))
            self.assertTrue(started[1].wait(2))
            queued = scheduler.submit("compute", lambda: "later")
            self.assertEqual(scheduler.snapshot()["running"], 2)
            self.assertEqual(scheduler.snapshot()["queued"], 1)
            self.assertEqual(
                scheduler.submit("scan", lambda: "scanned").result(timeout=2), "scanned"
            )
            self.assertTrue(queued.cancel())
            self.assertEqual(scheduler.snapshot()["queued"], 0)
            release.set()
            self.assertEqual(first.result(timeout=2), 0)
            self.assertEqual(second.result(timeout=2), 1)
        finally:
            release.set()
            scheduler.shutdown()

    def test_failed_lane_does_not_block_later_jobs(self):
        scheduler = RoadSurveyScheduler(workers=1, lane_factory=nullcontext)
        try:

            def bad():
                raise ValueError("sample")

            with self.assertRaises(ValueError):
                scheduler.submit("compute", bad).result(timeout=2)
            self.assertEqual(
                scheduler.submit("compute", lambda: 7).result(timeout=2), 7
            )
            self.assertEqual(scheduler.snapshot()["running"], 0)
        finally:
            scheduler.shutdown()

    def test_80gb_auto_batch_accounts_for_parallelism_and_resolution(self):
        two = automatic_batch(1280, 2, memory=(80, 70), workers=2)
        four = automatic_batch(1280, 2, memory=(80, 70), workers=4)
        self.assertGreater(two, 2)
        self.assertGreater(two, four)
        self.assertEqual(automatic_batch(1280, 2, memory=(8, 6), workers=1), 2)
        self.assertLess(
            automatic_batch((1088, 1920), 1, memory=(80, 70), workers=2), two
        )

    def test_oom_retries_smaller_batches_without_reordering_or_double_counting(self):
        import torch

        class Model:
            def predict(self, frames, **kwargs):
                if len(frames) > 2:
                    raise torch.cuda.OutOfMemoryError("simulated")
                return list(frames)

        state = {"batch_size": 8}
        with patch.object(torch.cuda, "empty_cache"):
            output = predict_with_backoff(Model(), list(range(7)), 640, state)
        self.assertEqual(output, list(range(7)))
        self.assertLessEqual(state["batch_size"], 2)

    def test_other_model_errors_are_not_retried_as_memory_failures(self):
        class Model:
            def predict(self, *args, **kwargs):
                raise RuntimeError("bad model")

        with self.assertRaisesRegex(RuntimeError, "bad model"):
            predict_with_backoff(Model(), [1, 2], 640, {"batch_size": 2})


if __name__ == "__main__":
    unittest.main()
