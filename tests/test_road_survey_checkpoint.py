import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services import road_survey_service as service
from services.road_survey_checkpoint import StreamCheckpoint, compact_retired_tracks


class FrameData:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FrameModel:
    def __init__(self):
        self.names = {0: "car"}
        self.inferred = []

    def predict(self, frames, **kwargs):
        from types import SimpleNamespace

        results = []
        for frame in frames:
            number = int(frame[0, 0, 0])
            self.inferred.append(number)
            x = 10 if number < 7 else 70
            boxes = np.array([[x, 10, x + 15, 40, 0.95, 0]], dtype=np.float32)
            results.append(
                SimpleNamespace(
                    orig_img=frame, boxes=SimpleNamespace(data=FrameData(boxes))
                )
            )
        return results


class FrameCapture:
    def __init__(self, path):
        self.number = int(path) * 6
        self.end = self.number + 6

    def isOpened(self):
        return True

    def grab(self):
        if self.number >= self.end:
            return False
        self.number += 1
        return True

    def read(self):
        if self.number >= self.end:
            return False, None
        frame = np.full((100, 100, 3), self.number, dtype=np.uint8)
        self.number += 1
        return True, frame

    def release(self):
        pass


class FrameCheckpointTest(unittest.TestCase):
    def test_retired_track_compaction_preserves_exact_tracker_output(self):
        from boxmot import ByteTrack
        from boxmot.trackers.bytetrack.basetrack import BaseTrack

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        def simulate(compact):
            BaseTrack._count = 0
            tracker = ByteTrack(track_buffer=2, frame_rate=25)
            outputs = []
            for frame in range(200):
                boxes = (
                    np.array([[10, 10, 30, 40, 0.95, 0]], dtype=np.float32)
                    if frame % 10 < 5
                    else np.empty((0, 6), dtype=np.float32)
                )
                outputs.append(tracker.update(boxes, image).tolist())
                if compact:
                    compact_retired_tracks(tracker)
            return outputs, len(tracker.removed_stracks)

        baseline, original_count = simulate(False)
        compacted, compact_count = simulate(True)
        self.assertEqual(compacted, baseline)
        self.assertGreater(original_count, 10)
        self.assertLess(compact_count, 3)

    def test_stop_resume_preserves_tracks_counts_and_exact_next_frame(self):
        config = {"class_mapping": {"car": "小客车"}}
        source = {"regions": {"主路": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}}}
        stream = {
            "key": "0:0",
            "direction": "北向南",
            "files": [
                {
                    "path": str(i),
                    "fps": 1,
                    "width": 100,
                    "height": 100,
                    "duration": 6,
                    "start": i * 6,
                    "end": i * 6 + 6,
                }
                for i in range(2)
            ],
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(service.cv2, "VideoCapture", FrameCapture),
            patch.object(service, "inference_settings", return_value=(640, 2)),
            patch(
                "services.road_survey_scheduler.predict_with_backoff",
                side_effect=lambda model, frames, *_: model.predict(frames),
            ),
        ):
            folder = Path(directory)
            first = FrameModel()

            def stop():
                if len(first.inferred) >= 4:
                    raise service.Cancelled("stop")

            with self.assertRaises(service.Cancelled):
                service.run_stream(
                    first,
                    config,
                    stream,
                    source,
                    folder / "events.jsonl",
                    check_cancel=stop,
                    checkpoint_path=folder / "state.checkpoint",
                )
            self.assertEqual(first.inferred, [0, 1, 2, 3])
            before = [
                json.loads(line)
                for line in (folder / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(before), 1)
            # An interrupted process may have written events beyond the committed snapshot.
            with (folder / "events.jsonl").open("a") as audit:
                audit.write("uncommitted tail\n")
            from boxmot.trackers.bytetrack.basetrack import BaseTrack

            # Simulate a process restart, where the library's global ID starts at zero.
            BaseTrack._count = 0
            second = FrameModel()
            resumed = service.run_stream(
                second,
                config,
                stream,
                source,
                folder / "events.jsonl",
                checkpoint_path=folder / "state.checkpoint",
                resume=True,
            )
            self.assertEqual(second.inferred, list(range(4, 12)))
            events = [
                json.loads(line)
                for line in (folder / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0], before[0])
            self.assertEqual(len({e["track_id"] for e in events}), 2)
            self.assertEqual(resumed["buckets"]["主路"]["0"]["小客车"], 2)
            self.assertEqual(len(resumed["coverage"]), 2)
            baseline = service.run_stream(
                FrameModel(), config, stream, source, folder / "baseline.jsonl"
            )
            self.assertEqual(resumed["buckets"], baseline["buckets"])
            self.assertEqual(resumed["coverage"], baseline["coverage"])
            third = FrameModel()
            finished = service.run_stream(
                third,
                config,
                stream,
                source,
                folder / "events.jsonl",
                checkpoint_path=folder / "state.checkpoint",
                resume=True,
            )
            self.assertEqual(third.inferred, [])
            self.assertEqual(finished["buckets"], baseline["buckets"])
            (folder / "events.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "明细与帧断点不一致"):
                service.run_stream(
                    FrameModel(),
                    config,
                    stream,
                    source,
                    folder / "events.jsonl",
                    checkpoint_path=folder / "state.checkpoint",
                    resume=True,
                )
            changed = FrameModel()
            changed_result = service.run_stream(
                changed,
                {"class_mapping": {"car": "出租车"}},
                stream,
                source,
                folder / "events.jsonl",
                checkpoint_path=folder / "state.checkpoint",
                resume=True,
            )
            self.assertEqual(changed.inferred, list(range(12)))
            self.assertEqual(changed_result["buckets"]["主路"]["0"], {"出租车": 2})

    def test_checkpoint_rejects_corruption_and_ignores_changed_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state"
            checkpoint = StreamCheckpoint(path, "first")
            with (Path(directory) / "audit").open("w") as audit:
                audit.write("one\n")
                checkpoint.save({"next_frame": 7}, audit)
            self.assertEqual(checkpoint.load()["next_frame"], 7)
            self.assertIsNone(StreamCheckpoint(path, "changed").load())
            path.write_bytes(b"broken snapshot")
            with self.assertRaisesRegex(ValueError, "断点读取失败"):
                checkpoint.load()


if __name__ == "__main__":
    unittest.main()
