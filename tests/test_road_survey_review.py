import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from services.road_survey_review import (
    ReviewWriter,
    decode_segment,
    generate_review,
    review_window,
)
from services.road_survey_service import Cancelled, probe_video


class ReviewTest(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required"
    )
    def test_mts_timestamp_offset_and_seek_match_sequential_decoding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "offset.MTS"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-itsoffset",
                    "0.56",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x64:rate=25:duration=5",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=stereo",
                    "-t",
                    "5",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "ac3",
                    "-f",
                    "mpegts",
                    "-output_ts_offset",
                    "100",
                    str(path),
                ],
                check=True,
            )
            metadata = probe_video(path)
            item = {
                "path": str(path),
                "start": 0,
                "end": metadata["duration"],
                **metadata,
            }
            trimmed = list(decode_segment(item, 2, 4, lambda: None))
            self.assertEqual(len(trimmed), 50)
            last = None
            for _, frame in decode_segment(item, 0, 2.04, lambda: None):
                last = frame
            self.assertTrue(np.array_equal(last, trimmed[0][1]))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_cancel_does_not_publish_incomplete_video(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cancel.mp4"
            stream = {
                "files": [
                    {"width": 64, "height": 64, "start": 0, "end": 300, "fps": 25}
                ]
            }

            def cancel():
                raise Cancelled("cancel")

            with self.assertRaises(Cancelled):
                generate_review(
                    object(),
                    {"class_mapping": {}},
                    stream,
                    {"regions": {}},
                    0,
                    5,
                    path,
                    check_cancel=cancel,
                )
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".partial.mp4").exists())

    def test_window_spans_video_boundary_and_truncates_at_source_end(self):
        stream = {
            "files": [{"start": 25200, "end": 26220}, {"start": 26220, "end": 27240}]
        }
        self.assertEqual(review_window(stream, 26100), (26100, 26400))
        self.assertEqual(review_window(stream, 27200), (27200, 27240))
        with self.assertRaises(ValueError):
            review_window(stream, 28000)
        with self.assertRaises(ValueError):
            review_window(stream, 25200, 600)

    @unittest.skipUnless(
        shutil.which("ffmpeg"), "FFmpeg required for encoded-video verification"
    )
    def test_encoded_duration_does_not_compress_missing_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "check.mp4"
            writer = ReviewWriter(path, 0, 4, 64, 64)
            writer.write_until(1, np.full((64, 64, 3), (0, 0, 200), np.uint8))
            writer.write_until(3, np.zeros((64, 64, 3), np.uint8))
            writer.write_until(4, np.full((64, 64, 3), (0, 200, 0), np.uint8))
            writer.close(True)
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 40)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertLess(float(frame.mean()), 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 35)
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertGreater(float(frame[:, :, 1].mean()), 150)
            cap.release()


if __name__ == "__main__":
    unittest.main()
