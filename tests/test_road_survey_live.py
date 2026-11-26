import unittest

import cv2
import numpy as np

from services.road_survey_live import (
    LiveFrames,
    finish_task,
    frames_for,
    render_preview,
)


class LivePreviewTest(unittest.TestCase):
    def test_camera_buffers_are_independent_and_finish_only_the_selected_road(self):
        first = frames_for(999001, "0:0")
        second = frames_for(999001, "1:0")
        other = frames_for(999002, "0:0")
        first.publish(b"camera-a")
        second.publish(b"camera-b")
        self.assertEqual(first.jpeg, b"camera-a")
        self.assertEqual(second.jpeg, b"camera-b")
        finish_task(999001)
        self.assertTrue(first.done)
        self.assertTrue(second.done)
        self.assertFalse(other.done)

    def test_multipart_delivers_latest_frame_and_cleans_up_viewer(self):
        live = LiveFrames()
        stream = live.stream()
        self.assertIn(b"Content-Type: image/jpeg", next(stream))
        self.assertEqual(live.viewers, 1)
        self.assertTrue(live.wants_frame())
        self.assertFalse(live.wants_frame())
        live.publish(b"older")
        live.publish(b"latest")
        self.assertTrue(next(stream).endswith(b"latest\r\n"))
        live.finish()
        next(stream)
        with self.assertRaises(StopIteration):
            next(stream)
        self.assertEqual(live.viewers, 0)
        live.reset()
        self.assertFalse(live.done)
        self.assertFalse(live.wants_frame())

    def test_preview_draws_boxes_and_returns_decodable_jpeg(self):
        frame = np.zeros((540, 960, 3), np.uint8)
        jpeg = render_preview(
            frame,
            [[100, 100, 250, 250, 7, 0.9, 0, 0]],
            {"主路": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
            {0: "小客车"},
            {"主路": 3, "辅路": 2},
            25200,
            "北向南",
        )
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, frame.shape)
        self.assertGreater(int(decoded.sum()), 0)


if __name__ == "__main__":
    unittest.main()
