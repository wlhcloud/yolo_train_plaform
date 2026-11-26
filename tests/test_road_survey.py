import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import load_workbook

from services import road_survey_service as service

REGION = {
    "mode": "line",
    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "line": [[0, 0.5], [1, 0.5]],
    "arrow": [[0.5, 0.2], [0.5, 0.8]],
}


def configuration(folder):
    return {
        "road_name": "测试道路",
        "spot_code": "01",
        "survey_date": "2026-11-10",
        "class_mapping": {"car": "小客车"},
        "directions": [
            {
                "name": name,
                "streams": [
                    {
                        "name": "机位1",
                        "start_time": "07:00:00",
                        "cards": [{"folder": str(folder), "gap_seconds": 0}],
                        "regions": {"主路": copy.deepcopy(REGION)},
                    }
                ],
            }
            for name in ["北向南", "南向北"]
        ],
    }


class RoadSurveyTest(unittest.TestCase):
    def test_flat_road_directories_have_no_invented_card_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            road = Path(directory) / "南二环永定门桥上11月11日（3小时）"
            names = [
                "东向西主路6：50-9：50",
                "西向东主路6：50-9：50",
                "西向东辅路6：50-9：50",
            ]
            for name in names:
                folder = road / name
                folder.mkdir(parents=True)
                (folder / "00000.MTS").touch()
            discovered = service.discover_roads(directory)[0]
            self.assertEqual(len(discovered["directions"]), 2)
            config = configuration(directory)
            config["road_name"] = discovered["road_name"]
            config["directions"] = discovered["directions"]
            for direction in config["directions"]:
                for source in direction["streams"]:
                    source["regions"] = {
                        "辅路" if "辅路" in source["name"] else "主路": REGION
                    }
            with patch.object(
                service,
                "probe_video",
                return_value={
                    "duration": 10800,
                    "fps": 25,
                    "width": 1440,
                    "height": 1080,
                },
            ):
                manifest = service.scan_manifest(config)
            self.assertEqual(len(manifest["streams"]), 3)
            for stream in manifest["streams"]:
                self.assertEqual(stream["files"][0]["card_label"], "")
                stream["files"][0].pop(
                    "card_label"
                )  # Existing saved manifests also work.
            with tempfile.TemporaryDirectory() as output:
                exports = service.export_workbooks(config, manifest, {}, output)
                self.assertEqual(len(exports), 4)
                for report in exports:
                    sheet = load_workbook(report["path"]).active
                    self.assertEqual(sheet["E3"].value, config["road_name"])
                    self.assertEqual(
                        sheet.max_row, 38
                    )  # Three hours, 36 five-minute rows.

    def test_card_labels_use_actual_folder_number_and_legacy_source(self):
        for folder, expected in [
            ("1", "一卡"),
            ("2", "二卡"),
            ("一卡", "一卡"),
            ("第2张卡", "二卡"),
            ("卡三", "三卡"),
            ("西向东主路6：50-9：50", ""),
        ]:
            self.assertEqual(service.card_directory_label(folder), expected)
        files = [
            {"path": r"E:\道路\西向东\2\00000.MTS", "card": 1, "start": 0, "end": 300}
        ]
        self.assertEqual(
            service.location_with_cards("道路", files, 0, 300), "道路（二卡）"
        )
        # A numbered nested folder is not a card when the saved source is an ordinary directory.
        source = {"cards": [{"folder": r"E:\道路\西向东主路"}]}
        self.assertEqual(
            service.location_with_cards("道路", files, 0, 300, source), "道路"
        )

    @patch("services.road_survey_scheduler.integer_setting", return_value=0)
    @patch(
        "services.road_survey_scheduler.automatic_batch",
        side_effect=lambda size, fallback: fallback,
    )
    def test_inference_resolution_and_safe_batch_sizes(self, auto_batch, server_batch):
        metadata = {"width": 1920, "height": 1080}
        self.assertEqual(service.inference_settings({}, metadata), (640, 16))
        self.assertEqual(
            service.inference_settings({"inference_resolution": "1280"}, metadata),
            (1280, 2),
        )
        self.assertEqual(
            service.inference_settings({"inference_resolution": "original"}, metadata),
            ((1088, 1920), 1),
        )
        config = configuration("ignored")
        config["inference_resolution"] = "bad"
        with self.assertRaisesRegex(ValueError, "分辨率"):
            service.validate_config(config)

    def test_excluded_label_is_still_visible_in_live_counts(self):
        region = {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        counter = service.RegionCounter({"主路": region}, {"作业车": service.EXCLUDE})
        events = counter.consume(
            [[10, 10, 30, 30, 1, 0.9, 0, 0]], 100, 100, 0, {0: "作业车"}
        )
        self.assertEqual(events[0]["category"], service.EXCLUDE)
        self.assertEqual(counter.label_counts["主路"]["作业车"], 1)
        self.assertEqual(counter.excluded, 1)

    def test_road_lookup_uses_directory_and_supports_older_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = configuration(root / "道路A" / "第一张")
            self.assertTrue(
                service.matches_road(config, str(root / "道路A"), "任意显示名称")
            )
            self.assertFalse(
                service.matches_road(config, str(root / "道路B"), "测试道路")
            )
            config["road_root"] = str(root / "道路A")
            self.assertTrue(service.matches_road(config, str(root / "道路A"), "道路A"))
            self.assertFalse(service.matches_road(config, str(root / "道路B"), "道路B"))

    def test_region_only_configuration_and_per_region_dedup(self):
        main = {"polygon": [[0, 0], [0.45, 0], [0.45, 1], [0, 1]]}
        side = {"polygon": [[0.55, 0], [1, 0], [1, 1], [0.55, 1]]}
        config = configuration("ignored")
        config["directions"][0]["streams"][0]["regions"] = {"主路": main, "辅路": side}
        service.validate_config(config, require_regions=True)
        counter = service.RegionCounter({"主路": main, "辅路": side}, {"car": "小客车"})

        def observe(x, second):
            return counter.consume(
                [[x - 5, 20, x + 5, 50, 1, 0.9, 0, 0]], 100, 100, second, {0: "car"}
            )

        self.assertEqual(len(observe(20, 0)), 1)
        self.assertEqual(observe(25, 1), [])
        self.assertEqual(observe(50, 2), [])
        self.assertEqual(observe(20, 3), [])
        self.assertEqual(observe(80, 4)[0]["road_type"], "辅路")
        self.assertEqual(observe(80, 5), [])

    def test_crossings_direction_dedup_and_finite_line(self):
        counter = service.RegionCounter({"主路": REGION}, {"car": "小客车"})

        def frame(y, tid=1, second=1):
            return counter.consume(
                [[40, y - 10, 60, y, tid, 0.9, 0, 0]], 100, 100, second, {0: "car"}
            )

        self.assertEqual(frame(40), [])
        events = frame(60, second=2)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["second"], 1.5)
        self.assertEqual(frame(40, second=3), [])
        self.assertEqual(frame(60, second=4), [])
        self.assertEqual(frame(60, tid=2, second=5), [])
        self.assertEqual(frame(40, tid=2, second=6), [])  # wrong direction
        self.assertIsNone(
            service.crossing_point([1.5, 0.4], [1.5, 0.6], REGION["line"])
        )

    def test_card_order_duration_gap_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "第一张", root / "第二张"
            first.mkdir()
            second.mkdir()
            for path in [
                first / "00010.MTS",
                first / "00002.MTS",
                second / "00000.MTS",
            ]:
                path.write_bytes(b"video")
            config = configuration(first)
            config["directions"] = config["directions"][:1]
            config["directions"][0]["streams"][0]["cards"].append(
                {"folder": str(second), "gap_seconds": 10}
            )
            with patch.object(
                service,
                "probe_video",
                return_value={"duration": 1020, "fps": 25, "width": 100, "height": 100},
            ):
                manifest = service.scan_manifest(config)
                files = manifest["streams"][0]["files"]
                self.assertEqual(
                    [f["name"] for f in files], ["00002.MTS", "00010.MTS", "00000.MTS"]
                )
                self.assertEqual([f["start"] for f in files], [25200, 26220, 27250])
                service.verify_manifest(config, manifest)
                config["directions"][0]["streams"][0]["cards"].append(
                    {"folder": str(first), "gap_seconds": 0}
                )
                with self.assertRaisesRegex(ValueError, "重复"):
                    service.scan_manifest(config)

    def test_template_exports_merge_counts_and_distinguish_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            config = configuration(temp)
            manifest = {
                "streams": [
                    {
                        "direction": "北向南",
                        "road_types": ["主路"],
                        "files": [
                            {
                                "start": 25200,
                                "end": 25440,
                                "card": 1,
                                "card_label": "一卡",
                            },
                            {
                                "start": 25440,
                                "end": 25800,
                                "card": 2,
                                "card_label": "二卡",
                            },
                        ],
                    }
                ]
            }
            completed = {
                "0:0": {
                    "direction": "北向南",
                    "buckets": {
                        "主路": {"25200": {"小客车": 2}, "25500": {"小客车": 3}}
                    },
                    "coverage": [
                        {"road_type": "主路", "start": 25200, "end": 25500},
                        {"road_type": "主路", "start": 25510, "end": 25800},
                    ],
                },
                "0:1": {
                    "direction": "北向南",
                    "buckets": {"主路": {"25800": {"小客车": 4}}},
                    "coverage": [],
                },
            }
            exports = service.export_workbooks(config, manifest, completed, temp)
            self.assertEqual(len(exports), 4)
            main = load_workbook(exports[0]["path"])
            self.assertEqual(
                main.sheetnames,
                [
                    "5分钟",
                    "15分钟",
                    "30分钟",
                    "1小时",
                    "断面15分钟",
                    "断面30分钟",
                    "断面1小时",
                ],
            )
            sheet = main.active
            self.assertEqual(sheet.max_column, 21)
            self.assertEqual(sheet["I3"].value, 2)
            self.assertEqual(sheet["I4"].value, 3)
            self.assertEqual(sheet["E3"].value, config["road_name"] + "（一卡、二卡）")
            self.assertEqual(sheet["E4"].value, config["road_name"] + "（二卡）")
            self.assertIsNone(sheet["H3"].comment)
            self.assertIn("290.00/300", sheet["H4"].comment.text)
            auxiliary = load_workbook(exports[1]["path"]).active
            self.assertIsNone(auxiliary["I3"].value)
            self.assertEqual(auxiliary["E3"].value, config["road_name"])
            self.assertIsNotNone(auxiliary["H3"].comment)
            self.assertIn("K1:L1", {str(m) for m in sheet.merged_cells.ranges})

    def test_overlapping_cameras_rejected_but_distinct_roads_allowed(self):
        config = configuration("test")
        source = copy.deepcopy(config["directions"][0]["streams"][0])
        config["directions"][0]["streams"].append(source)
        manifest = {
            "streams": [
                {
                    "direction": "北向南",
                    "direction_index": 0,
                    "stream_index": i,
                    "files": [{"start": 0, "end": 300}],
                }
                for i in range(2)
            ]
        }
        with self.assertRaisesRegex(ValueError, "重复覆盖"):
            service.validate_camera_coverage(config, manifest)
        source["regions"] = {"辅路": source["regions"].pop("主路")}
        service.validate_camera_coverage(config, manifest)

    def test_mts_probe_uses_frame_rate_not_field_rate(self):
        output = b'{"streams":[{"avg_frame_rate":"25/1","r_frame_rate":"50/1","duration":"35.78","width":1920,"height":1080}]}'
        with patch.object(service.subprocess, "check_output", return_value=output):
            self.assertEqual(service.probe_video("clip.MTS")["fps"], 25)

    def test_tracker_survives_card_boundary_and_counts_only_once(self):
        class Capture:
            def __init__(self, path):
                self.frames = iter([40, 60])

            def isOpened(self):
                return True

            def read(self):
                value = next(self.frames, None)
                return (
                    (False, None)
                    if value is None
                    else (True, np.full((100, 100, 3), value, np.uint8))
                )

            def release(self):
                pass

        class Data:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Model:
            names = {0: "car"}

            def predict(self, frames, **kwargs):
                return [
                    types.SimpleNamespace(
                        orig_img=f,
                        boxes=types.SimpleNamespace(
                            data=Data(
                                np.array(
                                    [
                                        [
                                            40,
                                            int(f[0, 0, 0]) - 10,
                                            60,
                                            int(f[0, 0, 0]),
                                            0.9,
                                            0,
                                        ]
                                    ]
                                )
                            )
                        ),
                    )
                    for f in frames
                ]

        instances = []

        class Tracker:
            def __init__(self, **kwargs):
                instances.append(self)

            def update(self, boxes, image):
                b = boxes[0]
                return [[*b[:4], 1, b[4], b[5], 0]]

        config = configuration("ignored")
        stream = {
            "key": "0:0",
            "direction": "北向南",
            "files": [
                {
                    "path": str(i),
                    "fps": 1,
                    "width": 100,
                    "height": 100,
                    "duration": 2,
                    "start": i * 2,
                    "end": i * 2 + 2,
                }
                for i in range(2)
            ],
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(service.cv2, "VideoCapture", Capture),
            patch(
                "services.road_survey_scheduler.automatic_batch",
                side_effect=lambda size, fallback: fallback,
            ),
            patch.dict(
                "sys.modules", {"boxmot": types.SimpleNamespace(ByteTrack=Tracker)}
            ),
        ):
            result = service.run_stream(
                Model(),
                config,
                stream,
                config["directions"][0]["streams"][0],
                Path(temp) / "events.jsonl",
            )
            self.assertEqual(len(instances), 1)
            self.assertEqual(result["buckets"]["主路"]["0"]["小客车"], 1)
            self.assertEqual(
                len(
                    (Path(temp) / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
