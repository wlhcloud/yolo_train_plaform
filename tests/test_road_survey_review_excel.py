import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from openpyxl import load_workbook

from services.road_survey_review_excel import build_review_excel, build_road_review_zip


class ReviewExcelTest(unittest.TestCase):
    def test_whole_road_zip_has_four_templates_and_matching_audits(self):
        audit, snapshot = self.fixture()
        config = copy.deepcopy(snapshot["config"])
        config["directions"] = [{"name": name} for name in ["北向南", "南向北"]]
        parts, outputs = [], []
        with tempfile.TemporaryDirectory() as directory:
            for index, direction in enumerate(config["directions"]):
                item = copy.deepcopy(audit)
                item.update(start=25200, end=25500, coverage=[[25200, 25500]])
                item["events"] = [item["events"][0]]
                item["config"] = config
                item["stream"] = {
                    "key": str(index),
                    "direction": direction["name"],
                    "name": "机位",
                }
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps(item), encoding="utf-8")
                parts.append(
                    {
                        "stream": item["stream"],
                        "source": snapshot["source"],
                        "start": 25200,
                        "end": 25500,
                    }
                )
                outputs.append({"key": str(index), "result": {"audit_path": str(path)}})
            package = build_road_review_zip(
                {"parts": outputs}, {"config": config, "parts": parts}
            )
            with zipfile.ZipFile(package) as archive:
                templates = [
                    n
                    for n in archive.namelist()
                    if n.endswith(".xlsx") and "/" not in n
                ]
                self.assertEqual(len(templates), 4)
                total = 0
                for name in templates:
                    workbook = load_workbook(
                        io.BytesIO(archive.read(name)), data_only=True
                    )
                    self.assertEqual(workbook.sheetnames, ["5分钟"])
                    sheet = workbook.active
                    self.assertEqual(sheet.max_row, 3)
                    self.assertEqual(sheet["H3"].value, "07:00-07:05")
                    total += sum(v or 0 for v in list(sheet.values)[2][8:])
                self.assertEqual(total, 2)
                self.assertEqual(
                    sum(
                        len(archive.read(n).splitlines())
                        for n in archive.namelist()
                        if n.endswith(".jsonl")
                    ),
                    2,
                )

    def fixture(self):
        config = {
            "road_name": "测试道路",
            "spot_code": "01",
            "survey_date": "2026-11-24",
            "system_model": "best.pt",
            "class_mapping": {
                "小客车": "小客车",
                "中巴车": "大客车非公交",
                "作业车": "不纳入模板",
            },
        }
        events = [
            {
                "second": 25495,
                "road_type": "主路",
                "class_name": "小客车",
                "category": "小客车",
                "track_id": 1,
                "confidence": 0.8,
                "file": "a.MTS",
            },
            {
                "second": 25506,
                "road_type": "主路",
                "class_name": "中巴车",
                "category": "大客车非公交",
                "track_id": 2,
                "confidence": 0.9,
                "file": "b.MTS",
            },
            {
                "second": 25510,
                "road_type": "辅路",
                "class_name": "作业车",
                "category": "不纳入模板",
                "track_id": 3,
                "confidence": 0.7,
                "file": "b.MTS",
            },
        ]
        stream = {"direction": "北向南", "name": "机位1"}
        audit = {
            "start": 25490,
            "end": 25520,
            "coverage": [[25490, 25500], [25505, 25520]],
            "events": events,
            "config": config,
            "stream": stream,
        }
        snapshot = {
            "config": config,
            "stream": stream,
            "source": {"regions": {"主路": {}, "辅路": {}}},
        }
        return audit, snapshot

    def test_template_mapping_coverage_and_raw_counts_reconcile(self):
        audit, snapshot = self.fixture()
        w = load_workbook(build_review_excel(audit, snapshot), data_only=True)
        self.assertEqual(w.sheetnames, ["5分钟", "车型汇总", "逐车明细", "抽查说明"])
        s = w["5分钟"]
        self.assertEqual(s.max_column, 21)
        self.assertEqual(s.max_row, 6)
        self.assertEqual(s["I3"].value, 1)
        self.assertEqual(s["O5"].value, 1)
        self.assertEqual(
            sum(sum(v or 0 for v in row[8:]) for row in list(s.values)[2:]), 2
        )
        self.assertIn("10.00/300", s["H3"].comment.text)
        self.assertIn("15.00/300", s["H5"].comment.text)
        totals = {row[0]: row[4] for row in list(w["车型汇总"].values)[1:]}
        self.assertEqual(totals["全部识别计数"], 3)
        self.assertEqual(totals["纳入模板计数"], 2)
        self.assertEqual(totals["不纳入模板计数"], 1)
        self.assertEqual(w["逐车明细"]["H4"].value, "否")
        self.assertEqual(w["逐车明细"].max_row, 4)
        notes = dict(list(w["抽查说明"].values)[1:])
        self.assertEqual(notes["停录或无录像（秒）"], 5)

    def test_unconfigured_road_is_blank_and_user_names_are_literal(self):
        audit, snapshot = self.fixture()
        snapshot["source"]["regions"].pop("辅路")
        audit["config"]["road_name"] = "=1+1"
        audit["events"] = [e for e in audit["events"] if e["road_type"] == "主路"]
        w = load_workbook(build_review_excel(audit, snapshot))
        self.assertIsNone(w["5分钟"]["I4"].value)
        self.assertIn("未配置", w["5分钟"]["H4"].comment.text)
        self.assertEqual(w["5分钟"]["E3"].data_type, "s")
        self.assertEqual(w["5分钟"]["E3"].value, "=1+1")


if __name__ == "__main__":
    unittest.main()
