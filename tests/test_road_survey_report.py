import io
import unittest
import zipfile

from openpyxl import load_workbook

from services.road_survey_report import (
    SHEETS,
    build_traffic_workbook,
    save_cached_workbook,
)
from services.road_survey_review_excel import build_review_zip


class TrafficTemplateTest(unittest.TestCase):
    def fixture(self):
        start = 6 * 3600 + 50 * 60
        config = {
            "road_name": "测试道路",
            "spot_code": "001",
            "survey_date": "2026-11-26",
        }
        buckets = {
            str(start + offset * 60): {"小客车": count}
            for offset, count in [(0, 2), (5, 3), (10, 5), (20, 7), (25, 11)]
        }
        intervals = [(start, start + 900), (start + 1200, start + 1800)]
        files = [
            {"start": start, "end": start + 720, "card_label": "一卡"},
            {"start": start + 720, "end": start + 1800, "card_label": "二卡"},
        ]
        return config, start, buckets, intervals, files

    def test_seven_sheets_formulas_cached_totals_start_boundaries_and_colors(self):
        config, start, buckets, intervals, files = self.fixture()
        output = save_cached_workbook(
            build_traffic_workbook(
                config, "北向南", "主路", start, start + 1800, buckets, intervals, files
            )
        ).getvalue()
        formulas = load_workbook(io.BytesIO(output))
        values = load_workbook(io.BytesIO(output), data_only=True)
        self.assertEqual(values.sheetnames, [name for name, *_ in SHEETS])
        expected = {
            "5分钟": [2, 3, 5, None, 7, 11],
            "15分钟": [10, 18],
            "30分钟": [28],
            "1小时": [28],
            "断面15分钟": [10, 18],
            "断面30分钟": [28],
            "断面1小时": [28],
        }
        for name, counts in expected.items():
            sheet = values[name]
            self.assertEqual(
                [sheet.cell(r, 9).value for r in range(3, sheet.max_row + 1)], counts
            )
            self.assertEqual(sum(v or 0 for v in counts), 28)
            self.assertEqual(sheet.max_column, 21)
            self.assertTrue(
                all(
                    sheet.cell(r, 7).value == "北向南"
                    for r in range(3, sheet.max_row + 1)
                )
            )
            self.assertFalse(
                any(
                    cell.fill.fgColor.rgb == "00FFF2CC" for row in sheet for cell in row
                )
            )
        self.assertEqual(values["15分钟"]["H3"].value, "06:50-07:05")
        self.assertEqual(values["30分钟"]["H3"].value, "06:50-07:20")
        self.assertEqual(values["1小时"]["H3"].value, "06:50-07:50")
        self.assertIsNone(values["15分钟"]["H3"].comment)
        self.assertIn("600.00/900", values["15分钟"]["H4"].comment.text)
        self.assertEqual(values["5分钟"]["E5"].value, "测试道路（一卡、二卡）")
        self.assertEqual(values["5分钟"]["I3"].fill.fgColor.rgb[-6:], "FF99CC")
        self.assertEqual(values["5分钟"]["I5"].fill.fgColor.rgb[-6:], "FF99CC")
        for row in range(6, 9):
            self.assertEqual(values["5分钟"].cell(row, 9).fill.fgColor.rgb[-6:], "CCFFCC")
        self.assertIsNone(values["5分钟"]["A3"].fill.patternType)
        self.assertEqual(values["断面15分钟"]["F3"].value, "断面")
        self.assertEqual(values["断面30分钟"]["S2"].value, "外卖电动")
        self.assertIn("SUMIF('5分钟'!", formulas["15分钟"]["I3"].value)
        self.assertEqual(formulas["断面15分钟"]["I3"].value, "='15分钟'!I3")
        self.assertEqual(formulas.calculation.calcMode, "auto")

    def test_no_coverage_stays_blank_in_every_sheet_and_midnight_dates_are_correct(
        self,
    ):
        config, _, _, _, _ = self.fixture()
        start, end = 86100, 86700
        buckets = {"86100": {"小客车": 2}, "86400": {"小客车": 3}}
        book = build_traffic_workbook(
            config, "西向东", "辅路", start, end, buckets, [(start, end)], []
        )
        values = load_workbook(save_cached_workbook(book), data_only=True)
        self.assertEqual(values["5分钟"]["C3"].value, 20261126)
        self.assertEqual(values["5分钟"]["C4"].value, 20261127)
        self.assertEqual(values["15分钟"]["H3"].value, "23:55-00:10")
        self.assertEqual(values["15分钟"].max_row, 3)
        empty = load_workbook(
            save_cached_workbook(
                build_traffic_workbook(config, "西向东", "辅路", start, end, {}, [], [])
            )
        )
        for sheet in empty:
            self.assertIsNone(sheet["I3"].value)
            self.assertIn("0.00/", sheet["H3"].comment.text)

    def test_legacy_review_zip_contains_two_seven_sheet_templates(self):
        config, start, _, _, files = self.fixture()
        stream = {"key": "0:0", "direction": "北向南", "name": "机位", "files": files}
        snapshot = {
            "config": config,
            "stream": stream,
            "source": {"regions": {"主路": {}}},
        }
        audit = {
            "config": config,
            "stream": stream,
            "start": start,
            "end": start + 300,
            "events": [],
            "coverage": [[start, start + 300]],
        }
        with zipfile.ZipFile(build_review_zip(audit, snapshot)) as archive:
            templates = [n for n in archive.namelist() if n.endswith("_抽查.xlsx")]
            self.assertEqual(len(templates), 2)
            for name in templates:
                self.assertEqual(
                    load_workbook(io.BytesIO(archive.read(name))).sheetnames,
                    [name for name, *_ in SHEETS],
                )


if __name__ == "__main__":
    unittest.main()
