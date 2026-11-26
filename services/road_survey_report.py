"""Seven-sheet traffic template, with start-aligned sums and cached Excel formulas."""

import io
import json
import math
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import CalcProperties

SHEETS = [
    ("5分钟", 300, None),
    ("15分钟", 900, "5分钟"),
    ("30分钟", 1800, "15分钟"),
    ("1小时", 3600, "30分钟"),
    ("断面15分钟", 900, "15分钟"),
    ("断面30分钟", 1800, "断面15分钟"),
    ("断面1小时", 3600, "断面30分钟"),
]
COLORS = ("FF99CC", "CCFFCC", "CC99FF", "FFCC99")


def build_traffic_workbook(
    config, direction, role, start, end, buckets, intervals, files
):
    from services.road_survey_service import (
        CATEGORIES,
        covered_seconds,
        location_with_cards,
    )

    metadata = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "config"
            / "traffic-survey-template.json"
        ).read_text(encoding="utf-8")
    )
    midnight = datetime.fromisoformat(config["survey_date"])
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.calculation = CalcProperties(
        calcMode="auto", fullCalcOnLoad=True, forceFullCalc=True
    )
    workbook._traffic_formula_cache = {}
    border = Side(style="thin", color="000000")
    for name, period, source_name in SHEETS:
        sheet = workbook.create_sheet(name)
        style = metadata["sheets"][name]
        sheet.append(
            [
                "编号",
                name + "编号",
                "调查日期",
                "地点编号",
                "地点名称",
                "调查道路",
                "运行方向",
                "时段",
                "小客车",
                "出租车",
                "公交车",
                "",
                "大货车",
                "小货车",
                "大客车\n(非公交)",
                "摩托车",
                "自行车",
                "",
                "",
                "三轮车",
                "",
            ]
        )
        sheet.append(
            [None] * 10
            + ["单机", "铰接"]
            + [None] * 4
            + [
                "电动",
                "普通",
                "外卖电动自行车" if period == 300 else "外卖电动",
                "电动",
                "快递货运",
            ]
        )
        first = start
        finish = start + math.ceil((end - start) / period) * period
        # Column B is the grouping key consumed by the next time-aggregation sheet.
        grouping = {300: 900, 900: 1800, 1800: 3600, 3600: 3600}[period]
        cached = workbook._traffic_formula_cache.setdefault(name, {})
        for number, slot in enumerate(range(first, finish, period), 1):
            when = midnight + timedelta(seconds=slot)
            until = midnight + timedelta(seconds=slot + period)
            seconds = covered_seconds(intervals, slot, slot + period)
            values = [
                sum(
                    buckets.get(str(s), {}).get(category, 0)
                    for s in range(slot, slot + period, 300)
                )
                for category in CATEGORIES
            ]
            sheet.append(
                [
                    number,
                    math.floor((slot - start) / grouping) + 1,
                    int(when.strftime("%Y%m%d")),
                    str(config.get("spot_code", "")),
                    location_with_cards(
                        config["road_name"], files, slot, slot + period
                    ),
                    "断面" if name.startswith("断面") else role,
                    direction,
                    when.strftime("%H:%M") + "-" + until.strftime("%H:%M"),
                    *[v if seconds > 0 else None for v in values],
                ]
            )
            row = sheet.max_row
            if source_name is not None and seconds > 0:
                source = workbook[source_name]
                for column, value in enumerate(values, 9):
                    letter = get_column_letter(column)
                    formula = (
                        f"='{source_name}'!{letter}{row}"
                        if name == "断面15分钟"
                        else f"=SUMIF('{source_name}'!$B$3:$B${source.max_row},$A{row},'{source_name}'!{letter}$3:{letter}${source.max_row})"
                    )
                    sheet.cell(row, column, formula)
                    cached[f"{letter}{row}"] = value
            if seconds < period - 0.05:
                sheet.cell(row, 8).comment = Comment(
                    f"不完整时段：有效录像覆盖 {seconds:.2f}/{period} 秒。仅统计已覆盖部分，不外推；无录像留空，不作为零流量。",
                    "道路调查",
                )
            for column in range(1, 9):
                cell = sheet.cell(row, column)
                if isinstance(cell.value, str):
                    cell.data_type = "s"
            if period == 300:
                fill = PatternFill("solid", fgColor=COLORS[((number - 1) // 3) % 4])
                for column in range(9, 22):
                    sheet.cell(row, column).fill = fill
            display_width = sum(
                2 if ord(char) > 127 else 1 for char in str(sheet.cell(row, 5).value)
            )
            sheet.row_dimensions[row].height = max(
                15, math.ceil(display_width / style["widths"][4]) * 15
            )
        for row in sheet:
            for cell in row:
                header = cell.row <= 2
                cell.font = Font(
                    name="宋体",
                    size=style["header_font_sizes"][cell.row - 1][cell.column - 1]
                    if header
                    else (12 if period == 300 and cell.column >= 9 else 11),
                )
                cell.alignment = Alignment(
                    horizontal="center", vertical="center", wrap_text=True
                )
                cell.border = Border(
                    left=border, right=border, top=border, bottom=border
                )
                if not header and cell.column >= 9:
                    cell.number_format = "0"
        for column in list(range(1, 11)) + list(range(13, 17)):
            sheet.merge_cells(
                start_row=1, start_column=column, end_row=2, end_column=column
            )
        for a, b in [(11, 12), (17, 19), (20, 21)]:
            sheet.merge_cells(start_row=1, start_column=a, end_row=1, end_column=b)
        for column, width in enumerate(style["widths"], 1):
            sheet.column_dimensions[get_column_letter(column)].width = width
        for row, height in enumerate(style["header_heights"], 1):
            sheet.row_dimensions[row].height = height
        if name.startswith("断面"):
            sheet["F1"].comment = Comment(
                f"本表仅为{direction}{role}的断面合计，已包含该区域各车道，不合并其他方向或另一主辅路。",
                "道路调查",
            )
        sheet.freeze_panes = "I3"
        sheet.print_title_rows = "1:2"
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_options.horizontalCentered = True
    return workbook


def save_cached_workbook(workbook, target=None):
    """Keep formulas editable and supply computed cached totals for previews/readers."""
    raw = io.BytesIO()
    workbook.save(raw)
    output = io.BytesIO()
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ET.register_namespace("", namespace)
    ET.register_namespace(
        "r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    with (
        zipfile.ZipFile(raw) as source,
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as destination,
    ):
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename.startswith(
                "xl/worksheets/sheet"
            ) and item.filename.endswith(".xml"):
                index = int(item.filename.rsplit("sheet", 1)[1][:-4]) - 1
                values = getattr(workbook, "_traffic_formula_cache", {}).get(
                    workbook.worksheets[index].title, {}
                )
                if values:
                    root = ET.fromstring(content)
                    for cell in root.iter(f"{{{namespace}}}c"):
                        if cell.attrib["r"] in values:
                            value = cell.find(f"{{{namespace}}}v")
                            if value is None:
                                value = ET.SubElement(cell, f"{{{namespace}}}v")
                            value.text = str(values[cell.attrib["r"]])
                    content = ET.tostring(root, encoding="utf-8")
            destination.writestr(item, content)
    output.seek(0)
    if target is not None:
        Path(target).write_bytes(output.getvalue())
    return output
