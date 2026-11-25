"""Read an existing review audit and export a workbook without rerunning inference."""

import io
import json
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from services.road_survey_service import (
    CATEGORIES,
    ROAD_TYPES,
    covered_seconds,
    export_workbooks,
    safe_name,
)


def build_road_review_zip(result, snapshot):
    """Use the formal exporter for all sampled directions, retaining audit data."""
    completed = {}
    manifest = {"streams": []}
    buffer = io.BytesIO()
    with (
        tempfile.TemporaryDirectory(prefix="road-review-") as directory,
        zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        for index, part in enumerate(snapshot["parts"]):
            key = part["stream"]["key"]
            output = next(p["result"] for p in result["parts"] if p["key"] == key)
            audit = json.loads(Path(output["audit_path"]).read_text(encoding="utf-8"))
            events = [
                e for e in audit["events"] if part["start"] <= e["second"] < part["end"]
            ]
            buckets = {}
            for event in events:
                if event.get("category") not in CATEGORIES:
                    continue
                values = buckets.setdefault(event["road_type"], {}).setdefault(
                    str(int(event["second"] // 300) * 300), {}
                )
                category = event["category"]
                values[category] = values.get(category, 0) + 1
            completed[key] = {
                "direction": part["stream"]["direction"],
                "buckets": buckets,
                "coverage": [
                    {
                        "start": max(a, part["start"]),
                        "end": min(b, part["end"]),
                        "road_type": role,
                    }
                    for a, b in audit["coverage"]
                    for role in part["source"]["regions"]
                    if a < part["end"] and b > part["start"]
                ],
            }
            manifest["streams"].append(
                {"files": [{"start": part["start"], "end": part["end"]}]}
            )
            prefix = safe_name(
                f"{index + 1}_{part['stream']['direction']}_{part['stream'].get('name', '机位')}"
            )
            archive.writestr(
                f"计数明细/{prefix}.jsonl",
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
            )
            archive.writestr(
                f"抽查明细/{prefix}.xlsx",
                build_review_excel(audit, {**snapshot, **part}).getvalue(),
            )
        exports = export_workbooks(snapshot["config"], manifest, completed, directory)
        for item in exports:
            archive.write(item["path"], arcname=item["name"])
        archive.writestr(
            "调查说明.json",
            json.dumps(
                {
                    "type": "整路全方向5分钟抽查样板",
                    "snapshot": snapshot,
                    "exports": [
                        {k: v for k, v in e.items() if k != "path"} for e in exports
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "抽查说明.txt",
            "各方向使用同一个固定5分钟时段，主路、辅路分别出表。\n本包为独立抽查复算，不替代整路正式统计。录像缺失标黄，未配置区域留空，不外推补齐。\n抽查明细目录保留各机位全部原始车型和逐车记录；带框视频在回放页面按机位下载。\n",
        )
    buffer.seek(0)
    return buffer


def build_review_zip(audit, snapshot):
    """Package the sampled direction in the same layout as a completed road."""
    report = build_review_excel(audit, snapshot).getvalue()
    config = audit.get("config") or snapshot["config"]
    stream = audit.get("stream") or snapshot["stream"]
    prefix = safe_name(f"{config['road_name']}_{stream['direction']}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for role in ROAD_TYPES:
            workbook = load_workbook(io.BytesIO(report))
            for sheet in list(workbook):
                if sheet.title != "5分钟":
                    workbook.remove(sheet)
            sheet = workbook.active
            for row in range(sheet.max_row, 2, -1):
                if sheet.cell(row, 6).value != role:
                    sheet.delete_rows(row)
            for number, row in enumerate(range(3, sheet.max_row + 1), 1):
                sheet.cell(row, 1, number)
            output = io.BytesIO()
            workbook.save(output)
            archive.writestr(
                safe_name(f"{prefix}_{role}_{config['survey_date']}_抽查") + ".xlsx",
                output.getvalue(),
            )
        archive.writestr("抽查汇总与逐车明细.xlsx", report)
        archive.writestr(
            "计数明细/抽查计数明细.json",
            json.dumps(audit, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "调查说明.json",
            json.dumps(
                {
                    "type": "抽查样板",
                    "snapshot": snapshot,
                    "start": audit["start"],
                    "end": audit["end"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "抽查说明.txt",
            "本包为所选机位、方向及时间范围的抽查样板，不是整条道路的正式结果。\n"
            "主路、辅路模板表各一份，每份只有5分钟Sheet。未抽查的其他方向不生成表格。\n"
            "未配置区域或无录像的时段留空；不足5分钟的时段标黄，不外推补齐。\n"
            "汇总与逐车明细保留全部原始车型；正式模板仅计入映射车型。\n"
            "带框视频可在抽查回放页面单独下载。\n",
        )
    buffer.seek(0)
    return buffer


def append_values(sheet, values):
    sheet.append(values)
    # User-provided names and filenames are literal report values, never formulas.
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str):
            cell.data_type = "s"


def style_sheet(sheet, header_rows=1):
    edge = Side(style="thin", color="CBD5E1")
    for row in sheet:
        for cell in row:
            cell.border = Border(left=edge, right=edge, top=edge, bottom=edge)
            cell.alignment = Alignment(
                horizontal="center", vertical="center", wrap_text=True
            )
            if cell.row <= header_rows:
                cell.fill = PatternFill("solid", fgColor="D9E1F2")
                cell.font = Font(bold=True, color="17365D")
    sheet.freeze_panes = f"A{header_rows + 1}"
    sheet.print_title_rows = f"1:{header_rows}"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    for col in range(1, sheet.max_column + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 18


def build_review_excel(audit, snapshot):
    config = audit.get("config") or snapshot["config"]
    stream = audit.get("stream") or snapshot["stream"]
    source = snapshot.get("source", {})
    regions = source.get("regions", {})
    start, end = float(audit["start"]), float(audit["end"])
    midnight = datetime.fromisoformat(config["survey_date"])
    absolute = lambda second: midnight + timedelta(seconds=second)
    stamp = lambda second: absolute(second).strftime("%Y-%m-%d %H:%M:%S")
    slot_name = lambda slot: (
        absolute(slot).strftime("%H:%M") + "-" + absolute(slot + 300).strftime("%H:%M")
    )
    events = sorted(
        (e for e in audit.get("events", []) if start <= e["second"] < end),
        key=lambda e: e["second"],
    )
    coverage = [
        (max(start, float(a)), min(end, float(b)))
        for a, b in audit.get("coverage", [])
        if a < end and b > start
    ]
    original = Counter((e["road_type"], e["class_name"]) for e in events)
    template = Counter(
        (int(e["second"] // 300) * 300, e["road_type"], e["category"])
        for e in events
        if e.get("category") in CATEGORIES
    )
    mapping = config.get("class_mapping", {})
    wb = Workbook()
    sheet = wb.active
    sheet.title = "5分钟"
    append_values(
        sheet,
        [
            "编号",
            "5分钟编号",
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
        ],
    )
    append_values(
        sheet,
        [None] * 9
        + ["出租车", "单机", "铰接"]
        + [None] * 4
        + ["电动", "普通", "外卖电动自行车", "电动", "快递货运"],
    )
    for col in list(range(1, 11)) + list(range(13, 17)):
        sheet.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)
    for a, b in [(11, 12), (17, 19), (20, 21)]:
        sheet.merge_cells(start_row=1, start_column=a, end_row=1, end_column=b)
    index = 0
    for number, slot in enumerate(
        range(int(start // 300) * 300, int((end - 1e-7) // 300) * 300 + 1, 300), 1
    ):
        for role in ROAD_TYPES:
            index += 1
            seconds = (
                covered_seconds(coverage, slot, slot + 300) if role in regions else 0
            )
            append_values(
                sheet,
                [
                    index,
                    number,
                    int(absolute(slot).strftime("%Y%m%d")),
                    str(config.get("spot_code", "")),
                    config["road_name"],
                    role,
                    stream["direction"],
                    slot_name(slot),
                    *[
                        template[(slot, role, category)] if seconds > 0 else None
                        for category in CATEGORIES
                    ],
                ],
            )
            if seconds < 299.95:
                note = (
                    f"抽查窗口：{stamp(start)} 至 {stamp(end)}；本时段录像覆盖 {seconds:.2f}/300 秒。"
                    "数值仅为已覆盖部分的独立复算，不代表完整5分钟流量，不外推补齐。"
                )
                if role not in regions:
                    note += "此机位未配置该主辅路区域，留空，不作为零流量。"
                sheet.cell(sheet.max_row, 8).comment = Comment(note, "道路调查")
                for cell in sheet[sheet.max_row]:
                    cell.fill = PatternFill("solid", fgColor="FFF2CC")
    style_sheet(sheet, 2)
    for col in range(1, 22):
        sheet.column_dimensions[get_column_letter(col)].width = (
            10 if col >= 9 else [8, 12, 13, 12, 28, 12, 12, 18][col - 1]
        )
    sheet.row_dimensions[1].height = 32
    sheet.freeze_panes = "I3"

    summary = wb.create_sheet("车型汇总")
    append_values(summary, ["模型原始车型", "模板对应", "主路", "辅路", "合计", "说明"])
    observed = {e["class_name"] for e in events}
    labels = list(
        dict.fromkeys(
            n
            for n in [*CATEGORIES, *mapping, *sorted(observed)]
            if n in mapping or n in observed
        )
    )
    for name in labels:
        main, side = original[("主路", name)], original[("辅路", name)]
        target = mapping.get(name, "")
        append_values(
            summary,
            [
                name,
                target,
                main,
                side,
                main + side,
                "保留抽查计数，未计入5分钟模板"
                if target not in CATEGORIES
                else "已按对应车型填入5分钟Sheet",
            ],
        )
    for title, predicate in [
        ("全部识别计数", lambda e: True),
        ("纳入模板计数", lambda e: e.get("category") in CATEGORIES),
        ("不纳入模板计数", lambda e: e.get("category") not in CATEGORIES),
    ]:
        totals = [
            sum(1 for e in events if e["road_type"] == role and predicate(e))
            for role in ROAD_TYPES
        ]
        append_values(
            summary, [title, "", *totals, sum(totals), "仅为所选抽查片段的计数"]
        )
    style_sheet(summary)
    summary.column_dimensions["A"].width = 22
    summary.column_dimensions["B"].width = 22
    summary.column_dimensions["F"].width = 40
    for row in range(summary.max_row - 2, summary.max_row + 1):
        for cell in summary[row]:
            cell.font = Font(bold=True)

    details = wb.create_sheet("逐车明细")
    append_values(
        details,
        [
            "序号",
            "记录时间",
            "5分钟时段",
            "主辅路",
            "追踪ID",
            "原始车型",
            "模板车型",
            "纳入模板",
            "置信度",
            "来源视频",
        ],
    )
    for i, event in enumerate(events, 1):
        when = absolute(event["second"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        append_values(
            details,
            [
                i,
                when,
                slot_name(int(event["second"] // 300) * 300),
                event["road_type"],
                event["track_id"],
                event["class_name"],
                event.get("category", ""),
                "是" if event.get("category") in CATEGORIES else "否",
                float(event.get("confidence", 0)),
                Path(event.get("file", "")).name,
            ],
        )
        details.cell(details.max_row, 9).number_format = "0.000"
    style_sheet(details)
    details.column_dimensions["B"].width = 27
    details.column_dimensions["J"].width = 40
    details.auto_filter.ref = details.dimensions

    notes = wb.create_sheet("抽查说明")
    append_values(notes, ["字段", "内容"])
    for key, value in [
        ("用途", "独立抽查复算，用于对照带框回放；不覆盖或替代正式统计"),
        ("道路", config["road_name"]),
        ("地点编号", str(config.get("spot_code", ""))),
        ("调查日期", config["survey_date"]),
        ("方向", stream["direction"]),
        ("机位", stream.get("name", "")),
        ("抽查开始", stamp(start)),
        ("抽查结束", stamp(end)),
        ("抽查时长（秒）", round(end - start, 3)),
        ("录像覆盖（秒）", round(covered_seconds(coverage, start, end), 3)),
        (
            "停录或无录像（秒）",
            round(end - start - covered_seconds(coverage, start, end), 3),
        ),
        ("模型", config.get("system_model", "best.pt")),
        ("推理分辨率", str(config.get("inference_resolution", "640"))),
        (
            "5分钟Sheet口径",
            "按固定5分钟时段分桶；主路、辅路分别列行，只统计映射到模板车型的事件",
        ),
        (
            "不完整时段",
            "黄色行与时段批注记录覆盖情况；未配置区域或完全无录像留空，不视为零流量",
        ),
        ("车型汇总口径", "保留全部原始车型，明确列出模板对应关系及未纳入模板的数量"),
        (
            "边界差异",
            "抽查预热最多10秒追踪；片段边界计数可能与完整道路连续推理略有差异",
        ),
    ]:
        append_values(notes, [key, value])
    style_sheet(notes)
    notes.column_dimensions["A"].width = 25
    notes.column_dimensions["B"].width = 100
    for row in range(2, notes.max_row + 1):
        notes.cell(row, 2).alignment = Alignment(
            horizontal="left", vertical="center", wrap_text=True
        )
        notes.row_dimensions[row].height = 32
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer
