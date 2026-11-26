"""Persistent road-survey jobs, kept separate from the legacy inference API."""

import io
import json
import os
import socket
import threading
import time
import uuid
import zipfile
from pathlib import Path

import cv2
from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)

from models import (
    Label,
    Project,
    ReasoningMaterial,
    RoadSurveyReview,
    RoadSurveyTask,
    db,
)
from services import road_survey_service as service
from services.road_survey_cameras import aggregate_progress, run_cameras
from services.road_survey_live import finish_task, frames_for, render_preview
from services.road_survey_scheduler import get_scheduler

road_survey = Blueprint("road_survey", __name__)
_jobs = {}
_review_jobs = {}
_futures = {}
_review_futures = {}
_lock = threading.Lock()
BUSY = {"queued", "running", "scan_queued", "scanning", "stopping"}


def task_for(project_id, task_id):
    task = RoadSurveyTask.query.filter_by(
        id=task_id, project_id=project_id
    ).first_or_404()
    if task.owner_host != socket.gethostname():
        abort(409, description="该任务的视频属于另一台机器，请在创建任务的机器上操作")
    if task.status in BUSY and task.worker_pid != os.getpid():
        # The old process cannot be resumed at a frame boundary. Completed streams remain usable.
        import psutil

        if not task.worker_pid or not psutil.pid_exists(task.worker_pid):
            task.status = "interrupted"
            task.error = (
                "服务重启或工作进程退出；可从最近保存的帧断点恢复，已完成机位保留。"
            )
            db.session.commit()
    return task


def serialize(task, detail=False):
    data = {
        "id": task.id,
        "name": task.name,
        "status": task.status,
        "error": task.error,
        "owner_host": task.owner_host,
        "progress": task.progress or {},
        "updated_at": task.updated_at.isoformat(),
        "result": task.result or {},
    }
    # Filesystem paths in audit/checkpoints are not rendered as user-facing implementation details.
    if detail:
        data.update(config=task.config, manifest=task.manifest)
    return data


@road_survey.errorhandler(ValueError)
@road_survey.errorhandler(OSError)
def bad_config(error):
    return jsonify(success=False, message=str(error)), 400


@road_survey.get("/project/<int:project_id>/road-surveys")
def page(project_id):
    project = Project.query.get_or_404(project_id)
    labels = Label.query.filter_by(project_id=project_id).order_by(Label.id).all()
    return render_template(
        "road_surveys.html",
        project=project,
        categories=service.CATEGORIES,
        labels=[l.name for l in labels],
        exclude=service.EXCLUDE,
    )


@road_survey.get("/api/project/<int:project_id>/road-surveys")
def list_tasks(project_id):
    Project.query.get_or_404(project_id)
    tasks = (
        RoadSurveyTask.query.filter_by(
            project_id=project_id, owner_host=socket.gethostname()
        )
        .order_by(RoadSurveyTask.id.desc())
        .all()
    )
    for task in tasks:
        task_for(project_id, task.id)
    return jsonify(tasks=[serialize(task) for task in tasks])


@road_survey.get("/api/project/<int:project_id>/road-surveys/runtime")
def runtime(project_id):
    Project.query.get_or_404(project_id)
    return jsonify(scheduler=get_scheduler().snapshot())


@road_survey.post("/api/project/<int:project_id>/road-surveys/discover")
def discover(project_id):
    Project.query.get_or_404(project_id)
    return jsonify(
        roads=service.discover_roads((request.get_json() or {}).get("root", ""))
    )


@road_survey.get("/api/project/<int:project_id>/road-surveys/find")
def find_road(project_id):
    Project.query.get_or_404(project_id)
    tasks = (
        RoadSurveyTask.query.filter_by(
            project_id=project_id, owner_host=socket.gethostname()
        )
        .order_by(RoadSurveyTask.id.desc())
        .all()
    )
    return jsonify(
        tasks=[
            serialize(task)
            for task in tasks
            if service.matches_road(
                task.config,
                request.args.get("road_root", ""),
                request.args.get("road_name", ""),
            )
        ]
    )


@road_survey.get("/api/project/<int:project_id>/road-surveys/material-regions")
def material_regions(project_id):
    Project.query.get_or_404(project_id)
    materials = (
        ReasoningMaterial.query.filter(
            ReasoningMaterial.project_id == project_id,
            ReasoningMaterial.material_type == 2,
            ReasoningMaterial.config.isnot(None),
        )
        .order_by(ReasoningMaterial.id.desc())
        .all()
    )
    choices = []
    for material in materials:
        config = material.config or {}
        labels = {str(label["id"]): label["name"] for label in config.get("labels", [])}
        for index, annotation in enumerate(config.get("annotations", [])):
            points = annotation.get("points", [])
            if len(points) < 3:
                continue
            choices.append(
                {
                    "material_id": material.id,
                    "filename": material.filename,
                    "annotation_index": index,
                    "label": labels.get(str(annotation.get("label_id")), "未命名区域"),
                    "polygon": [[p["x"], p["y"]] for p in points],
                }
            )
    return jsonify(regions=choices)


@road_survey.post("/api/project/<int:project_id>/road-surveys")
def create_task(project_id):
    Project.query.get_or_404(project_id)
    config = service.validate_config(request.get_json())
    task = RoadSurveyTask(
        project_id=project_id,
        name=config["road_name"],
        config=config,
        status="draft",
        owner_host=socket.gethostname(),
    )
    db.session.add(task)
    db.session.commit()
    return jsonify(task=serialize(task, True)), 201


@road_survey.route(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>", methods=["GET", "PUT"]
)
def detail(project_id, task_id):
    task = task_for(project_id, task_id)
    if request.method == "PUT":
        if task.status in BUSY:
            return jsonify(message="任务运行中，请先停止后修改"), 409
        config = service.validate_config(request.get_json())
        if task.config != config:
            if service.config_signature(task.config) != service.config_signature(
                config
            ):
                task.manifest = None
            task.result = None
            task.progress = None
            task.error = None
            task.config = config
            task.name = config["road_name"]
            task.status = "ready" if task.manifest else "draft"
            db.session.commit()
    return jsonify(task=serialize(task, True))


def schedule(task, mode, resume=False):
    with _lock:
        # Row lock prevents two requests (including different processes) claiming the same task.
        task = RoadSurveyTask.query.filter_by(id=task.id).with_for_update().one()
        if task.status in BUSY:
            db.session.rollback()
            return jsonify(message="任务已在队列或运行中"), 409
        stop = threading.Event()
        _jobs[task.id] = stop
        task.status = "scan_queued" if mode == "scan" else "queued"
        task.worker_pid = os.getpid()
        task.error = None
        if mode == "scan" or not resume:
            task.result = None
        task.progress = {"phase": task.status}
        db.session.commit()
        app = current_app._get_current_object()
        try:
            _futures[task.id] = get_scheduler().submit(
                "scan" if mode == "scan" else "control",
                worker,
                app,
                task.id,
                mode,
                stop,
                resume,
            )
        except Exception:
            _jobs.pop(task.id, None)
            task.status = "failed"
            task.error = "任务队列提交失败，请重试"
            db.session.commit()
            raise
    return jsonify(task=serialize(task)), 202


@road_survey.post("/api/project/<int:project_id>/road-surveys/<int:task_id>/scan")
def scan(project_id, task_id):
    return schedule(task_for(project_id, task_id), "scan")


@road_survey.post("/api/project/<int:project_id>/road-surveys/<int:task_id>/start")
def start(project_id, task_id):
    task = task_for(project_id, task_id)
    payload = request.get_json() or {}
    if payload.get("timeline_confirmed") is not True:
        raise ValueError("请确认视频顺序、起始时间和换卡间隔")
    service.validate_config(task.config, require_regions=True)
    service.verify_manifest(task.config, task.manifest)
    service.validate_camera_coverage(task.config, task.manifest)
    return schedule(task, "run", resume=bool(payload.get("resume")))


@road_survey.post("/api/project/<int:project_id>/road-surveys/<int:task_id>/stop")
def stop(project_id, task_id):
    task = task_for(project_id, task_id)
    with _lock:
        event = _jobs.get(task_id)
        if event:
            event.set()
            future = _futures.get(task_id)
            if future is not None and future.cancel():
                _futures.pop(task_id, None)
                _jobs.pop(task_id, None)
                task.status = "stopped"
                task.error = "已取消排队任务"
                finish_task(task_id)
            else:
                task.status = "stopping"
            db.session.commit()
    return jsonify(task=serialize(task))


@road_survey.get("/api/project/<int:project_id>/road-surveys/<int:task_id>/frame")
def frame(project_id, task_id):
    task = task_for(project_id, task_id)
    if not task.manifest:
        raise ValueError("请先扫描视频目录")
    stream = next(
        (s for s in task.manifest["streams"] if s["key"] == request.args.get("stream")),
        None,
    )
    if stream is None:
        abort(404)
    index = int(request.args.get("file", "0"))
    if not 0 <= index < len(stream["files"]):
        abort(404)
    item = stream["files"][index]
    cap = cv2.VideoCapture(item["path"])
    try:
        seconds = min(
            service.finite_number(request.args.get("second", 0)),
            max(0, item["duration"] - 1),
        )
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, image = cap.read()
        if not ok:
            raise ValueError("无法抽取视频画面")
        ratio = min(1, 1200 / image.shape[1])
        if ratio < 1:
            image = cv2.resize(image, None, fx=ratio, fy=ratio)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("画面编码失败")
        return send_file(
            io.BytesIO(encoded.tobytes()), mimetype="image/jpeg", max_age=0
        )
    finally:
        cap.release()


@road_survey.get("/api/project/<int:project_id>/road-surveys/<int:task_id>/download")
def download(project_id, task_id):
    task = task_for(project_id, task_id)
    if task.status != "finished":
        raise ValueError("任务尚未完成")
    result = task.result or {}
    if result.get("export_version") != service.EXPORT_VERSION:
        # Refresh report formatting for existing finished tasks without rerunning inference.
        with _lock:
            output_dir = Path(current_app.instance_path) / "road_surveys" / str(task_id)
            result = {
                **result,
                "exports": service.export_workbooks(
                    task.config, task.manifest, result.get("completed", {}), output_dir
                ),
                "export_version": service.EXPORT_VERSION,
            }
            task.result = result
            db.session.commit()
    exports = result.get("exports", [])
    index = request.args.get("index", type=int)
    if index is not None:
        if not 0 <= index < len(exports):
            abort(404)
        return send_file(
            exports[index]["path"],
            as_attachment=True,
            download_name=exports[index]["name"],
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in exports:
            archive.write(item["path"], arcname=item["name"])
        for value in result.get("completed", {}).values():
            archive.write(
                value["audit_path"],
                arcname="计数明细/" + Path(value["audit_path"]).name,
            )
        archive.writestr(
            "调查说明.json",
            json.dumps(
                {
                    "config": task.config,
                    "manifest": task.manifest,
                    "exports": [
                        {k: v for k, v in e.items() if k != "path"} for e in exports
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=service.safe_name(task.name) + "_调查结果.zip",
        mimetype="application/zip",
    )


@road_survey.get("/api/project/<int:project_id>/road-surveys/<int:task_id>/live")
def live_stream(project_id, task_id):
    task = task_for(project_id, task_id)
    streams = (task.manifest or {}).get("streams", [])
    key = request.args.get("stream") or (streams[0]["key"] if streams else None)
    if key is not None and not any(s["key"] == key for s in streams):
        abort(404)
    return Response(
        frames_for(task_id, key).stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


def get_review(project_id, task_id, review_id):
    task_for(project_id, task_id)
    review = RoadSurveyReview.query.filter_by(
        id=review_id, task_id=task_id
    ).first_or_404()
    if review.status in BUSY and review.worker_pid != os.getpid():
        import psutil

        if not review.worker_pid or not psutil.pid_exists(review.worker_pid):
            review.status = "failed"
            review.error = "服务重启导致抽查中断，请重新生成。"
            db.session.commit()
    return review


def review_data(review):
    result = {k: v for k, v in (review.result or {}).items() if not k.endswith("_path")}
    return {
        "id": review.id,
        "status": review.status,
        "error": review.error,
        "progress": review.progress or {},
        "result": result,
        "direction": review.snapshot["stream"]["direction"],
        "stream_name": review.snapshot["stream"].get("name", "机位"),
        "start": review.snapshot["start"],
        "end": review.snapshot["end"],
        "mapping": review.snapshot["config"].get("class_mapping", {}),
        "whole_road": bool(review.snapshot.get("parts")),
        "parts": [
            {
                "key": p["stream"]["key"],
                "direction": p["stream"]["direction"],
                "name": p["stream"].get("name", "机位"),
            }
            for p in review.snapshot.get("parts", [])
        ],
    }


@road_survey.route(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews",
    methods=["GET", "POST"],
)
def reviews(project_id, task_id):
    task = task_for(project_id, task_id)
    if request.method == "GET":
        items = (
            RoadSurveyReview.query.filter_by(task_id=task_id)
            .order_by(RoadSurveyReview.created_at.desc())
            .limit(20)
            .all()
        )
        return jsonify(
            reviews=[review_data(get_review(project_id, task_id, r.id)) for r in items]
        )
    payload = request.get_json() or {}
    if not task.manifest:
        raise ValueError("请先扫描并保存视频时间表")
    stream = next(
        (s for s in task.manifest["streams"] if s["key"] == payload.get("stream")), None
    )
    if stream is None:
        raise ValueError("请选择要抽查的方向和机位")
    config = json.loads(json.dumps(task.config))
    source = config["directions"][stream["direction_index"]]["streams"][
        stream["stream_index"]
    ]
    check_config = {
        **config,
        "directions": [{"name": stream["direction"], "streams": [source]}],
    }
    service.validate_config(check_config, require_regions=True)
    service.verify_manifest(config, task.manifest)
    from services.road_survey_review import review_window

    start, end = review_window(stream, payload.get("start_second"), 300)
    parts = []
    if payload.get("whole_road"):
        service.validate_config(config, require_regions=True)
        start = service.finite_number(payload.get("start_second"), 0, 7 * 86400)
        if start % 300:
            raise ValueError("整路抽查请选择整5分钟起点，例如07:00、07:05")
        end = start + 300
        for item in task.manifest["streams"]:
            a, b = review_window(item, start, 300)
            if b < end:
                raise ValueError(
                    f"{item['direction']} / {item.get('name', '机位')}不足5分钟，请调整起始时间"
                )
            source_item = config["directions"][item["direction_index"]]["streams"][
                item["stream_index"]
            ]
            parts.append({"stream": item, "source": source_item, "start": a, "end": b})
    with _lock:
        RoadSurveyTask.query.filter_by(id=task_id).with_for_update().one()
        if RoadSurveyReview.query.filter(
            RoadSurveyReview.task_id == task_id, RoadSurveyReview.status.in_(BUSY)
        ).first():
            db.session.rollback()
            return jsonify(message="该道路已有抽查任务，请等待完成或先取消"), 409
        review = RoadSurveyReview(
            id=uuid.uuid4().hex,
            task_id=task_id,
            status="queued",
            worker_pid=os.getpid(),
            snapshot={
                "config": config,
                "stream": stream,
                "source": source,
                "start": start,
                "end": end,
                "parts": parts,
            },
        )
        db.session.add(review)
        db.session.commit()
        stop = threading.Event()
        _review_jobs[review.id] = stop
        try:
            _review_futures[review.id] = get_scheduler().submit(
                "compute",
                review_worker,
                current_app._get_current_object(),
                review.id,
                stop,
            )
        except Exception:
            _review_jobs.pop(review.id, None)
            review.status = "failed"
            review.error = "抽查任务提交失败"
            db.session.commit()
            raise
    return jsonify(review=review_data(review)), 202


@road_survey.get(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews/<review_id>"
)
def review_status(project_id, task_id, review_id):
    return jsonify(review=review_data(get_review(project_id, task_id, review_id)))


@road_survey.post(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews/<review_id>/stop"
)
def stop_review(project_id, task_id, review_id):
    review = get_review(project_id, task_id, review_id)
    with _lock:
        event = _review_jobs.get(review.id)
        if event:
            event.set()
            future = _review_futures.get(review.id)
            if future is not None and future.cancel():
                _review_futures.pop(review.id, None)
                _review_jobs.pop(review.id, None)
                review.status = "stopped"
                review.error = "已取消排队中的抽查"
            else:
                review.status = "stopping"
            db.session.commit()
    return jsonify(review=review_data(review))


@road_survey.get(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews/<review_id>/video"
)
def review_video(project_id, task_id, review_id):
    review = get_review(project_id, task_id, review_id)
    if review.status != "finished":
        raise ValueError("抽查视频尚未生成完成")
    result = review.result
    if request.args.get("part"):
        result = next(
            (
                p["result"]
                for p in result.get("parts", [])
                if p["key"] == request.args["part"]
            ),
            None,
        )
        if result is None:
            abort(404)
    return send_file(
        result["video_path"],
        mimetype="video/mp4",
        conditional=True,
        as_attachment=request.args.get("download") == "1",
        download_name="抽查回放_" + review.id[:8] + ".mp4",
    )


@road_survey.get(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews/<review_id>/audit"
)
def review_audit(project_id, task_id, review_id):
    review = get_review(project_id, task_id, review_id)
    if review.status != "finished":
        raise ValueError("抽查明细尚未生成完成")
    return send_file(
        review.result["audit_path"],
        as_attachment=True,
        download_name="抽查计数明细.json",
    )


@road_survey.get(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews/<review_id>/excel"
)
@road_survey.get(
    "/api/project/<int:project_id>/road-surveys/<int:task_id>/reviews/<review_id>/download"
)
def review_excel(project_id, task_id, review_id):
    review = get_review(project_id, task_id, review_id)
    if review.status != "finished":
        raise ValueError("抽查尚未完成，暂不能导出Excel")
    from services.road_survey_review_excel import (
        build_review_excel,
        build_review_zip,
        build_road_review_zip,
    )

    audit = json.loads(Path(review.result["audit_path"]).read_text(encoding="utf-8"))
    is_zip = request.path.endswith("/download")
    if is_zip and review.snapshot.get("parts"):
        return send_file(
            build_road_review_zip(review.result, review.snapshot),
            as_attachment=True,
            download_name=service.safe_name(review.snapshot["config"]["road_name"])
            + "_全方向5分钟抽查样板.zip",
            mimetype="application/zip",
        )
    workbook = (build_review_zip if is_zip else build_review_excel)(
        audit, review.snapshot
    )
    name = service.safe_name(
        review.snapshot["config"]["road_name"]
        + "_"
        + review.snapshot["stream"]["direction"]
        + "_抽查_"
        + review.id[:8]
    ) + ("_样板结果.zip" if is_zip else ".xlsx")
    return send_file(
        workbook,
        as_attachment=True,
        download_name=name,
        mimetype="application/zip"
        if is_zip
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def review_worker(app, review_id, stop):
    with app.app_context():
        review = db.session.get(RoadSurveyReview, review_id)
        last_update = [0.0]

        def check_cancel():
            if stop.is_set():
                raise service.Cancelled("抽查生成已取消，不影响正式统计。")

        def progress(value):
            if time.monotonic() - last_update[0] >= 1:
                review.progress = json.loads(json.dumps(value))
                db.session.commit()
                last_update[0] = time.monotonic()

        try:
            check_cancel()
            review.status = "running"
            db.session.commit()
            from inference_manager import InferenceManager
            from services.road_survey_review import generate_review

            snapshot = review.snapshot
            config = snapshot["config"]
            task = db.session.get(RoadSurveyTask, review.task_id)
            model = InferenceManager(task.project_id).load_model(
                "system", config.get("system_model", "best.pt"), prefer_engine=False
            )
            missing = [
                name
                for name in model.names.values()
                if not config.get("class_mapping", {}).get(name)
            ]
            if missing:
                raise ValueError("请先保存车型对应：" + "、".join(missing))
            path = (
                Path(app.instance_path)
                / "road_surveys"
                / str(task.id)
                / "reviews"
                / (review.id + ".mp4")
            )
            parts = snapshot.get("parts") or [snapshot]
            outputs = []
            for index, part in enumerate(parts):
                check_cancel()
                part_path = (
                    path
                    if index == 0
                    else path.with_name(path.stem + f"_{index}" + path.suffix)
                )
                result = generate_review(
                    model,
                    config,
                    part["stream"],
                    part["source"],
                    part["start"],
                    part["end"],
                    part_path,
                    lambda value, i=index: progress(
                        {**value, "part": i + 1, "parts": len(parts)}
                    ),
                    check_cancel,
                )
                outputs.append({"key": part["stream"]["key"], "result": result})
            review.result = (
                {**outputs[0]["result"], "parts": outputs}
                if snapshot.get("parts")
                else outputs[0]["result"]
            )
            review.status = "finished"
            review.error = None
            review.progress = {
                "phase": "finished",
                "seconds": review.result["duration"],
                "duration": review.result["duration"],
            }
            db.session.commit()
        except Exception as error:
            db.session.rollback()
            review = db.session.get(RoadSurveyReview, review_id)
            review.status = (
                "stopped" if isinstance(error, service.Cancelled) else "failed"
            )
            review.error = str(error)
            db.session.commit()
            if not isinstance(error, service.Cancelled):
                app.logger.exception("抽查回放生成失败")
        finally:
            with _lock:
                _review_jobs.pop(review_id, None)
                _review_futures.pop(review_id, None)
            db.session.remove()


def camera_worker(
    app, project_id, task_id, config, stream, source, output_dir, stop, resume, report
):
    with app.app_context():
        live = frames_for(task_id, stream["key"])

        def check_cancel():
            if stop.is_set():
                raise service.Cancelled("道路任务已停止，机位断点已保存。")

        def preview(*args):
            if live.wants_frame():
                try:
                    live.publish(render_preview(*args))
                except Exception:
                    app.logger.exception("机位实时预览绘制失败，统计继续")

        try:
            check_cancel()
            report({"phase": "loading"})
            execution = os.environ.get("ROAD_SURVEY_EXECUTION", "thread")
            if execution == "process":
                from services.road_survey_process import run_camera_process

                prefix = output_dir / stream["key"].replace(":", "-")
                return run_camera_process(
                    app,
                    project_id,
                    config,
                    stream,
                    source,
                    prefix,
                    stop,
                    resume,
                    report,
                    live,
                )
            if execution != "thread":
                raise ValueError("ROAD_SURVEY_EXECUTION 仅支持 thread 或 process")
            from inference_manager import InferenceManager

            model = InferenceManager(project_id).load_model(
                "system", config.get("system_model", "best.pt"), prefer_engine=False
            )
            check_cancel()
            missing = [
                name
                for name in model.names.values()
                if not config.get("class_mapping", {}).get(name)
            ]
            if missing:
                raise ValueError("请明确车型映射：" + "、".join(missing))
            prefix = output_dir / stream["key"].replace(":", "-")
            return service.run_stream(
                model,
                config,
                stream,
                source,
                prefix.with_suffix(".jsonl"),
                report,
                check_cancel,
                preview=preview,
                checkpoint_path=prefix.with_suffix(".checkpoint"),
                resume=resume,
            )
        finally:
            live.finish()
            db.session.remove()


def worker(app, task_id, mode, stop, resume=False):
    with app.app_context():
        task = db.session.get(RoadSurveyTask, task_id)
        last_update = [0.0]
        latest_progress = [{}]
        live = frames_for(task_id) if mode == "run" else None
        if live is not None:
            live.reset()

        def check_cancel():
            if stop.is_set():
                raise service.Cancelled(
                    "任务已停止；当前帧断点已保存，恢复时继续处理，已完成机位保留。"
                )

        def progress(value):
            latest_progress[0] = value
            if time.monotonic() - last_update[0] >= 1:
                task.progress = value
                db.session.commit()
                last_update[0] = time.monotonic()

        try:
            check_cancel()
            config = task.config
            if mode == "scan":
                task.status = "scanning"
                db.session.commit()
                task.manifest = service.scan_manifest(config, progress, check_cancel)
                check_cancel()
                task.status = "ready"
                task.progress = {"phase": "ready", "files": task.manifest["file_count"]}
            else:
                task.status = "running"
                db.session.commit()
                service.verify_manifest(config, task.manifest)
                completed = {
                    key: value
                    for key, value in (task.result or {}).get("completed", {}).items()
                    if Path(value["audit_path"]).is_file()
                }
                output_dir = Path(app.instance_path) / "road_surveys" / str(task_id)
                states = {}
                for stream in task.manifest["streams"]:
                    value = completed.get(stream["key"])
                    states[stream["key"]] = {
                        "phase": "finished" if value else "queued",
                        "direction": stream["direction"],
                        "stream_name": stream.get("name", "机位"),
                        "counts": sum(
                            sum(v.values())
                            for slots in value.get("buckets", {}).values()
                            for v in slots.values()
                        )
                        if value
                        else 0,
                        "label_counts": value.get("label_counts", {}) if value else {},
                    }
                    if value is None:
                        frames_for(task_id, stream["key"]).reset()

                def publish(force=False):
                    value = aggregate_progress(states)
                    latest_progress[0] = value
                    if force:
                        task.progress = value
                        db.session.commit()
                        last_update[0] = time.monotonic()
                    else:
                        progress(value)

                def report(key, value):
                    states[key] = {**states[key], **value}
                    publish()

                def save_camera(key, value):
                    completed[key] = value
                    states[key] = {**states[key], "phase": "finished"}
                    task.result = {"completed": dict(completed)}
                    publish(force=True)

                project_id = task.project_id

                def run_camera(stream, report):
                    source = config["directions"][stream["direction_index"]]["streams"][
                        stream["stream_index"]
                    ]
                    return camera_worker(
                        app,
                        project_id,
                        task_id,
                        config,
                        stream,
                        source,
                        output_dir,
                        stop,
                        resume,
                        report,
                    )

                publish(force=True)
                try:
                    run_cameras(
                        get_scheduler(),
                        [
                            s
                            for s in task.manifest["streams"]
                            if s["key"] not in completed
                        ],
                        run_camera,
                        report,
                        save_camera,
                        stop,
                    )
                except service.Cancelled:
                    for key, value in states.items():
                        if value["phase"] != "finished":
                            states[key] = {**value, "phase": "stopped"}
                    publish(force=True)
                    raise
                check_cancel()
                exports = service.export_workbooks(
                    config, task.manifest, completed, output_dir
                )
                task.result = {
                    "completed": completed,
                    "exports": exports,
                    "export_version": service.EXPORT_VERSION,
                    "excluded": sum(r["excluded"] for r in completed.values()),
                }
                task.status = "finished"
                task.progress = {
                    **(task.progress or {}),
                    "phase": "finished",
                    "files": task.manifest["file_count"],
                }
            task.error = None
            db.session.commit()
        except Exception as error:
            db.session.rollback()
            task = db.session.get(RoadSurveyTask, task_id)
            task.status = (
                "stopped" if isinstance(error, service.Cancelled) else "failed"
            )
            task.error = str(error)
            db.session.commit()
            if not isinstance(error, service.Cancelled):
                app.logger.exception("道路调查任务 %s 失败", task_id)
        finally:
            if live is not None:
                finish_task(task_id)
            with _lock:
                _jobs.pop(task_id, None)
                _futures.pop(task_id, None)
            db.session.remove()
