from concurrent.futures import as_completed
import os
import io
import sys
import json
import tempfile
from typing import List
import uuid
import yaml
import shutil
import zipfile
import threading
import cv2
import math
import traceback
import posixpath
from PIL import Image as PILImage
from datetime import datetime
from managers.rtsp_threadpool_manager import rtsp_manager
from flask import (
    Blueprint,
    abort,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    current_app,
    send_file,
    Response,
)
from models import (
    AnnotationTask,
    AnnotationTaskItem,
    ReasoningMaterial,
    db,
    Project,
    Image,
    Label,
    Annotation,
    ExportRecord,
    LLMConfig,
)
from werkzeug.utils import secure_filename
from sqlalchemy import and_, not_, text, update
from sqlalchemy.exc import SQLAlchemyError
from app import project_task_locks, executor
from services.inference_service import inference_service
from services.video_service import video_service
from services.camera_service import camera_service
from services.reasoning_material_service import (
    ReasoningMaterialService,
    reasoning_material_service,
)
from utils import extract_first_frame
from project_dir_manager import ProjectDirManager
from loguru import logger as log

# 创建蓝图
main = Blueprint("main", __name__)

# 添加导入语句
from inference_manager import InferenceManager

# 动态导入服务模块
import importlib.util

# 导入训练服务
training_service_spec = importlib.util.spec_from_file_location(
    "training_service",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "services", "training_service.py"
    ),
)
training_service = importlib.util.module_from_spec(training_service_spec)
training_service_spec.loader.exec_module(training_service)

# 导入数据集服务
dataset_service_spec = importlib.util.spec_from_file_location(
    "dataset_service",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "services", "dataset_service.py"
    ),
)
dataset_service = importlib.util.module_from_spec(dataset_service_spec)
dataset_service_spec.loader.exec_module(dataset_service)

# 导入标注服务
annotation_service_spec = importlib.util.spec_from_file_location(
    "annotation_service",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "services", "annotation_service.py"
    ),
)
annotation_service = importlib.util.module_from_spec(annotation_service_spec)
annotation_service_spec.loader.exec_module(annotation_service)

# 导出服务
export_service_spec = importlib.util.spec_from_file_location(
    "export_service",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "services", "export_service.py"
    ),
)
export_service = importlib.util.module_from_spec(export_service_spec)
export_service_spec.loader.exec_module(export_service)

# 视频服务
video_service_spec = importlib.util.spec_from_file_location(
    "video_service",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "services", "video_service.py"
    ),
)
video_service = importlib.util.module_from_spec(video_service_spec)
video_service_spec.loader.exec_module(video_service)

# 图片服务
image_service_spec = importlib.util.spec_from_file_location(
    "image_service",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "services", "image_service.py"
    ),
)
image_service = importlib.util.module_from_spec(image_service_spec)
image_service_spec.loader.exec_module(image_service)

# 初始化服务
training_service_instance = training_service.TrainingService()
dataset_service_instance = dataset_service.DatasetService()
annotation_service_instance = annotation_service.AnnotationService()
export_service_instance = export_service.ExportService()
video_service = video_service.VideoService()
image_service_instance = image_service.ImageService()

# 全局字典用于管理RTSP流状态
rtsp_streams = {}


@main.route("/")
def index():
    projects = Project.query.order_by(Project.created_at.desc()).all()
    return render_template("index.html", projects=projects)


@main.route("/project/create", methods=["POST"])
def create_project():
    name = request.form.get("name")
    description = request.form.get("description")

    if not name:
        flash("项目名称不能为空", "error")
        return redirect(url_for("main.index"))

    project = Project(name=name, description=description)
    db.session.add(project)
    db.session.commit()

    flash(f'项目 "{name}" 创建成功', "success")
    return redirect(url_for("main.project_detail", project_id=project.id))


@main.route("/project/<int:project_id>")
def project_detail(project_id):
    project = Project.query.get_or_404(project_id)
    # 图片总数
    imgs_count = Image.query.filter_by(project_id=project_id).count()
    # 分页查询
    images_pagination = Image.query.filter_by(project_id=project_id).paginate(
        page=1, per_page=48, error_out=False
    )
    # 查询总标注数
    sql = """
    SELECT i.project_id, COUNT(i.*) AS label_imgs_count
    FROM yolov8_platform.image i
    WHERE i.project_id = :project_id
	and i.id in (select image_id from yolov8_platform.annotation group by image_id )
    GROUP BY i.project_id
"""
    result = db.session.execute(text(sql), {"project_id": project_id}).fetchall()

    # 查询标注框数
    sql = """
    SELECT i.project_id, COUNT(a.*) AS annotation_count
    FROM yolov8_platform.annotation a left join  yolov8_platform.image i on a.image_id=i.id
    WHERE i.project_id = :project_id
    GROUP BY i.project_id
"""
    annotationresult = db.session.execute(
        text(sql), {"project_id": project_id}
    ).fetchall()

    sql = """
    SELECT i.project_id, COUNT(i.*) AS label_count
    FROM yolov8_platform.label i
    WHERE i.project_id = :project_id
    GROUP BY i.project_id
"""
    labelresult = db.session.execute(text(sql), {"project_id": project_id}).fetchall()

    return render_template(
        "project_detail.html",
        project=project,
        label_imgs_count=len(result) > 0
        and result[0]
        or {
            "project_id": project_id,
            "label_imgs_count": 0,
        },
        imgs_count=imgs_count,
        annotation_count=len(annotationresult) > 0
        and annotationresult[0]
        or {
            "project_id": project_id,
            "annotation_count": 0,
        },
        label_count=len(labelresult) > 0
        and labelresult[0]
        or {
            "project_id": project_id,
            "label_count": 0,
        },
        images_pagination=images_pagination,
    )


@main.route("/project/<int:project_id>/delete", methods=["POST"])
def delete_project(project_id):
    project = Project.query.get_or_404(project_id)
    project_name = project.name

    # 删除项目相关的所有文件
    project_path = os.path.join("static/datasets", str(project_id))
    if os.path.exists(project_path):
        shutil.rmtree(project_path)

    # 删除项目记录
    db.session.delete(project)
    db.session.commit()

    flash(f'项目 "{project_name}" 已删除', "success")
    return redirect(url_for("main.index"))


# 图片管理
@main.route("/project/<int:project_id>/images")
def project_images(project_id):
    project = Project.query.get_or_404(project_id)
    # 获取分页参数
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    return render_template("images.html", page=page, per_page=per_page, project=project)


# 视频截图页面
@main.route("/project/<int:project_id>/video/capture")
def video_capture(project_id):
    project = Project.query.get_or_404(project_id)
    return render_template("video_capture.html", project=project)


# 图片管理
@main.route("/project/<int:project_id>/images/upload")
def images_upload(project_id):
    project = Project.query.get_or_404(project_id)
    # 图片总数
    imgs_count = Image.query.filter_by(project_id=project_id).count()
    # 获取分页参数
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    # 确保per_page在合理范围内
    per_page = min(per_page, 100)  # 最多显示100张图片

    # 分页查询图片
    images_pagination = Image.query.filter_by(project_id=project_id).paginate(
        page=page, per_page=per_page, error_out=False
    )

    # 查询总标注数
    sql = """
    SELECT i.project_id, COUNT(i.*) AS label_count
    FROM yolov8_platform.image i
    WHERE i.project_id = :project_id
	and i.id in (select image_id from yolov8_platform.annotation group by image_id )
    GROUP BY i.project_id
"""
    result = db.session.execute(text(sql), {"project_id": project_id}).fetchall()

    # 查询标注框数
    sql = """
    SELECT i.project_id, COUNT(a.*) AS annotation_count
    FROM yolov8_platform.annotation a left join  yolov8_platform.image i on a.image_id=i.id
    WHERE i.project_id = :project_id
    GROUP BY i.project_id
"""
    annotationresult = db.session.execute(
        text(sql), {"project_id": project_id}
    ).fetchall()

    return render_template(
        "images_upload.html",
        project=project,
        page=page,
        per_page=per_page,
        images=images_pagination,
        imgs_count=imgs_count,
        label_count=result
        and result[0]
        or {"project_id": project_id, "label_count": 0},
        annotation_count=annotationresult
        and annotationresult[0]
        or {"project_id": project_id, "annotation_count": 0},
        pagination=images_pagination,
    )


@main.route("/project/<int:project_id>/image/<int:image_id>/delete", methods=["POST"])
def delete_image(project_id, image_id):
    project = Project.query.get_or_404(project_id)
    image = Image.query.get_or_404(image_id)

    # 确保图片属于该项目
    if image.project_id != project_id:
        flash("无效的图片ID", "error")
        return redirect(url_for("main.images_upload", project_id=project_id))

    # 获取图片文件路径
    image_path = os.path.join(current_app.root_path, image.path)

    try:
        # 删除数据库记录
        db.session.delete(image)
        db.session.commit()

        # 删除图片文件
        if os.path.exists(image_path):
            os.remove(image_path)

        # 删除对应的YOLO格式标注文件（.txt文件）
        txt_file_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)

        flash(f'图片 "{image.original_filename}" 已删除', "success")
    except Exception as e:
        db.session.rollback()
        flash(f"删除图片时出错: {str(e)}", "error")

    return redirect(url_for("main.project_images", project_id=project_id))


@main.route("/project/<int:project_id>/upload_images", methods=["POST"])
def upload_images(project_id):
    project = Project.query.get_or_404(project_id)

    # 处理ZIP文件上传
    if "zip_file" in request.files:
        zip_file = request.files["zip_file"]
        if zip_file.filename != "":
            # 保存ZIP文件
            zip_path = os.path.join(
                "static/uploads", f"{project_id}_{zip_file.filename}"
            )
            zip_file.save(zip_path)

            # 解压ZIP文件
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                extract_path = os.path.join("static/uploads", str(project_id))
                os.makedirs(extract_path, exist_ok=True)
                zip_ref.extractall(extract_path)

            # 删除ZIP文件
            os.remove(zip_path)

            # 遍历解压后的文件并添加到数据库
            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, "static")
                        # 使用posixpath处理URL路径，确保在Windows上也使用正斜杠
                        relative_path = posixpath.join(*relative_path.split(os.sep))
                        img = PILImage.open(file_path)
                        width, height = img.size

                        image = Image(
                            filename=file,
                            original_filename=file,
                            path=relative_path,
                            project_id=project_id,
                            width=width,
                            height=height,
                        )
                        db.session.add(image)

            db.session.commit()
            flash("ZIP文件上传并解压成功", "success")
            return redirect(url_for("main.project_images", project_id=project_id))

    # 处理单个或多个图片上传
    if "images" in request.files:
        images = request.files.getlist("images")
        for image_file in images:
            if image_file.filename != "":
                # 保存图片
                filename = f"{project_id}_{int(datetime.now().timestamp()*1000)}_{image_file.filename}"
                file_path = os.path.join("static/uploads", filename)
                image_file.save(file_path)

                # 获取图片尺寸
                img = PILImage.open(file_path)
                width, height = img.size

                # 保存到数据库，确保路径正确
                relative_path = os.path.join("uploads", filename)
                # 使用posixpath处理URL路径，确保在Windows上也使用正斜杠
                relative_path = posixpath.join(*relative_path.split(os.sep))
                image = Image(
                    filename=filename,
                    original_filename=image_file.filename,
                    path=relative_path,
                    project_id=project_id,
                    width=width,
                    height=height,
                )
                db.session.add(image)

        db.session.commit()
        flash("图片上传成功", "success")

    return redirect(url_for("main.project_images", project_id=project_id))


@main.route("/project/<int:project_id>/delete_selected_images", methods=["POST"])
def delete_selected_images(project_id):
    """批量删除选中的图片"""
    project = Project.query.get_or_404(project_id)

    try:
        # 获取选中的图片ID列表
        image_ids_json = request.form.get("image_ids", "[]")
        image_ids = json.loads(image_ids_json)

        if not image_ids:
            flash("未选择任何图片", "warning")
            return redirect(url_for("main.project_images", project_id=project_id))

        # 使用图片服务删除选中的图片
        deleted_count, errors = image_service_instance.delete_images(image_ids)

        if errors:
            for error in errors:
                flash(error, "error")

        if deleted_count > 0:
            flash(f"成功删除 {deleted_count} 张图片", "success")
        elif not errors:
            flash("未删除任何图片", "warning")

    except Exception as e:
        flash(f"批量删除图片时出错: {str(e)}", "error")

    return redirect(url_for("main.project_images", project_id=project_id))


@main.route("/project/<int:project_id>/delete_unannotated_images", methods=["POST"])
def delete_unannotated_images(project_id):
    """删除项目中所有未标注的图片"""
    project = Project.query.get_or_404(project_id)

    try:
        # 使用图片服务删除所有未标注的图片
        deleted_count, errors = image_service_instance.delete_unannotated_images(
            project_id
        )

        if errors:
            for error in errors:
                flash(error, "error")

        if deleted_count > 0:
            flash(f"成功删除 {deleted_count} 张未标注的图片", "success")
        elif deleted_count == 0:
            flash("没有未标注的图片需要删除", "info")

    except Exception as e:
        flash(f"删除未标注图片时出错: {str(e)}", "error")

    return redirect(url_for("main.project_images", project_id=project_id))


# 标注功能
@main.route("/project/<int:project_id>/annotate")
def annotate(project_id, page=1, per_page=10):
    project = Project.query.get_or_404(project_id)
    images = Image.query.filter_by(project_id=project_id).all()

    # 如果没有图片，重定向到图片上传页面
    if not images:
        flash("请先上传图片再进行标注", "warning")
        return redirect(url_for("main.images_upload", project_id=project_id))

    task_annotate_list = AnnotationTask.query.filter_by(project_id=project_id)

    task_annotate_pagination = task_annotate_list.order_by(
        AnnotationTask.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        "task_annotate.html",
        project=project,
        images=images,
        pagination=task_annotate_pagination,
        task_list=task_annotate_pagination.items,
    )


# 不分配任务
@main.route("/project/<int:project_id>/annotate/<int:image_id>")
def annotate_image(project_id, image_id, page=1, per_page=10):
    project = Project.query.get_or_404(project_id)
    images = Image.query.filter_by(project_id=project_id).all()
    current_image = Image.query.get_or_404(image_id)

    return render_template(
        "annotate.html",
        project=project,
        images=images,
        current_image=current_image,
    )


# 通过任务分配形式标注图片
@main.route("/project/<int:project_id>/annotate_task_image/<int:image_id>")
def annotate_task_image(project_id, image_id, page=1, per_page=10):
    project = Project.query.get_or_404(project_id)
    images = Image.query.filter_by(project_id=project_id).all()
    current_image = Image.query.get_or_404(image_id)

    return render_template(
        "annotate_task_image.html",
        project=project,
        images=images,
        current_image=current_image,
    )


import os


@main.route("/api/project/<int:project_id>/labels")
def api_labels(project_id):
    labels_data = annotation_service_instance.get_labels(project_id)
    return jsonify(labels_data)


@main.route("/api/project/<int:project_id>/labels/create", methods=["POST"])
def api_create_label(project_id):
    data = request.get_json()
    name = data.get("name")
    color = data.get("color", "#0066ff")

    result = annotation_service_instance.create_label(project_id, name, color)
    return jsonify(result)


@main.route(
    "/api/project/<int:project_id>/labels/<int:label_id>/delete", methods=["POST"]
)
def api_delete_label(project_id, label_id):
    result = annotation_service_instance.delete_label(label_id)
    return jsonify(result)


@main.route(
    "/api/project/<int:project_id>/labels/<int:label_id>/update", methods=["POST"]
)
def api_update_label(project_id, label_id):
    data = request.get_json()
    name = data.get("name")
    color = data.get("color")

    result = annotation_service_instance.update_label(label_id, name, color)
    return jsonify(result)


@main.route("/api/image/<int:image_id>/annotations", methods=["GET"])
def api_annotations(image_id):
    # 查询任务项，更新任务进度
    annotations_data = annotation_service_instance.get_annotations(image_id)
    return jsonify(annotations_data)


import math


@main.route("/api/image/<int:image_id>/annotations/save", methods=["POST"])
def api_save_annotations(image_id):
    data = request.get_json()
    annotations = data.get("annotations", [])

    task_item: AnnotationTaskItem = AnnotationTaskItem.query.filter_by(
        image_id=image_id
    ).first()
    if task_item:
        if annotations and len(annotations) > 0:
            if not task_item.annotate_status:
                task_item.annotate_status = 1
                db.session.add(task_item)
                task_id = task_item.task_id
                db.session.execute(
                    update(AnnotationTask)
                    .where(AnnotationTask.id == task_id)
                    .values(completed_count=AnnotationTask.completed_count + 1)
                )
        else:
            if task_item.annotate_status:
                task_item.annotate_status = 0
                db.session.add(task_item)
                task_id = task_item.task_id
                db.session.execute(
                    update(AnnotationTask)
                    .where(AnnotationTask.id == task_id)
                    .values(completed_count=AnnotationTask.completed_count - 1)
                )

    result = annotation_service_instance.save_annotations(image_id, annotations)
    return jsonify(result)


# 数据集划分
@main.route("/project/<int:project_id>/dataset")
def project_dataset(project_id):
    project = Project.query.get_or_404(project_id)
    # 获取分页参数
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    # 确保per_page在合理范围内
    per_page = min(per_page, 100)  # 最多显示100张图片

    # 分页查询图片
    images_pagination = Image.query.filter_by(project_id=project_id).paginate(
        page=page, per_page=per_page, error_out=False
    )
    images = images_pagination.items
    # 统计各类型数据集数量
    train_count = Image.query.filter_by(
        project_id=project_id, dataset_type="train"
    ).count()
    val_count = Image.query.filter_by(project_id=project_id, dataset_type="val").count()
    test_count = Image.query.filter_by(
        project_id=project_id, dataset_type="test"
    ).count()
    unassigned_count = Image.query.filter_by(
        project_id=project_id, dataset_type="unassigned"
    ).count()

    return render_template(
        "dataset.html",
        project=project,
        train_count=train_count,
        images=images,
        pagination=images_pagination,
        val_count=val_count,
        test_count=test_count,
        unassigned_count=unassigned_count,
    )


@main.route("/api/project/<int:project_id>/dataset/update", methods=["POST"])
def api_update_dataset(project_id):
    data = request.get_json()
    image_ids = data.get("image_ids", [])
    dataset_type = data.get("dataset_type")

    result = dataset_service_instance.update_dataset(
        project_id, image_ids, dataset_type
    )
    return jsonify(result)


@main.route("/api/project/<int:project_id>/dataset/auto_assign", methods=["POST"])
def api_auto_assign_dataset(project_id):
    result = dataset_service_instance.auto_assign_dataset(project_id)
    return jsonify(result)


# 模型训练
@main.route("/project/<int:project_id>/train")
def project_train(project_id):
    project = Project.query.get_or_404(project_id)
    status = training_service_instance.get_training_status(project_id)
    return render_template("train.html", project=project, status=status)


@main.route("/api/project/<int:project_id>/train", methods=["POST"])
def api_start_training(project_id):
    try:
        # 获取训练参数
        data = request.get_json()
        epochs = data.get("epochs", 20)  # 默认20个epochs
        model_arch = data.get("model_arch", "yolov8n.pt")  # 默认使用yolov8n模型
        img_size = data.get("img_size", 640)  # 默认图像尺寸640
        batch_size = data.get("batch_size", 16)  # 默认批次大小16
        use_gpu = data.get("use_gpu", True)  # 默认使用GPU

        result = training_service_instance.start_training(
            project_id, epochs, model_arch, img_size, batch_size, use_gpu
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"启动训练失败: {str(e)}"})


@main.route("/api/project/<int:project_id>/train/stop", methods=["POST"])
def api_stop_training(project_id):
    result = training_service_instance.stop_training(project_id)
    return jsonify(result)


@main.route("/api/project/<int:project_id>/train/status")
def api_train_status(project_id):
    status = training_service_instance.get_training_status(project_id)
    return jsonify(status)


# 模型导出
@main.route("/project/<int:project_id>/export")
def project_export(project_id):
    project = Project.query.get_or_404(project_id)
    # 获取应用根路径
    root_path = current_app.root_path
    return render_template("export.html", project=project, root_path=root_path)


@main.route("/api/project/<int:project_id>/export/<format>", methods=["POST"])
def api_export_model(project_id, format):
    result = export_service_instance.export_model(project_id, format)
    return jsonify(result)


@main.route("/download/<path:filename>")
def download_file(filename):
    # 构造完整文件路径
    full_path = os.path.join(current_app.root_path, filename)

    # 检查文件是否存在
    if not os.path.exists(full_path):
        return "文件不存在", 404

    # 确定文件的MIME类型和下载文件名
    mimetype = "application/octet-stream"
    download_name = os.path.basename(filename)

    # 根据文件扩展名设置特定的MIME类型
    if filename.endswith(".onnx"):
        mimetype = "application/octet-stream"
        download_name = filename.split("/")[-1]  # 确保获取正确的文件名
    elif filename.endswith(".pt"):
        mimetype = "application/octet-stream"
        download_name = filename.split("/")[-1]
    elif filename.endswith(".torchscript"):
        mimetype = "application/octet-stream"
        download_name = filename.split("/")[-1]

    return send_file(
        full_path, as_attachment=True, mimetype=mimetype, download_name=download_name
    )


@main.route("/api/project/<int:project_id>/dataset/download")
def api_download_dataset(project_id):
    try:
        project = Project.query.get_or_404(project_id)

        # 检查数据集目录是否存在
        project_dir = os.path.join(
            current_app.root_path, "static/datasets", str(project_id)
        )
        if not os.path.exists(project_dir):
            # 如果目录不存在，重新组织一次
            dataset_service_instance.organize_dataset_directories(project_id)

        if not os.path.exists(project_dir):
            return jsonify({"success": False, "message": "数据集目录不存在"}), 404

        # 创建临时ZIP文件
        import tempfile
        import zipfile

        temp_dir = tempfile.mkdtemp()
        zip_filename = f"dataset_{project_id}.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)

        # 创建ZIP文件
        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_dir)
                    zipf.write(file_path, arcname)

        # 发送ZIP文件
        return send_file(
            zip_filepath,
            as_attachment=True,
            download_name=zip_filename,
            mimetype="application/zip",
        )

    except Exception as e:
        print(f"下载数据集时出错: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/project/<int:project_id>/rtsp")
def rtsp_capture(project_id):
    """RTSP截图页面"""
    project = Project.query.get_or_404(project_id)
    return render_template("rtsp_capture.html", project=project)


@main.route("/api/project/<int:project_id>/rtsp/start", methods=["POST"])
def api_start_rtsp_capture(project_id):
    """启动RTSP截图"""
    try:
        project = Project.query.get_or_404(project_id)
        data = request.get_json()

        rtsp_url = data.get("rtsp_url")
        interval = data.get("interval", 5)
        max_count = data.get("max_count", 10)

        result = camera_service.start_rtsp_capture(
            project_id, rtsp_url, interval, max_count
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/rtsp/manual_capture")
def api_rtsp_manual_capture(project_id):
    try:
        project = Project.query.get_or_404(project_id)
        data = request.get_json()
        # base64编码的图片
        screenshot_url = data.get("screenshot_url")
        result = camera_service.save_rtsp_manual_capture(project_id, screenshot_url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/rtsp/preview", methods=["POST"])
def api_preview_rtsp(project_id):
    """预览RTSP视频流"""
    try:
        project = Project.query.get_or_404(project_id)
        data = request.get_json()

        rtsp_url = data.get("rtsp_url")

        result = camera_service.preview_rtsp_stream(project_id, rtsp_url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/rtsp/stop", methods=["POST"])
def api_stop_rtsp_capture(project_id):
    """停止RTSP截图"""
    result = camera_service.stop_rtsp_capture(project_id)
    return jsonify(result)


@main.route("/api/project/<int:project_id>/rtsp/stop_preview", methods=["POST"])
def api_stop_rtsp_preview(project_id):
    """停止RTSP预览"""
    result = camera_service.stop_rtsp_preview(project_id)
    return jsonify(result)


@main.route("/api/project/<int:project_id>/rtsp/status")
def api_rtsp_status(project_id):
    """获取RTSP截图状态"""
    result = camera_service.get_rtsp_status(project_id)
    return jsonify(result)


@main.route("/api/project/<int:project_id>/rtsp/preview_status")
def api_rtsp_preview_status(project_id):
    """获取RTSP预览状态"""
    result = camera_service.get_rtsp_preview_status(project_id)
    return jsonify(result)


@main.route("/api/project/<int:project_id>/video/start_auto_capture", methods=["POST"])
def api_start_video_capture(project_id):
    """启动video截图（正确接收视频文件）"""
    try:
        project = Project.query.get_or_404(project_id)

        # 获取原文件对象
        video_file = request.files.get("video")
        if not video_file or video_file.filename == "":
            return jsonify({"success": False, "message": "请上传有效的视频文件"}), 400

        video_file_content = video_file.read()
        if not video_file_content:
            return jsonify({"success": False, "message": "上传的视频文件为空"}), 400

        try:
            interval = int(request.form.get("interval", 5))
            max_count = int(request.form.get("max_count", 10))
        except ValueError:
            return (
                jsonify({"success": False, "message": "截图间隔和最大数量必须为整数"}),
                400,
            )

        start_time = request.form.get("start_time", None)
        if start_time is not None:
            try:
                start_time = float(start_time)
                if start_time < 0:
                    return (
                        jsonify({"success": False, "message": "开始时间必须大于等于0"}),
                        400,
                    )
            except ValueError:
                return (
                    jsonify({"success": False, "message": "开始时间必须为数字"}),
                    400,
                )
        else:
            start_time = 0.04
        end_time = request.form.get("end_time", None)
        if end_time is not None:
            try:
                end_time = float(end_time)
                if end_time <= 0:
                    return (
                        jsonify({"success": False, "message": "结束时间必须大于0"}),
                        400,
                    )
            except ValueError:
                return (
                    jsonify({"success": False, "message": "结束时间必须为数字"}),
                    400,
                )
        else:
            end_time = None

        result = video_service.start_video_capture(
            project_id=project_id,
            video_file_content=video_file_content,
            interval=interval,
            max_count=max_count,
            start_time=start_time,
            end_time=end_time,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/project/<int:project_id>/video/manual_capture", methods=["POST"])
def api_manual_video_capture(project_id):
    """启动video手动截图（正确接收视频文件）"""
    try:
        project = Project.query.get_or_404(project_id)

        # 获取原文件对象
        video_file = request.files.get("video")
        if not video_file or video_file.filename == "":
            return jsonify({"success": False, "message": "请上传有效的视频文件"}), 400

        video_file_content = video_file.read()
        if not video_file_content:
            return jsonify({"success": False, "message": "上传的视频文件为空"}), 400

        video_current_time = float(request.form.get("video_current_time", 0))
        if video_current_time <= 0:
            return (
                jsonify({"success": False, "message": "视频当前时间必须大于0"}),
                400,
            )

        result = video_service.get_video_time_capture(
            project_id=project_id,
            video_file_content=video_file_content,
            video_current_time=video_current_time,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/project/<int:project_id>/video/stop", methods=["POST"])
def api_stop_video_capture(project_id):
    """停止video截图"""
    result = video_service.stop_video_capture(project_id)
    return jsonify(result)


@main.route("/api/project/<int:project_id>/video/status")
def api_video_status(project_id):
    """获取video截图状态"""
    result = video_service.get_video_status(project_id)
    return jsonify(result)


@main.route("/api/onvif/discover")
def api_discover_onvif_devices():
    """发现ONVIF设备"""
    result = camera_service.discover_onvif_devices()
    return jsonify(result)


@main.route("/api/onvif/<device_ip>/<device_port>/profiles", methods=["POST"])
def api_get_onvif_profiles(device_ip, device_port):
    """获取ONVIF设备的配置文件列表"""
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "message": "用户名和密码不能为空"})

        result = camera_service.get_onvif_profiles(
            device_ip, device_port, username, password
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/onvif/start", methods=["POST"])
def api_start_onvif_capture(project_id):
    """启动ONVIF截图"""
    try:
        project = Project.query.get_or_404(project_id)
        data = request.get_json()

        device_ip = data.get("device_ip")
        device_port = data.get("device_port")
        username = data.get("username")
        password = data.get("password")
        profile_token = data.get("profile_token")
        interval = data.get("interval", 5)
        max_count = data.get("max_count", 10)

        result = camera_service.start_onvif_capture(
            project_id,
            device_ip,
            device_port,
            username,
            password,
            profile_token,
            interval,
            max_count,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/onvif/stop", methods=["POST"])
def api_stop_onvif_capture(project_id):
    """停止ONVIF截图"""
    try:
        data = request.get_json()
        profile_token = data.get("profile_token")

        if not profile_token:
            return jsonify({"success": False, "message": "配置文件token不能为空"})

        result = camera_service.stop_onvif_capture(project_id, profile_token)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/onvif/status", methods=["POST"])
def api_onvif_status(project_id):
    """获取ONVIF截图状态"""
    try:
        data = request.get_json()
        profile_token = data.get("profile_token")

        if not profile_token:
            return jsonify({"success": False, "message": "配置文件token不能为空"})

        result = camera_service.get_onvif_status(project_id, profile_token)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@main.route("/project/<int:project_id>/inference")
def model_inference(project_id):
    """模型推理页面"""
    project = Project.query.get_or_404(project_id)

    # 检查系统模型
    inference_manager = InferenceManager(project_id)
    model_status = inference_manager.check_system_models()

    # 获取已存在的模型列表（从项目模型目录）
    existing_models = inference_manager.get_existing_models()

    # 获取已上传的模型列表（从uploaded_models目录）
    uploaded_models = inference_manager.get_uploaded_models()

    # 获取当前项目的标签列表
    project_labels = Label.query.filter_by(project_id=project_id).all()

    inference_labels = [label for label in project_labels if label.target]
    return render_template(
        "inference.html",
        project=project,
        best_model_exists=model_status["best_model_exists"],
        last_model_exists=model_status["last_model_exists"],
        existing_models=existing_models,
        uploaded_models=uploaded_models,
        project_labels=project_labels,
        inference_labels=inference_labels,
    )


@main.route("/project/<int:project_id>/inference/run", methods=["POST"])
def run_inference(project_id):
    """执行模型推理"""

    project = Project.query.get_or_404(project_id)

    project_lock = threading.Lock()
    with project_lock:
        if (
            project_id in project_task_locks
            and project_task_locks[project_id]["is_running"]
        ):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"项目 {project_id} 已有运行中的推理任务，请等待任务完成后再提交",
                    }
                ),
                409,
            )
        project_task_locks[project_id] = {
            "is_running": True,
            "total_materials": 0,
            "completed_materials": 0,
            "stop_event": threading.Event(),
        }

    try:
        # 获取表单数据
        model_type = request.form.get("model_type")
        system_model = request.form.get("system_model")
        uploaded_model = request.form.get("uploaded_model")
        existing_model = request.form.get("existing_model")
        material_id = request.form.get("material_id")

        # 获取上传的文件
        model_file = request.files.get("model_file")

        # 创建推理管理器
        labels = Label.query.filter_by(project_id=project_id).all()

        # 查询所有未推理的素材
        if material_id is not None:
            materials = [ReasoningMaterial.query.get_or_404(material_id)]
        else:
            materials = ReasoningMaterial.query.filter_by(
                project_id=project_id, status=0
            ).all()

        if not materials:
            with project_lock:
                project_task_locks[project_id]["is_running"] = False
                del project_task_locks[project_id]
            return jsonify({"success": False, "message": "没有需要推理的素材！"}), 400

        modeel_obj = {
            "model_type": model_type,
            "uploaded_model": uploaded_model,
            "existing_model": existing_model,
            "system_model": system_model,
            "model_file": model_file,
            "labels": labels,
        }
        executor.submit(
            inference_service.background_inference_controller,
            project_id,
            materials,
            modeel_obj,
        )

        return (
            jsonify(
                {
                    "success": True,
                    "project_id": project_id,
                    "message": "推理任务已提交至后台运行，可通过状态接口查询进度",
                    "total_materials": len(materials),
                }
            ),
            202,  # 202 Accepted：请求已接受，后台处理中
        )

    except Exception as e:
        # 主线程异常：释放锁并返回错误
        error_msg = f"任务提交失败：{str(e)}"
        current_app.logger.error(error_msg)
        with project_lock:
            if project_id in project_task_locks:
                project_task_locks[project_id]["is_running"] = False
                del project_task_locks[project_id]
        return jsonify({"success": False, "message": error_msg}), 500


# Flask后端示例：停止推理接口
@main.route("/project/<int:project_id>/stop_inference", methods=["POST"])
def stop_inference(project_id):
    log.info(f"开始执行停止推理任务")
    try:
        data = request.get_json() or {}
        material_id = data.get("materialId")
        if not material_id:
            task = project_task_locks.get(project_id)
            if not task:
                return

            task["stop_event"].set()
            task["is_running"] = False
        else:
            # 直接更新数据状态
            reasoning_material_service.update_material(
                material_id=material_id, status=2
            )

            stop_result = rtsp_manager.stop_rtsp_task(task_id=material_id)
            if not stop_result:
                return jsonify({"success": False, "message": "停止推理失败!"}), 404

        return jsonify({"success": True, "message": "停止推理成功"}), 200
    except Exception as e:
        db.session.rollback()
        log.error(f"停止推理失败：{str(e)}")
        return jsonify({"success": False, "message": f"停止推理失败：{str(e)}"}), 500


@main.route("/project/<int:project_id>/inference/status", methods=["GET"])
def get_inference_status(project_id):
    """查询后台推理任务进度"""
    # 1. 检查任务是否在运行
    if project_id in project_task_locks:
        task_info = project_task_locks[project_id]
        total = task_info["total_materials"]
        completed = task_info["completed_materials"]
        return (
            jsonify(
                {
                    "success": True,
                    "status": "running",
                    "progress": {
                        "total": total,
                        "completed": completed,
                        "processing": total - completed,
                        "percent": (
                            round(completed / total * 100, 2) if total > 0 else 0
                        ),
                    },
                }
            ),
            200,
        )

    # 2. 任务已完成：查询数据库统计
    materials = ReasoningMaterial.query.filter_by(project_id=project_id).all()
    if materials:
        success = len([m for m in materials if m.status == 2])
        failed = len([m for m in materials if m.status == 3])
        total = len(materials)
        return (
            jsonify(
                {
                    "success": True,
                    "status": "finished",
                    "summary": {"total": total, "success": success, "failed": failed},
                }
            ),
            200,
        )

    # 3. 任务不存在
    return jsonify({"success": False, "message": "任务不存在"}), 404


@main.route("/api/project/<int:project_id>/images/delete_selected", methods=["POST"])
def api_delete_selected_images(project_id):
    """删除选中的图片"""
    try:
        project = Project.query.get_or_404(project_id)
        data = request.get_json()
        image_ids = data.get("image_ids", [])

        if not image_ids:
            return jsonify({"success": False, "message": "未提供图片ID"})

        # 使用图片服务删除选中的图片
        deleted_count, errors = image_service_instance.delete_images(image_ids)

        if errors:
            return jsonify({"success": False, "message": "; ".join(errors)})

        return jsonify(
            {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"成功删除 {deleted_count} 张图片",
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": f"删除图片时出错: {str(e)}"})


@main.route("/api/project/<int:project_id>/images/delete_unannotated", methods=["POST"])
def api_delete_unannotated_images(project_id):
    """删除未标注的图片"""
    try:
        project = Project.query.get_or_404(project_id)

        # 使用图片服务删除所有未标注的图片
        deleted_count, errors = image_service_instance.delete_unannotated_images(
            project_id
        )

        if errors:
            return jsonify({"success": False, "message": "; ".join(errors)})

        return jsonify(
            {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"成功删除 {deleted_count} 张未标注图片",
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": f"删除未标注图片时出错: {str(e)}"})


# RTSP流媒体推理API
@main.route("/project/<int:project_id>/rtsp_stream")
def rtsp_stream_page(project_id):
    """RTSP实时流推理页面"""
    project = Project.query.get_or_404(project_id)

    # 检查系统模型
    inference_manager = InferenceManager(project_id)
    model_status = inference_manager.check_system_models()

    # 获取已存在的模型列表
    existing_models = inference_manager.get_existing_models()

    return render_template(
        "rtsp_stream.html",
        project=project,
        best_model_exists=model_status["best_model_exists"],
        last_model_exists=model_status["last_model_exists"],
        existing_models=existing_models,
    )


@main.route("/project/<int:project_id>/rtsp_stream/video_feed")
def rtsp_video_feed(project_id):
    """RTSP实时视频流"""
    from flask import Response
    import cv2
    import threading
    import time

    # 在请求上下文中获取参数
    rtsp_url = request.args.get("rtsp_url")
    model_type = request.args.get("model_type", "system")
    system_model = request.args.get("system_model")
    existing_model = request.args.get("existing_model")
    uploaded_model = request.args.get("uploaded_model")

    # 创建流ID
    stream_id = f"{project_id}_{rtsp_url}_{model_type}_{system_model or existing_model or uploaded_model}"

    # 停止之前的流（如果存在）
    if stream_id in rtsp_streams:
        print(f"停止之前的流: {stream_id}")
        rtsp_streams[stream_id]["stop_flag"] = True

    # 创建新的停止标志
    rtsp_streams[stream_id] = {"stop_flag": False}

    # 在请求上下文中创建推理管理器并加载模型
    try:
        print("正在加载模型...")
        inference_manager = InferenceManager(project_id)

        # 根据模型类型调用不同的加载方式
        if model_type == "uploaded":
            model = inference_manager.load_model("uploaded", model_file=uploaded_model)
        else:
            model = inference_manager.load_model(
                model_type, system_model, uploaded_model, existing_model
            )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")

        # 返回错误响应
        def error_generator():
            error_frame = generate_error_frame(f"模型加载失败: {str(e)}")
            ret, buffer = cv2.imencode(".jpg", error_frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        return Response(
            error_generator(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def generate_frames(
        rtsp_url,
        model_type,
        system_model,
        existing_model,
        model,
        inference_manager,
        stream_id,
    ):
        cap = None
        try:

            print(
                f"RTSP流参数: URL={rtsp_url}, 模型类型={model_type}, 系统模型={system_model}, 现有模型={existing_model}"
            )

            if not rtsp_url:
                print("错误: 未提供RTSP URL")
                return

            # 模型已在请求上下文中加载

            # 打开RTSP流
            print(f"正在连接RTSP流: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)

            # 设置连接超时和缓冲区大小
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                print(f"错误: 无法打开RTSP流 {rtsp_url}")
                # 生成错误图像
                error_frame = generate_error_frame(
                    "无法连接RTSP流，请检查地址和网络连接"
                )
                ret, buffer = cv2.imencode(".jpg", error_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                return

            print("RTSP流连接成功，开始处理帧...")
            frame_count = 0

            while True:
                # 检查停止标志
                if stream_id in rtsp_streams and rtsp_streams[stream_id]["stop_flag"]:
                    print(f"收到停止信号，退出流: {stream_id}")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("警告: 无法读取帧，可能是网络中断")
                    # 生成错误图像
                    error_frame = generate_error_frame("视频流中断，正在尝试重连...")
                    ret, buffer = cv2.imencode(".jpg", error_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                    time.sleep(1)  # 等待1秒后重试
                    continue

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧")

                # 进行推理
                results = model(frame)
                annotated_frame = results[0].plot()

                # 编码为JPEG
                ret, buffer = cv2.imencode(
                    ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if not ret:
                    print("警告: 帧编码失败")
                    continue

                frame_bytes = buffer.tobytes()

                # 生成多部分响应
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

                # 控制帧率
                time.sleep(0.033)  # 约30fps

        except Exception as e:
            print(f"RTSP流处理错误: {e}")
            import traceback

            traceback.print_exc()

            # 生成错误图像
            try:
                error_frame = generate_error_frame(f"处理错误: {str(e)}")
                ret, buffer = cv2.imencode(".jpg", error_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            except:
                pass
        finally:
            if cap is not None:
                cap.release()
                print("RTSP流已释放")
            # 清理流状态
            if stream_id in rtsp_streams:
                del rtsp_streams[stream_id]
                print(f"清理流状态: {stream_id}")

    return Response(
        generate_frames(
            rtsp_url,
            model_type,
            system_model,
            existing_model,
            model,
            inference_manager,
            stream_id,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@main.route("/api/project/<int:project_id>/rtsp_stream/stop", methods=["POST"])
def api_stop_rtsp_stream(project_id):
    """停止RTSP实时推理流"""
    try:
        data = request.get_json() or {}
        rtsp_url = data.get("rtsp_url", "")
        model_type = data.get("model_type", "system")
        system_model = data.get("system_model", "yolov8n")
        existing_model = data.get("existing_model")
        uploaded_model = data.get("uploaded_model")

        # 创建流ID（与video_feed中的逻辑一致）
        stream_id = f"{project_id}_{rtsp_url}_{model_type}_{system_model or existing_model or uploaded_model}"

        # 设置停止标志
        if stream_id in rtsp_streams:
            rtsp_streams[stream_id]["stop_flag"] = True
            print(f"设置停止标志: {stream_id}")
            return jsonify({"success": True, "message": "停止信号已发送"})
        else:
            print(f"未找到活动流: {stream_id}")
            return jsonify({"success": True, "message": "未找到活动流"})

    except Exception as e:
        print(f"停止RTSP流错误: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def generate_error_frame(message):
    """生成错误提示图像"""
    import numpy as np

    # 创建黑色背景
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 255)  # 红色
    thickness = 2

    # 计算文字位置
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)

    return frame


# LLM配置相关路由
@main.route("/llm-config")
def llm_config():
    """LLM配置页面"""
    return render_template("llm_config.html")


@main.route("/api/llm-configs", methods=["GET"])
def api_get_llm_configs():
    """获取所有LLM配置"""
    try:
        configs = LLMConfig.query.order_by(LLMConfig.created_at.desc()).all()
        config_list = []
        for config in configs:
            config_list.append(
                {
                    "id": config.id,
                    "name": config.name,
                    "base_url": config.base_url,
                    "model": config.model,
                    "is_active": config.is_active,
                    "created_at": config.created_at.isoformat(),
                }
            )
        return jsonify({"success": True, "configs": config_list})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm-configs", methods=["POST"])
def api_create_llm_config():
    """创建LLM配置"""
    try:
        data = request.get_json()

        # 验证必填字段
        required_fields = ["name", "base_url", "model"]
        for field in required_fields:
            if not data.get(field):
                return (
                    jsonify({"success": False, "message": f"{field} 是必填字段"}),
                    400,
                )

        # api_key 对于本地模型（如Ollama）可以为空
        api_key = data.get("api_key", "")

        # 创建新配置
        config = LLMConfig(
            name=data["name"],
            base_url=data["base_url"],
            api_key=api_key,
            model=data["model"],
        )

        db.session.add(config)
        db.session.commit()

        return jsonify(
            {"success": True, "message": "配置创建成功", "config_id": config.id}
        )
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm-configs/<int:config_id>", methods=["GET"])
def api_get_llm_config(config_id):
    """获取单个LLM配置"""
    try:
        config = LLMConfig.query.get_or_404(config_id)
        return jsonify(
            {
                "success": True,
                "config": {
                    "id": config.id,
                    "name": config.name,
                    "base_url": config.base_url,
                    "api_key": config.api_key,
                    "model": config.model,
                    "is_active": config.is_active,
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm-configs/<int:config_id>", methods=["PUT"])
def api_update_llm_config(config_id):
    """更新LLM配置"""
    try:
        config = LLMConfig.query.get_or_404(config_id)
        data = request.get_json()

        # 更新字段
        if "name" in data:
            config.name = data["name"]
        if "base_url" in data:
            config.base_url = data["base_url"]
        if "api_key" in data:
            config.api_key = data["api_key"]
        if "model" in data:
            config.model = data["model"]

        config.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({"success": True, "message": "配置更新成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm-configs/<int:config_id>", methods=["DELETE"])
def api_delete_llm_config(config_id):
    """删除LLM配置"""
    try:
        config = LLMConfig.query.get_or_404(config_id)
        db.session.delete(config)
        db.session.commit()

        return jsonify({"success": True, "message": "配置删除成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm-configs/<int:config_id>/activate", methods=["POST"])
def api_activate_llm_config(config_id):
    """激活LLM配置"""
    try:
        # 先将所有配置设为非激活状态
        LLMConfig.query.update({"is_active": False})

        # 激活指定配置
        config = LLMConfig.query.get_or_404(config_id)
        config.is_active = True
        config.updated_at = datetime.utcnow()

        db.session.commit()

        return jsonify({"success": True, "message": "配置激活成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm-configs/active", methods=["GET"])
def api_get_active_llm_config():
    """获取当前激活的LLM配置"""
    try:
        config = LLMConfig.query.filter_by(is_active=True).first()
        if config:
            return jsonify(
                {
                    "success": True,
                    "config": {
                        "id": config.id,
                        "name": config.name,
                        "base_url": config.base_url,
                        "api_key": config.api_key,
                        "model": config.model,
                    },
                }
            )
        else:
            return jsonify({"success": False, "message": "没有激活的LLM配置"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/llm/config/active", methods=["GET"])
def get_active_llm_config():
    """获取当前激活的LLM配置"""
    try:
        config = LLMConfig.query.filter_by(is_active=True).first()
        if config:
            return jsonify(
                {
                    "success": True,
                    "config": {
                        "id": config.id,
                        "name": config.name,
                        "base_url": config.base_url,
                        "model": config.model,
                    },
                }
            )
        else:
            return jsonify({"success": False, "message": "没有激活的LLM配置"})
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"LLM检测API错误: {e}")
        print(f"错误详情: {error_details}")
        return jsonify({"success": False, "message": str(e)})


@main.route("/api/project/<int:project_id>/upload_model", methods=["POST"])
def upload_model(project_id):
    """上传用户模型文件"""
    try:
        # 检查项目是否存在
        project = Project.query.get_or_404(project_id)

        # 检查是否有文件上传
        if "model_file" not in request.files:
            return jsonify({"success": False, "error": "没有选择文件"})

        file = request.files["model_file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "没有选择文件"})

        # 检查文件扩展名
        allowed_extensions = {".pt", ".pth", ".onnx"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify(
                {
                    "success": False,
                    "error": f'不支持的文件格式。支持的格式: {", ".join(allowed_extensions)}',
                }
            )

        # 创建上传目录
        upload_dir = os.path.join("projects", str(project_id), "uploaded_models")
        os.makedirs(upload_dir, exist_ok=True)

        # 生成安全的文件名
        filename = secure_filename(file.filename)
        # 如果文件已存在，添加时间戳
        if os.path.exists(os.path.join(upload_dir, filename)):
            name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}{ext}"

        # 保存文件
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        return jsonify(
            {"success": True, "filename": filename, "message": "模型上传成功"}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@main.route("/api/project/<int:project_id>/uploaded_models", methods=["GET"])
def get_uploaded_models(project_id):
    """获取已上传的模型列表"""
    try:
        # 检查项目是否存在
        project = Project.query.get_or_404(project_id)

        # 使用InferenceManager获取已上传的模型列表
        inference_manager = InferenceManager(project_id)
        models = inference_manager.get_uploaded_models()

        return jsonify({"success": True, "models": models})

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "models": []}), 500


@main.route("/api/project/<int:project_id>/ai-assist", methods=["GET"])
def api_get_project_ai_assist(project_id):
    """获取项目的AI辅助标注设置"""
    project = Project.query.get_or_404(project_id)
    return jsonify({"ai_assist_enabled": project.ai_assist_enabled})


@main.route("/api/project/<int:project_id>/ai-assist", methods=["PUT"])
def api_update_project_ai_assist(project_id):
    """更新项目的AI辅助标注设置"""
    project = Project.query.get_or_404(project_id)
    data = request.get_json()

    if "ai_assist_enabled" in data:
        project.ai_assist_enabled = bool(data["ai_assist_enabled"])
        db.session.commit()

        return jsonify(
            {"success": True, "ai_assist_enabled": project.ai_assist_enabled}
        )

    return jsonify({"error": "缺少必要参数"}), 400


@main.route("/api/llm/detect", methods=["POST"])
def llm_detect():
    """LLM检测接口"""
    try:
        data = request.get_json()
        project_id = data.get("project_id")
        image_data = data.get("image_data")

        if not project_id or not image_data:
            return jsonify({"success": False, "message": "缺少必要参数"}), 400

        # 获取项目和标签
        project = Project.query.get_or_404(project_id)
        labels = Label.query.filter_by(project_id=project_id).all()

        if not labels:
            return (
                jsonify({"success": False, "message": "项目中没有标签，请先创建标签"}),
                400,
            )

        # 使用LLM服务进行检测
        from services.llm_service import LLMService

        llm_service = LLMService()

        # 使用新的base64检测方法
        detections = llm_service.detect_objects_from_base64(image_data, labels)

        return jsonify({"success": True, "detections": detections})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/task-annotation", methods=["POST"])
def create_task_annotation():
    """
    创建标注任务（含批量创建任务明细）
    请求方式：POST
    请求体（JSON）：
    {
        "project_id": 1,          // 关联项目ID（必传）
        "start_time": "2025-01-01",// 任务开始日期（必传，格式：YYYY-MM-DD）
        "end_time": "2025-02-01",  // 任务结束日期（必传，格式：YYYY-MM-DD）
        "principal": "张三",       // 任务负责人（必传）
        "image_ids": [1, 2, 3]     // 待标注图片ID列表（必传，至少一个）
    }
    """
    try:
        # 1. 获取并校验请求数据（JSON格式）
        if not request.is_json:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "请求格式错误，请使用JSON格式提交数据",
                    }
                ),
                400,
            )

        req_data = request.get_json()
        # 提取必传参数
        project_id = req_data.get("project_id")
        start_time_str = req_data.get("start_time")
        end_time_str = req_data.get("end_time")
        principal = req_data.get("principal")
        image_ids = req_data.get("image_ids", [])

        # 2. 参数合法性校验
        # 校验非空参数
        missing_params = []
        if not project_id:
            missing_params.append("project_id")
        if not start_time_str:
            missing_params.append("start_time")
        if not end_time_str:
            missing_params.append("end_time")
        if not principal:
            missing_params.append("principal")
        if not isinstance(image_ids, list) or len(image_ids) == 0:
            missing_params.append("image_ids（需为非空列表）")

        if missing_params:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"缺少必要参数：{', '.join(missing_params)}",
                    }
                ),
                400,
            )

        # 校验日期格式和有效性
        try:
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d").date()
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d").date()
        except ValueError:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "日期格式错误，请使用YYYY-MM-DD格式（例如：2025-01-01）",
                    }
                ),
                400,
            )

        if start_time > end_time:
            return (
                jsonify({"success": False, "message": "开始时间不能晚于结束时间"}),
                400,
            )

        # 校验图片ID有效性（确保图片存在且归属当前项目）
        valid_image_ids = []
        for image_id in image_ids:
            image = Image.query.filter_by(id=image_id, project_id=project_id).first()
            if image:
                valid_image_ids.append(image_id)

        if len(valid_image_ids) == 0:
            return (
                jsonify(
                    {"success": False, "message": "未找到有效图片，无法创建任务明细"}
                ),
                400,
            )

        # 3. 业务逻辑处理：创建任务主表 + 批量创建任务明细
        # 创建标注任务主表记录
        new_task = AnnotationTask(
            project_id=project_id,
            start_time=start_time,
            end_time=end_time,
            principal=principal,
            total_count=len(valid_image_ids),
            completed_count=0,
            is_submitted=0,  # 初始状态：未提交
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.session.add(new_task)
        db.session.flush()  # 刷新获取新任务id，用于创建明细

        # 批量创建任务明细记录
        task_items = []
        for image_id in valid_image_ids:
            task_item = AnnotationTaskItem(
                task_id=new_task.id,
                image_id=image_id,
                annotate_status=0,  # 初始状态：未标注
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            task_items.append(task_item)

        if task_items:
            db.session.add_all(task_items)

        db.session.commit()

        # 5. 响应结果（两种选择：返回JSON数据 或 重定向到项目详情页）
        # 选项A：API接口标准返回（推荐，便于前端异步调用）
        return (
            jsonify(
                {
                    "success": True,
                    "message": "标注任务创建成功",
                    "data": {
                        "task_id": new_task.id,
                        "project_id": new_task.project_id,
                        "principal": new_task.principal,
                        "item_count": len(task_items),
                        "redirect_url": url_for(
                            "main.project_detail", project_id=project_id
                        ),
                    },
                }
            ),
            201,
        )

        # 选项B：重定向到项目详情页（对应你原代码的需求）
        # 注意：重定向仅适用于同步表单提交，API接口一般不推荐重定向
        # return redirect(url_for("main.project_detail", project_id=project_id))

    except SQLAlchemyError as e:
        # 数据库异常，回滚事务
        db.session.rollback()
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500

    except Exception as e:
        # 其他未知异常
        return jsonify({"success": False, "message": f"创建任务失败：{str(e)}"}), 500


@main.route("/api/task-annotation/<int:task_id>", methods=["PUT"])
def update_task_annotation(task_id):
    """
    修改标注任务（仅未提交任务可修改）
    请求方式：PUT
    请求地址：/api/task-annotation/1（task_id为任务主键ID）
    请求体（JSON，支持部分字段更新）：
    {
        "start_time": "2025-01-05",  // 可选：新任务开始日期
        "end_time": "2025-02-10",    // 可选：新任务结束日期
        "principal": "李四"          // 可选：新负责人姓名
        "image_ids": [4, 5, 6]       // 可选：新增的图片ID列表
    }
    """
    try:

        req_data = request.get_json()
        task: AnnotationTask = AnnotationTask.query.get(task_id)
        if not task:
            return (
                jsonify(
                    {"success": False, "message": f"未找到ID为{task_id}的标注任务"}
                ),
                404,
            )

        if task.is_submitted == 1:
            return jsonify({"success": False, "message": "该任务已提交，禁止修改"}), 403

        start_time_str = req_data.get("start_time")
        end_time_str = req_data.get("end_time")
        principal = req_data.get("principal")
        image_ids = req_data.get("image_ids", [])

        start_time = datetime.strptime(start_time_str, "%Y-%m-%d").date()
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d").date()

        task.start_time = start_time
        task.end_time = end_time
        task.principal = principal
        task.total_count = len(image_ids)
        task.completed_count = 0
        task.updated_at = datetime.utcnow()

        # 删除已有的任务项，重新添加新的任务项
        AnnotationTaskItem.query.filter_by(task_id=task_id).delete(
            synchronize_session=False
        )
        new_task_items = []
        for image_id in image_ids:
            task_item = AnnotationTaskItem(
                task_id=task.id,
                image_id=image_id,
                annotate_status=0,  # 初始状态：未标注
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            new_task_items.append(task_item)

        if new_task_items:
            db.session.add_all(new_task_items)

        db.session.commit()

        return (
            jsonify(
                {
                    "success": True,
                    "message": "标注任务修改成功",
                    "data": {
                        "task_id": task.id,
                        "project_id": task.project_id,
                        "start_time": task.start_time.strftime("%Y-%m-%d"),
                        "end_time": task.end_time.strftime("%Y-%m-%d"),
                        "principal": task.principal,
                    },
                }
            ),
            200,
        )

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500

    except Exception as e:
        return jsonify({"success": False, "message": f"修改任务失败：{str(e)}"}), 500


@main.route("/api/task-annotation/<int:task_id>", methods=["DELETE"])
def delete_task_annotation(task_id):
    """
    删除标注任务（仅未提交任务可删除，级联删除任务明细）
    请求方式：DELETE
    请求地址：/api/task-annotation/1（task_id为任务主键ID）
    """
    try:
        task = AnnotationTask.query.get(task_id)
        if not task:
            return (
                jsonify(
                    {"success": False, "message": f"未找到ID为{task_id}的标注任务"}
                ),
                404,
            )

        if task.is_submitted == 1:
            return jsonify({"success": False, "message": "该任务已提交，禁止删除"}), 403

        db.session.delete(task)

        # 删除任务项
        AnnotationTaskItem.query.filter_by(task_id=task_id).delete(
            synchronize_session=False
        )

        db.session.commit()

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"ID为{task_id}的标注任务及对应明细已成功删除",
                }
            ),
            200,
        )

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500

    except Exception as e:
        return jsonify({"success": False, "message": f"删除任务失败：{str(e)}"}), 500


@main.route("/api/task-annotation/item", methods=["DELETE"])
def delete_task_annotation_item():
    """
    删除标注任务明细项
    请求方式：DELETE
    请求体（JSON）：
    {
        "item_ids": [1, 2, 3]  // 待删除的任务明细项ID列表（必传，至少一个）
    }
    """
    try:
        data = request.get_json()
        item_ids = data.get("item_ids", [])
        if not isinstance(item_ids, list) or len(item_ids) == 0:
            return (
                jsonify({"success": False, "message": "item_ids必须是一个非空列表"}),
                400,
            )

        deleted_count = AnnotationTaskItem.query.filter(
            AnnotationTaskItem.id.in_(item_ids)
        ).delete(synchronize_session=False)

        db.session.commit()

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"成功删除{deleted_count}个任务明细项",
                }
            ),
            200,
        )

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500

    except Exception as e:
        return (
            jsonify({"success": False, "message": f"删除任务明细项失败：{str(e)}"}),
            500,
        )


@main.route("/api/task-annotation/<int:task_id>", methods=["GET"])
def get_task_annotation(task_id):
    """
    获取单个标注任务详情（用于编辑任务弹窗填充数据）
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    try:
        task = AnnotationTask.query.get(task_id)
        if not task:
            return (
                jsonify(
                    {"success": False, "message": f"未找到ID为{task_id}的标注任务"}
                ),
                404,
            )
        task_items_pagination = AnnotationTaskItem.query.filter_by(
            task_id=task_id
        ).paginate(page=page, per_page=per_page, error_out=False)
        task_items: List[AnnotationTaskItem] = task_items_pagination.items

        image_ids = [item.image_id for item in task_items]
        images = Image.query.filter(Image.id.in_(image_ids)).all()

        new_items = []
        for item in task_items:
            item: AnnotationTaskItem
            # 关联图片数据
            image: Image = next(
                (img for img in images if img.id == item.image_id), None
            )
            item.original_filename = image.original_filename
            new_items.append(
                {
                    "id": item.id,
                    "image_id": item.image_id,
                    "original_filename": item.original_filename,
                    "path": image.path,
                    "width": image.width,
                    "height": image.height,
                    "annotate_status": item.annotate_status,
                }
            )

        task_data = {
            "id": task.id,
            "project_id": task.project_id,
            "start_time": task.start_time.strftime("%Y-%m-%d"),
            "end_time": task.end_time.strftime("%Y-%m-%d"),
            "principal": task.principal,
            "is_submitted": task.is_submitted,
            "progress": "",  # 可从任务明细表计算后返回
            "task_items": new_items,
            "task_items_pagination": {
                "page": task_items_pagination.page,
                "total_pages": task_items_pagination.pages,
                "total_items": task_items_pagination.total,
            },
        }

        return jsonify({"success": True, "data": task_data}), 200

    except SQLAlchemyError as e:
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500


@main.route("/api/task-annotation/submit-multiple", methods=["POST"])
def sub_task_annotation():
    """
    提交标注任务（将任务状态设为已提交）
    请求方式：POST
    请求地址：/api/task-annotation/1/submit（task_id为任务主键ID）
    """
    try:
        data = request.get_json()
        task_ids = data.get("task_ids")
        if not isinstance(task_ids, list):
            return (
                jsonify({"success": False, "message": "task_ids必须是一个列表"}),
                400,
            )

        for task_id in task_ids:
            task = AnnotationTask.query.get(task_id)
            if not task:
                db.session.rollback()  # 若前面已有修改，回滚避免脏数据
                return (
                    jsonify(
                        {"success": False, "message": f"未找到ID为{task_id}的标注任务"}
                    ),
                    404,
                )
            if task.is_submitted == 1:
                continue
            task.is_submitted = 1
            task.updated_at = datetime.utcnow()

        db.session.commit()

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"选中的{len(task_ids)}个任务（有效未提交任务已成功提交）",
                }
            ),
            200,
        )

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500

    except Exception as e:
        return jsonify({"success": False, "message": f"提交任务失败：{str(e)}"}), 500


@main.get("/api/task-annotation/<int:project_id>/image-list")
def task_image_list(project_id):
    """图片列表页面"""
    project = Project.query.get_or_404(project_id)
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    per_page = min(per_page, 100)  # 最多显示100张图片

    subquery = (
        db.session.query(AnnotationTaskItem.image_id)
        .filter(AnnotationTaskItem.image_id == Image.id)
        .exists()
    )

    images_query = db.session.query(Image).filter(
        and_(
            Image.project_id == project_id,
            Image.dataset_type == "unassigned",
            not_(subquery),  # 不存在对应的AnnotationTaskItem记录
        )
    )

    images_pagination = images_query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    images: List[Image] = images_pagination.items

    # 后端将数据传递给 Jinja2 模板，渲染完整页面
    return render_template(
        "task_image_select.html",
        project=project,
        pagination=images_pagination,
        images=images,
    )


@main.route("/api/task-annotation/<int:task_id>", methods=["GET"])
def task_annotate_item_form(task_id):
    """
    获取单个标注任务详情（用于编辑任务弹窗填充数据）
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    try:
        task = AnnotationTask.query.get(task_id)
        if not task:
            return (
                jsonify(
                    {"success": False, "message": f"未找到ID为{task_id}的标注任务"}
                ),
                404,
            )
        task_items_pagination = AnnotationTaskItem.query.filter_by(
            task_id=task_id
        ).paginate(page=page, per_page=per_page, error_out=False)
        task_items: List[AnnotationTaskItem] = task_items_pagination.items

        image_ids = [item.image_id for item in task_items]
        images = Image.query.filter(Image.id.in_(image_ids)).all()

        new_items = []
        for item in task_items:
            item: AnnotationTaskItem
            # 关联图片数据
            image: Image = next(
                (img for img in images if img.id == item.image_id), None
            )
            item.original_filename = image.original_filename
            new_items.append(
                {
                    "id": item.id,
                    "image_id": item.image_id,
                    "original_filename": item.original_filename,
                    "path": image.path,
                    "annotate_status": item.annotate_status,
                }
            )

        task_data = {
            "id": task.id,
            "project_id": task.project_id,
            "start_time": task.start_time.strftime("%Y-%m-%d"),
            "end_time": task.end_time.strftime("%Y-%m-%d"),
            "principal": task.principal,
            "is_submitted": task.is_submitted,
            "progress": "",  # 可从任务明细表计算后返回
            "task_items": new_items,
            "task_items_pagination": {
                "page": task_items_pagination.page,
                "total_pages": task_items_pagination.pages,
                "total_items": task_items_pagination.total,
            },
        }

        return jsonify({"success": True, "data": task_data}), 200

    except SQLAlchemyError as e:
        return jsonify({"success": False, "message": f"数据库操作失败：{str(e)}"}), 500


@main.route(
    "/api/task-annotation/<int:project_id>/image/<int:task_id>", methods=["GET"]
)
def task_annotate_image(project_id, task_id, image_id=None):
    """任务标注页面"""
    # 获取图片id
    arg_image_id = request.args.get("image_id", type=int)
    image_id = arg_image_id if arg_image_id else image_id

    project = Project.query.get_or_404(project_id)

    task_image_relations = (
        db.session.query(AnnotationTaskItem, Image)
        .join(Image, AnnotationTaskItem.image_id == Image.id)
        .filter(AnnotationTaskItem.task_id == task_id)
        .order_by(AnnotationTaskItem.created_at.asc())
        .all()
    )

    images = []
    seen_image_ids = set()
    for relation, image in task_image_relations:
        if image.id not in seen_image_ids:
            seen_image_ids.add(image.id)
            image.task_item_id = relation.id  # 将任务明细ID附加到图片对象
            images.append(image)

    current_image = None
    if image_id:
        current_image = Image.query.get_or_404(image_id)
    else:
        current_image = images[0] if images else None

    return render_template(
        "annotate_task_img.html",
        project=project,
        images=images,
        current_image=current_image,
        task_id=task_id,
    )


@main.route("/api/material/<int:project_id>/page")
def inference_material(project_id):
    project = Project.query.get_or_404(project_id)
    return render_template("reasoning_material.html", project=project)


@main.route("/api/material/<int:project_id>/image")
def infer_image_material(project_id):
    project = Project.query.get_or_404(project_id)
    # 获取分页参数，默认第1页，每页10条
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)

    # 查询该项目下的图片素材并分页
    pagination = (
        ReasoningMaterial.query.filter_by(project_id=project_id, material_type=1)
        .order_by(ReasoningMaterial.created_at.desc())
        .paginate(
            page=page, per_page=per_page, error_out=False  # 超出页数不报错，返回空列表
        )
    )
    materials = pagination.items  # 当前页的数据

    # 转换 size_bytes -> KB 字符串
    for m in materials:
        m.size_kb = f"{(m.size_bytes or 0) / 1024:.1f} KB"
        # 转换状态显示
        if m.status == 0:
            m.status_text = "未开始"
            if m.config:
                m.status_text = "未开始，已配置"
            m.status_class = "badge-not-start"
        elif m.status == 1:
            m.status_text = "推理中"
            m.status_class = "badge-inferring"
        elif m.status == 2:
            m.status_text = "已推理"
            m.status_class = "badge-completed"
        else:
            m.status_text = "未知"
            m.status_class = "badge-secondary"

    return render_template(
        "infer/image_material.html",
        project=project,
        materials=materials,
        pagination=pagination,
    )


@main.route("/api/material/<int:project_id>/video")
def infer_video_material(project_id):
    project = Project.query.get_or_404(project_id)
    # 获取分页参数，默认第1页，每页10条
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)

    # 查询该项目下的视频素材并分页
    pagination = (
        ReasoningMaterial.query.filter_by(project_id=project_id, material_type=2)
        .order_by(ReasoningMaterial.created_at.desc())
        .paginate(
            page=page, per_page=per_page, error_out=False  # 超出页数不报错，返回空列表
        )
    )
    materials = pagination.items  # 当前页的数据

    # 转换 size_bytes -> KB 字符串
    for m in materials:
        m.size_kb = f"{(m.size_bytes or 0) / 1024:.1f} KB"
        # 转换状态显示
        if m.status == 0:
            m.status_text = "未开始"
            if m.config:
                m.status_text = "未开始，已配置"
            m.status_class = "badge-not-start"
        elif m.status == 1:
            m.status_text = "推理中"
            m.status_class = "badge-inferring"
        elif m.status == 2:
            m.status_text = "已推理"
            m.status_class = "badge-completed"
        else:
            m.status_text = "未知"
            m.status_class = "badge-secondary"

    return render_template(
        "infer/video_material.html",
        project=project,
        materials=materials,
        pagination=pagination,
    )


@main.route("/api/material/<int:project_id>/rtsp")
def infer_rtsp_material(project_id):

    project = Project.query.get_or_404(project_id)
    # 获取分页参数，默认第1页，每页10条
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)

    # 查询该项目下的RTSP并分页
    pagination = (
        ReasoningMaterial.query.filter_by(project_id=project_id, material_type=3)
        .order_by(ReasoningMaterial.created_at.desc())
        .paginate(
            page=page, per_page=per_page, error_out=False  # 超出页数不报错，返回空列表
        )
    )
    materials = pagination.items  # 当前页的数据

    for m in materials:
        if m.status == 0:
            m.status_text = "未开始"
            if m.config:
                m.status_text = "未开始，已配置"
            m.status_class = "badge-not-start"
        elif m.status == 1:
            m.status_text = "推理中"
            m.status_class = "badge-inferring"
        elif m.status == 2:
            m.status_text = "已推理"
            m.status_class = "badge-completed"
        else:
            m.status_text = "未知"
            m.status_class = "badge-secondary"

    return render_template(
        "infer/rtsp_material.html",
        project=project,
        materials=materials,
        pagination=pagination,
    )


@main.route("/api/material/<int:project_id>/upload_images", methods=["POST"])
def upload_images_material(project_id):
    project = Project.query.get_or_404(project_id)
    upload_base = ProjectDirManager.ensure_project_material_dir(project_id)

    # 处理ZIP文件上传
    if "zip_file" in request.files and request.files["zip_file"].filename != "":
        zip_file = request.files["zip_file"]
        zip_path = os.path.join(upload_base, zip_file.filename)

        # 保存并解压ZIP文件
        os.makedirs(upload_base, exist_ok=True)
        zip_file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(upload_base)
        os.remove(zip_path)

        # 遍历解压后的图片
        for root, dirs, files in os.walk(upload_base):
            for file in files:
                if file.lower().endswith(tuple(ReasoningMaterialService.IMAGE_EXTS)):
                    file_path = os.path.join(root, file)
                    reasoning_material_service.save_material_file(
                        src_path=file_path,
                        project_id=project_id,
                        upload_base=upload_base,
                        material_type=1,  # 图片
                        origin_filename=file,
                    )
        db.session.commit()

    # 处理单个/多个图片上传
    if "imageFiles" in request.files:
        images = request.files.getlist("imageFiles")
        for image_file in images:
            if image_file.filename != "":
                # 先保存临时文件
                temp_path = os.path.join(upload_base, image_file.filename)
                os.makedirs(upload_base, exist_ok=True)
                image_file.save(temp_path)

                # 调用核心方法处理
                reasoning_material_service.save_material_file(
                    src_path=temp_path,
                    project_id=project_id,
                    upload_base=upload_base,
                    material_type=1,  # 图片
                    origin_filename=image_file.filename,
                )
                # 删除临时文件（因为核心方法会重新保存）
                os.remove(temp_path)
        db.session.commit()

    return jsonify({"success": True}), 200


@main.route("/api/material/<int:project_id>/delete", methods=["POST"])
def delete_material(project_id):
    project = Project.query.get_or_404(project_id)
    data = request.get_json()

    if not data or "ids" not in data:
        return jsonify({"success": False, "message": "缺少参数 ids"}), 400

    ids = data["ids"]
    if not isinstance(ids, list) or len(ids) == 0:
        return jsonify({"success": False, "message": "ids 必须是非空列表"}), 400

    materials = ReasoningMaterial.query.filter(
        ReasoningMaterial.project_id == project_id, ReasoningMaterial.id.in_(ids)
    ).all()

    if not materials:
        return jsonify({"success": False, "message": "没有找到可删除的素材"}), 404

    try:
        for m in materials:
            # 图片视频
            if m.material_type in (1, 2):
                if m.path and os.path.exists(m.path):
                    os.remove(m.path)

            if m.cover_image and os.path.exists(m.cover_image):
                os.remove(m.cover_image)

            db.session.delete(m)

        db.session.commit()
        return jsonify(
            {"success": True, "message": f"成功删除 {len(materials)} 个素材"}
        )

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"delete_material error: {e}")
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


@main.route("/api/material/<int:project_id>/config/<int:id>", methods=["GET"])
def material_config_page(project_id, id):
    """
    推理素材配置页面
    """
    project = Project.query.get_or_404(project_id)

    materials = ReasoningMaterial.query.filter_by(
        id=id, project_id=project_id
    ).first_or_404()

    if materials.config:
        material_config_str = json.dumps(materials.config, ensure_ascii=False)
    else:
        material_config_str = ""

    return render_template(
        "material_config.html",
        project=project,
        materials=materials,
        material_config=material_config_str,
    )


@main.route("/api/material/<int:project_id>/config/save", methods=["POST"])
def save_material_config(project_id):
    """
    保存素材配置信息
    """
    data = request.get_json()

    if not data or "config" not in data:
        return jsonify({"success": False, "message": "缺少参数 config"}), 400

    annotations = data["config"]["annotations"]
    labels = data["config"]["labels"]
    material_id = data.get("id")

    if len(labels) == 0 or len(annotations) == 0:
        return jsonify({"success": False, "message": "请先配置区域信息！"}), 400

    material = ReasoningMaterial.query.filter_by(
        id=material_id, project_id=project_id
    ).first()

    material.config = data["config"]
    material.updated_at = datetime.utcnow()

    db.session.commit()

    return jsonify({"success": True}), 200


@main.route("/api/material/<int:project_id>/config/sync", methods=["POST"])
def sync_material_config(project_id):
    """
    将项目级配置同步到该项目下所有推理素材
    """
    data = request.get_json()

    if not data or "config" not in data:
        return jsonify({"success": False, "message": "缺少参数 config"}), 400

    config = data["config"]
    annotations = config.get("annotations", [])
    labels = config.get("labels", [])

    if not labels or not annotations:
        return jsonify({"success": False, "message": "请先配置区域信息！"}), 400

    materials = ReasoningMaterial.query.filter_by(project_id=project_id).all()

    if not materials:
        return jsonify({"success": False, "message": "该项目下暂无素材"}), 404

    now = datetime.utcnow()

    try:
        for material in materials:
            material.config = config
            material.updated_at = now

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": "同步失败"}), 500

    return jsonify({"success": True, "count": len(materials)}), 200


@main.route("/api/material/<int:project_id>/upload_videos", methods=["POST"])
def upload_videos_material(project_id):
    """启动video截图（正确接收视频文件）"""
    project = Project.query.get_or_404(project_id)
    upload_base = ProjectDirManager.ensure_project_material_dir(project_id)

    # 处理视频上传
    if "videoFiles" in request.files:
        videos = request.files.getlist("videoFiles")
        for video_file in videos:
            if video_file.filename != "":
                # 先保存临时文件
                temp_path = os.path.join(upload_base, video_file.filename)
                os.makedirs(upload_base, exist_ok=True)
                video_file.save(temp_path)

                # 调用核心方法处理
                reasoning_material_service.save_material_file(
                    src_path=temp_path,
                    project_id=project_id,
                    upload_base=upload_base,
                    material_type=2,  # 视频
                    origin_filename=video_file.filename,
                )
                # 删除临时文件
                os.remove(temp_path)
        db.session.commit()

    return jsonify({"success": True}), 200


@main.route("/api/material/<int:project_id>/upload_rtsp", methods=["POST"])
def upload_rtsp_material(project_id):
    """上传RTSP流"""
    project = Project.query.get_or_404(project_id)
    upload_base = ProjectDirManager.ensure_project_material_dir(project_id)

    try:
        # 获取请求参数
        filename = request.form.get("rtspName")
        rtsp_url = request.form.get("rtspUrl")
        start_time = request.form.get("startTime")
        end_time = request.form.get("endTime")

        # 参数校验
        if not filename:
            return jsonify({"success": False, "message": "参数 rtspName 不能为空"}), 400
        if not rtsp_url:
            return jsonify({"success": False, "message": "参数 rtspUrl 不能为空"}), 400

        # 调用核心方法处理RTSP
        reasoning_material_service.save_material_file(
            src_path=rtsp_url,
            project_id=project_id,
            upload_base=upload_base,
            material_type=3,  # RTSP
            origin_filename=filename,
            rtsp_url=rtsp_url,
            start_time=start_time,
            end_time=end_time,
        )
        db.session.commit()

        return jsonify({"success": True}), 200

    except Exception as e:
        current_app.logger.error(f"upload_rtsp_material error: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@main.route("/api/inference/labels/<int:label_id>/update", methods=["POST"])
def api_inference_update_label(label_id):
    target = request.form.get("target")

    label = Label.query.get_or_404(label_id)
    label.target = target

    db.session.commit()

    return jsonify({"success": True}), 200


@main.route("/api/inference/<int:project_id>/material", methods=["GET"])
def api_inference_material_page(project_id):
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    project = Project.query.get_or_404(project_id)

    # 获取当前项目的素材
    pagination = (
        ReasoningMaterial.query.filter_by(project_id=project_id)
        .order_by(ReasoningMaterial.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    materials = pagination.items

    for m in materials:
        if m.result and m.result.get("success"):
            m.total_count = m.result.get("result", {}).get("total_count", 0)
        else:
            m.total_count = 0
        if m.status == 0:
            m.status_text = "未开始"
            if m.config:
                m.status_text = "未开始，已配置"
            m.status_class = "badge-not-start"
        elif m.status == 1:
            m.status_text = "推理中"
            m.status_class = "badge-inferring"
        elif m.status == 2:
            m.status_text = "已推理"
            m.status_class = "badge-completed"
        else:
            m.status_text = "未知"
            m.status_class = "badge-secondary"

    return render_template(
        "infer/inference_material.html",
        project=project,
        materials=materials,
        pagination=pagination,
    )


@main.route("/api/inference/<int:project_id>/start", methods=["GET"])
def api_inference_start(project_id):
    project = Project.query.get_or_404(project_id)
    return jsonify({"success": True}), 200


@main.route("/api/inference/<int:project_id>/result/<int:id>", methods=["GET"])
def inference_result_page(project_id, id):
    """
    推理素材配置页面
    """
    project = Project.query.get_or_404(project_id)

    material = ReasoningMaterial.query.filter_by(
        id=id, project_id=project_id
    ).first_or_404()

    # if material.result:
    #     material_result_str = json.dumps(material.result, ensure_ascii=False)
    # else:
    #     material_result_str = ""

    return render_template(
        "infer/inference_result.html",
        project=project,
        material=material,
        material_result=material.result,
    )


@main.route("/api/server/parse", methods=["POST"])
def parse_server_dir():
    path = request.form.get("path")
    file_type = request.form.get("type")  # image / video

    if not path:
        return jsonify({"success": False, "message": "请填写合法路径"})

    abs_path = os.path.abspath(path)

    if not abs_path.startswith(ReasoningMaterialService.BASE_DIR):
        return jsonify({"success": False, "message": "非法路径"}), 403

    if not os.path.isdir(abs_path):
        return jsonify({"success": False, "message": "路径不是目录"})

    if file_type == "image":
        allow_exts = ReasoningMaterialService.IMAGE_EXTS
    elif file_type == "video":
        allow_exts = ReasoningMaterialService.VIDEO_EXTS
    else:
        allow_exts = (
            ReasoningMaterialService.IMAGE_EXTS | ReasoningMaterialService.VIDEO_EXTS
        )  # 不传 type 就都显示

    items = []
    for name in os.listdir(abs_path):
        full = os.path.join(abs_path, name)

        if os.path.isdir(full):
            items.append({"name": name, "path": full, "is_dir": True})
        else:
            ext = os.path.splitext(name)[1].lower()
            if ext not in allow_exts:
                continue

            items.append({"name": name, "path": full, "is_dir": False})

    return jsonify(
        {"success": True, "data": {"path": path, "items": items, "show_modal": True}}
    )


@main.route("/api/server/<int:project_id>/save", methods=["POST"])
def save_server_files(project_id):
    log.info(f"开始保存服务器素材")
    dirs = request.form.getlist("files")
    file_type = request.form.get("type")  # image / video

    if dirs is None or len(dirs) == 0:
        return jsonify({"success": False, "message": "请先选择素材分类"})

    upload_base = ProjectDirManager.ensure_project_material_dir(project_id)

    if file_type == "image":
        allow_exts = ReasoningMaterialService.IMAGE_EXTS
        material_type = 1
    elif file_type == "video":
        allow_exts = ReasoningMaterialService.VIDEO_EXTS
        material_type = 2
    else:
        allow_exts = (
            ReasoningMaterialService.IMAGE_EXTS | ReasoningMaterialService.VIDEO_EXTS
        )  # 不传 type 就都显示

    for dir_path in dirs:
        if not os.path.isdir(dir_path):
            continue  # 跳过不是目录的路径
        source_type = os.path.basename(dir_path.rstrip("/"))

        # 递归遍历目录
        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)

            if not os.path.isfile(full_path):
                continue

            ext = os.path.splitext(name)[1].lower()
            if ext not in allow_exts:
                continue

            reasoning_material_service.save_material_file(
                src_path=full_path,
                project_id=project_id,
                upload_base=upload_base,
                material_type=material_type,
                source_type=source_type,
                origin_filename=name,
            )
    db.session.commit()
    log.info(f"结束保存服务器素材")
    return jsonify({"success": True})


@main.route("/api/project/<int:project_id>/export_zip", methods=["GET"])
def export_materials_zip(project_id):
    project = Project.query.get_or_404(project_id)

    # 创建临时目录
    base_dir = tempfile.mkdtemp(prefix=f"export_project_{project_id}_")

    excel_map = {
        1: "image_excel.xlsx",
        2: "video_excel.xlsx",
        3: "rtsp_excel.xlsx",
    }

    excel_paths = []

    for material_type, excel_name in excel_map.items():
        materials = ReasoningMaterial.query.filter(
            ReasoningMaterial.project_id == project_id,
            ReasoningMaterial.material_type == material_type,
            ReasoningMaterial.status == 2,
        ).all()

        if not materials:
            continue

        data = ReasoningMaterialService.aggregate_by_category(materials, project.name)
        excel_path = os.path.join(base_dir, excel_name)
        ReasoningMaterialService.build_excel(data, excel_path)
        excel_paths.append(excel_path)

    # 生成 zip
    zip_name = f"materials_export_{project.name}.zip"
    zip_path = os.path.join(base_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in excel_paths:
            zipf.write(path, arcname=os.path.basename(path))

    # 直接返回 zip（浏览器下载）
    return send_file(
        zip_path, as_attachment=True, download_name=zip_name, mimetype="application/zip"
    )
