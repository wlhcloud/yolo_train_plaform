"""
交通量调查统计表批量生成脚本（独立命令行版，不依赖 Flask 运行环境）

用法示例：
    python export_survey.py "E:/调查视频/和平里东街 -路口南50米处 12-25 XT"
    python export_survey.py "E:/xxx" --model yolov8n.pt --project-id 1 --only-approaching

目录结构约定：
    根目录/
      ├─ 调查点名称（如 "和平里东街 -路口南50米处 12-25 XT"）
      │    └─ 调查道路（如 "和平里东街"）
      │         └─ 运行方向（如 "北向南"）
      │              ├─ 6：00-6：30.MTS
      │              └─ 6：30-7：00.MTS

视频文件名需包含时间段（支持全角冒号），按时间正序识别；
分类映射读取数据库 yolov8_platform.label 表（按 --project-id）；
调查日期取运行当天；每个「运行方向」文件夹生成一个
「交通量调查统计表_YYYY-MM-DD.xlsx」。
"""
import argparse
import os
import sys

from dotenv import dotenv_values


def load_labels_from_db(project_id):
    """从 .env 的 DATABASE_URL 读取 label 表：{英文类别: 中文类别} + 有序中文类别列表"""
    import psycopg2
    from urllib.parse import urlparse

    env = dotenv_values(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    db_url = env.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("请在 .env 中配置 DATABASE_URL")
    # postgresql+psycopg2://user:pass@host:port/dbname
    u = urlparse(db_url.replace("postgresql+psycopg2://", "postgresql://"))
    conn = psycopg2.connect(
        host=u.hostname,
        port=u.port or 5432,
        user=u.username,
        password=u.password,
        dbname=u.path.lstrip("/"),
    )
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, COALESCE(target, name) FROM yolov8_platform.label "
            "WHERE project_id = %s ORDER BY id",
            (project_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        raise RuntimeError(f"label 表中没有 project_id={project_id} 的分类映射，请先在推理页配置")
    labels_map = {name: target for name, target in rows}
    categories = list(dict.fromkeys(labels_map.values()))
    return labels_map, categories


def load_model(model_path):
    import numpy as np
    import torch
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {model_path}（设备: {device}）")
    model = YOLO(model_path).to(device)
    if device == "cuda":
        model = model.half()
    # 预热
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    return model


def persist_minute_stats(project_id, results, db_url):
    """把每分钟统计写入 yolov8_platform.survey_stat 表（同一地点/方向/日期先删后插，幂等）。"""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from models import SurveyStat

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.execute(text("CREATE SCHEMA IF NOT EXISTS yolov8_platform"))
        session.commit()
        SurveyStat.__table__.create(engine, checkfirst=True)

        total_rows = 0
        for r in results:
            session.query(SurveyStat).filter_by(
                project_id=project_id,
                survey_date=r["survey_date"],
                spot_name=r["spot"],
                road=r["road"],
                direction=r["direction"],
            ).delete(synchronize_session=False)
            for minute, counts in r.get("minute_rows", []):
                for cat, cnt in counts.items():
                    session.add(SurveyStat(
                        project_id=project_id,
                        survey_date=r["survey_date"],
                        spot_name=r["spot"],
                        road=r["road"],
                        direction=r["direction"],
                        minute_of_day=int(minute),
                        category=cat,
                        count=int(cnt),
                    ))
                    total_rows += 1
        session.commit()
        print(f"每分钟统计已入库 survey_stat：{len(results)} 个方向文件夹，{total_rows} 条分钟记录")
    except Exception as e:
        session.rollback()
        print(f"警告：每分钟统计入库失败（不影响 Excel 生成）: {e}")
    finally:
        session.close()
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="批量视频推理并生成交通量调查统计表 Excel")
    parser.add_argument("root", help="视频根文件夹（其下为 调查点/道路/方向 三级目录）")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 模型文件路径")
    parser.add_argument("--project-id", type=int, default=1,
                        help="项目ID（用于读取 label 表的中文分类映射）")
    parser.add_argument("--only-approaching", action="store_true",
                        help="仅统计运动轨迹由远到近的物体")
    parser.add_argument("--no-db", action="store_true",
                        help="不把每分钟统计写入数据库 survey_stat 表")
    args = parser.parse_args()

    from services import survey_service

    labels_map, categories = load_labels_from_db(args.project_id)
    print(f"分类映射（{len(categories)} 类）: {categories}")

    model = load_model(args.model)

    def _cb(done, total, current_file, frame_done, frame_total):
        msg = f"[{done}/{total}] {current_file}"
        if frame_total:
            msg += f" 帧进度 {frame_done}/{frame_total}"
        print(msg, flush=True)

    results = survey_service.run_survey(
        root=args.root,
        model=model,
        labels_map=labels_map,
        categories=categories,
        only_approaching=args.only_approaching,
        progress_cb=_cb,
    )

    # 每分钟统计入库（survey_stat 表）
    if not args.no_db:
        env = dotenv_values(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".env"))
        db_url = env.get("DATABASE_URL", "")
        if db_url:
            persist_minute_stats(args.project_id, results, db_url)
        else:
            print("警告：.env 未配置 DATABASE_URL，跳过数据库存储")

    print("\n===== 完成 =====")
    for r in results:
        print(f"{r['spot']} / {r['road']} / {r['direction']}："
              f"{r['video_count']} 个视频，{r['track_count']} 个目标 -> {r['excel_path']}")


if __name__ == "__main__":
    sys.exit(main())
