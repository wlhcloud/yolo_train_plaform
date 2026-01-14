import os
import sqlite3
import argparse

def parse_yolo_line(line):
    """解析一行 YOLO 格式标注"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        return (class_id, x_center, y_center, width, height)
    except ValueError:
        return None

def main(label_dir, db_path):
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 确保 annotations 表存在（可选）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            label_id INTEGER NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            FOREIGN KEY (image_id) REFERENCES image(id)
        )
    ''')
    conn.commit()

    # 获取所有图像文件名 -> id 的映射（假设 filename 不含路径和扩展名）
    cursor.execute("SELECT id, filename FROM image where project_id=4 ")
    image_records = cursor.fetchall()
    
    # 构建 {basename: id} 映射（去除扩展名，只保留主文件名）
    # 例如：'001.jpg' -> '001'
    filename_to_id = {}
    for img_id, fname in image_records:
        base_name = os.path.splitext(fname)[0]  # 去掉 .jpg / .png 等
        filename_to_id[base_name] = img_id

    # 遍历 label_dir 中所有 .txt 文件
    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith('.txt'):
            continue

        base_name = os.path.splitext(txt_file)[0]  # 如 '001'
        if base_name not in filename_to_id:
            print(f"⚠️ 跳过 {txt_file}：未在 images 表中找到匹配的图像文件名")
            continue

        image_id = filename_to_id[base_name]
        txt_path = os.path.join(label_dir, txt_file)

        annotations = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                anno = parse_yolo_line(line)
                if anno is None:
                    print(f"⚠️ {txt_file} 第 {line_num} 行格式无效，跳过")
                    continue
                annotations.append((image_id,) + anno)

        # 批量插入
        if annotations:
            cursor.executemany('''
                INSERT INTO annotations (image_id, label_id, x, y, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', annotations)
            conn.commit()
            print(f"✅ 已导入 {len(annotations)} 条标注：{txt_file} → image_id={image_id}")

    conn.close()
    print("🎉 所有标注导入完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将 YOLO 格式的标注文件批量导入 SQLite 数据库')
    parser.add_argument('--label_dir', type=str, required=True, help='标注文件 (.txt) 所在文件夹路径')
    parser.add_argument('--db', type=str, required=True, help='SQLite 数据库文件路径')
    args = parser.parse_args()
    main(args.label_dir, args.db)


# 执行语句
# python import_annotations_to_sqlite.py --label_dir ./static/datasets/4/labels/train --db ./instance/yolov8_platform.db
