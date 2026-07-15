"""
COCO JSON -> YOLO TXT 数据集格式转换脚本

功能：
    将 COCO 格式的 JSON 标注文件转换为 YOLO 格式的 TXT 标注文件

使用方式：
    cd rsod-agent-platform/backend
    python tools/convert_coco.py
"""

import os
import random
import shutil
import sys
from pathlib import Path

# 项目根目录：tools/ → 项目根目录/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COCO_JSON_FILE = os.path.join(PROJECT_ROOT, "datasets/rsod/raw/annotations/instances_train.json")
RAW_IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets/rsod/raw/images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets/rsod/yolo_dataset")
CLASS_MAPPING = {"aircraft": 0, "oiltank": 1, "overpass": 2, "playground": 3}
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def coco_to_yolo(json_file, output_dir, class_mapping=None):
    import json
    os.makedirs(output_dir, exist_ok=True)
    stats = {"total": 0, "converted": 0, "skipped": 0, "errors": [], "image_files": []}

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  [错误] COCO JSON 读取失败: {str(e)}")
        stats["errors"].append(str(e))
        return stats

    if class_mapping is None:
        category_mapping = {cat["id"]: idx for idx, cat in enumerate(coco_data.get("categories", []))}
    else:
        cat_name_to_id = {cat["name"]: cat["id"] for cat in coco_data.get("categories", [])}
        category_mapping = {}
        for cat_name, class_id in class_mapping.items():
            if cat_name in cat_name_to_id:
                category_mapping[cat_name_to_id[cat_name]] = class_id

    image_annotations = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    images = coco_data.get("images", [])
    stats["total"] = len(images)
    print(f"  COCO 转换开始：{len(images)} 张图像")

    for img_info in images:
        img_id = img_info["id"]
        img_width = img_info["width"]
        img_height = img_info["height"]
        file_name = img_info["file_name"]
        if img_width <= 0 or img_height <= 0:
            stats["skipped"] += 1
            continue
        stats["image_files"].append(file_name)

        yolo_lines = []
        for ann in image_annotations.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in category_mapping:
                continue
            class_id = category_mapping[cat_id]
            x_min, y_min, bbox_w, bbox_h = ann["bbox"]
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            x_min = max(0, min(x_min, img_width))
            y_min = max(0, min(y_min, img_height))
            bbox_w = min(bbox_w, img_width - x_min)
            bbox_h = min(bbox_h, img_height - y_min)
            x_center = (x_min + bbox_w / 2.0) / img_width
            y_center = (y_min + bbox_h / 2.0) / img_height
            width = bbox_w / img_width
            height = bbox_h / img_height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        img_stem = Path(file_name).stem
        txt_file = Path(output_dir) / f"{img_stem}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
        stats["converted"] += 1

    print(f"  转换结果: 总计 {stats['total']}, 成功 {stats['converted']}, 跳过 {stats['skipped']}")
    return stats


def split_dataset(image_files, temp_label_dir):
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    splits = {"train": image_files[:train_end], "val": image_files[train_end:val_end], "test": image_files[val_end:]}
    for split_name, files in splits.items():
        img_out = os.path.join(OUTPUT_DIR, "images", split_name)
        lbl_out = os.path.join(OUTPUT_DIR, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for filename in files:
            src_image = os.path.join(RAW_IMAGE_DIR, filename)
            if os.path.exists(src_image):
                shutil.copy2(src_image, os.path.join(img_out, filename))
                basename = os.path.splitext(filename)[0]
                label_file = os.path.join(temp_label_dir, f"{basename}.txt")
                if os.path.exists(label_file):
                    shutil.copy2(label_file, os.path.join(lbl_out, f"{basename}.txt"))
                else:
                    open(os.path.join(lbl_out, f"{basename}.txt"), "w").close()
        print(f"  {split_name}: {len(files)} 个")


def generate_yaml():
    class_names = sorted(CLASS_MAPPING.keys(), key=lambda x: CLASS_MAPPING[x])
    yaml_content = f"path: ./{os.path.basename(OUTPUT_DIR)}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: {len(class_names)}\nnames:\n"
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  配置文件已生成: {yaml_path}")


def main():
    print("=" * 70)
    print("      COCO -> YOLO 数据集转换流程")
    print("=" * 70)

    if not os.path.exists(COCO_JSON_FILE):
        print(f"\n[错误] COCO JSON 文件不存在: {COCO_JSON_FILE}")
        sys.exit(1)
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"\n[错误] 原始图片目录不存在: {RAW_IMAGE_DIR}")
        sys.exit(1)

    print("\n[1] COCO转YOLO格式")
    temp_label_dir = os.path.join(OUTPUT_DIR, "temp_labels")
    os.makedirs(temp_label_dir, exist_ok=True)
    stats = coco_to_yolo(COCO_JSON_FILE, temp_label_dir, CLASS_MAPPING)
    if stats["total"] == 0:
        shutil.rmtree(temp_label_dir, ignore_errors=True)
        sys.exit(1)

    print("\n[2] 划分数据集")
    split_dataset(stats["image_files"], temp_label_dir)

    print("\n[3] 生成data.yaml")
    generate_yaml()

    shutil.rmtree(temp_label_dir, ignore_errors=True)
    print("\n" + "=" * 70)
    print(f"  处理完成！输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
