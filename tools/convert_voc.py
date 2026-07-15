#!/usr/bin/env python3
"""
COCO JSON → YOLO TXT 数据集格式转换脚本
功能：
 将 COCO 格式的 JSON 标注文件转换为 YOLO 格式的 TXT 标注文件
 支持完整的数据集处理流程：转换 + 划分 + yaml生成
COCO 格式说明：
 COCO JSON 文件包含三个主要部分：
 - images: 图像信息列表，包含 id, file_name, width, height
 - categories: 类别信息列表，包含 id, name
 - annotations: 标注信息列表，包含 image_id, category_id, bbox
 COCO bbox 格式：[x_min, y_min, width, height]（像素坐标）
 YOLO bbox 格式：[x_center, y_center, width, height]（归一化坐标）
使用方式：
 cd rsod-agent-platform/backend
 python ../tools/convert_coco.py
 配置说明（修改下方配置区域）：
 - COCO_JSON_FILE: COCO JSON 标注文件路径
 - CLASS_MAPPING: 类别名称到数字ID的映射
 - OUTPUT_DIR: YOLO 格式输出目录
输出目录结构：
 yolo_dataset/
 ├── data.yaml
 ├── images/
 │   ├── train/
 │   ├── val/
 │   └── test/
 └── labels/
     ├── train/
     ├── val/
     └── test/
"""

import os
import random
import shutil
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ── 配置区域（根据你的数据集修改）────────────────────
# COCO JSON 标注文件路径
COCO_JSON_FILE = os.path.join(
    PROJECT_ROOT, "datasets/rsod/raw/annotations/instances_train.json"
)
# 原始图片目录（与 COCO JSON 中 file_name 对应的图片位置）
RAW_IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets/rsod/raw/images")
# YOLO 格式输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets/rsod/yolo_dataset")
# 类别映射（格式：{类别名称: 类别ID}）
# 如果设为 None，则按 COCO categories 中的顺序自动编号（从 0 开始）
CLASS_MAPPING = {
    "aircraft": 0,
    "oiltank": 1,
    "overpass": 2,
    "playground": 3,
}
# 数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1（剩余部分）
# 随机种子（保证划分结果可重复）
RANDOM_SEED = 42
# 支持的图片扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def coco_to_yolo(json_file: str, output_dir: str, class_mapping: Optional[dict] = None) -> dict:
    """
    将 COCO JSON 格式转换为 YOLO TXT 格式
    参数：
    json_file: COCO JSON 标注文件路径
    output_dir: YOLO TXT 输出目录
    class_mapping: 类别映射字典 {类别名称: 类别ID}，为 None 时自动编号
    返回：
    转换统计信息字典 {"total": 总数, "converted": 成功数, "skipped": 跳过数, "errors": 错误列表}
    转换流程：
    1. 读取 COCO JSON 文件
    2. 构建类别映射（COCO category_id → YOLO class_id）
    3. 按 image_id 分组所有标注
    4. 遍历每张图像，将 COCO bbox 转换为 YOLO 归一化坐标
    5. 每张图像输出一个 .txt 文件
    """
    import json
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)
    stats = {"total": 0, "converted": 0, "skipped": 0, "errors": [], "image_files": []}
    # 读取 COCO JSON 文件
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f" [错误] COCO JSON 读取失败: {str(e)}")
        stats["errors"].append(str(e))
        return stats
    # 构建类别映射
    if class_mapping is None:
        # 自动按 categories 顺序编号（从 0 开始）
        category_mapping = {
            cat["id"]: idx for idx, cat in enumerate(coco_data.get("categories", []))
        }
        print(f" 自动类别映射: {category_mapping}")
    else:
        # 使用用户提供的映射：建立 coco_category_id → class_id 的关系
        cat_name_to_id = {
            cat["name"]: cat["id"] for cat in coco_data.get("categories", [])
        }
        category_mapping = {}
        for cat_name, class_id in class_mapping.items():
            if cat_name in cat_name_to_id:
                category_mapping[cat_name_to_id[cat_name]] = class_id
            else:
                print(f" [警告] COCO categories 中未找到类别 '{cat_name}'")
    # 按 image_id 分组标注
    image_annotations = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    # 遍历每张图像进行转换
    images = coco_data.get("images", [])
    stats["total"] = len(images)
    print(f" COCO 转换开始：{len(images)} 张图像")
    for img_info in images:
        img_id = img_info["id"]
        img_width = img_info["width"]
        img_height = img_info["height"]
        file_name = img_info["file_name"]
        # 检查图像尺寸有效性
        if img_width <= 0 or img_height <= 0:
            print(f" [警告] 图像 {file_name} 尺寸无效，跳过")
            stats["skipped"] += 1
            continue
        # 收集图像文件名（用于后续划分）
        stats["image_files"].append(file_name)
        # 转换标注
        yolo_lines = []
        for ann in image_annotations.get(img_id, []):
            # 检查类别是否在映射中
            cat_id = ann["category_id"]
            if cat_id not in category_mapping:
                continue
            class_id = category_mapping[cat_id]
            # COCO bbox: [x_min, y_min, width, height]（像素坐标）
            x_min, y_min, bbox_w, bbox_h = ann["bbox"]
            # 过滤无效框
            if bbox_w <= 0 or bbox_h <= 0:
                continue
            # 边界值裁剪：确保坐标不超出图像范围
            x_min = max(0, min(x_min, img_width))
            y_min = max(0, min(y_min, img_height))
            bbox_w = min(bbox_w, img_width - x_min)
            bbox_h = min(bbox_h, img_height - y_min)
            # COCO [x_min, y_min, w, h] → YOLO [x_center, y_center, w, h]（归一化）
            x_center = (x_min + bbox_w / 2.0) / img_width
            y_center = (y_min + bbox_h / 2.0) / img_height
            width = bbox_w / img_width
            height = bbox_h / img_height
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
        # 保存 YOLO 标注文件
        img_stem = Path(file_name).stem
        txt_file = Path(output_dir) / f"{img_stem}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
        stats["converted"] += 1
    print(
        f" 转换结果: 总计 {stats['total']}, 成功 {stats['converted']}, 跳过 {stats['skipped']}"
    )
    return stats


def split_dataset(image_files, temp_label_dir):
    """
    按比例划分数据集并复制文件
    参数：
    image_files: 图像文件名列表
    temp_label_dir: 临时标注目录
    返回：
    None
    """
    # 设置随机种子，保证划分结果可重复
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    # 计算划分索引
    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }
    # 复制图片和标注到对应目录
    for split_name, files in splits.items():
        img_out = os.path.join(OUTPUT_DIR, "images", split_name)
        lbl_out = os.path.join(OUTPUT_DIR, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for filename in files:
            # 复制图片
            src_image = os.path.join(RAW_IMAGE_DIR, filename)
            if os.path.exists(src_image):
                dst_image = os.path.join(img_out, filename)
                shutil.copy2(src_image, dst_image)
            # 复制标注（如果存在）
            basename = os.path.splitext(filename)[0]
            label_file = os.path.join(temp_label_dir, f"{basename}.txt")
            if os.path.exists(label_file):
                shutil.copy2(label_file, os.path.join(lbl_out, f"{basename}.txt"))
            else:
                open(os.path.join(lbl_out, f"{basename}.txt"), "w").close()
        print(f" {split_name}: {len(files)} 个")


def generate_yaml():
    """生成 YOLO 数据集配置文件"""
    # 获取类别列表
    if CLASS_MAPPING:
        class_names = sorted(CLASS_MAPPING.keys(), key=lambda x: CLASS_MAPPING[x])
    else:
        class_names = []
    # 构建 YAML 内容
    yaml_content = f"""path: ./{os.path.basename(OUTPUT_DIR)}
train: images/train
val: images/val
test: images/test
nc: {len(class_names)}
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f" {i}: {name}\n"
    # 写入 YAML 文件
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f" 配置文件已生成: {yaml_path}")


def main():
    """主函数：执行完整的 COCO → YOLO 转换流程"""
    print("=" * 70)
    print(" COCO → YOLO 数据集转换流程")
    print("=" * 70)
    # 检查 COCO JSON 文件是否存在
    if not os.path.exists(COCO_JSON_FILE):
        print(f"\n[错误] COCO JSON 文件不存在: {COCO_JSON_FILE}")
        print("请检查配置区域中的 COCO_JSON_FILE 路径")
        sys.exit(1)
    # 检查原始图片目录是否存在
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"\n[错误] 原始图片目录不存在: {RAW_IMAGE_DIR}")
        print("请检查配置区域中的 RAW_IMAGE_DIR 路径")
        sys.exit(1)
    # ── 步骤1：COCO转YOLO格式 ──
    print("\n[1] COCO转YOLO格式")
    # 创建临时标注目录（用于存放转换后的TXT文件）
    temp_label_dir = os.path.join(OUTPUT_DIR, "temp_labels")
    os.makedirs(temp_label_dir, exist_ok=True)
    # 执行转换
    stats = coco_to_yolo(COCO_JSON_FILE, temp_label_dir, CLASS_MAPPING)
    if stats["total"] == 0:
        print("\n[错误] 未找到任何图像数据，请检查 COCO JSON 文件")
        shutil.rmtree(temp_label_dir, ignore_errors=True)
        sys.exit(1)
    # ── 步骤2：划分数据集 ──
    print("\n[2] 划分数据集")
    split_dataset(stats["image_files"], temp_label_dir)
    # ── 步骤3：生成data.yaml ──
    print("\n[3] 生成data.yaml")
    generate_yaml()
    # 清理临时目录
    shutil.rmtree(temp_label_dir, ignore_errors=True)
    # 输出完成信息
    print("\n" + "=" * 70)
    print(f" 处理完成！输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
