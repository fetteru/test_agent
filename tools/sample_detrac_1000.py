"""
UA-DETRAC 抽取 10000 条样本用于模型训练

步骤：
  1. 遍历所有视频帧，随机无放回抽取 10000 张图片+对应标注
  2. 生成精简 XML 标注文件
  3. 复用 DataConverter 做 VOC → YOLO 转换
  4. 复用 DatasetSplitter 划分 train/val/test

使用方式：
    cd test_agent
    python tools/sample_detrac_1000.py
"""

import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== 配置区 =====================
RAW_IMG = Path(PROJECT_ROOT) / "myprojectdata" / "DETRAC-Images" / "DETRAC-Images"
RAW_XML = (
    Path(PROJECT_ROOT)
    / "myprojectdata"
    / "DETRAC-Train-Annotations-XML"
    / "DETRAC-Train-Annotations-XML"
)

OUTPUT = Path(PROJECT_ROOT) / "datasets" / "UA-DETRAC" / "detrac_1000_subset"
OUT_IMG = OUTPUT / "images"
OUT_XML = OUTPUT / "xml_annotations"

SAMPLE_NUM = 10000
random.seed(42)
# ==================================================


def main():
    # 创建输出文件夹
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_XML.mkdir(parents=True, exist_ok=True)

    # 1. 搜集全部图片绝对路径
    print("[1] 搜集全部图片...")
    all_img_paths = []
    for seq_dir in RAW_IMG.iterdir():
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        xml_path = RAW_XML / f"{seq_name}.xml"
        if not xml_path.exists():
            continue
        for img in seq_dir.glob("*.jpg"):
            all_img_paths.append(
                {
                    "img_path": img,
                    "seq_name": seq_name,
                    "xml_path": xml_path,
                }
            )

    print(f"  全集总图片数量：{len(all_img_paths)}")

    # 2. 随机抽样
    print(f"\n[2] 随机抽样 {SAMPLE_NUM} 张...")
    random.shuffle(all_img_paths)
    sample_list = all_img_paths[:SAMPLE_NUM]

    # 按视频序列分组
    seq_sample_map = defaultdict(list)
    for item in sample_list:
        seq_sample_map[item["seq_name"]].append(item)

    # 3. 逐个处理每个视频的 XML
    print("\n[3] 处理标注...")
    for seq_name, item_list in seq_sample_map.items():
        xml_file = item_list[0]["xml_path"]
        tree = ET.parse(xml_file)
        root = tree.getroot()

        picked_frame_ids = set()
        for item in item_list:
            img_stem = item["img_path"].stem
            frame_id = int(img_stem.replace("img", ""))
            picked_frame_ids.add(frame_id)
            shutil.copy2(item["img_path"], OUT_IMG / item["img_path"].name)

        new_root = ET.Element("sequence")
        new_root.set("name", seq_name)

        for frame_elem in root.findall("frame"):
            fid = int(frame_elem.get("num", frame_elem.get("id", 0)))
            if fid in picked_frame_ids:
                new_root.append(frame_elem)

        new_tree = ET.ElementTree(new_root)
        new_tree.write(
            OUT_XML / f"{seq_name}.xml", encoding="utf-8", xml_declaration=True
        )

    print(f"  抽样完成，共 {len(sample_list)} 张图片")
    print(f"  图片：{OUT_IMG}")
    print(f"  标注：{OUT_XML}")

    # 4. UA-DETRAC XML → YOLO TXT 直接转换
    print("\n[4] UA-DETRAC → YOLO 格式转换...")
    CLASS_NAMES = ["car", "bus", "van", "others"]
    CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    labels_dir = OUTPUT / "labels_all"
    labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for seq_name, item_list in seq_sample_map.items():
        xml_file = OUT_XML / f"{seq_name}.xml"
        if not xml_file.exists():
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 构建 {帧号: [targets]} 映射
        frame_data = {}
        for frame_elem in root.findall("frame"):
            fid = int(frame_elem.get("num", frame_elem.get("id", 0)))
            targets = []
            for target in frame_elem.iter("target"):
                box = target.find("box")
                attr = target.find("attribute")
                if box is None:
                    continue
                xmin = float(box.get("left", 0))
                ymin = float(box.get("top", 0))
                w = float(box.get("width", 0))
                h = float(box.get("height", 0))
                vtype = (
                    attr.get("vehicle_type", "others") if attr is not None else "others"
                )
                class_name = vtype if vtype in CLASS_MAPPING else "others"
                targets.append((class_name, xmin, ymin, xmin + w, ymin + h))
            frame_data[fid] = targets

        # 为每张抽样图片生成 YOLO TXT
        for item in item_list:
            img_stem = item["img_path"].stem
            frame_id = int(img_stem.replace("img", ""))
            targets = frame_data.get(frame_id, [])

            lines = []
            for cn, xmin, ymin, xmax, ymax in targets:
                if xmax <= xmin or ymax <= ymin:
                    continue
                cid = CLASS_MAPPING[cn]
                xc = max(0.0, min(1.0, (xmin + xmax) / 2.0 / 960))
                yc = max(0.0, min(1.0, (ymin + ymax) / 2.0 / 540))
                bw = max(0.0, min(1.0, (xmax - xmin) / 960))
                bh = max(0.0, min(1.0, (ymax - ymin) / 540))
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            txt_path = labels_dir / f"{img_stem}.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            converted += 1

    print(f"  转换完成：{converted} 个标注文件")

    # 5. 划分数据集
    print("\n[5] 划分数据集 (8:1:1)...")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
    from app.training.dataset_splitter import DatasetSplitter

    splitter = DatasetSplitter()
    final_dir = Path(PROJECT_ROOT) / "datasets" / "UA-DETRAC" / "detrac_1000_final"

    split_stats = splitter.organize_dataset(
        image_dir=str(OUT_IMG),
        label_dir=str(labels_dir),
        output_dir=str(final_dir),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # 6. 生成 data.yaml
    print("\n[6] 生成 data.yaml...")
    splitter.generate_data_yaml(
        output_dir=str(final_dir),
        class_names=["car", "bus", "van", "others"],
    )

    print(f"\n{'=' * 60}")
    print("  完成！")
    print("  数据集：{final_dir}")
    print(f"  train: {split_stats.get('train', 0)} 张")
    print(f"  val: {split_stats.get('val', 0)} 张")
    print(f"  test: {split_stats.get('test', 0)} 张")
    print("\n  下一步训练：")
    print("    python tools/train_on_cloud.py --epochs 5 --batch 4 --device cpu")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
