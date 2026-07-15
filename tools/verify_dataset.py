#!/usr/bin/env python3
"""
YOLO 数据集验证脚本

功能：
    对转换后的 YOLO 格式数据集进行全面验证

使用方式：
    cd rsod-agent-platform
    python tools/verify_dataset.py
    python tools/verify_dataset.py /path/to/dataset
"""

import os
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_yaml_classes(dataset_dir):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return {}
    names = {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            in_names = False
            for line in f:
                line = line.strip()
                if line.startswith("names:"):
                    in_names = True
                    continue
                if in_names and line:
                    if line[0].isdigit():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            names[int(parts[0].strip())] = parts[1].strip()
                elif in_names and not line:
                    break
    except Exception:
        pass
    return names


def verify_dataset(dataset_dir):
    results = {
        "total_images": 0, "total_labels": 0, "total_annotations": 0,
        "missing_labels": [], "missing_images": [], "empty_labels": 0,
        "invalid_format": [], "out_of_range": [], "class_distribution": {},
        "class_names": {}, "bbox_stats": {
            "total": 0, "avg_width": 0, "avg_height": 0,
            "max_width": 0, "max_height": 0,
            "min_width": float("inf"), "min_height": float("inf"),
            "small_boxes": 0, "large_boxes": 0,
        },
        "split_stats": {}, "has_warnings": False,
    }

    dataset_path = Path(dataset_dir)
    class_names = load_yaml_classes(dataset_path)
    results["class_names"] = class_names

    for split in ["train", "val", "test"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split
        split_result = {"images": 0, "labels": 0, "annotations": 0, "missing_labels": 0, "missing_images": 0, "class_distribution": {}}

        if not img_dir.exists():
            if split != "test":
                print(f"[WARNING] Missing dir: {img_dir}")
            results["split_stats"][split] = split_result
            continue
        if not lbl_dir.exists():
            print(f"[WARNING] Missing dir: {lbl_dir}")
            results["split_stats"][split] = split_result
            continue

        image_files = {f.stem for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
        label_files = {f.stem for f in lbl_dir.iterdir() if f.suffix == ".txt"}
        missing_labels = image_files - label_files
        missing_images = label_files - image_files

        split_result["images"] = len(image_files)
        split_result["labels"] = len(label_files)
        split_result["missing_labels"] = len(missing_labels)
        split_result["missing_images"] = len(missing_images)

        results["missing_labels"].extend([f"{split}/{name}" for name in missing_labels])
        results["missing_images"].extend([f"{split}/{name}" for name in missing_images])
        results["total_images"] += len(image_files)
        results["total_labels"] += len(label_files)

        bbox_widths = []
        bbox_heights = []

        for label_file in lbl_dir.glob("*.txt"):
            content = label_file.read_text(encoding="utf-8").strip()
            if not content:
                results["empty_labels"] += 1
                continue
            for line_num, line in enumerate(content.split("\n"), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    results["invalid_format"].append(f"{split}/{label_file.name}:{line_num}")
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    results["invalid_format"].append(f"{split}/{label_file.name}:{line_num}")
                    continue
                results["class_distribution"][class_id] = results["class_distribution"].get(class_id, 0) + 1
                split_result["class_distribution"][class_id] = split_result["class_distribution"].get(class_id, 0) + 1
                results["total_annotations"] += 1
                split_result["annotations"] += 1
                try:
                    coords = [float(v) for v in parts[1:]]
                    x_center, y_center, width, height = coords
                    for i, v in enumerate(coords):
                        if v < 0 or v > 1:
                            field_names = ["x_center", "y_center", "width", "height"]
                            results["out_of_range"].append(f"{split}/{label_file.name}:{line_num} {field_names[i]}={v:.6f}")
                            break
                    bbox_widths.append(width)
                    bbox_heights.append(height)
                except ValueError:
                    results["invalid_format"].append(f"{split}/{label_file.name}:{line_num}")

        if bbox_widths:
            results["bbox_stats"]["total"] += len(bbox_widths)
            results["bbox_stats"]["avg_width"] += sum(bbox_widths)
            results["bbox_stats"]["avg_height"] += sum(bbox_heights)
            results["bbox_stats"]["max_width"] = max(results["bbox_stats"]["max_width"], max(bbox_widths))
            results["bbox_stats"]["max_height"] = max(results["bbox_stats"]["max_height"], max(bbox_heights))
            results["bbox_stats"]["min_width"] = min(results["bbox_stats"]["min_width"], min(bbox_widths))
            results["bbox_stats"]["min_height"] = min(results["bbox_stats"]["min_height"], min(bbox_heights))
            results["bbox_stats"]["small_boxes"] += sum(1 for w, h in zip(bbox_widths, bbox_heights) if w * h < 0.001)
            results["bbox_stats"]["large_boxes"] += sum(1 for w, h in zip(bbox_widths, bbox_heights) if w * h > 0.5)

        results["split_stats"][split] = split_result

    if results["bbox_stats"]["total"] > 0:
        results["bbox_stats"]["avg_width"] /= results["bbox_stats"]["total"]
        results["bbox_stats"]["avg_height"] /= results["bbox_stats"]["total"]

    return results


def print_report(results):
    print("\n" + "=" * 70)
    print("              YOLO Dataset Verification Report")
    print("=" * 70)

    print(f"\n  [Summary]")
    print(f"    Total images: {results['total_images']}")
    print(f"    Total labels: {results['total_labels']}")
    print(f"    Total annotations: {results['total_annotations']}")
    print(f"    Empty labels: {results['empty_labels']}")

    print(f"\n  [Split Stats]")
    for split in ["train", "val", "test"]:
        stats = results["split_stats"].get(split, {})
        print(f"    {split}: {stats.get('images', 0)} images, {stats.get('labels', 0)} labels, {stats.get('annotations', 0)} annotations")

    if results["missing_labels"]:
        print(f"\n  [WARNING] Missing labels ({len(results['missing_labels'])}):")
        for name in results["missing_labels"][:5]:
            print(f"    - {name}")
        results["has_warnings"] = True

    if results["missing_images"]:
        print(f"\n  [WARNING] Missing images ({len(results['missing_images'])}):")
        for name in results["missing_images"][:5]:
            print(f"    - {name}")
        results["has_warnings"] = True

    if results["invalid_format"]:
        print(f"\n  [ERROR] Invalid format ({len(results['invalid_format'])}):")
        for item in results["invalid_format"][:5]:
            print(f"    - {item}")
        results["has_warnings"] = True

    if results["out_of_range"]:
        print(f"\n  [WARNING] Out of range ({len(results['out_of_range'])}):")
        for item in results["out_of_range"][:5]:
            print(f"    - {item}")
        results["has_warnings"] = True

    print(f"\n  [Class Distribution]")
    class_names = results["class_names"]
    total = results["total_annotations"] or 1
    if results["class_distribution"]:
        max_count = max(results["class_distribution"].values())
        for class_id in sorted(results["class_distribution"].keys()):
            count = results["class_distribution"][class_id]
            percentage = (count / total) * 100
            class_name = class_names.get(class_id, f"class_{class_id}")
            bar_length = int((count / max_count) * 40) if max_count > 0 else 0
            bar = "#" * bar_length + "." * (40 - bar_length)
            print(f"    {class_id:2d}. {class_name:12s} {count:6d} ({percentage:5.2f}%) {bar}")

    print(f"\n  [Bbox Stats]")
    bbox = results["bbox_stats"]
    if bbox["total"] > 0:
        print(f"    Total bboxes: {bbox['total']}")
        print(f"    Avg width: {bbox['avg_width']:.4f}, Avg height: {bbox['avg_height']:.4f}")
        print(f"    Small (<0.001): {bbox['small_boxes']}, Large (>0.5): {bbox['large_boxes']}")

    print(f"\n{'=' * 70}")
    if results["has_warnings"]:
        print("  Result: WARNING - issues found, please fix")
    else:
        print("  Result: PASS - dataset ready for training")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "rsod", "yolo_dataset")
    if len(sys.argv) > 1:
        DATASET_DIR = sys.argv[1]
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset dir not found: {DATASET_DIR}")
        sys.exit(1)
    results = verify_dataset(DATASET_DIR)
    print_report(results)
