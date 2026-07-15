"""
修复 VOC XML 标注文件中的越界坐标

使用方式：
    cd rsod-agent-platform/backend
    python tools/fix_xml_coords.py
"""

import os
import xml.etree.ElementTree as ET

# 项目根目录：tools/ → 项目根目录/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_DIR = os.path.join(PROJECT_ROOT, "datasets/rsod/raw/annotations")


def fix_xml_file(xml_path):
    stats = {"fixed": 0, "deleted": 0, "total": 0}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            return stats
        width_elem = size.find("width")
        height_elem = size.find("height")
        if width_elem is None or height_elem is None:
            return stats
        if width_elem.text is None or height_elem.text is None:
            return stats

        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
        if img_width <= 0 or img_height <= 0:
            return stats

        objects = root.findall("object")
        stats["total"] = len(objects)
        to_delete = []

        for obj in objects:
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin_elem = bbox.find("xmin")
            ymin_elem = bbox.find("ymin")
            xmax_elem = bbox.find("xmax")
            ymax_elem = bbox.find("ymax")
            if xmin_elem is None or ymin_elem is None or xmax_elem is None or ymax_elem is None:
                continue
            if xmin_elem.text is None or ymin_elem.text is None or xmax_elem.text is None or ymax_elem.text is None:
                continue

            xmin = float(xmin_elem.text)
            ymin = float(ymin_elem.text)
            xmax = float(xmax_elem.text)
            ymax = float(ymax_elem.text)
            changed = False

            if xmin < 0:
                xmin = 0
                changed = True
            if ymin < 0:
                ymin = 0
                changed = True
            if xmax > img_width:
                xmax = img_width
                changed = True
            if ymax > img_height:
                ymax = img_height
                changed = True

            if xmax <= xmin or ymax <= ymin:
                to_delete.append(obj)
                stats["deleted"] += 1
                continue

            if changed:
                xmin_elem.text = str(int(xmin))
                ymin_elem.text = str(int(ymin))
                xmax_elem.text = str(int(xmax))
                ymax_elem.text = str(int(ymax))
                stats["fixed"] += 1

        for obj in to_delete:
            root.remove(obj)

        if stats["fixed"] > 0 or stats["deleted"] > 0:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    except Exception as e:
        print(f"  [错误] 处理 {os.path.basename(xml_path)}: {str(e)}")

    return stats


def main():
    print("=" * 70)
    print("      修复 VOC XML 标注文件中的越界坐标")
    print("=" * 70)

    total_stats = {"fixed": 0, "deleted": 0, "total": 0, "files_affected": 0}

    if not os.path.exists(ANNOTATION_DIR):
        print(f"\n[错误] 标注目录不存在: {ANNOTATION_DIR}")
        return

    xml_files = [f for f in os.listdir(ANNOTATION_DIR) if f.lower().endswith(".xml")]
    print(f"\n找到 {len(xml_files)} 个 XML 文件")
    print("正在处理...")

    for xml_file in xml_files:
        xml_path = os.path.join(ANNOTATION_DIR, xml_file)
        stats = fix_xml_file(xml_path)
        if stats["fixed"] > 0 or stats["deleted"] > 0:
            total_stats["files_affected"] += 1
            print(f"  {xml_file}: 修复 {stats['fixed']} 个, 删除 {stats['deleted']} 个")
        total_stats["fixed"] += stats["fixed"]
        total_stats["deleted"] += stats["deleted"]
        total_stats["total"] += stats["total"]

    print("\n" + "=" * 70)
    print("  修复完成！")
    print(f"  总计目标数: {total_stats['total']}")
    print(f"  修复越界: {total_stats['fixed']} 个")
    print(f"  删除无效: {total_stats['deleted']} 个")
    print(f"  影响文件: {total_stats['files_affected']} 个")
    print("=" * 70)


if __name__ == "__main__":
    main()
