"""
数据集划分与目录组织工具
职责：
 - 将图像和标注文件按指定比例划分到 train/val/test 目录
 - 自动创建目录结构
 - 验证图像与标注的配对完整性
使用方式：
 from app.training.dataset_splitter import DatasetSplitter
 splitter = DatasetSplitter()
 splitter.organize_dataset(
     image_dir="datasets/rsod/yolo_dataset/images",
     label_dir="datasets/rsod/yolo_dataset/labels",
     output_dir="datasets/rsod",
     train_ratio=0.8,
     val_ratio=0.1,
     test_ratio=0.1
 )
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """数据集划分与目录组织工具"""

    @staticmethod
    def organize_dataset(
        image_dir: str,
        label_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """
        将图像和标注文件按比例划分到 train/val/test 目录
        Args:
            image_dir: 原始图像目录（所有图像在同一目录）
            label_dir: 原始标注目录（所有标注在同一目录）
            output_dir: 输出目录（将创建标准 YOLO 目录结构）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子（确保可重复）
        Returns:
            划分统计信息
        """
        random.seed(seed)
        # 获取所有图像文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = sorted(
            [
                f
                for f in Path(image_dir).iterdir()
                if f.suffix.lower() in image_extensions
            ]
        )
        if not image_files:
            logger.error("图像目录 %s 下未找到图像文件", image_dir)
            return {"error": "未找到图像文件"}
        logger.info("找到 %d 张图像，开始划分...", len(image_files))
        # 随机打乱
        random.shuffle(image_files)
        # 计算划分边界
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        splits = {
            "train": image_files[:train_end],
            "val": image_files[train_end:val_end],
            "test": image_files[val_end:],
        }
        # 创建目录结构并复制文件
        stats = {"train": 0, "val": 0, "test": 0, "missing_labels": []}
        for split_name, files in splits.items():
            # 创建目录
            img_out = Path(output_dir) / "images" / split_name
            lbl_out = Path(output_dir) / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for img_file in files:
                # 复制图像
                shutil.copy2(img_file, img_out / img_file.name)
                # 查找并复制对应的标注文件
                label_file = Path(label_dir) / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, lbl_out / label_file.name)
                    stats[split_name] += 1
                else:
                    # 没有标注文件时创建空文件
                    empty_label = lbl_out / f"{img_file.stem}.txt"
                    empty_label.touch()
                    stats["missing_labels"].append(img_file.name)
                    stats[split_name] += 1
                    logger.warning(
                        "图像 %s 无对应标注文件，已创建空标注", img_file.name
                    )
        logger.info(
            "数据集划分完成：train=%d, val=%d, test=%d, 缺失标注=%d",
            stats["train"],
            stats["val"],
            stats["test"],
            len(stats["missing_labels"]),
        )
        return stats

    @staticmethod
    def generate_data_yaml(
        output_dir: str,
        class_names: list,
        class_names_cn: Optional[list] = None,
    ) -> str:
        """
        自动生成 data.yaml 配置文件
        Args:
            output_dir: 数据集根目录
            class_names: 类别名称列表（英文，按 class_id 顺序）
            class_names_cn: 类别中文名称列表（可选，用于前端显示）
        Returns:
            生成的 data.yaml 文件路径
        """
        import yaml

        data_config = {
            "path": f"./{os.path.basename(output_dir)}",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }
        if class_names_cn:
            data_config["names_cn"] = {i: name for i, name in enumerate(class_names_cn)}
        yaml_path = Path(output_dir) / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data_config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("data.yaml 已生成：%s", yaml_path)
        return str(yaml_path)
