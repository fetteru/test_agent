import argparse
import json
import os
import sys
from typing import Optional

# ── 项目路径 ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_data_yaml(weights_path: str) -> str:
    """
    根据权重路径自动查找 data.yaml
    训练输出目录结构：
    runs/train/task_xxxxxxxx/
    ├── weights/best.pt
    └── ... (data.yaml 的 path 指向数据集目录)
    策略：
    1. 从权重路径向上查找训练输出目录
    2. 读取 args.yaml 获取 data 字段
    3. 如果找不到，尝试在 datasets/ 目录下查找
    """
    # 从权重路径向上查找
    task_dir = os.path.dirname(os.path.dirname(weights_path))
    # 尝试读取 args.yaml（Ultralytics 训练时保存的训练参数）
    args_yaml = os.path.join(task_dir, "args.yaml")
    if os.path.exists(args_yaml):
        with open(args_yaml, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("data:"):
                    data_path = line.split(":", 1)[1].strip()
                    if os.path.exists(data_path):
                        return data_path
    # 在 datasets/ 目录下查找
    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
    if os.path.exists(datasets_dir):
        for scene_dir in os.listdir(datasets_dir):
            yaml_path = os.path.join(
                datasets_dir, scene_dir, "yolo_dataset", "data.yaml"
            )
            if os.path.exists(yaml_path):
                return yaml_path
    return ""


def run_evaluation(
    weights_path: str,
    data_yaml: str,
    split: str = "val",
    conf: float = 0.001,
    iou: float = 0.6,
    img_size: int = 640,
    device: str = "cpu",
    save_output: bool = True,
    output_dir: Optional[str] = None,
) -> dict:
    """
    运行模型评估
    参数：
    weights_path: 模型权重路径（best.pt）
    data_yaml: data.yaml 路径
    split: 评估数据集划分（val / test / train）
    conf: 置信度阈值
    iou: NMS IoU 阈值
    img_size: 图像尺寸
    device: 评估设备（cpu / 0）
    save_output: 是否保存评估输出（混淆矩阵等图片）
    output_dir: 输出目录
    返回：
    评估结果字典
    """
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(" 模型评估")
    print(f" 权重文件: {weights_path}")
    print(f" 数据集: {data_yaml}")
    print(f" 评估集: {split}")
    print(f" 置信度: {conf}")
    print(f" IoU: {iou}")
    print(f" 设备: {device}")
    print(f"{'=' * 60}\n")

    # 加载模型
    model = YOLO(weights_path)

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(weights_path))

    # 运行验证
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=img_size,
        device=device,
        save_json=True,
        save_txt=True,
        save_conf=True,
        plots=save_output,
        project=output_dir,
        name="eval",
        exist_ok=True,
        verbose=True,
    )

    # 解析评估结果
    # model.names 返回 dict[int, str]，但类型推断可能不准确
    class_names: dict = model.names if isinstance(model.names, dict) else {}
    report = parse_evaluation_results(results, class_names)
    # 打印评估报告
    print_report(report)

    # 保存 JSON 报告
    if save_output:
        report_path = os.path.join(output_dir, "eval", "eval_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n评估报告已保存到: {report_path}")

    return report


def parse_evaluation_results(results, class_names: dict) -> dict:
    """
    解析 Ultralytics val() 返回结果
    参数：
    results: model.val() 的返回对象
    class_names: 类别 ID → 名称映射
    返回：
    结构化的评估报告字典
    """
    # 整体指标
    report = {
        "overall": {
            "precision": float(results.box.mp),  # mean precision
            "recall": float(results.box.mr),  # mean recall
            "map50": float(results.box.map50),  # mAP@0.50
            "map50_95": float(results.box.map),  # mAP@0.50:0.95
            "map75": float(results.box.map75)
            if hasattr(results.box, "map75")
            else None,
        },
        "per_class": {},
    }

    # 每类指标
    if results.box.ap is not None:
        for i, ap50 in enumerate(results.box.ap50):
            class_name = class_names.get(i, f"class_{i}")
            ap50_95 = results.box.ap[i] if i < len(results.box.ap) else 0.0
            report["per_class"][class_name] = {
                "ap50": float(ap50),
                "ap50_95": float(ap50_95),
                "instances": int(results.box.np[i])
                if hasattr(results.box, "np")
                else None,
            }
    return report


def print_report(report: dict):
    """
    打印格式化的评估报告
    参数：
    report: parse_evaluation_results 返回的报告字典
    """
    overall = report["overall"]
    print(f"\n{'=' * 60}")
    print(" 评估报告")
    print(f"{'=' * 60}")
    print("\n ▸ 整体指标:")
    print(f" {'指标':<16} {'值':>10}")
    print(f" {'─' * 26}")
    print(f" {'Precision':<16} {overall['precision']:>10.4f}")
    print(f" {'Recall':<16} {overall['recall']:>10.4f}")
    print(f" {'mAP@50':<16} {overall['map50']:>10.4f}")
    print(f" {'mAP@50-95':<16} {overall['map50_95']:>10.4f}")

    # 每类指标排序输出
    per_class = report["per_class"]
    if per_class:
        print("\n ▸ 每类 AP（按 mAP50 降序）:")
        print(f" {'类别':<20} {'AP@50':>8} {'AP@50-95':>10}")
        print(f" {'─' * 38}")
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]["ap50"],
            reverse=True,
        )
        for class_name, metrics in sorted_classes:
            print(
                f" {class_name:<20} {metrics['ap50']:>8.4f} {metrics['ap50_95']:>10.4f}"
            )
        # 标出弱势类别
        weak_classes = [name for name, m in sorted_classes if m["ap50"] < 0.5]
        if weak_classes:
            print(f"\n ⚠ 弱势类别（AP@50 < 0.5）: {', '.join(weak_classes)}")
            print(" 建议: 增加这些类别的训练样本，或检查标注质量")
    print(f"\n{'=' * 60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 模型评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
 # 评估 best.pt（自动查找 data.yaml）
 python tools/evaluate_model.py --weights runs/train/task_xxxxxxxx/weights/best.pt
 # 在测试集上评估
 python tools/evaluate_model.py --weights path/to/best.pt --split test
 # 自定义阈值
 python tools/evaluate_model.py --weights path/to/best.pt --conf 0.25 --iou 0.45
 # 保存评估报告
 python tools/evaluate_model.py --weights path/to/best.pt --output models/eval_report
 """,
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="模型权重路径（best.pt）",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="data.yaml 路径（默认自动查找）",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="评估数据集划分（默认: val）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="置信度阈值（默认: 0.001）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU 阈值（默认: 0.6）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="图像尺寸（默认: 640）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="评估设备（cpu / 0）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="评估报告输出目录",
    )
    args = parser.parse_args()

    # 验证权重文件
    if not os.path.exists(args.weights):
        print(f"[错误] 权重文件不存在: {args.weights}")
        sys.exit(1)

    # 查找 data.yaml
    data_yaml = args.data
    if not data_yaml:
        data_yaml = find_data_yaml(args.weights)
        if not data_yaml or not os.path.exists(data_yaml):
            print("[错误] 未找到 data.yaml，请使用 --data 参数指定")
            sys.exit(1)

    # 运行评估
    run_evaluation(
        weights_path=args.weights,
        data_yaml=data_yaml,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
        device=args.device,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
