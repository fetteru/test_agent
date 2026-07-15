"""
模型训练服务

职责：
  - 封装 YOLOv11 训练启动、监控、停止逻辑
  - 支持本地 CPU 训练和 GPU 训练
  - 训练在后台线程中执行，不阻塞 API 请求
  - 实时解析训练指标并写入数据库

使用方式：
  from app.training.training_service import training_service

  task = training_service.start_training(
      db=db, user_id=current_user.id, scene_id=scene.id,
      config={"model_name": "yolov11n", "epochs": 50, "batch_size": 8}
  )
"""

import csv
import os
import threading
import uuid
from datetime import datetime
from typing import Optional

from app.config.settings import settings
from app.core.logger import get_logger
from app.database.session import SessionLocal
from app.entity.db_models import TrainingMetric, TrainingTask

logger = get_logger(__name__)

_running_tasks: dict = {}
_running_lock = threading.Lock()
_stop_events: dict = {}  # 存储每个训练任务的停止标志


def _safe_float(value):
    """安全地将字符串转换为浮点数，失败时返回 None"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class TrainingService:
    """模型训练服务 — 封装 YOLOv11 训练全流程"""

    @staticmethod
    def start_training(db, user_id: int, scene_id: int, config: dict) -> TrainingTask:
        """创建并启动训练任务"""
        task_uuid = str(uuid.uuid4())[:8]

        data_yaml = config.get("data_yaml")
        dataset_path = config.get("dataset_path", "")
        if not data_yaml and dataset_path:
            yaml_candidate = os.path.join(dataset_path, "data.yaml")
            if os.path.exists(yaml_candidate):
                data_yaml = yaml_candidate

        task = TrainingTask(
            user_id=user_id,
            scene_id=scene_id,
            task_uuid=task_uuid,
            status="pending",
            model_name=config.get("model_name", "yolov11n"),
            epochs=config.get("epochs", 50),
            img_size=config.get("img_size", 640),
            batch_size=config.get("batch_size", 8),
            device=config.get("device", "cpu"),
            optimizer=config.get("optimizer", "SGD"),
            lr0=config.get("lr0", 0.01),
            augment_config=config.get("augment_config"),
            dataset_path=dataset_path,
            data_yaml=data_yaml,
        )
        db.add(task)
        db.commit()
        db.refresh(task)

        # 创建停止事件
        stop_event = threading.Event()
        _stop_events[task_uuid] = stop_event

        thread = threading.Thread(
            target=TrainingService._run_training,
            args=(task.id, task.task_uuid, config, stop_event),
            daemon=True,
            name=f"train-{task_uuid}",
        )
        thread.start()

        logger.info(
            "训练任务已启动：task_id=%d, uuid=%s, model=%s, epochs=%d",
            task.id, task_uuid, task.model_name, task.epochs,
        )
        return task

    @staticmethod
    def _run_training(task_id: int, task_uuid: str, config: dict, stop_event: Optional[threading.Event] = None):
        """在后台线程中执行 YOLOv11 训练"""
        db = SessionLocal()
        data_yaml = None
        original_content = None
        original_cwd = os.getcwd()

        try:
            task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
            if not task:
                logger.error("训练任务不存在：task_id=%d", task_id)
                return

            task.status = "running"  # type: ignore
            task.started_at = datetime.now()  # type: ignore
            db.commit()

            from ultralytics import YOLO

            model_name = config.get("model_name", "yolov11n")
            # original_cwd 是 backend/，模型在项目根目录 models/
            project_root = os.path.dirname(original_cwd)

            # 模型名称标准化映射：支持 yolov11n 和 yolo11n 两种命名
            model_aliases = [
                model_name,  # 原始名称，如 yolov11n
                model_name.replace("yolov", "yolo"),  # 替换命名，如 yolo11n
                model_name.replace("yolo", "yolov"),  # 反向替换，如 yolov11n
            ]

            # 在本地 models/ 目录中查找
            model_path = None
            for alias in model_aliases:
                local_model = os.path.join(project_root, "models", f"{alias}.pt")
                if os.path.exists(local_model):
                    model_path = local_model
                    break

            # 如果本地找不到，直接使用模型名称（Ultralytics 会自动下载）
            if model_path is None:
                # 只允许下载 YOLOv11 系列模型，防止误下载其他版本
                allowed_prefixes = ("yolov11", "yolo11")
                if not any(model_name.lower().startswith(p) for p in allowed_prefixes):
                    raise ValueError(f"不支持的模型: {model_name}，仅支持 YOLOv11 系列")
                model_path = model_name
                logger.info("本地未找到模型，将使用 Ultralytics 自动下载: %s", model_name)

            logger.info("加载预训练模型：%s", model_path)
            model = YOLO(model_path)

            with _running_lock:
                _running_tasks[task_uuid] = model

            data_yaml = config.get("data_yaml", "")
            if not data_yaml:
                dataset_path = config.get("dataset_path", "")
                data_yaml = os.path.join(dataset_path, "data.yaml")

            if not os.path.exists(data_yaml):
                raise FileNotFoundError(f"data.yaml 不存在：{data_yaml}")

            data_yaml_dir = os.path.dirname(data_yaml)

            with open(data_yaml, "r", encoding="utf-8") as f:
                original_content = f.read()

            # 将 path 字段替换为绝对路径，确保 Ultralytics 能找到图片
            # 逐行处理，避免正则表达式处理 Windows 路径时的问题
            lines = original_content.split('\n')
            modified_lines = []
            for line in lines:
                if line.strip().startswith('path:'):
                    # 替换 path 字段为绝对路径
                    modified_lines.append(f'path: {data_yaml_dir}')
                else:
                    modified_lines.append(line)
            modified_content = '\n'.join(modified_lines)
            with open(data_yaml, "w", encoding="utf-8") as f:
                f.write(modified_content)
            logger.info("已更新 data.yaml 路径：%s", data_yaml_dir)

            # 自动检测设备：无 GPU 则降级到 CPU
            import torch
            device = config.get("device", "cpu")
            if device != "cpu" and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，自动降级到 CPU 训练")
                device = "cpu"

            train_kwargs = {
                "data": data_yaml,
                "epochs": config.get("epochs", 50),
                "imgsz": config.get("img_size", 640),
                "batch": config.get("batch_size", 8),
                "device": device,
                "optimizer": config.get("optimizer", "SGD"),
                "lr0": config.get("lr0", 0.01),
                "project": os.path.join(project_root, settings.TRAIN_OUTPUT_DIR),
                "name": f"task_{task_uuid}",
                "exist_ok": True,
                "verbose": True,
                "save": True,
                "plots": False,
                "workers": 2,  # 限制数据加载进程数，避免资源耗尽
            }

            def on_train_epoch_end(trainer):
                try:
                    # 检查是否需要停止训练
                    if stop_event and stop_event.is_set():
                        logger.info("收到停止信号，终止训练：task=%s", task_uuid)
                        # 设置 trainer 的 stop 标志
                        if hasattr(trainer, 'stop'):
                            trainer.stop = True
                        return

                    epoch = trainer.epoch + 1
                    metrics = trainer.metrics or {}

                    # 使用独立的数据库会话，避免阻塞训练
                    metric_db = SessionLocal()
                    try:
                        metric_record = TrainingMetric(
                            task_id=task_id,
                            epoch=epoch,
                            box_loss=float(metrics.get("metrics/box_loss", 0) if isinstance(metrics, dict) else 0),
                            cls_loss=float(metrics.get("metrics/cls_loss", 0) if isinstance(metrics, dict) else 0),
                            dfl_loss=float(metrics.get("metrics/dfl_loss", 0) if isinstance(metrics, dict) else 0),
                            precision=float(metrics.get("metrics/precision(B)", 0) if isinstance(metrics, dict) else 0),
                            recall=float(metrics.get("metrics/recall(B)", 0) if isinstance(metrics, dict) else 0),
                            map50=float(metrics.get("metrics/mAP50(B)", 0) if isinstance(metrics, dict) else 0),
                            map50_95=float(metrics.get("metrics/mAP50-95(B)", 0) if isinstance(metrics, dict) else 0),
                        )
                        metric_db.add(metric_record)

                        # 更新任务进度
                        task_record = metric_db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
                        if task_record:
                            total_epochs = config.get("epochs", 50)
                            task_record.current_epoch = epoch
                            task_record.progress = int((epoch / total_epochs) * 100)  # type: ignore

                        metric_db.commit()
                        logger.info("训练进度：task=%s epoch=%d/%d, mAP50=%.4f", task_uuid, epoch, total_epochs,
                                   float(metrics.get("metrics/mAP50(B)", 0) if isinstance(metrics, dict) else 0))
                    except Exception as db_err:
                        logger.warning("数据库写入异常（不影响训练）：%s", str(db_err))
                        metric_db.rollback()
                    finally:
                        metric_db.close()

                except Exception as e:
                    logger.warning("训练回调异常（不影响训练）：%s", str(e))

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            logger.info("开始训练：data=%s, epochs=%d", data_yaml, train_kwargs["epochs"])
            model.train(**train_kwargs)

            task.status = "completed"  # type: ignore
            task.progress = 100  # type: ignore
            task.current_epoch = config.get("epochs", 50)
            task.completed_at = datetime.now()  # type: ignore
            db.commit()

            project_path = os.path.join(project_root, settings.TRAIN_OUTPUT_DIR)
            TrainingService._parse_final_results(db, task_id, task_uuid, config, project_path)

            logger.info("训练完成：task_id=%d, uuid=%s", task_id, task_uuid)

        except FileNotFoundError as e:
            logger.error("训练文件缺失：task_id=%d, error=%s", task_id, str(e))
            if task:
                task.status = "failed"  # type: ignore
                task.error_message = str(e)  # type: ignore
                db.commit()

        except KeyboardInterrupt:
            # 用户请求停止训练
            logger.info("训练被用户中断：task_id=%d", task_id)
            if task and task.status == "running":  # type: ignore
                task.status = "cancelled"  # type: ignore
                task.completed_at = datetime.now()  # type: ignore
                db.commit()

        except Exception as e:
            logger.error("训练异常：task_id=%d, error=%s", task_id, str(e), exc_info=True)
            if task:
                task.status = "failed"  # type: ignore
                task.error_message = str(e)[:2000]  # type: ignore
                db.commit()

        finally:
            if data_yaml and original_content:
                try:
                    with open(data_yaml, "w", encoding="utf-8") as f:
                        f.write(original_content)
                except Exception:
                    pass
            try:
                os.chdir(original_cwd)
            except Exception:
                pass
            with _running_lock:
                _running_tasks.pop(task_uuid, None)
            _stop_events.pop(task_uuid, None)
            db.close()

    @staticmethod
    def _parse_final_results(db, task_id: int, task_uuid: str, config: dict, project_path: Optional[str] = None):
        """训练完成后从 results.csv 解析最终指标"""
        if project_path is None:
            project_path = settings.TRAIN_OUTPUT_DIR

        results_csv = os.path.join(project_path, f"task_{task_uuid}", "results.csv")
        if not os.path.exists(results_csv):
            logger.warning("results.csv 不存在：%s", results_csv)
            return

        try:
            existing_epochs = set()
            existing = db.query(TrainingMetric).filter(TrainingMetric.task_id == task_id).all()
            for m in existing:
                existing_epochs.add(m.epoch)

            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    epoch = int(row.get("epoch", 0)) + 1
                    if epoch in existing_epochs:
                        continue

                    metric = TrainingMetric(
                        task_id=task_id,
                        epoch=epoch,
                        box_loss=_safe_float(row.get("train/box_loss", "")),
                        cls_loss=_safe_float(row.get("train/cls_loss", "")),
                        dfl_loss=_safe_float(row.get("train/dfl_loss", "")),
                        precision=_safe_float(row.get("metrics/precision(B)", "")),
                        recall=_safe_float(row.get("metrics/recall(B)", "")),
                        map50=_safe_float(row.get("metrics/mAP50(B)", "")),
                        map50_95=_safe_float(row.get("metrics/mAP50-95(B)", "")),
                        lr=_safe_float(row.get("lr/pg0", "")),
                    )
                    db.add(metric)

            db.commit()
            logger.info("results.csv 解析完成")
        except Exception as e:
            logger.warning("results.csv 解析异常：%s", str(e))
            db.rollback()

    @staticmethod
    def get_training_status(db, task_id: int) -> dict:
        """获取训练任务状态"""
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return {"error": "训练任务不存在"}

        latest_metric = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.task_id == task_id)
            .order_by(TrainingMetric.epoch.desc())
            .first()
        )

        with _running_lock:
            is_running = task.task_uuid in _running_tasks

        return {
            "task": {
                "id": task.id,
                "task_uuid": task.task_uuid,
                "status": task.status,
                "model_name": task.model_name,
                "epochs": task.epochs,
                "current_epoch": task.current_epoch,
                "progress": task.progress,
                "device": task.device,
                "batch_size": task.batch_size,
                "img_size": task.img_size,
                "started_at": str(task.started_at) if task.started_at else None,
                "completed_at": str(task.completed_at) if task.completed_at else None,
                "error_message": task.error_message,
            },
            "latest_metric": {
                "epoch": latest_metric.epoch,
                "box_loss": latest_metric.box_loss,
                "cls_loss": latest_metric.cls_loss,
                "dfl_loss": latest_metric.dfl_loss,
                "precision": latest_metric.precision,
                "recall": latest_metric.recall,
                "map50": latest_metric.map50,
                "map50_95": latest_metric.map50_95,
                "lr": latest_metric.lr,
            } if latest_metric else None,
            "is_running": is_running,
        }

    @staticmethod
    def get_training_metrics(db, task_id: int) -> list:
        """获取训练任务的所有 epoch 指标"""
        metrics = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.task_id == task_id)
            .order_by(TrainingMetric.epoch.asc())
            .all()
        )
        return [
            {
                "epoch": m.epoch,
                "box_loss": m.box_loss,
                "cls_loss": m.cls_loss,
                "dfl_loss": m.dfl_loss,
                "precision": m.precision,
                "recall": m.recall,
                "map50": m.map50,
                "map50_95": m.map50_95,
                "lr": m.lr,
            }
            for m in metrics
        ]

    @staticmethod
    def stop_training(db, task_id: int) -> dict:
        """停止正在运行的训练任务"""
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return {"error": "训练任务不存在"}

        if task.status != "running":
            return {"error": f"任务当前状态为 {task.status}，无法停止"}

        # 设置停止事件，让训练线程在下一个 epoch 结束时停止
        stop_event = _stop_events.get(task.task_uuid)
        if stop_event:
            stop_event.set()
            logger.info("已设置停止标志：task_id=%d, uuid=%s", task_id, task.task_uuid)
        else:
            logger.warning("未找到停止标志：task_id=%d, uuid=%s", task_id, task.task_uuid)

        # 尝试直接设置 trainer 的 stop 标志
        with _running_lock:
            model = _running_tasks.get(task.task_uuid)
            if model and hasattr(model, 'trainer') and model.trainer is not None:
                if hasattr(model.trainer, 'stop'):
                    model.trainer.stop = True
                    logger.info("已设置 trainer.stop 标志：task_id=%d", task_id)

        task.status = "cancelled"  # type: ignore
        task.completed_at = datetime.now()  # type: ignore
        db.commit()

        logger.info("训练任务已停止：task_id=%d", task_id)
        return {"message": "训练任务已停止", "task_id": task_id}

    @staticmethod
    def get_task_list(db, user_id: Optional[int] = None, limit: int = 20) -> list:
        """获取训练任务列表"""
        query = db.query(TrainingTask)
        if user_id:
            query = query.filter(TrainingTask.user_id == user_id)

        tasks = query.order_by(TrainingTask.created_at.desc()).limit(limit).all()
        return [
            {
                "id": t.id,
                "task_uuid": t.task_uuid,
                "status": t.status,
                "model_name": t.model_name,
                "epochs": t.epochs,
                "current_epoch": t.current_epoch,
                "progress": t.progress,
                "device": t.device,
                "created_at": str(t.created_at),
                "started_at": str(t.started_at) if t.started_at else None,
                "completed_at": str(t.completed_at) if t.completed_at else None,
            }
            for t in tasks
        ]

    @staticmethod
    def parse_results_csv(results_csv_path: str) -> list:
        """独立解析 results.csv 文件"""
        metrics = []
        if not os.path.exists(results_csv_path):
            return metrics
        with open(results_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                metrics.append({
                    "epoch": int(row.get("epoch", 0)) + 1,
                    "box_loss": _safe_float(row.get("train/box_loss", "")),
                    "cls_loss": _safe_float(row.get("train/cls_loss", "")),
                    "dfl_loss": _safe_float(row.get("train/dfl_loss", "")),
                    "precision": _safe_float(row.get("metrics/precision(B)", "")),
                    "recall": _safe_float(row.get("metrics/recall(B)", "")),
                    "map50": _safe_float(row.get("metrics/mAP50(B)", "")),
                    "map50_95": _safe_float(row.get("metrics/mAP50-95(B)", "")),
                    "lr": _safe_float(row.get("lr/pg0", "")),
                })
        return metrics

    @staticmethod
    def validate_model(
        db,
        task_id: int,
        split: str = "val",
        conf: float = 0.001,
        iou: float = 0.6,
    ) -> dict:
        """
        对已完成训练的模型执行验证集评估

        流程：
          1. 查找训练任务对应的 best.pt 路径
          2. 加载模型并运行 model.val()
          3. 解析评估结果
          4. 将评估指标写入 ModelVersion 表
          5. 返回结构化评估报告
        """
        from ultralytics import YOLO

        # 查找训练任务
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return {"error": "训练任务不存在"}

        if task.status != "completed":
            return {"error": f"训练任务状态为 {task.status}，只有已完成的任务才能评估"}

        # 定位 best.pt
        original_cwd = os.getcwd()
        # 项目根目录是 backend/ 的上级目录
        project_root = os.path.dirname(original_cwd)
        weights_path = os.path.join(
            project_root,
            settings.TRAIN_OUTPUT_DIR,
            f"task_{task.task_uuid}",
            "weights",
            "best.pt",
        )

        if not os.path.exists(weights_path):
            return {"error": f"模型权重不存在: {weights_path}"}

        # 定位 data.yaml
        data_yaml = task.data_yaml
        if not data_yaml or not os.path.exists(data_yaml):
            if task.dataset_path:
                data_yaml = os.path.join(task.dataset_path, "data.yaml")
            if not data_yaml or not os.path.exists(data_yaml):
                return {"error": "data.yaml 不存在"}

        logger.info(
            "开始模型评估: task_id=%d, weights=%s, split=%s",
            task_id, weights_path, split,
        )

        try:
            # 加载模型并评估
            model = YOLO(weights_path)
            results = model.val(
                data=data_yaml,
                split=split,
                conf=conf,
                iou=iou,
                imgsz=task.img_size,
                device="cpu",
                save_json=True,
                plots=True,
                project=os.path.join(original_cwd, settings.TRAIN_OUTPUT_DIR),
                name=f"task_{task.task_uuid}",
                exist_ok=True,
                verbose=False,
            )

            # 解析评估结果
            overall = {
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
            }

            per_class = {}
            if results.box.ap is not None and len(results.box.ap) > 0:
                # model.names 返回 dict[int, str]，但类型推断可能不准确
                names_dict: dict = model.names if isinstance(model.names, dict) else {}
                for i, ap50 in enumerate(results.box.ap50):
                    class_name = names_dict.get(i, f"class_{i}")
                    ap50_95 = results.box.ap[i] if i < len(results.box.ap) else 0.0
                    per_class[class_name] = {
                        "ap50": round(float(ap50), 4),
                        "ap50_95": round(float(ap50_95), 4),
                    }

            report = {
                "task_id": task_id,
                "task_uuid": task.task_uuid,
                "split": split,
                "overall": overall,
                "per_class": per_class,
            }

            # 更新或创建 ModelVersion 记录
            from app.entity.db_models import ModelVersion, DetectionScene

            scene = (
                db.query(DetectionScene)
                .filter(DetectionScene.id == task.scene_id)
                .first()
            )

            model_version = (
                db.query(ModelVersion)
                .filter(ModelVersion.training_task_id == task_id)
                .first()
            )

            if not model_version:
                existing_count = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.scene_id == task.scene_id)
                    .count()
                )
                version = f"v{existing_count + 1}.0.0"

                model_version = ModelVersion(
                    scene_id=task.scene_id,
                    training_task_id=task_id,
                    version=version,
                    model_name=f"{task.model_name}_{scene.name}_{version}" if scene else f"{task.model_name}_{version}",
                    model_type=task.model_name,
                    model_path=weights_path,
                    map50=overall["map50"],
                    map50_95=overall["map50_95"],
                    precision=overall["precision"],
                    recall=overall["recall"],
                    per_class_ap=per_class,
                    file_size=os.path.getsize(weights_path),
                    description=f"训练任务 {task.task_uuid} 自动产出",
                )
                db.add(model_version)
            else:
                model_version.map50 = overall["map50"]
                model_version.map50_95 = overall["map50_95"]
                model_version.precision = overall["precision"]
                model_version.recall = overall["recall"]
                model_version.per_class_ap = per_class

            db.commit()
            report["model_version_id"] = model_version.id
            report["model_version"] = model_version.version

            logger.info(
                "模型评估完成: task_id=%d, mAP50=%.4f, mAP50-95=%.4f",
                task_id, overall["map50"], overall["map50_95"],
            )

            return report

        except Exception as e:
            logger.error("模型评估异常: task_id=%d, error=%s", task_id, str(e), exc_info=True)
            return {"error": f"评估失败: {str(e)}"}

    @staticmethod
    def export_model(
        db,
        task_id: int,
        version: Optional[str] = None,
        description: Optional[str] = None,
        set_default: bool = False,
        upload_minio: bool = True,
    ) -> dict:
        """
        导出训练好的模型为正式版本

        流程：
          1. 复制 best.pt 到 models/ 目录
          2. 运行评估获取最终指标
          3. 保存评估报告 JSON
          4. 创建 ModelVersion 记录
          5. 可选上传到 MinIO
        """
        import shutil
        import json
        from app.entity.db_models import ModelVersion, DetectionScene

        # 查找训练任务
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return {"error": "训练任务不存在"}

        if task.status != "completed":
            return {"error": f"训练任务状态为 {task.status}，只有已完成的任务才能导出"}

        # 定位 best.pt
        original_cwd = os.getcwd()
        # 项目根目录是 backend/ 的上级目录
        project_root = os.path.dirname(original_cwd)
        weights_path = os.path.join(
            project_root,
            settings.TRAIN_OUTPUT_DIR,
            f"task_{task.task_uuid}",
            "weights",
            "best.pt",
        )

        if not os.path.exists(weights_path):
            return {"error": f"模型权重不存在: {weights_path}"}

        # 获取场景信息
        scene = (
            db.query(DetectionScene)
            .filter(DetectionScene.id == task.scene_id)
            .first()
        )
        if not scene:
            return {"error": "关联场景不存在"}

        # 生成版本号
        if not version:
            existing_count = (
                db.query(ModelVersion)
                .filter(ModelVersion.scene_id == task.scene_id)
                .count()
            )
            version = f"v{existing_count + 1}.0.0"

        # 创建导出目录
        export_dir = os.path.join(original_cwd, "models", f"{scene.name}_{version}")
        os.makedirs(export_dir, exist_ok=True)

        # 复制模型文件
        exported_weight = os.path.join(export_dir, "best.pt")
        shutil.copy2(weights_path, exported_weight)
        logger.info("模型文件已复制: %s -> %s", weights_path, exported_weight)

        # 复制评估图表
        task_output_dir = os.path.join(original_cwd, settings.TRAIN_OUTPUT_DIR, f"task_{task.task_uuid}")
        eval_plots = ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "results.png"]
        for plot_name in eval_plots:
            src = os.path.join(task_output_dir, plot_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(export_dir, plot_name))

        # 运行评估获取最终指标
        eval_result = TrainingService.validate_model(db, task_id, split="val")
        overall = eval_result.get("overall", {})
        per_class = eval_result.get("per_class", {})

        # 保存评估报告 JSON
        report = {
            "version": version,
            "model_name": task.model_name,
            "scene": scene.name,
            "training_task": task.task_uuid,
            "evaluation": {"split": "val", "overall": overall, "per_class": per_class},
            "training_config": {
                "epochs": task.epochs, "batch_size": task.batch_size,
                "img_size": task.img_size, "optimizer": task.optimizer,
                "lr0": task.lr0, "device": task.device,
            },
            "exported_at": datetime.now().isoformat(),
        }
        report_path = os.path.join(export_dir, "eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 上传到 MinIO
        minio_url = None
        if upload_minio:
            try:
                from app.storage.minio_client import MinIOClient
                minio_client = MinIOClient()
                object_name = f"models/{scene.name}/{version}/best.pt"
                minio_url = minio_client.upload_file(object_name, exported_weight)
                logger.info("模型已上传 MinIO: %s", minio_url)
            except Exception as e:
                logger.warning("MinIO 上传失败（不影响导出）: %s", str(e))

        # 创建/更新 ModelVersion 记录
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.training_task_id == task_id)
            .first()
        )

        if model_version:
            model_version.version = version
            model_version.model_path = exported_weight
            model_version.minio_url = minio_url
            model_version.map50 = overall.get("map50")
            model_version.map50_95 = overall.get("map50_95")
            model_version.precision = overall.get("precision")
            model_version.recall = overall.get("recall")
            model_version.per_class_ap = per_class
            model_version.file_size = os.path.getsize(exported_weight)
            model_version.description = description or f"训练任务 {task.task_uuid} 导出"
        else:
            model_version = ModelVersion(
                scene_id=task.scene_id,
                training_task_id=task_id,
                version=version,
                model_name=f"{task.model_name}_{scene.name}_{version}",
                model_type=task.model_name,
                model_path=exported_weight,
                minio_url=minio_url,
                map50=overall.get("map50"),
                map50_95=overall.get("map50_95"),
                precision=overall.get("precision"),
                recall=overall.get("recall"),
                per_class_ap=per_class,
                file_size=os.path.getsize(exported_weight),
                description=description or f"训练任务 {task.task_uuid} 导出",
            )
            db.add(model_version)

        # 设置默认模型
        if set_default:
            db.query(ModelVersion).filter(
                ModelVersion.scene_id == task.scene_id,
                ModelVersion.id != model_version.id,
            ).update({"is_default": False})
            model_version.is_default = True  # type: ignore

        db.commit()
        db.refresh(model_version)

        logger.info("模型导出完成: scene=%s, version=%s, mAP50=%.4f", scene.name, version, overall.get("map50", 0))

        return {
            "model_version_id": model_version.id,
            "version": version,
            "model_name": model_version.model_name,
            "model_path": exported_weight,
            "export_dir": export_dir,
            "minio_url": minio_url,
            "file_size": model_version.file_size,
            "evaluation": {"map50": overall.get("map50"), "map50_95": overall.get("map50_95"), "precision": overall.get("precision"), "recall": overall.get("recall"), "per_class": per_class},
            "is_default": model_version.is_default,
            "message": f"模型已导出为版本 {version}",
        }

    @staticmethod
    def get_model_download_path(db, task_id: int) -> dict:
        """
        获取训练任务的模型权重文件路径（用于下载）
        """
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return {"error": "训练任务不存在"}

        original_cwd = os.getcwd()
        # 项目根目录是 backend/ 的上级目录
        project_root = os.path.dirname(original_cwd)

        # 优先返回 best.pt
        best_path = os.path.join(project_root, settings.TRAIN_OUTPUT_DIR, f"task_{task.task_uuid}", "weights", "best.pt")
        if os.path.exists(best_path):
            return {"file_path": best_path, "filename": f"best_{task.task_uuid}.pt", "file_size": os.path.getsize(best_path)}

        # 备选 last.pt
        last_path = os.path.join(project_root, settings.TRAIN_OUTPUT_DIR, f"task_{task.task_uuid}", "weights", "last.pt")
        if os.path.exists(last_path):
            return {"file_path": last_path, "filename": f"last_{task.task_uuid}.pt", "file_size": os.path.getsize(last_path)}

        return {"error": "模型权重文件不存在"}


training_service = TrainingService()
