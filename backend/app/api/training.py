"""
训练相关 API 路由

接口列表：
  - POST   /api/training/start              启动训练任务
  - GET    /api/training/tasks               获取训练任务列表
  - GET    /api/training/status/{task_id}    获取训练状态
  - GET    /api/training/metrics/{task_id}   获取训练指标历史
  - POST   /api/training/stop/{task_id}      停止训练任务
  - GET    /api/training/results/{task_uuid}  获取 results.csv 原始数据
  - POST   /api/training/validate/{task_id}  模型评估
  - POST   /api/training/export/{task_id}    模型导出
  - GET    /api/training/download/{task_id}  下载模型
  - POST   /api/training/predict             上传测试图验证
"""

import os
import tempfile
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.config.settings import settings
from app.core.logger import get_logger
from app.database.session import get_db
from app.entity.db_models import TrainingTask
from app.entity.schemas import TrainingTaskCreate, ModelValidateRequest, ModelExportRequest
from app.training.training_service import training_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/training", tags=["模型训练"])


@router.post("/start")
async def start_training(
    request: TrainingTaskCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """启动模型训练任务"""
    config = {
        "model_name": request.model_name,
        "epochs": request.epochs,
        "img_size": request.img_size,
        "batch_size": request.batch_size,
        "device": request.device,
        "optimizer": request.optimizer,
        "lr0": request.lr0,
        "augment_config": request.augment_config,
    }

    from app.entity.db_models import DetectionScene

    scene = (
        db.query(DetectionScene).filter(DetectionScene.id == request.scene_id).first()
    )
    if not scene:
        raise HTTPException(status_code=404, detail="检测场景不存在")

    api_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(api_dir)
    backend_dir = os.path.dirname(app_dir)
    project_root = os.path.dirname(backend_dir)
    dataset_path = os.path.join(project_root, "datasets", str(scene.name))
    config["dataset_path"] = dataset_path

    # 查找 data.yaml（优先 yolo_dataset 子目录，其次场景根目录）
    data_yaml_candidates = [
        os.path.join(dataset_path, "yolo_dataset", "data.yaml"),
        os.path.join(dataset_path, "data.yaml"),
    ]
    data_yaml = None
    for candidate in data_yaml_candidates:
        if os.path.exists(candidate):
            data_yaml = candidate
            break

    if data_yaml:
        config["data_yaml"] = data_yaml
    else:
        raise HTTPException(
            status_code=400,
            detail="data.yaml 不存在，请先完成数据集准备",
        )

    try:
        task = training_service.start_training(
            db=db,
            user_id=current_user.id,
            scene_id=request.scene_id,
            config=config,
        )
    except Exception as e:
        logger.error("启动训练失败：%s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"启动训练失败：{str(e)}")

    logger.info(
        "用户 %s 启动训练任务：scene=%s, model=%s, epochs=%d",
        current_user.username,
        scene.name,
        request.model_name,
        request.epochs,
    )

    return {
        "id": task.id,
        "task_uuid": task.task_uuid,
        "status": task.status,
        "model_name": task.model_name,
        "epochs": task.epochs,
        "message": "训练任务已创建，正在后台启动",
    }


@router.get("/tasks")
async def list_training_tasks(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """获取当前用户的训练任务列表"""
    tasks = training_service.get_task_list(db, user_id=current_user.id)
    return {"total": len(tasks), "items": tasks}


@router.get("/status/{task_id}")
async def get_training_status(
    task_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """获取训练任务状态"""
    status = training_service.get_training_status(db, task_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@router.get("/metrics/{task_id}")
async def get_training_metrics(
    task_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """获取训练任务的所有 epoch 指标"""
    metrics = training_service.get_training_metrics(db, task_id)
    return {"task_id": task_id, "total": len(metrics), "metrics": metrics}


@router.post("/stop/{task_id}")
async def stop_training(
    task_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """停止正在运行的训练任务"""
    result = training_service.stop_training(db, task_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/results/{task_uuid}")
async def get_results_csv(
    task_uuid: str,
    current_user=Depends(get_current_user),
):
    """获取 Ultralytics 生成的原始 results.csv 文件"""
    results_path = os.path.join(
        settings.TRAIN_OUTPUT_DIR, f"task_{task_uuid}", "results.csv"
    )
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="results.csv 文件不存在")
    return FileResponse(
        path=results_path,
        media_type="text/csv",
        filename=f"training_results_{task_uuid}.csv",
    )


@router.post("/validate/{task_id}")
async def validate_model(
    task_id: int,
    request: Optional[ModelValidateRequest] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    对已完成训练的模型执行评估

    - 在验证集或测试集上运行 model.val()
    - 返回 mAP、Precision、Recall 等指标
    - 返回每类 AP 分析
    - 自动创建/更新 ModelVersion 记录
    """
    if request is None:
        request = ModelValidateRequest()

    result = training_service.validate_model(
        db=db,
        task_id=task_id,
        split=request.split,
        conf=request.conf,
        iou=request.iou,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    logger.info(
        "用户 %s 评估模型: task_id=%d, mAP50=%.4f",
        current_user.username, task_id,
        result.get("overall", {}).get("map50", 0),
    )

    return result


@router.post("/export/{task_id}")
async def export_model(
    task_id: int,
    request: Optional[ModelExportRequest] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    导出训练好的模型为正式版本

    - 复制 best.pt 到 models/ 目录
    - 运行评估获取最终指标
    - 保存评估报告 JSON
    - 创建 ModelVersion 记录
    - 可选上传到 MinIO
    """
    if request is None:
        request = ModelExportRequest(version=None, description=None, set_default=False, upload_minio=True)

    result = training_service.export_model(
        db=db,
        task_id=task_id,
        version=request.version,
        description=request.description,
        set_default=request.set_default,
        upload_minio=request.upload_minio,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    logger.info(
        "用户 %s 导出模型: task_id=%d, version=%s",
        current_user.username, task_id, result.get("version"),
    )

    return result


@router.get("/download/{task_id}")
async def download_model(
    task_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    下载训练好的模型权重文件（best.pt）

    返回文件下载响应，浏览器直接保存文件
    """
    result = training_service.get_model_download_path(db, task_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    logger.info(
        "用户 %s 下载模型: task_id=%d, file=%s",
        current_user.username, task_id, result["filename"],
    )

    return FileResponse(
        path=result["file_path"],
        media_type="application/octet-stream",
        filename=result["filename"],
    )


@router.post("/predict")
async def predict_test_image(
    file: UploadFile = File(..., description="测试图片"),
    task_id: int = Form(..., description="训练任务 ID（使用哪个模型）"),
    conf: float = Form(0.25, description="置信度阈值"),
    iou: float = Form(0.45, description="NMS IoU 阈值"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    上传测试图片，使用训练好的模型进行预测

    用于快速验证模型训练效果：
    1. 上传一张不在训练集/验证集中的测试图片
    2. 使用 best.pt 进行推理
    3. 返回检测结果（标注图 + 检测统计）
    """
    from ultralytics import YOLO
    import base64

    # 验证文件类型
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}，支持: {', '.join(allowed_types)}",
        )

    # 查找训练任务
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    if task.status != "completed":  # type: ignore
        raise HTTPException(status_code=400, detail="训练任务未完成，无法进行预测")

    # 定位 best.pt
    # 项目根目录是 backend/ 的上级目录
    project_root = os.path.dirname(os.getcwd())
    weights_path = os.path.join(
        project_root,
        settings.TRAIN_OUTPUT_DIR,
        f"task_{task.task_uuid}",
        "weights",
        "best.pt",
    )
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="模型权重文件不存在")

    # 保存上传文件到临时目录
    file_ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 加载模型并推理
        model = YOLO(weights_path)
        results = model.predict(
            source=tmp_path,
            conf=conf,
            iou=iou,
            imgsz=task.img_size,
            device="cpu",
            save=False,
            verbose=False,
        )

        # 解析检测结果
        result = results[0]
        detections = []
        total_objects = 0

        # model.names 返回 dict[int, str]，但类型推断可能不准确
        names_dict: dict = model.names if isinstance(model.names, dict) else {}

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = names_dict.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                })
                total_objects += 1

        # 生成标注图
        annotated_img = result.plot()

        # 将标注图编码为 base64
        _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # 统计各类别数量
        class_counts = {}
        for det in detections:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        # 获取推理时间，处理可能的 None 值
        inference_time = result.speed.get("inference", 0) if result.speed else 0

        return {
            "task_id": task_id,
            "task_uuid": task.task_uuid,
            "filename": file.filename,
            "total_objects": total_objects,
            "detections": detections,
            "class_counts": class_counts,
            "annotated_image": annotated_base64,
            "inference_time": round(float(inference_time or 0), 2),
        }

    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
