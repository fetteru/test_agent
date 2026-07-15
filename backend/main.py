from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.api.auth import router as auth_router
from app.core.exceptions import register_exception_handlers
from app.middleware.request_logger import RequestLogMiddleware
from app.api.health import router as health_router
from app.api.training import router as training_router

def init_minio():
    """初始化 MinIO 存储桶"""
    from app.storage.minio_client import MinIOClient
    try:
        minio_client = MinIOClient()
        print(f"MinIO 存储桶 '{minio_client.bucket_name}' 初始化完成")
    except Exception as e:
        print(f"MinIO 初始化失败: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应⽤⽣命周期管理"""
    # 启动时执⾏
    print("正在初始化服务...")
    init_minio()
    yield
    # 关闭时执⾏（如果需要）
    print("服务已关闭")


# 创建 FastAPI 实例
app = FastAPI(
    title="RSOD Agent Platform",
    version="0.1.0",
    description="基于 YOLOv11 的⽬标检测智能体平台 API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── 注册全局异常处理器 ─────────────────────────────────
register_exception_handlers(app)

# ── CORS 中间件配置 ──────────────────────────────────
# 允许前端跨域请求后端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 请求⽇志中间件（在 CORS 之后执⾏）
app.add_middleware(RequestLogMiddleware)

# ── 注册路由 ─────────────────────────────────────────
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(training_router)


@app.get("/")
def root():
    return {
        "message": "欢迎使⽤ RSOD Agent Platform",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                 host="0.0.0.0", 
                 port=8000, 
                 reload=True,
                 reload_excludes=[".venv", "__pycache__", ".git", "logs"]
                 )