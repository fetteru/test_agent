"""
pytest 全局 Fixtures
Fixtures 是 pytest 的核⼼概念，⽤于：
 - 创建测试所需的前置条件（如数据库连接、测试客户端）
 - 在所有测试⽤例间共享资源
 - ⾃动清理测试产⽣的数据
conftest.py 中的 fixtures 对所有测试⽂件可⽤，⽆需显式导⼊。
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.session import Base, get_db

# ── 测试数据库 ────────────────────────────────────────
# 使⽤ SQLite 内存数据库进⾏测试，避免依赖 PostgreSQL
# 优点：速度快、隔离性好、⽆需清理
# 注意：SQLite 不⽀持 PostgreSQL 特有功能（如 JSON 字段），
# 但对于基础 CRUD 测试⾜够
TEST_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 特有参数
)
TestSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


def override_get_db():
    """覆盖 FastAPI 的 get_db 依赖，使⽤测试数据库"""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── 导⼊所有模型（确保 Base.metadata 包含所有表定义）───
from app.entity import db_models  # noqa: E402, F401
from main import app  # noqa: E402

# 覆盖依赖
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
     创建测试数据库表（所有测试共享）
     scope="session"：整个测试会话只执⾏⼀次
     autouse=True：⾃动应⽤，⽆需在测试函数中显式引⽤
     """
    Base.metadata.create_all(bind=test_engine)
    yield
    # 测试结束后清理
    Base.metadata.drop_all(bind=test_engine)
    # 关键：销毁数据库引擎，彻底释放 test.db 文件占用句柄
    test_engine.dispose()
    # 删除 SQLite ⽂件
    import os
    if os.path.exists("./test.db"):
        os.remove("./test.db")


@pytest.fixture
def client():
    """
     提供 FastAPI 测试客户端
     ⽤法：
     def test_health(client):
     response = client.get("/api/health")
     assert response.status_code == 200

     """
    return TestClient(app)


@pytest.fixture
def db_session():
    """
     提供独⽴的数据库会话（每个测试⽤独⽴事务）
     ⽤法：
     def test_create_user(db_session):
     user = User(username="test", ...)
     db_session.add(user)
     db_session.commit()
     """
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()