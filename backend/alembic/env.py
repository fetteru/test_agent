"""
Alembic 数据库迁移配置⽂件
核⼼作⽤：
 1. 配置 SQLAlchemy 模型元数据来源（Base.metadata）
 2. 提供在线/离线两种迁移模式
 3. 连接 alembic.ini 中的数据库配置
使⽤流程：
 1. 修改模型后执⾏：alembic revision --autogenerate -m "描述"
 2. 执⾏迁移：alembic upgrade head
 3. 回滚迁移：alembic downgrade -1
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 将项⽬根⽬录加⼊ Python 路径，确保能导⼊ app 模块
import sys
from pathlib import Path
# __file__ = alembic/env.py
# __file__.parent = alembic/
# __file__.parent.parent = backend/（项⽬根⽬录）
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导⼊ SQLAlchemy 的 Base 类，⽤于获取模型元数据
from app.database.session import Base
# 关键：导⼊所有模型模块，触发模型类的定义和注册
# 原理：当 Python 执⾏此 import 时，会执⾏ db_models.py 中的所有类定义
# 模型类继承⾃ Base，会⾃动注册到 Base.metadata 中
# 虽然 db_models 变量没被直接使⽤，但这个导⼊是必须的
from app.entity import db_models

# 获取 Alembic 配置对象（从 alembic.ini 读取配置）
config = context.config

# 配置 Python ⽇志系统（读取 alembic.ini 中的 [loggers] 等配置）
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置迁移⽬标元数据：告诉 Alembic 从哪⾥读取模型结构
# Alembic 通过对⽐ Base.metadata 和数据库当前状态来⽣成迁移脚本
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
     离线迁移模式（不连接数据库）
     使⽤场景：
     - 需要⽣成 SQL 脚本⽂件，⼿动执⾏或在其他环境执⾏
     - 数据库驱动不可⽤，但需要⽣成迁移 SQL
     ⼯作⽅式：
     - 只使⽤数据库 URL 和元数据⽣成 SQL 语句
     - 不创建数据库连接，直接输出 SQL 到控制台或⽂件
     """
    # 从 alembic.ini 读取数据库连接 URL
    url = config.get_main_option("sqlalchemy.url")
    # 配置迁移上下⽂
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,  # 将参数值直接嵌⼊ SQL（便于⽣成独⽴脚本）
        dialect_opts={"paramstyle": "named"},  # 使⽤命名参数⻛格
    )
    # 执⾏迁移
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
     在线迁移模式（连接数据库执⾏）
     使⽤场景：
     - 本地开发环境，直接连接数据库执⾏迁移
     - CI/CD 流程中⾃动执⾏迁移
     ⼯作⽅式：
     - 创建数据库 Engine 和连接
     - 对⽐数据库当前状态与模型定义
     - 直接在数据库中执⾏迁移操作
     """
    # 从配置创建数据库连接引擎
    connectable = engine_from_config(
        # 获取配置段（默认是 [alembic]）
        config.get_section(config.config_ini_section, {}),
        # 只读取以 "sqlalchemy." 开头的配置项
        prefix="sqlalchemy.",
        # 使⽤ NullPool（迁移完成后⽴即释放连接，避免连接泄漏）
        poolclass=pool.NullPool,
    )
    # 建⽴数据库连接
    with connectable.connect() as connection:
        # 配置迁移上下⽂，关联到当前连接
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        # 在事务中执⾏迁移（确保原⼦性，失败可回滚）
        with context.begin_transaction():
            context.run_migrations()


# 根据执⾏模式选择迁移⽅式
# 通过命令⾏参数 --offline 或环境变量控制
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()