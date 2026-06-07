"""你谁啊 — 后端服务入口

薄代理层，职责：
1. AI API 代理 — 转发对话请求到 Claude/DeepSeek，隐藏 API Key
2. IAP 收据验证 — 验证 App Store / Google Play 内购票据
3. 速率限制 — 防止 API 滥用
4. 健康检查 — 服务可用性监控

不存储用户数据 — 所有用户数据均在客户端。
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware.rate_limit import RateLimitMiddleware
from .routes import ai_proxy, health, payment

# 配置 structlog 输出（使用 stdlib LoggerFactory 以兼容 filter_by_level）
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    logger.info("Server starting", environment=settings.environment)
    yield
    logger.info("Server shutting down")


def create_app() -> FastAPI:
    """创建 FastAPI 应用并注册所有路由和中间件"""
    settings = get_settings()

    app = FastAPI(
        title="你谁啊 API",
        description="AI 代理 + IAP 验证服务",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    # CORS — 仅允许 App 客户端
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # App 客户端不固定源
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "X-Client-Version"],
        max_age=3600,
    )

    # 速率限制
    app.add_middleware(RateLimitMiddleware)

    # 注册路由
    app.include_router(health.router, tags=["health"])
    app.include_router(ai_proxy.router, prefix="/api", tags=["ai"])
    app.include_router(payment.router, prefix="/api/payment", tags=["payment"])

    return app


app = create_app()
