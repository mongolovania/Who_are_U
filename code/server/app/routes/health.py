"""健康检查路由"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """服务健康检查"""
    return {
        "status": "ok",
        "service": "who-are-u-api",
        "version": "0.1.0",
    }


@router.get("/health/ready")
async def ready():
    """就绪检查 — 用于 K8s readiness probe"""
    from ..config import get_settings

    settings = get_settings()

    if not settings.ai_api_key:
        return {
            "status": "not_ready",
            "reason": "AI_API_KEY not configured",
        }

    return {"status": "ready"}
