"""速率限制中间件

简单的滑动窗口速率限制。
防止单个客户端过度使用 AI 代理。
"""

import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于客户端 IP 的滑动窗口速率限制"""

    def __init__(self, app):
        super().__init__(app)
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # 跳过非 AI 路由
        if not request.url.path.startswith("/api/chat"):
            return await call_next(request)

        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"

        now = time.time()
        window = 60  # 1 分钟窗口

        # 清理过期记录
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < window
        ]

        # 检查限制
        if len(self._requests[client_ip]) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "请求过于频繁，请稍后再试",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

        self._requests[client_ip].append(now)
        response = await call_next(request)
        return response
