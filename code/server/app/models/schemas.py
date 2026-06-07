"""共享数据模型 / Pydantic Schemas

初期为服务端验证和文档定义。
后续可与客户端共享类型定义。
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应格式"""

    error: str
    code: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    service: str
    version: str
