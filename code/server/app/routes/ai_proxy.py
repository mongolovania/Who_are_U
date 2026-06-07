"""AI 代理路由 — 转发对话请求到 AI 服务

职责：
- 接收客户端对话请求
- 匿名化后转发到 Claude/DeepSeek API
- 返回 AI 回复
- 不持久化任何对话内容
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..services.ai_service import AIService, AIServiceError, AIRateLimitError

router = APIRouter()


class ChatRequest(BaseModel):
    """对话请求 — 不包含任何用户标识"""

    messages: list[dict] = Field(
        ...,
        description="对话历史（已由客户端匿名化）",
        max_length=100,
    )
    system_prompt: str | None = Field(
        default=None,
        description="系统提示词（5阶段对应模板）",
    )
    max_tokens: int = Field(default=4096, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    """对话响应"""

    content: str
    finish_reason: str | None = None
    usage: dict | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送对话到 AI 服务

    POST /api/chat

    请求体中的 messages 应为匿名化的对话历史。
    服务端不存储任何对话内容，仅做转发。
    """
    settings = get_settings()

    if not settings.ai_api_key:
        raise HTTPException(
            status_code=503,
            detail="AI 服务未配置",
        )

    try:
        ai_service = AIService(settings)
        response = await ai_service.chat(
            messages=request.messages,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return response

    except AIServiceError as e:
        raise HTTPException(
            status_code=502,
            detail=f"AI 服务错误: {e.message}",
        )
    except AIRateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {e.retry_after}s 后重试",
            headers={"Retry-After": str(e.retry_after)},
        )
