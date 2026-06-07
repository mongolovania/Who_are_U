"""AI 服务封装

对 Claude/DeepSeek API 的封装层：
- 统一请求/响应格式
- 错误处理和重试
- Token 使用跟踪
- httpx 客户端生命周期管理
"""

from dataclasses import dataclass

import httpx
import structlog

from ..config import Settings

logger = structlog.get_logger(__name__)


class AIServiceError(Exception):
    """AI 服务通用错误"""

    def __init__(self, message: str):
        self.message = message


class AIRateLimitError(Exception):
    """AI 服务速率限制错误"""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after


class AIService:
    """AI 对话服务 — 每次请求创建新实例以隔离连接"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _build_client(self) -> httpx.AsyncClient:
        """构建带认证头的 httpx 客户端"""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self.settings.ai_api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict:
        """发送对话请求

        Args:
            messages: 匿名化对话历史
            system_prompt: 阶段模板系统提示词
            max_tokens: 最大回复 token 数
            temperature: 温度参数

        Returns:
            {"content": "...", "finish_reason": "...", "usage": {...}}

        Raises:
            AIServiceError: AI 服务调用失败
            AIRateLimitError: 请求频率超限
        """
        body = {
            "model": self.settings.ai_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
        }

        if system_prompt:
            body["system"] = system_prompt

        async with self._build_client() as client:
            try:
                response = await client.post(
                    f"{self.settings.ai_base_url}/messages",
                    json=body,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    raise AIRateLimitError(retry_after)

                if response.status_code != 200:
                    logger.error(
                        "AI API error",
                        status=response.status_code,
                        body=response.text[:500],
                    )
                    raise AIServiceError(f"AI API 返回 {response.status_code}")

                data = response.json()
                content, finish_reason = self._extract_response(data)

                usage = data.get("usage", {})
                logger.info("AI chat completed", tokens=usage.get("total_tokens", 0))

                return {
                    "content": content,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }

            except (AIServiceError, AIRateLimitError):
                raise
            except httpx.TimeoutException:
                raise AIServiceError("AI 服务响应超时")
            except Exception as e:
                logger.error("AI 调用异常", error=str(e))
                raise AIServiceError(f"AI 调用失败: {e}")

    @staticmethod
    def _extract_response(data: dict) -> tuple[str, str | None]:
        """从 AI 响应中提取文本内容和结束原因

        支持 OpenAI 兼容格式和 Anthropic 原生格式。
        """
        if "choices" in data:
            # OpenAI / DeepSeek 兼容格式
            choice = data["choices"][0]
            return choice["message"]["content"], choice.get("finish_reason")

        if "content" in data:
            # Anthropic 原生格式
            content_blocks = data.get("content", [])
            text = content_blocks[0].get("text", "") if content_blocks else ""
            return text, data.get("stop_reason")

        return "", None
