# ============================================================
# Module: LLM Gateway (llm_gateway.py)
# Unified LLM interface with model routing.
#
# Strategy (from v6 plan):
#   Main inference: DeepSeek-V3 (~$0.14/$0.28 per 1M tokens)
#   Lightweight tasks: Gemini 2.5 Flash (free tier, 30 req/min)
#   Embedding: Gemini embedding-001 (free tier, 1500 req/day)
# ============================================================

from __future__ import annotations

import asyncio
import os
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, AsyncStream

from token_counter import count_tokens, estimate_cost, TokenUsage, build_usage

logger = logging.getLogger("memory_palace.llm")


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and the service is unavailable."""
    pass


class TaskType(Enum):
    """Task categories mapped to models."""
    INFERENCE = "inference"        # Main chat/decision
    LIGHTWEIGHT = "lightweight"    # Tagging, extraction, dehydration
    EMBEDDING = "embedding"        # Vector generation


@dataclass
class ModelConfig:
    model: str
    base_url: str
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0


@dataclass
class LLMGateway:
    """Routes LLM calls to the appropriate model based on task type."""

    config: dict = field(default_factory=dict)

    # Client cache
    _clients: dict[str, AsyncOpenAI] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._inference_cfg = self._build_inference()
        self._lightweight_cfg = self._build_lightweight()
        self._embedding_cfg = self._build_embedding()

        # ── Retry config ─────────────────────────────────
        self._max_retries: int = 3
        self._retry_base_delay: float = 1.0   # seconds
        self._retry_max_delay: float = 16.0

        # ── Circuit breaker ──────────────────────────────
        self._failure_count: int = 0
        self._circuit_open: bool = False
        self._circuit_open_since: float = 0.0
        self._circuit_reset_delay: float = 60.0  # 1 minute
        self._failure_threshold: int = 5

        # ── Cost tracking ────────────────────────────────
        self._total_cost_rmb: float = 0.0
        self._total_tokens: int = 0

        # ── Token counting ───────────────────────────────
        self._track_tokens: bool = True

    # ── Config builders ──────────────────────────────────

    def _build_inference(self) -> ModelConfig:
        dehy = self.config.get("dehydration", {})
        return ModelConfig(
            model=os.environ.get("MP_INFERENCE_MODEL", dehy.get("model", "deepseek-chat")),
            base_url=os.environ.get("MP_INFERENCE_BASE_URL", dehy.get("base_url", "https://api.deepseek.com/v1")),
            api_key=os.environ.get("OMBRE_API_KEY", dehy.get("api_key", "")),
            max_tokens=4096,
            temperature=0.7,
            timeout=90.0,
        )

    def _build_lightweight(self) -> ModelConfig:
        """Gemini 2.5 Flash for free-tier lightweight tasks."""
        return ModelConfig(
            model=os.environ.get("MP_LIGHTWEIGHT_MODEL", "gemini-2.5-flash"),
            base_url=os.environ.get(
                "MP_LIGHTWEIGHT_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("OMBRE_API_KEY", "")),
            max_tokens=1024,
            temperature=0.1,
            timeout=30.0,
        )

    def _build_embedding(self) -> ModelConfig:
        embed = self.config.get("embedding", {})
        return ModelConfig(
            model=os.environ.get("MP_EMBEDDING_MODEL", embed.get("model", "gemini-embedding-001")),
            base_url=os.environ.get(
                "MP_EMBEDDING_BASE_URL",
                embed.get("base_url", "")
                or "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("OMBRE_API_KEY", "")),
            max_tokens=1,
            temperature=0.0,
            timeout=30.0,
        )

    # ── Client management ────────────────────────────────

    def _get_client(self, cfg: ModelConfig) -> AsyncOpenAI:
        key = f"{cfg.model}@{cfg.base_url}"
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=cfg.timeout,
            )
        return self._clients[key]

    # ── Public API ───────────────────────────────────────

    # ── Circuit breaker ──────────────────────────────────

    def _check_circuit(self) -> None:
        """Check if circuit breaker is open. Raise if service is unavailable."""
        if self._circuit_open:
            elapsed = time.time() - self._circuit_open_since
            if elapsed >= self._circuit_reset_delay:
                # Try to close circuit (half-open)
                logger.info("Circuit breaker: half-open, allowing retry")
                self._circuit_open = False
                self._failure_count = 0
            else:
                remaining = self._circuit_reset_delay - elapsed
                raise CircuitBreakerOpenError(
                    f"Circuit breaker open. Reset in {remaining:.0f}s. "
                    f"Using fallback response."
                )

    def _record_success(self) -> None:
        """Record a successful call — reset failure count."""
        self._failure_count = 0
        self._circuit_open = False

    def _record_failure(self) -> None:
        """Record a failed call — potentially open circuit."""
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            logger.error(
                f"Circuit breaker OPEN after {self._failure_count} failures"
            )
            self._circuit_open = True
            self._circuit_open_since = time.time()

    # ── Chat with retry ──────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        task: TaskType = TaskType.INFERENCE,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request with retry + circuit breaker.

        Retry strategy:
          - Up to 3 retries with exponential backoff (1s, 2s, 4s)
          - Circuit breaker opens after 5 consecutive failures
          - Circuit resets after 60s
        """
        self._check_circuit()

        cfg = self._inference_cfg if task == TaskType.INFERENCE else self._lightweight_cfg
        client = self._get_client(cfg)

        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        # Count input tokens
        prompt_tokens = 0
        if self._track_tokens:
            full_text = system + "".join(m.get("content", "") for m in messages) if system else "".join(m.get("content", "") for m in messages)
            prompt_tokens = count_tokens(full_text, cfg.model)

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=cfg.model,
                    messages=oai_messages,
                    temperature=temperature if temperature is not None else cfg.temperature,
                    max_tokens=max_tokens or cfg.max_tokens,
                )
                content = resp.choices[0].message.content or ""

                # Track tokens and cost
                if self._track_tokens and resp.usage:
                    completion_tokens = resp.usage.completion_tokens
                    usage = build_usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_tokens=getattr(resp.usage, 'prompt_tokens_details', None) or {},
                        model=cfg.model,
                    )
                    self._total_cost_rmb += usage.cost_estimate
                    self._total_tokens += usage.total_tokens

                self._record_success()
                return content

            except Exception as e:
                last_error = e
                self._record_failure()

                if attempt < self._max_retries:
                    delay = min(
                        self._retry_base_delay * (2 ** attempt),
                        self._retry_max_delay,
                    )
                    logger.warning(
                        f"LLM chat attempt {attempt + 1}/{self._max_retries + 1} "
                        f"failed [{cfg.model}]: {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"LLM chat failed after {self._max_retries + 1} attempts "
                        f"[{cfg.model}]: {e}"
                    )

        # All retries exhausted — return fallback
        if self._circuit_open:
            return "我现在有点累，休息一下... 待会再聊好吗？"
        raise last_error  # type: ignore[misc]

    # ── Streaming chat ────────────────────────────────────

    async def chat_stream(
        self,
        messages: list[dict],
        task: TaskType = TaskType.INFERENCE,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming chat completion. Yields tokens as they arrive.
        Used by SSE endpoint.

        Usage:
            async for token in gateway.chat_stream(messages, system=prompt):
                yield f"data: {json.dumps({'token': token})}\\n\\n"
            yield "data: [DONE]\\n\\n"
        """
        self._check_circuit()

        cfg = self._inference_cfg if task == TaskType.INFERENCE else self._lightweight_cfg
        client = self._get_client(cfg)

        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        try:
            stream: AsyncStream = await client.chat.completions.create(
                model=cfg.model,
                messages=oai_messages,
                temperature=temperature if temperature is not None else cfg.temperature,
                max_tokens=max_tokens or cfg.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            self._record_success()

        except Exception as e:
            self._record_failure()
            logger.error(f"LLM stream failed [{cfg.model}]: {e}")
            yield "[STREAM_ERROR]"

    # ── Cost tracking ────────────────────────────────────

    @property
    def total_cost_rmb(self) -> float:
        return round(self._total_cost_rmb, 6)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def chat_with_json(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        task: TaskType = TaskType.LIGHTWEIGHT,
    ) -> str:
        """Chat with JSON response format (for structured extraction)."""
        cfg = self._lightweight_cfg if task == TaskType.LIGHTWEIGHT else self._inference_cfg
        client = self._get_client(cfg)

        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        try:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=oai_messages,
                temperature=0.1,
                max_tokens=cfg.max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or "{}"
        except Exception as e:
            logger.error(f"LLM JSON chat failed [{cfg.model}]: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector."""
        cfg = self._embedding_cfg
        client = self._get_client(cfg)
        try:
            resp = await client.embeddings.create(
                model=cfg.model,
                input=text[:2000],
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    @property
    def inference_model(self) -> str:
        return self._inference_cfg.model

    @property
    def lightweight_model(self) -> str:
        return self._lightweight_cfg.model

    @property
    def embedding_model(self) -> str:
        return self._embedding_cfg.model
