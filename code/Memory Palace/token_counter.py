# ============================================================
# Module: Token Counter (token_counter.py)
# Approximate token counting for cost tracking.
# Token 近似计数 — 用于 LLM 成本追踪。
#
# Priority: tiktoken > character-based estimation
# tiktoken is the most accurate for OpenAI/DeepSeek models.
# Fallback: CJK characters ≈ 1.5 tokens, English words ≈ 1.3 tokens.
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("memory_palace.token_counter")

# Try to import tiktoken for accurate counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    # DeepSeek-V3 uses a similar tokenizer to OpenAI's cl100k_base
    _DEFAULT_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _DEFAULT_ENCODING = None


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0       # tokens that hit the prompt cache
    cost_estimate: float = 0.0   # estimated cost in RMB


@dataclass
class CostTracker:
    """Cumulative cost tracking for a user/session."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_rmb: float = 0.0
    call_count: int = 0
    calls: list[TokenUsage] = field(default_factory=list)

    def record(self, usage: TokenUsage):
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cached_tokens += usage.cached_tokens
        self.total_cost_rmb += usage.cost_estimate
        self.call_count += 1
        self.calls.append(usage)

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_rmb": round(self.total_cost_rmb, 4),
            "call_count": self.call_count,
            "avg_tokens_per_call": self.total_tokens // max(1, self.call_count),
        }


# ── Pricing (RMB per 1M tokens) ──────────────────────────

MODEL_PRICING = {
    "deepseek-chat": {"input": 1.0, "output": 2.0},       # DeepSeek-V3
    "deepseek-reasoner": {"input": 4.0, "output": 16.0},   # DeepSeek-R1
    "gpt-4o": {"input": 17.5, "output": 52.5},             # GPT-4o
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},     # Free tier
    "gemini-embedding-001": {"input": 0.0, "output": 0.0}, # Free tier
}


# ── Token counting ────────────────────────────────────────

def count_tokens(text: str, model: str = "deepseek-chat") -> int:
    """
    Count tokens in text.
    Uses tiktoken if available, otherwise falls back to estimation.
    """
    if not text:
        return 0

    if _TIKTOKEN_AVAILABLE and _DEFAULT_ENCODING:
        try:
            return len(_DEFAULT_ENCODING.encode(text))
        except Exception as e:
            logger.debug(f"tiktoken encode failed: {e}")

    # Fallback: character-based estimation
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """
    Character-based token estimation.
    CJK characters ≈ 1.5 tokens each
    English/other ≈ 0.25 tokens per character (≈1.3 per word)
    """
    cjk_count = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
    other_count = len(text) - cjk_count
    return int(cjk_count * 1.5 + other_count * 0.25)


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "deepseek-chat",
    cached_prompt_tokens: int = 0,
) -> float:
    """
    Estimate cost in RMB.

    DeepSeek-V3: ¥1/M input, ¥2/M output
    GPT-4o: $2.50/M input, $10/M output (≈¥17.5/¥52.5)
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["deepseek-chat"])

    # Cached tokens: DeepSeek offers discount on cached prompts
    # Assume 50% discount on cached input tokens
    billable_input = prompt_tokens - cached_prompt_tokens * 0.5

    input_cost = (billable_input / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def build_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    model: str = "deepseek-chat",
) -> TokenUsage:
    """Build a TokenUsage with cost estimate."""
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_tokens=cached_tokens,
        cost_estimate=round(estimate_cost(prompt_tokens, completion_tokens, model, cached_tokens), 6),
    )


# ── Cost alert ────────────────────────────────────────────

class CostAlert:
    """Monthly cost monitoring with alerts."""

    def __init__(self, monthly_budget_rmb: float = 10.0, alert_threshold: float = 0.8):
        self.monthly_budget = monthly_budget_rmb
        self.alert_threshold = alert_threshold  # alert at 80% of budget
        self._tracker = CostTracker()

    def record(self, usage: TokenUsage):
        self._tracker.record(usage)

    def should_alert(self) -> bool:
        return self._tracker.total_cost_rmb >= self.monthly_budget * self.alert_threshold

    def is_over_budget(self) -> bool:
        return self._tracker.total_cost_rmb >= self.monthly_budget

    def status(self) -> dict:
        s = self._tracker.summary()
        s["budget_rmb"] = self.monthly_budget
        s["budget_used_pct"] = round(self._tracker.total_cost_rmb / self.monthly_budget * 100, 1)
        s["alert"] = self.should_alert()
        s["over_budget"] = self.is_over_budget()
        return s

    def reset_monthly(self):
        self._tracker = CostTracker()
