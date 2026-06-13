# ============================================================
# Module: Cold Start (cold_start.py)
# L0: Cold-start strategies for new users.
# L0：冷启动策略 — 新用户没有个人数据时的默认行为。
#
# Core principle (Design §2.2):
#   "Minimum Assumption" — for cold-start users, assume every
#   session is important. Store everything, return everything,
#   use LLM general knowledge for emotion/intent understanding.
#   "最小假设原则" — 冷启动用户假设每次会话都重要。
#
# Design references:
#   - Vapnik: Structural Risk Minimization
#     → Model complexity should match data amount
#   - Adomavicius: Cold-start recommendation
#     → Use content-based + general prior when no collaborative data
#   - Narayanan: LLM as prior
#     → LLM training corpus = general human behavior prior
# ============================================================

from __future__ import annotations

import logging
from typing import Optional

from memory_node import (
    MemoryNode, MemoryType, BucketType, ValenceArousal,
    DDILevel, COLD_STRATEGY,
)

logger = logging.getLogger("memory_palace.cold_start")


class ColdStartPolicy:
    """
    Cold-start policy engine for new users (DDI = COLD).

    When we have ZERO personal data, we rely on:
      1. LLM general knowledge (what would most people feel?)
      2. Session meta-signals (time of day, message length, keywords)
      3. Conservative defaults (assume importance, don't decay)
    """

    # ── Storage gate (always store for cold users) ─────────

    def should_store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CHAT,
        valence: float = 0.5,
        arousal: float = 0.3,
    ) -> tuple[bool, str]:
        """
        COLD storage gate: store everything unless it's trivially short.

        Returns (should_store, reason).

        Minimum assumption: every user utterance may be important.
        Only filter out purely functional messages (< 5 chars).
        """
        stripped = content.strip()
        if len(stripped) < 5:
            return False, "too_short"
        return True, "cold_start_store_all"

    # ── Importance estimation (no personal baseline) ───────

    def estimate_importance(
        self,
        content: str,
        arousal: float = 0.3,
        message_length: int = 0,
        session_hour: int = 12,
    ) -> int:
        """
        Estimate importance without personal data.

        Heuristics (LLM knowledge prior proxy):
          - Content length: longer = more invested → higher importance
          - Arousal > 0.7: emotionally charged → higher importance
          - Late night (0-5h): vulnerable hours → slightly higher
          - Default: 5 (neutral)
        """
        score = 5  # default baseline

        length = message_length or len(content)
        if length > 200:
            score += 2
        elif length > 80:
            score += 1
        elif length < 15:
            score -= 1

        if arousal > 0.7:
            score += 2
        elif arousal > 0.5:
            score += 1

        if 0 <= session_hour <= 5:
            score += 1  # 凌晨脆弱时段

        return max(1, min(10, score))

    # ── Emotion estimation (LLM prior proxy) ───────────────

    def estimate_emotion(
        self,
        content: str,
        session_hour: int = 12,
    ) -> ValenceArousal:
        """
        Estimate emotion from content surface signals.
        For cold users, this is a simple keyword heuristic.
        Real LLM-based estimation is done by dehydrator.analyze().
        """
        # Simple keyword heuristics as fallback
        positive_words = {"开心", "高兴", "好", "爱", "喜欢", "棒", "成功", "拿到", "收到",
                          "谢谢", "感恩", "期待", "兴奋", "激动", "幸福", "满意", "赞"}
        negative_words = {"难过", "伤心", "焦虑", "害怕", "生气", "烦", "累", "失败", "失去",
                          "担心", "压力", "痛苦", "绝望", "无聊", "孤独", "迷茫", "崩溃"}

        pos_count = sum(1 for w in positive_words if w in content)
        neg_count = sum(1 for w in negative_words if w in content)

        if pos_count > neg_count:
            valence = 0.7
        elif neg_count > pos_count:
            valence = 0.3
        else:
            valence = 0.5

        # Arousal: urgency signals
        exclamation = content.count("!") + content.count("！")
        question = content.count("?") + content.count("？")
        arousal = 0.3
        if exclamation > 2:
            arousal = 0.7
        elif exclamation > 0:
            arousal = 0.5
        if 0 <= session_hour <= 5:
            arousal = max(arousal, 0.6)

        return ValenceArousal(valence=valence, arousal=arousal)

    # ── Retrieval strategy ─────────────────────────────────

    def get_retrieval_limit(self, ddi: float = 0.0) -> int:
        """COLD users: return ALL memories (there aren't many)."""
        if ddi < 5:
            return 50  # very new: show everything
        elif ddi < 10:
            return 30
        return 20  # approaching WARM

    # ── Decay strategy ─────────────────────────────────────

    def get_decay_config(self, ddi: float = 0.0) -> dict:
        """COLD users: NO decay. Protect early memories."""
        return {
            "decay_enabled": False,
            "decay_lambda": 0.0,
            "archive_threshold": 0.0,  # never archive
            "auto_resolve_enabled": False,
        }

    # ── Emotion strategy ───────────────────────────────────

    def get_emotion_mode(self, ddi: float = 0.0) -> str:
        """
        COLD: warm_default — use max empathy, assume positive intent.
        As approaching WARM (DDI > 5): llm_prior — use LLM general knowledge.
        """
        return "llm_prior" if ddi > 5 else "warm_default"


# ── Singleton ──────────────────────────────────────────────

cold_start = ColdStartPolicy()
