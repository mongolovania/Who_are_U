# ============================================================
# Module: Flashbulb Detector (flashbulb_detector.py)
# L2: Flashbulb memory detection based on Brown & Kulik (1977).
# L2：闪光灯记忆检测 — Brown & Kulik (1977)
#
# Flashbulb memories are vivid, long-lasting memories of
# surprising and emotionally arousing events.
# 闪光灯记忆是对惊讶且高情绪唤醒事件的生动、持久记忆。
#
# Trigger conditions (triple trigger):
#   - High surprise (surprise > 0.7)
#   - High personal relevance (personal_relevance > 0.7)
#   - High emotional arousal (arousal > 0.8)
#   All three simultaneously → "Print Now!" → flashbulb memory
#
# Flashbulb memory characteristics:
#   - Decay coefficient ×0.5 (slower forgetting)
#   - Retrieval priority boost
#   - Store "reception context" (where, with whom, emotional state)
#     — not just the event itself
#
# For COLD users: use LLM to judge if this seems like a
# flashbulb-worthy event (general knowledge prior).
# For WARM+ users: use personal baselines for detection.
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from memory_node import ValenceArousal

logger = logging.getLogger("memory_palace.flashbulb")


@dataclass
class FlashbulbContext:
    """Reception context stored with flashbulb memories."""
    location: str = ""            # Where the user was
    company: str = ""             # Who they were with
    emotional_state: str = ""     # Their emotional state
    ongoing_activity: str = ""    # What they were doing
    consequences: str = ""        # Immediate aftermath
    is_flashbulb: bool = False
    surprise_score: float = 0.0
    personal_relevance: float = 0.0


class FlashbulbDetector:
    """
    Brown & Kulik (1977) flashbulb memory detector.

    Triple-trigger model:
      1. Surprise: how unexpected was this event?
      2. Personal relevance: how much does this matter to the user?
      3. Emotional arousal: how intense is the emotional reaction?

    When all three exceed thresholds, the "Print Now!" mechanism
    activates, creating a flashbulb memory with special properties.
    """

    def __init__(self):
        # Detection thresholds (can be tuned per DDI level)
        self.surprise_threshold: float = 0.7
        self.relevance_threshold: float = 0.7
        self.arousal_threshold: float = 0.8

        # Flashbulb protection multipliers
        self.decay_multiplier: float = 0.5    # Half decay speed
        self.retrieval_boost: float = 2.0     # Double retrieval priority
        self.importance_boost: int = 3         # +3 importance points

        # Personal baselines for WARM+ users
        self._personal_arousal_mean: float = 0.3
        self._personal_arousal_std: float = 0.15
        self._has_personal_baseline: bool = False

    # ── Main detection ─────────────────────────────────────

    def detect(
        self,
        content: str,
        emotion: ValenceArousal,
        surprise: float = 0.0,
        personal_relevance: float = 0.5,
        use_llm: bool = False,
        llm_gateway=None,
    ) -> tuple[bool, FlashbulbContext]:
        """
        Detect if the current memory qualifies as a flashbulb.

        Returns (is_flashbulb, context).

        For COLD users without personal baselines:
          - Uses absolute thresholds (conservative: requires clearer signal)
          - Can optionally use LLM to judge surprise/relevance
        For WARM+ users with personal baselines:
          - Uses relative thresholds (arousal > personal_mean + 2*std)
        """
        # Adjust arousal threshold based on personal baseline
        arousal_threshold = self.arousal_threshold
        if self._has_personal_baseline:
            # Personal: flag if arousal is 2 std above personal mean
            personal_threshold = self._personal_arousal_mean + 2 * self._personal_arousal_std
            arousal_threshold = min(self.arousal_threshold, personal_threshold)

        # Triple trigger check
        surprise_ok = surprise >= self.surprise_threshold
        relevance_ok = personal_relevance >= self.relevance_threshold
        arousal_ok = emotion.arousal >= arousal_threshold

        is_flashbulb = surprise_ok and relevance_ok and arousal_ok

        context = FlashbulbContext(
            is_flashbulb=is_flashbulb,
            surprise_score=surprise,
            personal_relevance=personal_relevance,
            emotional_state=_describe_emotional_state(emotion),
        )

        if is_flashbulb:
            logger.info(
                f"Flashbulb detected! surprise={surprise:.2f}, "
                f"relevance={personal_relevance:.2f}, arousal={emotion.arousal:.2f}"
            )

        return is_flashbulb, context

    # ── Heuristic detection (no LLM, fast path) ────────────

    def detect_heuristic(
        self,
        content: str,
        arousal: float,
        valence: float,
    ) -> tuple[bool, float, float]:
        """
        Fast heuristic flashbulb detection (no LLM call).
        Uses surface signals from the content.

        Returns (is_flashbulb, surprise_estimate, relevance_estimate).
        """
        # Surprise signals
        surprise_keywords = {
            "突然": 0.3, "竟然": 0.3, "没想到": 0.3, "震惊": 0.4,
            "不可思议": 0.4, "天啊": 0.3, "居然": 0.3, "意外": 0.25,
            "惊呆了": 0.4, "难以置信": 0.4, "第一次": 0.2, "从未": 0.2,
        }
        surprise = 0.0
        for kw, weight in surprise_keywords.items():
            if kw in content:
                surprise += weight
        surprise = min(1.0, surprise)

        # Exclamation marks amplify surprise
        exclamations = content.count("!") + content.count("！")
        surprise = min(1.0, surprise + exclamations * 0.1)

        # Personal relevance signals
        relevance_keywords = {
            "我": 0.05, "我的": 0.1, "人生": 0.15, "一辈子": 0.2,
            "最重要": 0.2, "改变": 0.15, "转折": 0.2, "命运": 0.2,
            "永远": 0.15, "决定": 0.1,
        }
        relevance = 0.3  # baseline: most things people say are personally relevant
        for kw, weight in relevance_keywords.items():
            if kw in content:
                relevance += weight
        relevance = min(1.0, relevance)

        is_flashbulb = (
            surprise >= self.surprise_threshold
            and relevance >= self.relevance_threshold
            and arousal >= self.arousal_threshold
        )

        return is_flashbulb, surprise, relevance

    # ── LLM-based detection (for ambiguous cases) ──────────

    async def detect_with_llm(
        self,
        content: str,
        emotion: ValenceArousal,
        llm_gateway,
    ) -> tuple[bool, FlashbulbContext]:
        """
        Use LLM to judge if this is a flashbulb-worthy event.
        For COLD users where we lack personal baselines.
        """
        prompt = f"""Analyze if this is a "flashbulb memory" event.

Flashbulb memories (Brown & Kulik, 1977) are vivid, long-lasting memories formed during surprising, emotionally intense, personally consequential events.

User's statement: "{content[:500]}"
Emotional arousal: {emotion.arousal:.2f} (0=calm, 1=extremely excited/agitated)

Return JSON:
{{
  "is_flashbulb": true/false,
  "surprise": 0.0-1.0 (how unexpected is this event?),
  "personal_relevance": 0.0-1.0 (how consequential for this person's life?),
  "reasoning": "brief explanation"
}}"""

        try:
            import json
            response = await llm_gateway.chat_with_json(
                messages=[{"role": "user", "content": prompt}],
                system="You are a cognitive psychology expert. Respond with JSON only.",
            )
            result = json.loads(response)
            is_fb = result.get("is_flashbulb", False)
            surprise = result.get("surprise", 0.0)
            relevance = result.get("personal_relevance", 0.5)
            return self.detect(content, emotion, surprise, relevance)
        except Exception as e:
            logger.warning(f"LLM flashbulb detection failed: {e}")
            # Fall back to heuristic
            is_fb, surprise, relevance = self.detect_heuristic(content, emotion.arousal, emotion.valence)
            return self.detect(content, emotion, surprise, relevance)

    # ── Protection application ─────────────────────────────

    def apply_protection(self, importance: int) -> int:
        """
        Apply flashbulb protection to importance score.
        Flashbulb memories get +3 importance and are pinned.
        """
        return min(10, importance + self.importance_boost)

    def get_decay_multiplier(self) -> float:
        """Flashbulb memories decay at half speed."""
        return self.decay_multiplier

    def get_retrieval_boost(self) -> float:
        """Flashbulb memories get double retrieval priority."""
        return self.retrieval_boost

    # ── Baseline management ────────────────────────────────

    def update_personal_baseline(self, arousal_values: list[float]):
        """Update personal arousal baseline from accumulated data."""
        if len(arousal_values) < 10:
            return
        self._personal_arousal_mean = sum(arousal_values) / len(arousal_values)
        variance = sum((a - self._personal_arousal_mean) ** 2 for a in arousal_values) / (len(arousal_values) - 1)
        self._personal_arousal_std = max(variance ** 0.5, 0.05)
        self._has_personal_baseline = True


# ── Helpers ────────────────────────────────────────────────

def _describe_emotional_state(emotion: ValenceArousal) -> str:
    """Describe emotional state in Chinese based on Russell circumplex."""
    v, a = emotion.valence, emotion.arousal
    if v > 0.6 and a > 0.6:
        return "兴奋激动"
    elif v > 0.6 and a < 0.4:
        return "平静满足"
    elif v < 0.4 and a > 0.6:
        return "焦虑不安"
    elif v < 0.4 and a < 0.4:
        return "低落消沉"
    elif a > 0.6:
        return "情绪高涨"
    elif a < 0.4:
        return "情绪平稳"
    else:
        return "中性"
