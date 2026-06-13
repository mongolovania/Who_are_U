# ============================================================
# Module: Importance Fusion (importance_fusion.py)
# L2: Multi-signal importance scoring.
# L2：多信号融合重要性评分
#
# 7-signal fusion model (Design §4.2):
#   1. statistical_deviation  — Schank (0-1)
#   2. emotional_intensity    — Bower (0-1)
#   3. emotional_meaning      — Bower (0-1, semantic)
#   4. user_explicit_mark     — 0 or 1
#   5. retrieval_frequency    — Ebbinghaus (normalized)
#   6. association_density    — A-MEM (in-degree/out-degree)
#   7. working_self_match     — Conway (0-1)
#
# Layered computation:
#   Sync path (at storage time): signals 1, 2, 4, 5 (cheap, no LLM)
#   Async path (background):     signals 3, 6, 7 (requires semantics or graph)
#
# Importance is not a one-time judgment — it's an emergent property.
# It evolves over time as:
#   - Association density grows → importance rises
#   - Retrieval frequency grows → importance rises
#   - Working Self changes → importance is re-evaluated
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("memory_palace.importance")


@dataclass
class ImportanceResult:
    """Multi-signal importance assessment."""
    sync_score: float          # 1-10, instantaneous (sync path)
    async_score: float         # 1-10, after deep processing (async path)
    emergent_score: float      # 1-10, evolving over time
    signals: dict              # individual signal values for debugging
    flashbulb: bool = False
    flashbulb_boost: float = 0.0


class ImportanceFusion:
    """
    7-signal importance fusion engine.

    The importance of a memory is NOT a static property assigned
    at creation time. It EMERGES over time from multiple signals.

    Signal sources:
      - Statistical: is this unusual for this user? (script_deviation)
      - Emotional: how intense is the emotion? (raw valence/arousal)
      - Semantic: what does this MEAN to the user? (LLM, async)
      - Explicit: did the user mark this as important? (hold importance param)
      - Retrieval: how often is this memory accessed? (activation_count)
      - Graph: how connected is this in the memory graph? (memory_graph)
      - WS: how relevant is this to active goals? (working_self)
    """

    def __init__(self):
        # Signal weights for sync path
        self.sync_weights = {
            "statistical_deviation": 0.25,
            "emotional_intensity": 0.35,
            "user_explicit_mark": 0.30,
            "retrieval_frequency": 0.10,
        }

        # Signal weights for async path (adds semantic + graph + WS)
        self.async_weights = {
            "statistical_deviation": 0.15,
            "emotional_intensity": 0.20,
            "emotional_meaning": 0.20,
            "user_explicit_mark": 0.15,
            "retrieval_frequency": 0.10,
            "association_density": 0.10,
            "working_self_match": 0.10,
        }

    # ── Sync path (storage time, <10ms, no LLM) ────────────

    def compute_sync(
        self,
        content: str = "",
        valence: float = 0.5,
        arousal: float = 0.3,
        user_importance: int = 5,
        activation_count: int = 0,
        script_deviation_score: float = 0.0,
        is_flashbulb: bool = False,
    ) -> ImportanceResult:
        """
        Compute importance at storage time (sync, cheap, <10ms).
        存储时即时重要性评分（同步·低成本·<10ms）。

        Uses signals 1, 2, 4, 5 only.
        """
        signals = {}

        # Signal 1: Statistical deviation (0-1 → 0-10)
        signals["statistical_deviation"] = script_deviation_score * 10

        # Signal 2: Emotional intensity (0-1 → 0-10)
        # Combine valence extremity (distance from neutral) with arousal
        valence_extremity = abs(valence - 0.5) * 2  # 0-1
        emotional_intensity = (valence_extremity * 0.3 + arousal * 0.7)
        signals["emotional_intensity"] = emotional_intensity * 10

        # Signal 4: User explicit mark (1-10, directly from hold())
        signals["user_explicit_mark"] = float(user_importance)

        # Signal 5: Retrieval frequency (normalized)
        # log(1 + count) to prevent runaway
        import math
        retrieval_norm = min(1.0, math.log(1 + activation_count) / math.log(1 + 50))
        signals["retrieval_frequency"] = retrieval_norm * 10

        # Weighted sync score
        sync_score = sum(
            signals.get(sig, 5.0) * weight
            for sig, weight in self.sync_weights.items()
        )
        sync_score = max(1.0, min(10.0, sync_score))

        # Flashbulb boost
        flashbulb_boost = 3.0 if is_flashbulb else 0.0

        return ImportanceResult(
            sync_score=round(sync_score + flashbulb_boost, 2),
            async_score=round(sync_score, 2),  # will be updated later
            emergent_score=round(sync_score, 2),
            signals=signals,
            flashbulb=is_flashbulb,
            flashbulb_boost=flashbulb_boost,
        )

    # ── Async path (background, may use LLM) ────────────────

    async def compute_async(
        self,
        sync_result: ImportanceResult,
        content: str = "",
        graph_edge_count: int = 0,
        working_self_match: float = 0.0,
        llm_gateway=None,
        use_llm: bool = False,
    ) -> ImportanceResult:
        """
        Compute full importance with async signals (background).
        后台深度评分（异步·可调用轻量LLM）。

        Adds signals 3, 6, 7 to the sync score.
        """
        signals = dict(sync_result.signals)

        # Signal 3: Emotional meaning (semantic, 0-10)
        if use_llm and llm_gateway and content:
            signals["emotional_meaning"] = await self._compute_emotional_meaning(
                content, llm_gateway
            )
        else:
            # Fallback: use emotional intensity as proxy
            signals["emotional_meaning"] = signals.get("emotional_intensity", 5.0)

        # Signal 6: Association density (0-10)
        import math
        density_norm = min(1.0, math.log(1 + graph_edge_count) / math.log(1 + 20))
        signals["association_density"] = density_norm * 10

        # Signal 7: Working Self match (0-1 → 0-10)
        signals["working_self_match"] = working_self_match * 10

        # Weighted async score
        async_score = sum(
            signals.get(sig, 5.0) * weight
            for sig, weight in self.async_weights.items()
        )
        async_score = max(1.0, min(10.0, async_score))

        # Flashbulb boost applied in async too
        if sync_result.flashbulb:
            async_score += sync_result.flashbulb_boost

        sync_result.async_score = round(async_score, 2)
        sync_result.signals = signals
        return sync_result

    # ── Emergent evolution ─────────────────────────────────

    def evolve(
        self,
        current: ImportanceResult,
        new_edge_count: int = 0,
        new_retrieval_count: int = 0,
        new_ws_match: float = 0.0,
    ) -> ImportanceResult:
        """
        Evolve importance over time — it's an emergent property.
        重要性随时间演化——不是一次性判定，而是涌现属性。

        Called periodically (dream cycle) to update emergent_score.
        """
        import math

        # Association density growth → importance rises
        density_delta = math.log(1 + new_edge_count) * 0.5

        # Retrieval frequency growth → importance rises
        retrieval_delta = math.log(1 + new_retrieval_count) * 0.3

        # Working Self relevance → importance adjustment
        ws_delta = (new_ws_match - 0.5) * 2.0  # -1 to +1

        emergent = current.emergent_score + density_delta + retrieval_delta + ws_delta
        current.emergent_score = round(max(1.0, min(10.0, emergent)), 2)
        return current

    # ── LLM emotional meaning ──────────────────────────────

    async def _compute_emotional_meaning(
        self,
        content: str,
        llm_gateway,
    ) -> float:
        """
        Use lightweight LLM to judge the emotional significance
        of a memory. Returns 0-10.

        The question is not "how intense is the emotion?" (that's
        signal 2) but "how meaningful is this emotionally?"
        — does it represent a turning point, a deep insight, etc.?
        """
        prompt = f"""Rate the EMOTIONAL MEANING of this memory on a 1-10 scale.

"Emotional meaning" is NOT the same as emotional intensity.
A quiet realization can have high meaning (e.g., "I finally understood why I keep doing this").
A dramatic event might have low meaning (e.g., "I stubbed my toe and it hurt a lot").

Memory content: "{content[:400]}"

Return JSON:
{{"emotional_meaning": 1-10, "reasoning": "one sentence"}}"""

        try:
            import json
            response = await llm_gateway.chat_with_json(
                messages=[{"role": "user", "content": prompt}],
                system="You rate the emotional significance of personal memories. JSON only.",
            )
            result = json.loads(response)
            return float(result.get("emotional_meaning", 5.0))
        except Exception as e:
            logger.warning(f"LLM emotional meaning failed: {e}")
            return 5.0

    # ── Content-type-specific weights ──────────────────────

    def get_weights_for_type(self, memory_type: str) -> dict:
        """
        Get signal weights adjusted for content type (Design §六).
        按内容类型调整信号权重。

        chat:      情绪强度40% + 闪光灯25%
        decision:  WS匹配35% + 关联密度25%
        milestone: 用户显式标记40% + 闪光灯30%
        emotion:   情绪强度50% + 情绪意义25%
        """
        type_weights = {
            "chat": {
                "emotional_intensity": 0.40,
                "statistical_deviation": 0.10,
                "emotional_meaning": 0.15,
                "flashbulb": 0.25,
                "user_explicit_mark": 0.05,
                "retrieval_frequency": 0.05,
            },
            "decision": {
                "working_self_match": 0.35,
                "association_density": 0.25,
                "user_explicit_mark": 0.15,
                "emotional_meaning": 0.10,
                "emotional_intensity": 0.05,
                "flashbulb": 0.10,
            },
            "milestone": {
                "user_explicit_mark": 0.40,
                "flashbulb": 0.30,
                "emotional_intensity": 0.10,
                "emotional_meaning": 0.10,
                "association_density": 0.10,
            },
            "emotion": {
                "emotional_intensity": 0.50,
                "emotional_meaning": 0.25,
                "statistical_deviation": 0.10,
                "flashbulb": 0.10,
                "user_explicit_mark": 0.05,
            },
        }
        return type_weights.get(memory_type, self.async_weights)
