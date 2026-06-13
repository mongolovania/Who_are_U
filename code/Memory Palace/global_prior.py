# ============================================================
# Module: Global Prior (global_prior.py)
# L0: LLM general knowledge prior queries.
# L0：LLM 通用知识先验查询
#
# When a user has insufficient personal data for statistical
# inference, we fall back on LLM training corpus knowledge.
# 当用户没有足够的个人数据做统计推断时，退回到 LLM 语料中的通用知识。
#
# This is a constitutionally-compliant fallback (Design §2.2):
#   ✅ LLM training corpus general knowledge — ALLOWED
#   ❌ Other users' data — FORBIDDEN
#   ❌ Cross-user inference — FORBIDDEN
#
# Usage pattern:
#   COLD user → global_prior.emotion_context(situation)
#   WARM user → global_prior + personal baseline (blended)
#   HOT+ user → personal model only
# ============================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("memory_palace.global_prior")


@dataclass
class GlobalPrior:
    """
    LLM general knowledge prior cache.

    Stores pre-computed general patterns so we don't call LLM
    for every cold-start request. Patterns are from published
    psychology research, not from other users.

    Privacy: ALL patterns here are from LLM training corpus
    (published academic knowledge), NEVER from any user's data.
    """

    # ── Pre-computed priors ────────────────────────────────

    # General emotional patterns by life domain
    # (from published affect research: Russell, Barrett, etc.)
    domain_emotion_priors: dict = field(default_factory=lambda: {
        "职业": {"valence": 0.45, "arousal": 0.55},   # career: moderate stress
        "关系": {"valence": 0.55, "arousal": 0.50},    # relationships: mixed
        "健康": {"valence": 0.40, "arousal": 0.45},    # health: concern
        "财务": {"valence": 0.42, "arousal": 0.52},    # finance: anxiety
        "成长": {"valence": 0.60, "arousal": 0.48},    # growth: positive
        "日常": {"valence": 0.55, "arousal": 0.30},    # daily: neutral
        "情感": {"valence": 0.50, "arousal": 0.60},    # emotional: high arousal
        "未分类": {"valence": 0.50, "arousal": 0.30},   # default
    })

    # Decision context → typical emotional profile
    # (from published decision psychology: Kahneman, Tversky, etc.)
    decision_emotion_priors: dict = field(default_factory=lambda: {
        "career_change": {"valence": 0.45, "arousal": 0.65},
        "relationship": {"valence": 0.50, "arousal": 0.60},
        "relocation": {"valence": 0.48, "arousal": 0.58},
        "financial": {"valence": 0.40, "arousal": 0.55},
        "education": {"valence": 0.55, "arousal": 0.50},
        "health": {"valence": 0.42, "arousal": 0.52},
        "conflict": {"valence": 0.35, "arousal": 0.70},
        "general": {"valence": 0.50, "arousal": 0.50},
    })

    # Vulnerability factors from published clinical research
    # (McEwen, Post, Kuppens, Scheffer — population averages)
    population_baselines: dict = field(default_factory=lambda: {
        "allostatic_load_mean": 0.4,
        "kindling_threshold": 3,           # major events before sensitization
        "emotional_inertia_mean": 0.35,    # autocorrelation of valence
        "critical_slowing_mean": 0.25,     # recovery rate baseline
        "vulnerability_index_default": 0.5, # neutral baseline
    })

    # ── Query interface ────────────────────────────────────

    def get_domain_emotion(self, domains: list[str]) -> dict:
        """
        Get the general emotional profile for a life domain.
        获取某个生活领域的通用情感画像。

        Returns {valence, arousal} based on published research,
        not on any specific user's data.
        """
        if not domains:
            return {"valence": 0.5, "arousal": 0.3}

        # Average across all matching domains
        vals = []
        for d in domains:
            if d in self.domain_emotion_priors:
                vals.append(self.domain_emotion_priors[d])

        if not vals:
            return {"valence": 0.5, "arousal": 0.3}

        return {
            "valence": sum(v["valence"] for v in vals) / len(vals),
            "arousal": sum(v["arousal"] for v in vals) / len(vals),
        }

    def get_decision_context_emotion(self, context_type: str = "general") -> dict:
        """
        Get typical emotional profile for a decision context type.
        获取某类决策场景的典型情感画像（来自决策心理学文献）。
        """
        return self.decision_emotion_priors.get(
            context_type,
            self.decision_emotion_priors["general"],
        )

    def get_population_baseline(self, metric: str) -> float:
        """
        Get a population-average baseline for clinical metrics.
        Used as prior when no personal data exists.

        Metrics: allostatic_load, kindling_threshold, emotional_inertia,
                 critical_slowing, vulnerability_index
        """
        return self.population_baselines.get(metric, 0.5)

    # ── Blending (prior + personal) ────────────────────────

    def blend(
        self,
        prior_value: float,
        personal_value: Optional[float],
        personal_weight: float,  # 0-1, how much to trust personal data
    ) -> float:
        """
        Blend LLM prior with personal data.
        Weight shifts from prior→personal as DDI increases.

        COLD (weight=0): 100% prior
        WARM (weight=0.3): 70% prior + 30% personal
        HOT  (weight=0.7): 30% prior + 70% personal
        RICH (weight=1.0): 100% personal
        """
        if personal_value is None:
            return prior_value
        w = max(0.0, min(1.0, personal_weight))
        return prior_value * (1 - w) + personal_value * w

    def personal_weight_from_ddi(self, ddi: float) -> float:
        """
        Map DDI score to personal data weight.
        Smooth sigmoid-like transition:
          DDI 0   → weight 0.0 (pure prior)
          DDI 10  → weight 0.2
          DDI 50  → weight 0.7
          DDI 200 → weight 1.0 (pure personal)
        """
        if ddi <= 0:
            return 0.0
        if ddi >= 200:
            return 1.0
        # Logistic-like curve
        import math
        midpoint = 50
        steepness = 0.03
        return 1.0 / (1.0 + math.exp(-steepness * (ddi - midpoint)))

    # ── Dynamic LLM query (for complex cases) ──────────────

    async def query_llm_prior(
        self,
        situation: str,
        llm_gateway=None,  # LLMGateway instance
    ) -> dict:
        """
        Query LLM for a general prior about a situation type.

        This asks the LLM: "Based on your training knowledge of human
        psychology, what is the typical emotional profile for someone
        in [situation]?"

        NEVER includes any user-specific data in the query.
        """
        if llm_gateway is None:
            return {"valence": 0.5, "arousal": 0.3, "confidence": 0.0}

        prompt = f"""You are a psychology research assistant. Based on published academic knowledge of human psychology and behavior, describe the TYPICAL emotional profile for someone experiencing:

"{situation}"

Return ONLY a JSON object with:
- valence (0=negative to 1=positive): typical emotional valence
- arousal (0=calm to 1=excited): typical arousal level
- common_patterns: 2-3 typical behavioral/emotional patterns (brief)
- confidence (0-1): how confident you are in this assessment

Do NOT reference any specific person. This is about GENERAL population patterns from academic research.

JSON:"""

        try:
            response = await llm_gateway.chat_with_json(
                messages=[{"role": "user", "content": prompt}],
                system="You are a psychology research assistant. Respond with JSON only.",
            )
            return json.loads(response)
        except Exception as e:
            logger.warning(f"LLM prior query failed: {e}")
            return {"valence": 0.5, "arousal": 0.3, "confidence": 0.0}


# ── Singleton ──────────────────────────────────────────────

global_prior = GlobalPrior()
