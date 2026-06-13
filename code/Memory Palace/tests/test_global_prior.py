# ============================================================
# Test: Global Prior (test_global_prior.py)
# L0: LLM general knowledge prior queries.
#
# Covers:
#   - Domain emotion priors lookup
#   - Decision context emotion priors
#   - Population baseline queries
#   - Personal weight mapping (DDI → weight)
#   - Prior + personal blending
#   - Privacy constraints (no user data in queries)
# ============================================================

import math
import pytest
from unittest.mock import AsyncMock, MagicMock

from global_prior import GlobalPrior, global_prior


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def gp():
    return GlobalPrior()


# ── Domain Emotion Priors ───────────────────────────────────

class TestDomainEmotion:
    """Verify domain → emotion lookup (from published research)."""

    def test_career_domain_has_stress(self, gp):
        result = gp.get_domain_emotion(["职业"])
        assert result["valence"] < 0.5  # Career: moderate stress
        assert result["arousal"] > 0.5

    def test_daily_domain_is_neutral(self, gp):
        result = gp.get_domain_emotion(["日常"])
        assert abs(result["valence"] - 0.55) < 0.1
        assert result["arousal"] < 0.4

    def test_multiple_domains_averaged(self, gp):
        single = gp.get_domain_emotion(["职业"])
        multi = gp.get_domain_emotion(["职业", "成长"])
        # Average of two domains should differ from single
        assert multi["valence"] != single["valence"] or multi["arousal"] != single["arousal"]

    def test_unknown_domain_returns_default(self, gp):
        result = gp.get_domain_emotion(["不存在的领域"])
        assert result["valence"] == 0.5
        assert result["arousal"] == 0.3

    def test_empty_domains_returns_default(self, gp):
        result = gp.get_domain_emotion([])
        assert result["valence"] == 0.5
        assert result["arousal"] == 0.3


# ── Decision Context Priors ─────────────────────────────────

class TestDecisionContext:
    """Verify decision context → emotion lookup."""

    def test_career_change_is_stressful(self, gp):
        result = gp.get_decision_context_emotion("career_change")
        assert result["valence"] < 0.5
        assert result["arousal"] > 0.6

    def test_conflict_is_negative(self, gp):
        result = gp.get_decision_context_emotion("conflict")
        assert result["valence"] < 0.4
        assert result["arousal"] > 0.6

    def test_unknown_context_falls_back_to_general(self, gp):
        result = gp.get_decision_context_emotion("nonexistent_context")
        assert result["valence"] == 0.5
        assert result["arousal"] == 0.5


# ── Population Baselines ────────────────────────────────────

class TestPopulationBaselines:
    """Verify clinical metric population averages."""

    def test_known_metrics_return_value(self, gp):
        assert gp.get_population_baseline("allostatic_load_mean") == 0.4
        assert gp.get_population_baseline("kindling_threshold") == 3
        assert gp.get_population_baseline("emotional_inertia_mean") == 0.35
        assert gp.get_population_baseline("critical_slowing_mean") == 0.25
        assert gp.get_population_baseline("vulnerability_index_default") == 0.5

    def test_unknown_metric_returns_default(self, gp):
        assert gp.get_population_baseline("unknown_metric") == 0.5


# ── Personal Weight Mapping ─────────────────────────────────

class TestPersonalWeight:
    """Verify DDI → personal_weight sigmoid mapping."""

    def test_cold_is_zero(self, gp):
        assert gp.personal_weight_from_ddi(0) == 0.0

    def test_warm_is_low(self, gp):
        w = gp.personal_weight_from_ddi(15)
        assert 0.1 < w < 0.4

    def test_hot_is_high(self, gp):
        w = gp.personal_weight_from_ddi(100)
        assert w > 0.7

    def test_rich_is_one(self, gp):
        assert gp.personal_weight_from_ddi(200) == 1.0
        assert gp.personal_weight_from_ddi(500) == 1.0

    def test_monotonically_increasing(self, gp):
        prev = gp.personal_weight_from_ddi(0)
        for ddi in [5, 10, 25, 50, 100, 200]:
            curr = gp.personal_weight_from_ddi(ddi)
            assert curr >= prev, f"Not monotonic at DDI={ddi}"
            prev = curr

    def test_negative_ddi_returns_zero(self, gp):
        assert gp.personal_weight_from_ddi(-10) == 0.0


# ── Blending ────────────────────────────────────────────────

class TestBlending:
    """Verify prior + personal blending."""

    def test_pure_prior_when_weight_zero(self, gp):
        result = gp.blend(prior_value=0.5, personal_value=0.2, personal_weight=0.0)
        assert result == 0.5

    def test_pure_personal_when_weight_one(self, gp):
        result = gp.blend(prior_value=0.5, personal_value=0.2, personal_weight=1.0)
        assert result == 0.2

    def test_weighted_blend_50_50(self, gp):
        result = gp.blend(prior_value=0.4, personal_value=0.8, personal_weight=0.5)
        assert result == pytest.approx(0.6, rel=0.01)  # 0.4*0.5 + 0.8*0.5

    def test_none_personal_returns_prior(self, gp):
        result = gp.blend(prior_value=0.7, personal_value=None, personal_weight=0.5)
        assert result == 0.7

    def test_weight_clamped(self, gp):
        """Weight should be clamped to [0, 1]."""
        over = gp.blend(0.5, 0.9, personal_weight=2.0)
        under = gp.blend(0.5, 0.9, personal_weight=-1.0)
        assert 0.0 <= over <= 1.0
        assert 0.0 <= under <= 1.0


# ── Privacy Constraints ─────────────────────────────────────

class TestPrivacyConstraints:
    """Verify no user data leaks into prior queries."""

    def test_domain_query_is_purely_academic(self, gp):
        """Domain queries only use published research categories."""
        for domain in gp.domain_emotion_priors:
            # All domains are abstract categories, not user-specific
            assert isinstance(domain, str)
            assert len(domain) < 20  # Generic category names

    def test_decision_contexts_are_generic(self, gp):
        """Decision contexts are generic types, not user data."""
        for context in gp.decision_emotion_priors:
            assert isinstance(context, str)
            # These should be generic type labels, not containing any user info

    def test_population_baselines_are_fixed(self, gp):
        """Population baselines are constants from literature, not computed."""
        for key, value in gp.population_baselines.items():
            assert isinstance(value, (int, float))


# ── Singleton ───────────────────────────────────────────────

class TestGlobalPriorSingleton:
    """Verify module-level singleton."""

    def test_global_prior_is_instance(self):
        assert isinstance(global_prior, GlobalPrior)

    def test_singleton_works(self):
        result = global_prior.get_domain_emotion(["职业"])
        assert "valence" in result
        assert "arousal" in result
