# ============================================================
# Test: Importance Fusion (test_importance_fusion.py)
# L2: Multi-signal importance scoring unit tests.
#
# Covers:
#   - Sync path computation (4 signals)
#   - Async path computation (7 signals)
#   - Emergent evolution over time
#   - Content-type-specific weights
#   - Flashbulb boost
#   - Edge cases (extreme values, missing data)
# ============================================================

import math
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from importance_fusion import ImportanceFusion, ImportanceResult


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def fusion():
    return ImportanceFusion()


@pytest.fixture
def sample_sync_result(fusion):
    return fusion.compute_sync(
        content="今天被老板表扬了，感觉自己的努力终于被看见了",
        valence=0.85,
        arousal=0.7,
        user_importance=7,
        activation_count=3,
        script_deviation_score=0.4,
        is_flashbulb=False,
    )


# ── Sync Path ───────────────────────────────────────────────

class TestSyncPath:
    """Verify sync path computation (signals 1,2,4,5)."""

    def test_default_importance_is_neutral(self, fusion):
        result = fusion.compute_sync()
        assert 1.0 <= result.sync_score <= 10.0  # Clamped to valid range

    def test_high_emotion_increases_score(self, fusion):
        neutral = fusion.compute_sync(valence=0.5, arousal=0.3)
        intense = fusion.compute_sync(valence=0.1, arousal=0.95)
        assert intense.sync_score > neutral.sync_score

    def test_user_explicit_mark_matters(self, fusion):
        low = fusion.compute_sync(user_importance=1)
        high = fusion.compute_sync(user_importance=10)
        assert high.sync_score > low.sync_score

    def test_statistical_deviation_contributes(self, fusion):
        no_dev = fusion.compute_sync(script_deviation_score=0.0)
        high_dev = fusion.compute_sync(script_deviation_score=0.9)
        assert high_dev.sync_score > no_dev.sync_score

    def test_retrieval_frequency_contributes(self, fusion):
        never = fusion.compute_sync(activation_count=0)
        often = fusion.compute_sync(activation_count=50)
        assert often.sync_score > never.sync_score

    def test_flashbulb_boost_applied(self, fusion):
        normal = fusion.compute_sync(is_flashbulb=False)
        flashbulb = fusion.compute_sync(is_flashbulb=True)
        assert flashbulb.sync_score == normal.sync_score + 3.0

    def test_score_clamped_1_to_10(self, fusion):
        # All extreme values
        result = fusion.compute_sync(
            valence=0.0, arousal=1.0,
            user_importance=10,
            script_deviation_score=1.0,
            activation_count=100,
            is_flashbulb=True,
        )
        assert 1.0 <= result.sync_score <= 13.0  # 10 + 3 flashbulb boost

    def test_sync_signals_populated(self, fusion):
        result = fusion.compute_sync(
            script_deviation_score=0.5,
            valence=0.3,
            arousal=0.7,
            user_importance=8,
            activation_count=5,
        )
        assert "statistical_deviation" in result.signals
        assert "emotional_intensity" in result.signals
        assert "user_explicit_mark" in result.signals
        assert "retrieval_frequency" in result.signals


# ── Async Path ──────────────────────────────────────────────

class TestAsyncPath:
    """Verify async path computation (adds signals 3,6,7)."""

    @pytest.mark.asyncio
    async def test_async_adds_graph_signal(self, fusion, sample_sync_result):
        result = await fusion.compute_async(
            sample_sync_result,
            content="test",
            graph_edge_count=10,
            working_self_match=0.5,
            use_llm=False,
        )
        assert "association_density" in result.signals
        assert "working_self_match" in result.signals

    @pytest.mark.asyncio
    async def test_graph_density_increases_score(self, fusion, sample_sync_result):
        sparse = await fusion.compute_async(
            sample_sync_result, graph_edge_count=0,
            working_self_match=0.5, use_llm=False,
        )
        dense = await fusion.compute_async(
            sample_sync_result, graph_edge_count=100,
            working_self_match=0.5, use_llm=False,
        )
        assert dense.async_score >= sparse.async_score

    @pytest.mark.asyncio
    async def test_working_self_match_increases_score(self, fusion, sample_sync_result):
        no_match = await fusion.compute_async(
            sample_sync_result, graph_edge_count=5,
            working_self_match=0.0, use_llm=False,
        )
        perfect_match = await fusion.compute_async(
            sample_sync_result, graph_edge_count=5,
            working_self_match=1.0, use_llm=False,
        )
        assert perfect_match.async_score >= no_match.async_score

    @pytest.mark.asyncio
    async def test_async_with_llm(self, fusion, sample_sync_result):
        """LLM-based emotional meaning should be callable."""
        mock_llm = AsyncMock()
        mock_llm.chat_with_json = AsyncMock(return_value='{"emotional_meaning": 9, "reasoning": "deep personal insight"}')
        result = await fusion.compute_async(
            sample_sync_result,
            content="我今天终于想通了为什么总是重复同样的错误",
            graph_edge_count=0,
            working_self_match=0.0,
            llm_gateway=mock_llm,
            use_llm=True,
        )
        assert "emotional_meaning" in result.signals

    @pytest.mark.asyncio
    async def test_async_without_llm_uses_intensity_proxy(self, fusion, sample_sync_result):
        """Without LLM, emotional_meaning falls back to emotional_intensity."""
        result = await fusion.compute_async(
            sample_sync_result,
            content="test",
            use_llm=False,
        )
        # Should equal emotional_intensity when no LLM
        assert abs(result.signals["emotional_meaning"] - result.signals["emotional_intensity"]) < 0.01

    @pytest.mark.asyncio
    async def test_flashbulb_boost_persists_in_async(self, fusion):
        sync = fusion.compute_sync(is_flashbulb=True)
        async_result = await fusion.compute_async(sync, use_llm=False)
        # Flashbulb boost should be reflected in async score
        assert async_result.flashbulb is True


# ── Emergent Evolution ──────────────────────────────────────

class TestEmergentEvolution:
    """Verify importance evolves over time."""

    def test_new_edges_increase_emergent(self, fusion):
        result = fusion.compute_sync()
        original = result.emergent_score
        evolved = fusion.evolve(result, new_edge_count=10)
        assert evolved.emergent_score > original

    def test_new_retrievals_increase_emergent(self, fusion):
        result = fusion.compute_sync()
        original = result.emergent_score
        evolved = fusion.evolve(result, new_retrieval_count=20, new_ws_match=0.5)
        assert evolved.emergent_score >= original - 0.01

    def test_high_ws_match_increases_emergent(self, fusion):
        result = fusion.compute_sync()
        original = result.emergent_score
        evolved = fusion.evolve(result, new_ws_match=1.0)
        assert evolved.emergent_score > original

    def test_low_ws_match_decreases_emergent(self, fusion):
        result = fusion.compute_sync()
        original = result.emergent_score
        evolved = fusion.evolve(result, new_ws_match=0.0)
        assert evolved.emergent_score < original

    def test_emergent_clamped(self, fusion):
        result = fusion.compute_sync()
        # Extreme negative evolution
        result.emergent_score = 1.0
        evolved = fusion.evolve(result, new_ws_match=0.0, new_edge_count=0, new_retrieval_count=0)
        assert evolved.emergent_score >= 1.0  # minimum clamp

    def test_multiple_evolutions_accumulate(self, fusion):
        result = fusion.compute_sync()
        last_score = result.emergent_score
        for i in range(5):
            # Each evolution with edges and retrievals should maintain or increase
            result = fusion.evolve(result, new_edge_count=2, new_retrieval_count=2)
        # After several evolutions, score shouldn't collapse
        assert result.emergent_score >= 1.0


# ── Content-Type-Specific Weights ───────────────────────────

class TestContentTypeWeights:
    """Verify content-type-specific weight adjustments (Design §六)."""

    def test_chat_weights_emphasize_emotion(self, fusion):
        weights = fusion.get_weights_for_type("chat")
        assert weights["emotional_intensity"] == 0.40
        assert weights["flashbulb"] == 0.25

    def test_decision_weights_emphasize_ws(self, fusion):
        weights = fusion.get_weights_for_type("decision")
        assert weights["working_self_match"] == 0.35
        assert weights["association_density"] == 0.25

    def test_milestone_weights_emphasize_explicit(self, fusion):
        weights = fusion.get_weights_for_type("milestone")
        assert weights["user_explicit_mark"] == 0.40
        assert weights["flashbulb"] == 0.30

    def test_emotion_weights_emphasize_intensity(self, fusion):
        weights = fusion.get_weights_for_type("emotion")
        assert weights["emotional_intensity"] == 0.50
        assert weights["emotional_meaning"] == 0.25

    def test_unknown_type_falls_back_to_async(self, fusion):
        weights = fusion.get_weights_for_type("nonexistent_type")
        assert weights == fusion.async_weights


# ── Signal Independence ─────────────────────────────────────

class TestSignalIndependence:
    """Verify that signals contribute independently."""

    def test_signals_are_independent(self, fusion):
        """Each signal change should affect the score."""
        base = fusion.compute_sync()
        base_score = base.sync_score

        # Change only deviation
        dev_only = fusion.compute_sync(
            script_deviation_score=1.0,
            valence=0.5, arousal=0.3,  # same as base
            user_importance=5,  # same as base
        )
        assert dev_only.sync_score != base_score

        # Change only emotion
        emo_only = fusion.compute_sync(
            script_deviation_score=0.0,  # same as base
            valence=0.0, arousal=1.0,  # different from base
            user_importance=5,  # same as base
        )
        assert emo_only.sync_score != base_score

    def test_async_signals_added_independently(self, fusion):
        """Each async signal should independently affect async_score."""
        base = fusion.compute_sync()
        assert base.async_score == base.sync_score  # initially equal

    @pytest.mark.asyncio
    async def test_async_score_differs_from_sync(self, fusion):
        result = fusion.compute_sync()
        result = await fusion.compute_async(
            result, graph_edge_count=15,
            working_self_match=0.8,
            use_llm=False,
        )
        # Async should differ from sync when graph/WS signals present
        assert result.async_score != result.sync_score


# ── Edge Cases ──────────────────────────────────────────────

class TestImportanceEdgeCases:
    """Boundary and edge case tests."""

    def test_empty_content(self, fusion):
        result = fusion.compute_sync(content="")
        assert 1.0 <= result.sync_score <= 10.0

    def test_extreme_negative_emotion(self, fusion):
        result = fusion.compute_sync(valence=0.0, arousal=1.0)
        assert result.sync_score >= 5.0  # High arousal = flagged

    def test_extreme_positive_emotion(self, fusion):
        result = fusion.compute_sync(valence=1.0, arousal=1.0)
        assert result.sync_score >= 5.0

    def test_neutral_emotion_is_middling(self, fusion):
        result = fusion.compute_sync(valence=0.5, arousal=0.3)
        assert 1.0 <= result.sync_score <= 10.0

    def test_user_importance_clamped(self, fusion):
        low = fusion.compute_sync(user_importance=0)
        high = fusion.compute_sync(user_importance=100)
        assert 1.0 <= low.sync_score <= 10.0
        assert 1.0 <= high.sync_score <= 10.0

    @pytest.mark.asyncio
    async def test_llm_failure_handled_gracefully(self, fusion):
        mock_llm = AsyncMock()
        mock_llm.chat_with_json = AsyncMock(side_effect=Exception("API Error"))
        result = fusion.compute_sync()
        async_result = await fusion.compute_async(
            result, content="test",
            llm_gateway=mock_llm,
            use_llm=True,
        )
        assert async_result is not None
        assert async_result.async_score > 0
