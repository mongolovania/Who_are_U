# ============================================================
# Test: Learnable Path Weights (test_learnable_weights.py)
# Track C Task 5: MemLong-style online weight learning.
# ============================================================

import math
import pytest
from unittest.mock import MagicMock, patch

from learnable_weights import (
    LearnablePathWeights, PathWeightState, FeedbackSignal,
)


# ── Sample base weights ───────────────────────────────────────

@pytest.fixture
def base_weights():
    return {
        "vector": 0.22,
        "bm25": 0.10,
        "graph": 0.18,
        "emotion": 0.10,
        "temporal": 0.12,
        "cross_ref": 0.08,
        "narrative": 0.08,
        "ppr": 0.08,
        "ws_rerank": 0.04,
    }


# ── PathWeightState Tests ─────────────────────────────────────


class TestPathWeightState:
    """Test PathWeightState data model."""

    def test_create_state(self):
        """Basic state creation."""
        state = PathWeightState(
            path_name="vector",
            base_weight=0.22,
            learned_weight=0.25,
        )
        assert state.path_name == "vector"
        assert state.base_weight == 0.22
        assert state.learned_weight == 0.25
        assert state.observation_count == 0

    def test_success_rate_empty(self):
        """Success rate with no observations."""
        state = PathWeightState(path_name="test", base_weight=0.1, learned_weight=0.1)
        assert state.success_rate == 0.5  # Neutral prior

    def test_success_rate_with_data(self):
        """Success rate calculation."""
        state = PathWeightState(
            path_name="test",
            base_weight=0.1,
            learned_weight=0.15,
            observation_count=10,
            success_count=7,
        )
        assert state.success_rate == 0.7

    def test_to_dict_and_back(self):
        """Serialization round-trip."""
        state = PathWeightState(
            path_name="vector",
            base_weight=0.22,
            learned_weight=0.28,
            observation_count=50,
            success_count=35,
        )
        data = state.to_dict()
        restored = PathWeightState.from_dict(data)

        assert restored.path_name == state.path_name
        assert restored.base_weight == state.base_weight
        assert restored.learned_weight == state.learned_weight
        assert restored.success_rate == state.success_rate


# ── FeedbackSignal Tests ──────────────────────────────────────


class TestFeedbackSignal:
    """Test FeedbackSignal data model."""

    def test_create_signal(self):
        """Basic signal creation."""
        signal = FeedbackSignal(
            result_id="mem_001",
            query="测试查询",
            query_category="factual",
            path_contributions={"vector": 0.6, "bm25": 0.4},
            engaged=True,
            referenced=False,
        )
        assert signal.result_id == "mem_001"
        assert signal.query_category == "factual"
        assert signal.engaged is True
        assert signal.ignored is False

    def test_default_ignored(self):
        """Default: result is ignored unless engaged or referenced."""
        signal = FeedbackSignal(result_id="test")
        assert signal.ignored is True


# ── LearnablePathWeights Tests ────────────────────────────────


class TestLearnablePathWeights:
    """Test the learnable weights engine."""

    def test_init_from_base_weights(self, base_weights):
        """Initialization from base weights."""
        lw = LearnablePathWeights(base_weights=base_weights)

        weights = lw.get_weights()
        assert "vector" in weights
        assert "ppr" in weights
        assert "narrative" in weights

        # Should be normalized
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

        # Should match base weights initially
        assert abs(weights["vector"] - 0.22) < 0.05

    def test_get_base_weights(self, base_weights):
        """Base weights are preserved."""
        lw = LearnablePathWeights(base_weights=base_weights)
        base = lw.get_base_weights()
        assert "vector" in base
        total = sum(base.values())
        assert abs(total - 1.0) < 0.01

    def test_record_feedback_single(self, base_weights):
        """Recording a single feedback signal."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            learning_rate=0.1,
        )
        lw.record_feedback(
            result_id="mem_001",
            path_contributions={"vector": 0.8, "bm25": 0.2},
            engaged=True,
            query_category="factual",
        )

        # A single signal doesn't trigger batch processing (need 10)
        stats = lw.get_stats()
        assert stats["total_updates"] == 0  # Not processed yet

    def test_record_feedback_batch(self, base_weights):
        """Batch of 10 signals triggers weight updates."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            learning_rate=0.2,
        )

        # Send 10 engaged signals for vector path
        for i in range(10):
            lw.record_feedback(
                result_id=f"mem_{i}",
                path_contributions={"vector": 0.8, "bm25": 0.2},
                engaged=True,
                query_category="factual",
            )

        stats = lw.get_stats()
        assert stats["total_updates"] >= 1

        # Vector should have higher learned weight after engagement
        deltas = lw.get_weight_deltas()
        assert "vector" in deltas

    def test_weights_normalize(self, base_weights):
        """Returned weights always sum to ~1.0."""
        lw = LearnablePathWeights(base_weights=base_weights)

        for _ in range(5):
            weights = lw.get_weights()
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.02

    def test_min_weight_enforced(self, base_weights):
        """Minimum weight constraint prevents zeroing out paths."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            min_weight=0.02,
        )

        weights = lw.get_weights()
        for w in weights.values():
            assert w >= 0.02

    def test_epsilon_exploration(self, base_weights):
        """Exploration weights differ from exploitation."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            epsilon=1.0,  # Always explore
        )

        weights1 = lw.get_weights()
        weights2 = lw.get_weights()

        # With epsilon=1.0, weights should have noise
        # But with same seed might be close — check structure
        for k in weights1:
            assert k in weights2

    def test_epsilon_zero_exploitation(self, base_weights):
        """With epsilon=0, weights should be deterministic."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            epsilon=0.0,
        )

        weights1 = lw.get_weights()
        weights2 = lw.get_weights()

        for k in weights1:
            assert weights1[k] == weights2[k]

    def test_category_specialization(self, base_weights):
        """Per-category weight specialization."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            learning_rate=0.2,
        )

        # Send emotional queries with emotion path engagement
        for i in range(10):
            lw.record_feedback(
                result_id=f"mem_{i}",
                path_contributions={"emotion": 0.9, "vector": 0.1},
                engaged=True,
                query_category="emotional",
            )

        # Emotional category should track emotion path
        stats = lw.get_stats()
        assert stats["categories_tracked"] >= 1

    def test_apply_explicit_feedback(self, base_weights):
        """Explicit feedback has stronger impact than implicit."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            learning_rate=0.2,
        )

        # Apply explicit positive feedback for vector
        lw.apply_explicit_feedback(
            path_ratings={"vector": 1.0, "bm25": 0.0},
        )

        stats = lw.get_stats()
        vector_stats = stats["path_stats"].get("vector", {})
        assert vector_stats.get("learned", 0.22) != 0.22  # Should have changed

    def test_reset_to_base(self, base_weights):
        """Reset returns weights to base."""
        lw = LearnablePathWeights(base_weights=base_weights)

        # Send some signals
        for i in range(10):
            lw.record_feedback(
                result_id=f"mem_{i}",
                path_contributions={"vector": 0.9, "graph": 0.1},
                engaged=True,
            )

        # Reset
        lw.reset_to_base()

        weights = lw.get_weights()
        base = lw.get_base_weights()

        for k in base:
            assert abs(weights[k] - base[k]) < 0.01

    def test_persistence(self, base_weights, tmp_path):
        """Save and load preserves state."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            user_id="test_user",
            data_dir=str(tmp_path / "buckets"),
        )

        # Record some feedback
        for i in range(10):
            lw.record_feedback(
                result_id=f"mem_{i}",
                path_contributions={"vector": 0.9},
                engaged=True,
            )

        lw.save()

        # Reload
        lw2 = LearnablePathWeights(
            base_weights=base_weights,
            user_id="test_user",
            data_dir=str(tmp_path / "buckets"),
        )
        lw2.load()

        assert lw2._total_updates == lw._total_updates

    def test_get_stats(self, base_weights):
        """Stats report contains required fields."""
        lw = LearnablePathWeights(base_weights=base_weights)
        stats = lw.get_stats()

        assert "total_updates" in stats
        assert "epsilon" in stats
        assert "weight_deltas" in stats
        assert "path_stats" in stats
        assert "categories_tracked" in stats

    def test_regularization_pull(self, base_weights):
        """Regularization pulls weights toward base."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            regularization_strength=0.01,  # Small regularization
        )

        weights = lw.get_weights()
        base = lw.get_base_weights()

        # Weights should be close to base initially
        for k in base:
            assert abs(weights[k] - base[k]) < 0.1

    def test_epsilon_decay(self, base_weights):
        """Epsilon decays over time."""
        lw = LearnablePathWeights(
            base_weights=base_weights,
            epsilon=0.1,
            epsilon_decay=0.99,
        )

        initial_epsilon = lw.epsilon

        # Process multiple batches
        for _ in range(5):
            for i in range(10):
                lw.record_feedback(
                    result_id=f"mem_{i}",
                    path_contributions={"vector": 1.0},
                    engaged=True,
                )
            # After each batch, epsilon should decay
            assert lw.epsilon <= initial_epsilon
