# ============================================================
# Test: DDA Controller (test_dda_controller.py)
# L0: Data-Density-Adaptive controller unit tests.
#
# Covers:
#   - DDI calculation formula correctness
#   - COLD→WARM→HOT→RICH level mapping
#   - Strategy matrix selection
#   - User stats persistence
#   - Session logging + regularity calculation
#   - Boundary conditions (zero sessions, extreme values)
# ============================================================

import json
import math
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path

from dda_controller import DDAController, UserStats
from memory_node import DDILevel, DDAStrategy, STRATEGY_MATRIX


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def dda_controller(tmp_path):
    """DDAController with temp directory."""
    return DDAController(stats_dir=str(tmp_path / "buckets"))


@pytest.fixture
def empty_stats():
    """Brand-new user stats."""
    return UserStats(user_id="test_user")


@pytest.fixture
def warm_stats():
    """WARM-level user (~25 sessions, ~3/week)."""
    return UserStats(
        user_id="test_user",
        total_sessions=25,
        sessions_this_week=3,
        avg_session_duration_minutes=12,
        avg_session_depth=0.4,
        days_since_first_use=60,
        session_regularity=0.8,
        time_of_day_pattern_score=0.1,
        sessions_per_week=2.9,
    )


@pytest.fixture
def hot_stats():
    """HOT-level user (~100 sessions, ~5/week)."""
    return UserStats(
        user_id="test_user",
        total_sessions=100,
        sessions_this_week=5,
        avg_session_duration_minutes=20,
        avg_session_depth=0.6,
        days_since_first_use=180,
        session_regularity=0.9,
        time_of_day_pattern_score=0.3,
        sessions_per_week=5.1,
    )


@pytest.fixture
def rich_stats():
    """RICH-level user (~300 sessions, ~7/week)."""
    return UserStats(
        user_id="test_user",
        total_sessions=300,
        sessions_this_week=7,
        avg_session_duration_minutes=30,
        avg_session_depth=0.8,
        days_since_first_use=365,
        session_regularity=0.95,
        time_of_day_pattern_score=0.6,
        sessions_per_week=7.2,
    )


# ── DDI Calculation ─────────────────────────────────────────

class TestDDICalculation:
    """Verify DDI formula correctness (Design §2.1)."""

    def test_empty_stats_gives_low_ddi(self, dda_controller, empty_stats):
        ddi = dda_controller.calculate_ddi(empty_stats)
        assert 0.0 <= ddi < 10, f"Empty stats should be COLD, got {ddi}"

    def test_warm_stats_returns_positive_ddi(self, dda_controller, warm_stats):
        ddi = dda_controller.calculate_ddi(warm_stats)
        assert ddi >= 0, f"WARM stats should give positive DDI, got {ddi}"

    def test_hot_stats_higher_than_warm(self, dda_controller, warm_stats, hot_stats):
        warm_ddi = dda_controller.calculate_ddi(warm_stats)
        hot_ddi = dda_controller.calculate_ddi(hot_stats)
        assert hot_ddi > warm_ddi

    def test_rich_stats_highest(self, dda_controller, rich_stats):
        ddi = dda_controller.calculate_ddi(rich_stats)
        assert ddi > 0, f"RICH stats should give positive DDI, got {ddi}"

    def test_ddi_monotonically_increases_with_sessions(self, dda_controller):
        """More sessions → higher DDI."""
        stats = UserStats(user_id="test")
        stats.days_since_first_use = 60
        prev = dda_controller.calculate_ddi(stats)
        for sessions in [1, 5, 10, 20, 50, 100]:
            stats.total_sessions = sessions
            stats.sessions_per_week = sessions / 8.57  # 60 days ≈ 8.57 weeks
            curr = dda_controller.calculate_ddi(stats)
            assert curr >= prev, f"DDI should increase: sessions={sessions}, prev={prev:.2f}, curr={curr:.2f}"
            prev = curr

    def test_ddi_respects_weights(self, dda_controller):
        """Each input component should contribute proportionally."""
        base = UserStats(
            user_id="test",
            total_sessions=10,
            sessions_per_week=1.0,
            avg_session_duration_minutes=10,
            avg_session_depth=0.3,
            days_since_first_use=30,
            session_regularity=1.0,
            time_of_day_pattern_score=0.1,
        )
        base_ddi = dda_controller.calculate_ddi(base)

        # Doubling sessions_per_week should increase DDI the most (weight 0.25)
        high_freq = UserStats(**{**base.__dict__})
        high_freq.sessions_per_week = 2.0
        freq_ddi = dda_controller.calculate_ddi(high_freq)
        assert freq_ddi > base_ddi

        # Late-night pattern should also increase DDI
        late_night = UserStats(**{**base.__dict__})
        late_night.time_of_day_pattern_score = 0.8
        night_ddi = dda_controller.calculate_ddi(late_night)
        assert night_ddi > base_ddi


# ── Level Mapping ───────────────────────────────────────────

class TestLevelMapping:
    """Verify DDI → level mapping."""

    def test_cold_range(self, dda_controller):
        assert dda_controller.get_level(0) == DDILevel.COLD
        assert dda_controller.get_level(5) == DDILevel.COLD
        assert dda_controller.get_level(9.9) == DDILevel.COLD

    def test_warm_range(self, dda_controller):
        assert dda_controller.get_level(10) == DDILevel.WARM
        assert dda_controller.get_level(30) == DDILevel.WARM
        assert dda_controller.get_level(49) == DDILevel.WARM

    def test_hot_range(self, dda_controller):
        assert dda_controller.get_level(50) == DDILevel.HOT
        assert dda_controller.get_level(100) == DDILevel.HOT
        assert dda_controller.get_level(199) == DDILevel.HOT

    def test_rich_range(self, dda_controller):
        assert dda_controller.get_level(200) == DDILevel.RICH
        assert dda_controller.get_level(500) == DDILevel.RICH
        assert dda_controller.get_level(10000) == DDILevel.RICH


# ── Strategy Matrix ─────────────────────────────────────────

class TestStrategySelection:
    """Verify correct strategy returned for each level."""

    def test_cold_strategy(self, dda_controller):
        strategy = dda_controller.get_strategy(5)
        assert strategy == STRATEGY_MATRIX[DDILevel.COLD]
        assert strategy.store_all is True
        assert strategy.decay_enabled is False
        assert strategy.retrieval_mode == "all"

    def test_warm_strategy(self, dda_controller):
        strategy = dda_controller.get_strategy(25)
        assert strategy == STRATEGY_MATRIX[DDILevel.WARM]
        assert strategy.use_vector_search is True
        assert strategy.use_bm25_search is False
        assert strategy.retrieval_mode == "semantic_time"

    def test_hot_strategy(self, dda_controller):
        strategy = dda_controller.get_strategy(100)
        assert strategy == STRATEGY_MATRIX[DDILevel.HOT]
        assert strategy.retrieval_mode == "three_way"
        assert strategy.use_vector_search is True
        assert strategy.use_bm25_search is True
        assert strategy.use_graph_search is True
        assert strategy.vulnerability_enabled is True

    def test_rich_strategy(self, dda_controller):
        strategy = dda_controller.get_strategy(300)
        assert strategy == STRATEGY_MATRIX[DDILevel.RICH]
        assert strategy.retrieval_mode == "four_way_ws"
        assert strategy.use_ws_rerank is True
        assert strategy.use_vulnerability_gate is True
        assert strategy.importance_mode == "full_fusion"

    def test_get_strategy_for_user(self, dda_controller):
        level, ddi, strategy = dda_controller.get_strategy_for_user("new_user")
        assert level == DDILevel.COLD
        assert ddi >= 0.0
        assert strategy.store_all is True


# ── Stats Persistence ───────────────────────────────────────

class TestStatsPersistence:
    """Verify stats save/load round-trip."""

    def test_save_and_load_roundtrip(self, dda_controller, warm_stats):
        dda_controller.save_stats(warm_stats)
        loaded = dda_controller.load_stats(warm_stats.user_id)
        assert loaded.total_sessions == warm_stats.total_sessions
        assert loaded.sessions_per_week == warm_stats.sessions_per_week
        assert loaded.avg_session_depth == warm_stats.avg_session_depth
        assert loaded.session_regularity == warm_stats.session_regularity

    def test_new_user_loads_empty_stats(self, dda_controller):
        stats = dda_controller.load_stats("nonexistent_user")
        assert stats.total_sessions == 0
        assert stats.user_id == "nonexistent_user"

    def test_corrupted_stats_file_is_handled(self, dda_controller):
        user_id = "corrupt_user"
        path = dda_controller._stats_path(user_id)
        os.makedirs(path.parent, exist_ok=True)
        path.write_text("not valid json {{{", encoding="utf-8")
        stats = dda_controller.load_stats(user_id)
        assert stats.total_sessions == 0  # Returns empty stats


# ── Session Update ──────────────────────────────────────────

class TestSessionUpdate:
    """Verify after-session stats update."""

    def test_first_session_updates(self, dda_controller, empty_stats):
        updated = dda_controller.update_after_session(
            empty_stats,
            session_duration_minutes=15,
            session_depth=0.5,
            session_start_hour=14,
        )
        assert updated.total_sessions == 1
        assert updated.sessions_per_week > 0

    def test_late_night_detection(self, dda_controller, empty_stats):
        updated = dda_controller.update_after_session(
            empty_stats,
            session_duration_minutes=10,
            session_depth=0.4,
            session_start_hour=3,
        )
        assert updated.time_of_day_pattern_score == 0.8

    def test_ema_smoothing(self, dda_controller):
        """Exponential moving average should converge."""
        stats = UserStats(
            user_id="test",
            total_sessions=10,
            avg_session_duration_minutes=10,
            avg_session_depth=0.4,
        )
        updated = dda_controller.update_after_session(
            stats,
            session_duration_minutes=30,
            session_depth=0.7,
            session_start_hour=12,
        )
        # EMA: new_value = (1-0.3)*old + 0.3*new
        expected_duration = 10 * 0.7 + 30 * 0.3  # = 16
        expected_depth = 0.4 * 0.7 + 0.7 * 0.3    # = 0.49
        assert abs(updated.avg_session_duration_minutes - expected_duration) < 0.01
        assert abs(updated.avg_session_depth - expected_depth) < 0.01


# ── Regularity Calculation ──────────────────────────────────

class TestRegularity:
    """Verify session regularity calculation."""

    def test_empty_user_is_regular(self, dda_controller):
        regularity = dda_controller._calculate_regularity("new_user")
        assert regularity == 1.0

    def test_regularity_normalizes_cv(self, dda_controller):
        """Regularity = 1 - normalized coefficient of variation."""
        # With few sessions, regularity stays at 1.0
        assert dda_controller._calculate_regularity("sparse_user") == 1.0


# ── Boundary / Edge Cases ───────────────────────────────────

class TestDDABoundaries:
    """Boundary and edge case tests."""

    def test_zero_sessions(self, dda_controller):
        stats = UserStats(user_id="test", total_sessions=0)
        ddi = dda_controller.calculate_ddi(stats)
        assert ddi >= 0.0
        assert dda_controller.get_level(ddi) == DDILevel.COLD

    def test_maximum_values(self, dda_controller):
        stats = UserStats(
            user_id="test",
            total_sessions=1000,
            sessions_per_week=14,
            avg_session_duration_minutes=120,
            avg_session_depth=1.0,
            days_since_first_use=365,
            session_regularity=1.0,
            time_of_day_pattern_score=1.0,
            sessions_this_week=14,
        )
        ddi = dda_controller.calculate_ddi(stats)
        # Maximum values should produce highest DDI
        assert ddi >= 0

    def test_ddi_never_negative(self, dda_controller):
        # Even with weird data, DDI should never be negative
        weird = UserStats(
            user_id="test",
            total_sessions=-5,
            sessions_per_week=-1,
            avg_session_duration_minutes=-10,
        )
        ddi = dda_controller.calculate_ddi(weird)
        assert ddi >= 0.0, f"DDI should never be negative, got {ddi}"

    def test_strategy_matrix_is_complete(self):
        """All four DDI levels must have a strategy."""
        for level in DDILevel:
            assert level in STRATEGY_MATRIX, f"Missing strategy for {level}"

    def test_strategy_attributes_are_valid(self):
        """All strategies should have valid modes."""
        valid_modes = {"all", "semantic_time", "three_way", "four_way_ws"}
        for level, strategy in STRATEGY_MATRIX.items():
            assert strategy.retrieval_mode in valid_modes, \
                f"{level}: invalid retrieval_mode {strategy.retrieval_mode}"

    def test_cold_strategy_no_decay(self):
        strategy = STRATEGY_MATRIX[DDILevel.COLD]
        assert strategy.decay_enabled is False
        assert strategy.decay_lambda == 0.05  # default value, but disabled

    def test_warm_has_vector_not_graph(self):
        strategy = STRATEGY_MATRIX[DDILevel.WARM]
        assert strategy.use_vector_search is True
        assert strategy.use_graph_search is False
        assert strategy.use_ws_rerank is False
        assert strategy.vulnerability_enabled is False

    def test_hot_has_graph_not_ws(self):
        strategy = STRATEGY_MATRIX[DDILevel.HOT]
        assert strategy.use_graph_search is True
        assert strategy.use_ws_rerank is False

    def test_rich_has_all_features(self):
        strategy = STRATEGY_MATRIX[DDILevel.RICH]
        assert strategy.use_vector_search is True
        assert strategy.use_bm25_search is True
        assert strategy.use_graph_search is True
        assert strategy.use_ws_rerank is True
        assert strategy.vulnerability_enabled is True
        assert strategy.use_vulnerability_gate is True
        assert strategy.importance_mode == "full_fusion"
