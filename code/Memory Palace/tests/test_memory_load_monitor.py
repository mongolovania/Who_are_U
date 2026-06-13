# ============================================================
# Test: Memory Load Monitor (test_memory_load_monitor.py)
# L0: Adaptive sleep cycle triggering tests.
#
# Covers:
#   - Load computation (all 5 metrics)
#   - Sleep recommendation (DDI-adaptive)
#   - COLD/WARM/HOT/RICH thresholds
#   - Sleep history tracking
#   - Persistence (save/load)
#   - Edge cases (empty state)
# ============================================================

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from memory_load_monitor import (
    MemoryLoadMonitor, MemoryLoad, SleepRecommendation,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def monitor(tmp_path):
    return MemoryLoadMonitor(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def mock_bucket_mgr():
    mgr = MagicMock()
    # Return empty list by default
    mgr.list_all.return_value = []
    return mgr


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.get_graph_stats.return_value = {
        "node_count": 10,
        "edge_count": 25,
        "active_edge_count": 20,
    }
    return graph


# ── Load computation ─────────────────────────────────────────

class TestComputeLoad:
    """Memory load computation."""

    def test_empty_state(self, monitor):
        """Load from empty state."""
        load = monitor.compute_load(dda_level="COLD")
        assert isinstance(load, MemoryLoad)
        assert load.new_memories_since_last_sleep == 0
        assert load.load_score >= 0.0

    def test_load_with_graph(self, monitor, mock_graph):
        """Load computation with graph data."""
        load = monitor.compute_load(graph=mock_graph, dda_level="WARM")
        assert load.edge_density == 2.0  # 20 active edges / 10 nodes

    def test_load_score_range(self, monitor, mock_graph):
        """Load score should be between 0 and 1."""
        load = monitor.compute_load(graph=mock_graph, dda_level="HOT")
        assert 0.0 <= load.load_score <= 1.0

    def test_consolidation_need_computed(self, monitor, mock_graph):
        """Consolidation need should be a distinct metric."""
        load = monitor.compute_load(graph=mock_graph, dda_level="WARM")
        assert 0.0 <= load.consolidation_need <= 1.0

    def test_all_five_metrics_present(self, monitor, mock_graph):
        """Load should contain all 5 core metrics."""
        load = monitor.compute_load(bucket_mgr=MagicMock(), graph=mock_graph)
        assert hasattr(load, 'new_memories_since_last_sleep')
        assert hasattr(load, 'avg_importance_since_last_sleep')
        assert hasattr(load, 'edge_density')
        assert hasattr(load, 'emotional_volatility')
        assert hasattr(load, 'time_since_last_sleep_hours')

    def test_time_since_last_sleep_never_slept(self, monitor):
        """Never slept → high pseudo-value."""
        load = monitor.compute_load()
        assert load.time_since_last_sleep_hours > 100


# ── Sleep recommendation ─────────────────────────────────────

class TestRecommendSleepCycle:
    """Sleep cycle recommendation."""

    def test_cold_user_light_sleep(self, monitor):
        """COLD users get minimal sleep recommendations (only when forced)."""
        load = MemoryLoad(
            new_memories_since_last_sleep=3,
            load_score=0.2,
            consolidation_need=0.1,
            time_since_last_sleep_hours=50,  # > 48h max → forced
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="COLD")
        assert isinstance(rec, SleepRecommendation)
        # COLD: even when forced, stages should be minimal
        if rec.should_sleep:
            assert rec.recommended_intensity == "light"

    def test_cold_user_below_threshold(self, monitor):
        """COLD user below threshold should not sleep."""
        load = MemoryLoad(
            load_score=0.1,
            consolidation_need=0.1,
            time_since_last_sleep_hours=1.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="COLD")
        assert rec.should_sleep is False

    def test_rich_user_sleeps_proactively(self, monitor):
        """RICH user should sleep at lower thresholds."""
        load = MemoryLoad(
            load_score=0.3,
            consolidation_need=0.3,
            edge_density=4.0,  # High density
            time_since_last_sleep_hours=4.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="RICH")
        assert rec.should_sleep is True

    def test_high_emotional_volatility_triggers_sleep(self, monitor):
        """High emotional volatility → REM-like processing needed."""
        load = MemoryLoad(
            emotional_volatility=0.25,  # Above threshold
            time_since_last_sleep_hours=4.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="HOT")
        assert rec.should_sleep is True

    def test_too_soon_after_last_sleep(self, monitor):
        """Should not sleep within min_hours_between_sleep."""
        monitor._last_sleep_at = datetime.now(timezone.utc).isoformat()
        load = MemoryLoad(
            load_score=0.9,
            time_since_last_sleep_hours=0.1,  # Just slept
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="RICH")
        assert rec.should_sleep is False
        assert "too_soon" in rec.reason

    def test_recommendation_includes_stages(self, monitor):
        """Sleep recommendation should specify which stages to run."""
        load = MemoryLoad(
            load_score=0.7,
            consolidation_need=0.65,
            emotional_volatility=0.2,
            time_since_last_sleep_hours=12.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="HOT")
        if rec.should_sleep:
            assert len(rec.recommended_stages) > 0

    def test_critical_load_deep_sleep(self, monitor):
        """Critical load → deep sleep with all 5 stages."""
        load = MemoryLoad(
            load_score=0.9,
            consolidation_need=0.85,
            edge_density=8.0,
            emotional_volatility=0.3,
            time_since_last_sleep_hours=24.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="RICH")
        if rec.should_sleep:
            assert rec.recommended_intensity == "deep"
            assert len(rec.recommended_stages) == 5


# ── Sleep tracking ───────────────────────────────────────────

class TestRecordSleep:
    """Sleep history tracking."""

    def test_record_updates_last_sleep(self, monitor):
        """Recording sleep should update last_sleep_at."""
        old_time = monitor._last_sleep_at
        monitor.record_sleep_complete({
            "cycle_id": "c1",
            "duration_seconds": 12.5,
            "stages_completed": ["REPLAY", "PRUNE", "CONSOLIDATE"],
            "memories_processed": 50,
        })
        assert monitor._last_sleep_at != old_time
        assert len(monitor._sleep_history) == 1

    def test_history_capped_at_30(self, monitor):
        """Sleep history should be capped at 30 entries."""
        for i in range(35):
            monitor.record_sleep_complete({
                "cycle_id": f"c{i}",
                "duration_seconds": 10.0,
                "stages_completed": [],
                "memories_processed": i,
            })
        assert len(monitor._sleep_history) <= 30


# ── Persistence ──────────────────────────────────────────────

class TestPersistence:
    """Save/load state."""

    def test_save_and_load(self, monitor):
        """State should survive save/load cycle."""
        monitor._last_sleep_at = "2026-06-01T00:00:00"
        monitor._sleep_history = [{"completed_at": "2026-06-01T00:00:00"}]
        monitor.save()

        # Create new instance and load
        new_monitor = MemoryLoadMonitor(
            user_id="test_user",
            data_dir=str(monitor.data_dir.parent),  # Same parent dir
        )
        # Override data_dir to match
        new_monitor.data_dir = monitor.data_dir
        new_monitor.load()

        assert new_monitor._last_sleep_at == "2026-06-01T00:00:00"
        assert len(new_monitor._sleep_history) == 1

    def test_load_nonexistent_file(self, monitor):
        """Loading when no file exists should not fail."""
        monitor.data_dir = Path("/nonexistent/path")
        monitor.load()
        assert monitor._loaded is True


# ── Stats ────────────────────────────────────────────────────

class TestStats:
    """Monitor statistics."""

    def test_empty_stats(self, monitor):
        stats = monitor.get_stats()
        assert stats["total_sleep_cycles"] == 0
        assert stats["recommendations_generated"] == 0

    def test_stats_after_sleep(self, monitor):
        monitor.record_sleep_complete({
            "cycle_id": "c1",
            "duration_seconds": 15.0,
            "stages_completed": [],
            "memories_processed": 20,
        })
        stats = monitor.get_stats()
        assert stats["total_sleep_cycles"] == 1
        assert stats["avg_sleep_duration_seconds"] == 15.0


# ── Data model ───────────────────────────────────────────────

class TestDataModels:
    """MemoryLoad and SleepRecommendation."""

    def test_load_has_computed_at(self):
        load = MemoryLoad()
        assert load.computed_at != ""

    def test_load_to_dict(self):
        load = MemoryLoad(
            new_memories_since_last_sleep=5,
            edge_density=2.5,
            load_score=0.5,
        )
        d = load.to_dict()
        assert d["new_memories_since_last_sleep"] == 5
        assert d["edge_density"] == 2.5

    def test_recommendation_has_generated_at(self):
        rec = SleepRecommendation(should_sleep=False)
        assert rec.generated_at != ""

    def test_recommendation_to_dict(self):
        load = MemoryLoad(load_score=0.6)
        rec = SleepRecommendation(
            should_sleep=True,
            urgency=0.7,
            recommended_stages=["REPLAY", "PRUNE", "CONSOLIDATE"],
            recommended_intensity="normal",
            reason="high_load",
            load=load,
        )
        d = rec.to_dict()
        assert d["should_sleep"] is True
        assert d["load"] is not None


# ── DDA-adaptive thresholds ──────────────────────────────────

class TestDDAThresholds:
    """Different DDA levels have different thresholds."""

    def test_cold_highest_threshold(self, monitor):
        """COLD users should have the highest threshold (sleep least)."""
        # COLD: only forced sleep
        load = MemoryLoad(
            load_score=0.4,  # Below COLD threshold of 0.70
            consolidation_need=0.3,
            time_since_last_sleep_hours=2.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="COLD")
        assert rec.should_sleep is False

    def test_rich_lowest_threshold(self, monitor):
        """RICH users have lowest threshold (sleep most proactively)."""
        load = MemoryLoad(
            load_score=0.25,  # Above RICH threshold of 0.20
            consolidation_need=0.3,  # Above RICH threshold of 0.25
            time_since_last_sleep_hours=4.0,
        )
        rec = monitor.recommend_sleep_cycle(load, dda_level="RICH")
        assert rec.should_sleep is True
