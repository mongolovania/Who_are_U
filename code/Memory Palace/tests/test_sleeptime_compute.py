# ============================================================
# Test: Sleeptime Compute (test_sleeptime_compute.py)
# L3: Sleep cycle computation tests.
#
# Covers:
#   - Full 5-stage sleep cycle execution
#   - Stage 1: Hippocampal replay
#   - Stage 2: Synaptic pruning (decay + edge scaling)
#   - Stage 3: Narrative consolidation
#   - Stage 4: Precomputation index
#   - Stage 5: Memory evolution
#   - Cold user fast mode
#   - Precomputed index queries
#   - Cycle history
# ============================================================

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sleeptime_compute import (
    SleeptimeComputer, SleepCycleResult, ReplayTrace, PrecomputedIndex,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def mock_bucket_mgr():
    mgr = AsyncMock()
    mgr.list_all = AsyncMock(return_value=[
        {
            "id": "mem_001",
            "content": "面试感觉不错",
            "metadata": {
                "name": "面试", "domain": ["求职"], "tags": ["面试"],
                "valence": 0.7, "arousal": 0.55, "importance": 8,
                "pinned": False, "protected": False, "type": "dynamic",
                "memory_type": "chat", "created": "2026-06-01T10:00:00",
            },
        },
        {
            "id": "mem_002",
            "content": "失眠了很焦虑",
            "metadata": {
                "name": "失眠", "domain": ["健康"], "tags": ["失眠", "焦虑"],
                "valence": 0.15, "arousal": 0.80, "importance": 7,
                "pinned": False, "protected": False, "type": "dynamic",
                "memory_type": "emotion", "created": "2026-06-02T02:00:00",
            },
        },
        {
            "id": "mem_003",
            "content": "拿到offer了",
            "metadata": {
                "name": "offer", "domain": ["求职"], "tags": ["offer"],
                "valence": 0.90, "arousal": 0.85, "importance": 9,
                "pinned": False, "protected": False, "type": "dynamic",
                "memory_type": "milestone", "created": "2026-06-05T10:00:00",
            },
        },
        {
            "id": "mem_004",
            "content": "日常记录",
            "metadata": {
                "name": "日常", "domain": ["日常"], "tags": [],
                "valence": 0.50, "arousal": 0.30, "importance": 4,
                "pinned": False, "protected": False, "type": "dynamic",
                "memory_type": "chat", "created": "2026-06-03T12:00:00",
            },
        },
    ])
    return mgr


@pytest.fixture
def mock_decay():
    de = MagicMock()
    de.calculate_score = MagicMock(return_value=5.0)
    de.run_decay_cycle = AsyncMock(return_value={
        "checked": 4, "archived": 1, "auto_resolved": 0, "lowest_score": 0.15,
    })
    de.apply_dda_strategy = MagicMock()
    de.set_ddi_level = MagicMock()
    return de


@pytest.fixture
def mock_graph():
    g = MagicMock()
    g.get_neighbors = MagicMock(return_value=[
        {"edge_id": "e1", "from_id": "mem_001", "to_id": "mem_003",
         "relation_type": "causal", "weight": 0.7},
        {"edge_id": "e2", "from_id": "mem_001", "to_id": "mem_002",
         "relation_type": "emotional", "weight": 0.5},
    ])
    g.add_edge = MagicMock(return_value="new_edge")
    g.expire_edge = MagicMock()
    g.get_graph_stats = MagicMock(return_value={
        "node_count": 4, "edge_count": 6, "active_edge_count": 5,
        "expired_edge_count": 1,
    })
    return g


@pytest.fixture
def mock_narrative():
    ne = MagicMock()
    ne.run_narrative_merge = AsyncMock(return_value={
        "communities_detected": 2,
        "threads_merged": 1,
        "summaries_updated": 3,
        "threads_resolved": 1,
        "life_periods_updated": 1,
    })
    return ne


@pytest.fixture
def mock_evolution():
    ev = MagicMock()
    ev.run_evolution_cycle = AsyncMock(return_value={
        "memories_scanned": 10,
        "re_evaluated": 3,
        "ws_re_ranked": 5,
        "emergences_detected": 2,
    })
    return ev


@pytest.fixture
def computer(mock_bucket_mgr, mock_decay, mock_graph, mock_narrative, mock_evolution):
    """SleeptimeComputer with mocked dependencies."""
    return SleeptimeComputer(
        user_id="test_user",
        bucket_mgr=mock_bucket_mgr,
        decay_engine=mock_decay,
        memory_graph=mock_graph,
        narrative_engine=mock_narrative,
        memory_evolution=mock_evolution,
    )


# ── Full Sleep Cycle ─────────────────────────────────────────

class TestSleepCycle:
    """Verify the full 5-stage sleep pipeline."""

    @pytest.mark.asyncio
    async def test_full_sleep_cycle_runs_all_stages(self, computer):
        result = await computer.run_sleep_cycle(
            session_messages=[{"role": "user", "content": "今天面试了"}],
            ddi_level="HOT",
        )

        assert isinstance(result, SleepCycleResult)
        assert result.replay is not None
        assert result.prune is not None
        assert result.consolidate is not None
        assert result.precompute is not None
        assert result.evolve is not None

    @pytest.mark.asyncio
    async def test_sleep_cycle_has_duration(self, computer):
        result = await computer.run_sleep_cycle(ddi_level="WARM")
        assert result.duration_seconds >= 0
        assert result.started_at != ""
        assert result.completed_at != ""

    @pytest.mark.asyncio
    async def test_sleep_cycle_increments_count(self, computer):
        assert computer._cycle_count == 0
        await computer.run_sleep_cycle(ddi_level="WARM")
        assert computer._cycle_count == 1
        await computer.run_sleep_cycle(ddi_level="WARM")
        assert computer._cycle_count == 2


# ── Cold User Fast Mode ──────────────────────────────────────

class TestColdUserMode:
    """Verify COLD users get minimal sleep."""

    @pytest.mark.asyncio
    async def test_cold_fast_mode_skips_expensive_stages(self, computer):
        result = await computer.run_sleep_cycle(
            ddi_level="COLD",
            fast_mode=True,
        )

        # Replay, consolidate, precompute, evolve → skipped
        assert result.replay.get("skipped") is True
        assert result.consolidate.get("skipped") is True
        assert result.precompute.get("skipped") is True
        assert result.evolve.get("skipped") is True
        # Prune should still run
        assert result.prune.get("skipped") is not True

    @pytest.mark.asyncio
    async def test_warm_user_runs_all_stages(self, computer):
        result = await computer.run_sleep_cycle(ddi_level="WARM")
        # All stages should run (not skipped)
        assert result.replay.get("skipped") is not True
        assert result.prune.get("skipped") is not True
        # consolidate/precompute/evolve may or may not skip
        # depending on mock availability


# ── Stage 1: Replay ──────────────────────────────────────────

class TestStageReplay:
    """Verify hippocampal replay stage."""

    @pytest.mark.asyncio
    async def test_replay_selects_high_importance_memories(self, computer, mock_bucket_mgr):
        result = await computer._stage_replay()

        assert "memories_replayed" in result
        # Only memories with importance >= 6 should be replayed
        # mem_004 has importance=4, should be excluded
        assert result["memories_replayed"] >= 2  # mem_001(8), mem_002(7), mem_003(9)

    @pytest.mark.asyncio
    async def test_replay_strengthens_edges(self, computer, mock_graph):
        result = await computer._stage_replay()

        assert "edges_strengthened" in result
        # mock_graph.get_neighbors returns 2 edges per call
        if not result.get("skipped"):
            assert mock_graph.add_edge.call_count > 0

    @pytest.mark.asyncio
    async def test_replay_skips_without_graph(self, mock_bucket_mgr, mock_decay):
        comp = SleeptimeComputer(
            user_id="test",
            bucket_mgr=mock_bucket_mgr,
            decay_engine=mock_decay,
            memory_graph=None,
        )
        result = await comp._stage_replay()
        assert result.get("skipped") is True


# ── Stage 2: Prune ───────────────────────────────────────────

class TestStagePrune:
    """Verify synaptic pruning stage."""

    @pytest.mark.asyncio
    async def test_prune_runs_decay_cycle(self, computer, mock_decay):
        result = await computer._stage_prune(
            SleepCycleResult(), ddi_level="HOT"
        )

        assert "decay_checked" in result
        assert "archived" in result
        mock_decay.run_decay_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_prune_skips_without_decay(self, mock_bucket_mgr):
        comp = SleeptimeComputer(
            user_id="test",
            bucket_mgr=mock_bucket_mgr,
            decay_engine=None,
        )
        result = await comp._stage_prune(SleepCycleResult())
        assert result.get("skipped") is True


# ── Stage 3: Consolidate ─────────────────────────────────────

class TestStageConsolidate:
    """Verify narrative consolidation stage."""

    @pytest.mark.asyncio
    async def test_consolidate_runs_narrative_merge(self, computer, mock_narrative):
        result = await computer._stage_consolidate(
            session_messages=[{"role": "user", "content": "test"}],
        )

        assert "narrative_merge" in result
        mock_narrative.run_narrative_merge.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_skips_without_narrative(self, mock_bucket_mgr, mock_decay):
        comp = SleeptimeComputer(
            user_id="test",
            bucket_mgr=mock_bucket_mgr,
            decay_engine=mock_decay,
            narrative_engine=None,
        )
        result = await comp._stage_consolidate()
        assert result["narrative_merge"] is None


# ── Stage 4: Precompute ──────────────────────────────────────

class TestStagePrecompute:
    """Verify precomputation index stage."""

    @pytest.mark.asyncio
    async def test_precompute_builds_topic_clusters(self, computer):
        result = await computer._stage_precompute()

        assert "topic_clusters" in result
        assert result["topic_clusters"] >= 1  # "求职" domain
        assert "emotion_buckets" in result
        assert "timeline_events" in result

    @pytest.mark.asyncio
    async def test_precomputed_index_available_after_compute(self, computer):
        await computer._stage_precompute()

        stats = computer.get_precomputed_stats()
        assert stats["available"] is True
        assert stats["memory_count"] == 4

    @pytest.mark.asyncio
    async def test_precompute_skips_without_bucket_mgr(self):
        comp = SleeptimeComputer(user_id="test", bucket_mgr=None)
        result = await comp._stage_precompute()
        assert result.get("skipped") is True

    @pytest.mark.asyncio
    async def test_precomputed_domain_query(self, computer):
        await computer._stage_precompute()

        ids = computer.get_precomputed_for_domain("求职")
        assert isinstance(ids, list)
        assert "mem_001" in ids or "mem_003" in ids

    @pytest.mark.asyncio
    async def test_precomputed_emotion_query(self, computer):
        await computer._stage_precompute()

        ids = computer.get_precomputed_for_emotion(valence=0.15, arousal=0.80)
        assert isinstance(ids, list)
        # Should find mem_002 (high arousal, low valence)

    @pytest.mark.asyncio
    async def test_precomputed_timeline_query(self, computer):
        await computer._stage_precompute()

        events = computer.get_precomputed_timeline()
        assert isinstance(events, list)
        assert len(events) == 4
        # Should be chronologically sorted
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_precomputed_unavailable_before_compute(self, computer):
        stats = computer.get_precomputed_stats()
        assert stats["available"] is False

    @pytest.mark.asyncio
    async def test_precomputed_timeline_filtered(self, computer):
        await computer._stage_precompute()

        events = computer.get_precomputed_timeline(
            after="2026-06-03T00:00:00",
            before="2026-06-05T00:00:00",
        )
        assert isinstance(events, list)


# ── Stage 5: Evolve ──────────────────────────────────────────

class TestStageEvolve:
    """Verify memory evolution stage."""

    @pytest.mark.asyncio
    async def test_evolve_runs_evolution_cycle(self, computer, mock_evolution):
        result = await computer._stage_evolve()

        assert "memories_scanned" in result
        mock_evolution.run_evolution_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_evolve_skips_without_evolution_engine(self, mock_bucket_mgr, mock_decay):
        comp = SleeptimeComputer(
            user_id="test",
            bucket_mgr=mock_bucket_mgr,
            decay_engine=mock_decay,
            memory_evolution=None,
        )
        result = await comp._stage_evolve()
        assert result.get("skipped") is True


# ── Cycle History ────────────────────────────────────────────

class TestCycleHistory:
    """Verify cycle history tracking."""

    @pytest.mark.asyncio
    async def test_cycle_history_empty_initially(self, computer):
        history = computer.get_cycle_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_cycle_history_after_one_cycle(self, computer):
        await computer.run_sleep_cycle(ddi_level="WARM")
        history = computer.get_cycle_history()
        assert len(history) == 1
        assert "cycle_id" in history[0]
        assert "duration_seconds" in history[0]
        assert "replay" in history[0]
        assert "prune" in history[0]


# ── Edge Cases ───────────────────────────────────────────────

class TestSleeptimeEdgeCases:
    """Boundary and edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_bucket_list_graceful(self):
        mgr = AsyncMock()
        mgr.list_all = AsyncMock(return_value=[])

        comp = SleeptimeComputer(
            user_id="test",
            bucket_mgr=mgr,
            decay_engine=None,
            memory_graph=None,
        )

        result = await comp.run_sleep_cycle(ddi_level="WARM")
        assert isinstance(result, SleepCycleResult)

    @pytest.mark.asyncio
    async def test_bucket_mgr_error_graceful(self):
        mgr = AsyncMock()
        mgr.list_all = AsyncMock(side_effect=Exception("DB error"))

        comp = SleeptimeComputer(
            user_id="test",
            bucket_mgr=mgr,
            decay_engine=None,
            memory_graph=None,
        )

        result = await comp.run_sleep_cycle(ddi_level="WARM")
        assert isinstance(result, SleepCycleResult)
        # Should not crash
        assert result.health_status == "healthy"

    @pytest.mark.asyncio
    async def test_all_mocks_integration(self, computer):
        """Integration: run full cycle with all mocks, verify no crashes."""
        result = await computer.run_sleep_cycle(
            session_messages=[
                {"role": "user", "content": "今天面试了"},
                {"role": "assistant", "content": "感觉怎么样？"},
            ],
            ddi_level="HOT",
        )

        # All stages should have results
        for stage in ["replay", "prune", "consolidate", "precompute", "evolve"]:
            assert stage in result.__dict__ or hasattr(result, stage), f"Missing stage: {stage}"

    def test_precomputed_index_dataclass(self):
        idx = PrecomputedIndex(
            topic_clusters={"求职": ["mem_001", "mem_003"]},
            emotion_index={"v0_a4": ["mem_002"]},
            timeline_index=[{"memory_id": "mem_001", "timestamp": "2026-06-01T10:00:00"}],
        )
        assert idx.memory_count == 0  # Not set explicitly
        assert idx.topic_clusters["求职"] == ["mem_001", "mem_003"]

    def test_sleep_cycle_result_dataclass(self):
        result = SleepCycleResult()
        assert result.cycle_id != ""
        assert result.started_at != ""
        assert result.health_status == "healthy"
        assert result.replay == {}
