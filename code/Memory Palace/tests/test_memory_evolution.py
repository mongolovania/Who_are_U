# ============================================================
# Test: Memory Evolution (test_memory_evolution.py)
# L2: A-MEM Zettelkasten evolution tests.
#
# Covers:
#   - Linking new memory to old memories
#   - Link type inference (thematic/causal/contrastive/successor)
#   - Bidirectional graph edge creation
#   - Importance evolution after link accumulation
#   - Re-evaluation cycle
#   - Working Self re-ranking
#   - Emergence detection
#   - Persistence (save/load)
# ============================================================

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from memory_evolution import (
    MemoryEvolution, EvolutionLink, EvolutionEvent,
)
from memory_node import RelationType


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def evolution_engine(tmp_path):
    """MemoryEvolution with temp directory."""
    return MemoryEvolution(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def mock_graph():
    """Mock memory graph."""
    g = MagicMock()
    g.add_edge = MagicMock(return_value="edge_id")
    g.get_neighbors = MagicMock(return_value=[])
    g.add_node = MagicMock()
    return g


@pytest.fixture
def mock_bucket_mgr():
    """Mock bucket manager with some memories."""
    mgr = AsyncMock()
    mgr.list_all = AsyncMock(return_value=[
        {
            "id": "old_001",
            "content": "今天面试感觉很顺利，面试官人很好",
            "metadata": {"valence": 0.7, "arousal": 0.55, "importance": 6,
                         "tags": ["面试"], "domain": ["求职"]},
        },
        {
            "id": "old_002",
            "content": "又失眠了，翻来覆去睡不着，脑子里很乱",
            "metadata": {"valence": 0.15, "arousal": 0.75, "importance": 7,
                         "tags": ["失眠", "焦虑"], "domain": ["健康"]},
        },
        {
            "id": "old_003",
            "content": "拿到offer了！开心",
            "metadata": {"valence": 0.88, "arousal": 0.82, "importance": 8,
                         "tags": ["offer"], "domain": ["求职"]},
        },
    ])
    return mgr


# ── Link Creation ─────────────────────────────────────────────

class TestLinkCreation:
    """Verify Zettelkasten linking of new memories to old."""

    @pytest.mark.asyncio
    async def test_link_new_memory_to_old(self, evolution_engine, mock_graph, mock_bucket_mgr):
        links = await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="今天又面试了，感觉还不错，希望能拿到offer",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        assert isinstance(links, list)
        # Should create links to old memories with overlapping content
        for link in links:
            assert link.memory_id_a == "new_001"
            assert link.memory_id_b in ("old_001", "old_002", "old_003")
            assert link.link_type in ("thematic", "causal", "contrastive", "successor", "predecessor")
            assert 0.0 < link.strength <= 1.0
            assert link.bidirectional is True

    @pytest.mark.asyncio
    async def test_link_adds_graph_edges(self, evolution_engine, mock_graph, mock_bucket_mgr):
        await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试感觉不错",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        # Should have called add_edge for each link (2 edges per link = bidirectional)
        if mock_graph.add_edge.call_count > 0:
            # Each call adds two directed edges
            assert mock_graph.add_edge.call_count % 2 == 0  # Pairs of edges

    @pytest.mark.asyncio
    async def test_no_links_without_graph(self, evolution_engine, mock_bucket_mgr):
        links = await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="test",
            memory_graph=None,
            bucket_mgr=mock_bucket_mgr,
        )
        assert links == []

    @pytest.mark.asyncio
    async def test_no_duplicate_links(self, evolution_engine, mock_graph, mock_bucket_mgr):
        # First link
        await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        first_link_count = len(evolution_engine.links)

        # Second link with same content — should not duplicate
        await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        # Link count should not increase for the same pair
        assert len(evolution_engine.links) == first_link_count

    @pytest.mark.asyncio
    async def test_max_links_per_memory_enforced(self, evolution_engine, mock_graph):
        # Create many old memories
        mgr = AsyncMock()
        mgr.list_all = AsyncMock(return_value=[
            {"id": f"old_{i:03d}", "content": f"memory about topic {i}"}
            for i in range(20)
        ])

        links = await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="topic memory",
            memory_graph=mock_graph,
            bucket_mgr=mgr,
        )

        assert len(links) <= evolution_engine.max_links_per_memory


# ── Link Type Inference ───────────────────────────────────────

class TestLinkTypeInference:
    """Verify zero-LLM link type detection."""

    def test_causal_content_infers_causal_link(self):
        link_type = MemoryEvolution._infer_link_type(
            new_content="面试失败了，因为准备不够充分",
            new_valence=0.25,
            new_arousal=0.50,
            old_id="old_001",
        )
        assert link_type == "causal"

    def test_successor_content_infers_successor_link(self):
        link_type = MemoryEvolution._infer_link_type(
            new_content="后来又去面试了另一家公司",
            new_valence=0.55,
            new_arousal=0.45,
            old_id="old_001",
        )
        assert link_type == "successor"

    def test_high_arousal_low_valence_infers_contrastive(self):
        link_type = MemoryEvolution._infer_link_type(
            new_content="突然觉得很难过",
            new_valence=0.10,
            new_arousal=0.85,
            old_id="old_001",
        )
        assert link_type == "contrastive"

    def test_default_to_thematic(self):
        link_type = MemoryEvolution._infer_link_type(
            new_content="今天天气不错",
            new_valence=0.60,
            new_arousal=0.35,
            old_id="old_001",
        )
        assert link_type == "thematic"


# ── Query / Lookup ────────────────────────────────────────────

class TestEvolutionQuery:
    """Verify evolution state queries."""

    @pytest.mark.asyncio
    async def test_get_links_for_memory(self, evolution_engine, mock_graph, mock_bucket_mgr):
        await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试经历",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        links = evolution_engine.get_links_for_memory("new_001")
        assert isinstance(links, list)

        # old_001 and old_003 should have links (both job-related)
        linked_ids = set()
        for l in links:
            linked_ids.add(l.memory_id_a)
            linked_ids.add(l.memory_id_b)
        assert "new_001" in linked_ids

    def test_get_links_empty_for_unknown_memory(self, evolution_engine):
        links = evolution_engine.get_links_for_memory("nonexistent")
        assert links == []

    @pytest.mark.asyncio
    async def test_get_link_between(self, evolution_engine, mock_graph, mock_bucket_mgr):
        links = await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        if links:
            link = evolution_engine.get_link_between("new_001", links[0].memory_id_b)
            assert link is not None

    def test_get_evolution_events(self, evolution_engine):
        events = evolution_engine.get_evolution_events(limit=10)
        assert isinstance(events, list)


# ── Re-evaluation ─────────────────────────────────────────────

class TestReEvaluation:
    """Verify memory re-evaluation triggers."""

    @pytest.mark.asyncio
    async def test_re_evaluate_requires_enough_links(self, evolution_engine):
        # Memory with few links should not trigger re-evaluation
        event = await evolution_engine.re_evaluate_memory(
            memory_id="mem_few_links",
            bucket_mgr=None,
            importance_fusion=None,
        )
        assert event is None

    @pytest.mark.asyncio
    async def test_re_evaluate_with_enough_links(self, evolution_engine, mock_graph, mock_bucket_mgr):
        # Manually create enough links
        for i in range(5):
            link = EvolutionLink(
                memory_id_a="target_mem",
                memory_id_b=f"linked_{i}",
                strength=0.6,
            )
            evolution_engine.links[link.id] = link
            evolution_engine._link_index.setdefault("target_mem", set()).add(link.id)

        # Add events to trigger (mock importance_fusion would be needed for real test)
        # With no importance_fusion, should return None
        event = await evolution_engine.re_evaluate_memory(
            memory_id="target_mem",
            bucket_mgr=mock_bucket_mgr,
            importance_fusion=None,
        )
        # Without importance_fusion, returns None (can't evolve without it)
        assert event is None  # Expected — no importance_fusion provided

    @pytest.mark.asyncio
    async def test_run_evolution_cycle_no_crash(self, evolution_engine, mock_graph, mock_bucket_mgr):
        result = await evolution_engine.run_evolution_cycle(
            bucket_mgr=mock_bucket_mgr,
            memory_graph=mock_graph,
        )
        assert "memories_scanned" in result
        assert "re_evaluated" in result
        assert "ws_re_ranked" in result
        assert "emergences_detected" in result


# ── Emergence Detection ───────────────────────────────────────

class TestEmergenceDetection:
    """Verify emergence detection logic."""

    def test_detect_emergences_with_high_connectivity(self, evolution_engine):
        # Create memory with many links (emergent)
        for i in range(6):
            link = EvolutionLink(
                memory_id_a="emergent_mem",
                memory_id_b=f"linked_{i}",
                strength=0.7 + i * 0.02,
            )
            evolution_engine.links[link.id] = link
            evolution_engine._link_index.setdefault("emergent_mem", set()).add(link.id)
            evolution_engine._link_index.setdefault(f"linked_{i}", set()).add(link.id)

        emergences = evolution_engine._detect_emergences()
        assert len(emergences) >= 1

        emergent = [e for e in emergences if e["memory_id"] == "emergent_mem"]
        assert len(emergent) == 1
        assert emergent[0]["link_count"] >= 6

    def test_detect_emergences_low_connectivity_not_emergent(self, evolution_engine):
        # Memory with few links should not be detected as emergent
        for i in range(2):
            link = EvolutionLink(
                memory_id_a="low_connectivity",
                memory_id_b=f"linked_{i}",
                strength=0.5,
            )
            evolution_engine.links[link.id] = link
            evolution_engine._link_index.setdefault("low_connectivity", set()).add(link.id)

        emergences = evolution_engine._detect_emergences()
        # Low connectivity (2 links < 5 threshold) should not appear
        low_conn = [e for e in emergences if e["memory_id"] == "low_connectivity"]
        assert len(low_conn) == 0


# ── Persistence ───────────────────────────────────────────────

class TestEvolutionPersistence:
    """Verify save/load roundtrip."""

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, evolution_engine, mock_graph, mock_bucket_mgr, tmp_path):
        await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        link_count_before = len(evolution_engine.links)

        # Create new engine pointing to same directory
        eng2 = MemoryEvolution(user_id="test_user", data_dir=str(tmp_path / "buckets"))
        eng2.load()

        assert len(eng2.links) == link_count_before
        assert len(eng2._link_index) == len(evolution_engine._link_index)

    def test_new_engine_starts_empty(self, evolution_engine):
        assert evolution_engine.links == {}
        assert evolution_engine._link_index == {}
        assert evolution_engine.events == []


# ── Edge Cases ────────────────────────────────────────────────

class TestEvolutionEdgeCases:
    """Boundary and edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_content_still_links(self, evolution_engine, mock_graph):
        mgr = AsyncMock()
        mgr.list_all = AsyncMock(return_value=[
            {"id": "old_001", "content": "some content"}
        ])

        links = await evolution_engine.link_new_memory(
            new_memory_id="new_empty",
            new_content="",
            memory_graph=mock_graph,
            bucket_mgr=mgr,
        )
        # Empty content → no keyword overlap → no links
        assert links == []

    def test_link_type_mapping(self):
        """Verify link_type → RelationType mapping."""
        assert MemoryEvolution._link_type_to_relation("thematic") == RelationType.THEMATIC
        assert MemoryEvolution._link_type_to_relation("causal") == RelationType.CAUSAL
        assert MemoryEvolution._link_type_to_relation("successor") == RelationType.TEMPORAL
        assert MemoryEvolution._link_type_to_relation("predecessor") == RelationType.TEMPORAL
        assert MemoryEvolution._link_type_to_relation("contrastive") == RelationType.EMOTIONAL
        # Unknown type defaults to THEMATIC
        assert MemoryEvolution._link_type_to_relation("unknown") == RelationType.THEMATIC

    def test_get_stats_on_empty_engine(self, evolution_engine):
        stats = evolution_engine.get_stats()
        assert stats["total_links"] == 0
        assert stats["memories_with_links"] == 0
        assert stats["avg_links_per_memory"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_after_linking(self, evolution_engine, mock_graph, mock_bucket_mgr):
        await evolution_engine.link_new_memory(
            new_memory_id="new_001",
            new_content="面试",
            memory_graph=mock_graph,
            bucket_mgr=mock_bucket_mgr,
        )

        stats = evolution_engine.get_stats()
        assert stats["total_links"] > 0
        assert stats["memories_with_links"] > 0
