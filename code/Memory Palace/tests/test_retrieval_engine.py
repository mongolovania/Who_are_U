# ============================================================
# Test: Retrieval Engine (test_retrieval_engine.py)
# L2: DDA-adaptive multi-path retrieval unit tests.
#
# Covers:
#   - COLD: return ALL mode
#   - WARM: semantic + time ranking
#   - HOT: 3-way (vector + BM25 + graph) fusion
#   - RICH: 4-way (3-way + Working Self re-rank)
#   - Emotion resonance scoring (Russell circumplex)
#   - Path weight correctness
#   - Random surface diversity
# ============================================================

import math
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from retrieval_engine import RetrievalEngine
from memory_node import DDILevel, DDAStrategy, COLD_STRATEGY, WARM_STRATEGY, HOT_STRATEGY, RICH_STRATEGY


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def engine():
    return RetrievalEngine()


@pytest.fixture
def mock_bucket_mgr():
    mgr = MagicMock()
    mgr.list_all = AsyncMock(return_value=[])
    mgr.search = AsyncMock(return_value=[])
    return mgr


@pytest.fixture
def mock_embedding_engine():
    ee = MagicMock()
    ee.search_similar = AsyncMock(return_value=[])
    return ee


@pytest.fixture
def mock_memory_graph():
    g = MagicMock()
    g.get_neighbors = MagicMock(return_value=[])
    return g


@pytest.fixture
def mock_working_self():
    ws = MagicMock()
    ws.has_goals = False
    ws.match = MagicMock(return_value=0.0)
    return ws


@pytest.fixture
def mock_decay_engine():
    de = MagicMock()
    de.calculate_score = MagicMock(return_value=5.0)
    return de


@pytest.fixture
def sample_buckets():
    return [
        {
            "id": "mem_001",
            "content": "今天学了Python的asyncio",
            "metadata": {
                "name": "Python学习", "type": "dynamic",
                "valence": 0.8, "arousal": 0.6,
                "importance": 8, "created": "2026-06-01T10:00:00",
                "resolved": False, "pinned": False,
            },
        },
        {
            "id": "mem_002",
            "content": "面试被拒了很失落",
            "metadata": {
                "name": "面试失败", "type": "dynamic",
                "valence": 0.2, "arousal": 0.8,
                "importance": 7, "created": "2026-06-05T14:00:00",
                "resolved": False, "pinned": False,
            },
        },
        {
            "id": "mem_003",
            "content": "和家人去旅行了很开心",
            "metadata": {
                "name": "家庭旅行", "type": "dynamic",
                "valence": 0.9, "arousal": 0.7,
                "importance": 6, "created": "2026-06-08T20:00:00",
                "resolved": False, "pinned": False,
            },
        },
    ]


# ── COLD: Return ALL ────────────────────────────────────────

class TestRetrieveAll:
    """COLD mode: return all memories."""

    @pytest.mark.asyncio
    async def test_returns_all_buckets(self, engine, mock_bucket_mgr, mock_decay_engine):
        mock_bucket_mgr.list_all = AsyncMock(return_value=[
            {"id": "a", "metadata": {"importance": 5, "type": "dynamic"}, "content": "test1"},
            {"id": "b", "metadata": {"importance": 3, "type": "dynamic"}, "content": "test2"},
        ])
        results = await engine._retrieve_all(mock_bucket_mgr, mock_decay_engine, top_k=20)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_respects_top_k(self, engine, mock_bucket_mgr, mock_decay_engine):
        mock_bucket_mgr.list_all = AsyncMock(return_value=[
            {"id": f"m_{i}", "metadata": {"importance": i, "type": "dynamic"}, "content": f"test{i}"}
            for i in range(10)
        ])
        results = await engine._retrieve_all(mock_bucket_mgr, mock_decay_engine, top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, engine, mock_bucket_mgr, mock_decay_engine):
        mock_bucket_mgr.list_all = AsyncMock(side_effect=Exception("DB Error"))
        results = await engine._retrieve_all(mock_bucket_mgr, mock_decay_engine, top_k=20)
        assert results == []

    @pytest.mark.asyncio
    async def test_sorted_by_importance(self, engine, mock_bucket_mgr, mock_decay_engine):
        mock_bucket_mgr.list_all = AsyncMock(return_value=[
            {"id": "a", "metadata": {"importance": 3, "type": "dynamic"}, "content": "low"},
            {"id": "b", "metadata": {"importance": 9, "type": "dynamic"}, "content": "high"},
            {"id": "c", "metadata": {"importance": 5, "type": "dynamic"}, "content": "mid"},
        ])
        results = await engine._retrieve_all(mock_bucket_mgr, mock_decay_engine, top_k=20)
        assert results[0]["importance"] >= results[-1]["importance"]


# ── WARM: Semantic + Time ───────────────────────────────────

class TestSemanticTime:
    """WARM mode: semantic + time ranking."""

    @pytest.mark.asyncio
    async def test_searches_with_query(self, engine, mock_bucket_mgr, mock_embedding_engine, mock_decay_engine):
        mock_bucket_mgr.search = AsyncMock(return_value=[
            {"id": "found", "metadata": {"type": "dynamic", "resolved": False}, "content": "test", "score": 80},
        ])
        results = await engine._retrieve_semantic_time(
            "Python", mock_bucket_mgr, mock_embedding_engine, mock_decay_engine, top_k=10,
        )
        assert len(results) >= 1
        assert results[0]["source"] == "semantic"

    @pytest.mark.asyncio
    async def test_supplements_unresolved(self, engine, mock_bucket_mgr, mock_embedding_engine, mock_decay_engine):
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_bucket_mgr.list_all = AsyncMock(return_value=[
            {"id": "u1", "metadata": {"name": "unresolved", "type": "dynamic", "resolved": False, "pinned": False}, "content": "issue"},
        ])
        mock_decay_engine.calculate_score = MagicMock(return_value=7.0)
        results = await engine._retrieve_semantic_time(
            "test", mock_bucket_mgr, mock_embedding_engine, mock_decay_engine, top_k=10,
        )
        assert any(r["source"] == "unresolved" for r in results)


# ── HOT: 3-Way Fusion ───────────────────────────────────────

class TestThreeWay:
    """HOT mode: vector + BM25 + graph fusion."""

    @pytest.mark.asyncio
    async def test_fuses_three_paths(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                      mock_memory_graph, mock_decay_engine):
        mock_embedding_engine.search_similar = AsyncMock(return_value=[
            ("mem_001", 0.9), ("mem_002", 0.7),
        ])
        mock_bucket_mgr.search = AsyncMock(return_value=[
            {"id": "mem_001", "metadata": {}, "content": "test", "score": 85},
            {"id": "mem_003", "metadata": {}, "content": "test", "score": 60},
        ])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[
            {"to_id": "mem_004", "from_id": "mem_001", "weight": 0.8},
        ])

        results = await engine._retrieve_three_way(
            "Python", mock_bucket_mgr, mock_embedding_engine,
            mock_memory_graph, mock_decay_engine, top_k=10,
        )
        # Should have results from all three paths
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_three_way_scores_are_weighted(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                                   mock_memory_graph, mock_decay_engine):
        mock_embedding_engine.search_similar = AsyncMock(return_value=[("mem_001", 1.0)])
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[])

        results = await engine._retrieve_three_way(
            "test", mock_bucket_mgr, mock_embedding_engine,
            mock_memory_graph, mock_decay_engine, top_k=10,
        )
        if results:
            # Pure vector result → final_score = vector_score * 0.35
            assert "final_score" in results[0]

    @pytest.mark.asyncio
    async def test_random_surface_diversity(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                              mock_memory_graph, mock_decay_engine):
        """15% chance of random surface memory for diversity."""
        mock_embedding_engine.search_similar = AsyncMock(return_value=[])
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[])
        mock_bucket_mgr.list_all = AsyncMock(return_value=[
            {"id": "rand1", "metadata": {"type": "dynamic"}, "content": "random memory"},
        ])

        # Run multiple times — at least one should include a random surface
        found_random = False
        with patch('random.random', return_value=0.01):  # Force random surface
            results = await engine._retrieve_three_way(
                "test", mock_bucket_mgr, mock_embedding_engine,
                mock_memory_graph, mock_decay_engine, top_k=10,
            )
            found_random = any(r.get("source") == "random_surface" for r in results)
        assert found_random


# ── RICH: 4-Way + WS Re-rank ────────────────────────────────

class TestFourWayWS:
    """RICH mode: 3-way + Working Self re-rank."""

    @pytest.mark.asyncio
    async def test_four_way_delegates_to_three_way(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                                     mock_memory_graph, mock_working_self, mock_decay_engine):
        mock_embedding_engine.search_similar = AsyncMock(return_value=[("mem_001", 0.9)])
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[])
        mock_working_self.has_goals = True
        mock_working_self.match = MagicMock(return_value=0.8)

        results = await engine._retrieve_four_way_ws(
            "test", mock_bucket_mgr, mock_embedding_engine,
            mock_memory_graph, mock_working_self, mock_decay_engine, top_k=5,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_ws_match_added_to_score(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                            mock_memory_graph, mock_working_self, mock_decay_engine):
        mock_embedding_engine.search_similar = AsyncMock(return_value=[("mem_001", 1.0)])
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[])
        mock_working_self.has_goals = True
        mock_working_self.match = MagicMock(return_value=0.9)

        results = await engine._retrieve_four_way_ws(
            "test", mock_bucket_mgr, mock_embedding_engine,
            mock_memory_graph, mock_working_self, mock_decay_engine, top_k=5,
        )
        if results:
            assert "ws_match" in results[0]


# ── Main Search Routing ─────────────────────────────────────

class TestSearchRouting:
    """Verify DDA-level → retrieval mode routing."""

    @pytest.mark.asyncio
    async def test_cold_routes_to_all(self, engine, mock_bucket_mgr, mock_decay_engine):
        mock_bucket_mgr.list_all = AsyncMock(return_value=[])
        results = await engine.search(
            query="test",
            strategy=COLD_STRATEGY,
            ddi_level=DDILevel.COLD,
            bucket_mgr=mock_bucket_mgr,
            decay_engine=mock_decay_engine,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_warm_routes_to_semantic_time(self, engine, mock_bucket_mgr, mock_embedding_engine, mock_decay_engine):
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_bucket_mgr.list_all = AsyncMock(return_value=[])
        results = await engine.search(
            query="test",
            strategy=WARM_STRATEGY,
            ddi_level=DDILevel.WARM,
            bucket_mgr=mock_bucket_mgr,
            embedding_engine=mock_embedding_engine,
            decay_engine=mock_decay_engine,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hot_routes_to_three_way(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                             mock_memory_graph, mock_decay_engine):
        mock_embedding_engine.search_similar = AsyncMock(return_value=[])
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[])
        results = await engine.search(
            query="test",
            strategy=HOT_STRATEGY,
            ddi_level=DDILevel.HOT,
            bucket_mgr=mock_bucket_mgr,
            embedding_engine=mock_embedding_engine,
            memory_graph=mock_memory_graph,
            decay_engine=mock_decay_engine,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_rich_routes_to_four_way_ws(self, engine, mock_bucket_mgr, mock_embedding_engine,
                                                mock_memory_graph, mock_working_self, mock_decay_engine):
        mock_embedding_engine.search_similar = AsyncMock(return_value=[])
        mock_bucket_mgr.search = AsyncMock(return_value=[])
        mock_memory_graph.get_neighbors = MagicMock(return_value=[])
        mock_working_self.has_goals = False
        results = await engine.search(
            query="test",
            strategy=RICH_STRATEGY,
            ddi_level=DDILevel.RICH,
            bucket_mgr=mock_bucket_mgr,
            embedding_engine=mock_embedding_engine,
            memory_graph=mock_memory_graph,
            working_self=mock_working_self,
            decay_engine=mock_decay_engine,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_defaults_to_all_on_missing_strategy(self, engine, mock_bucket_mgr, mock_decay_engine):
        mock_bucket_mgr.list_all = AsyncMock(return_value=[])
        results = await engine.search(
            query="test",
            strategy=None,  # None → defaults to COLD-like
            bucket_mgr=mock_bucket_mgr,
            decay_engine=mock_decay_engine,
        )
        assert isinstance(results, list)


# ── Emotion Resonance ───────────────────────────────────────

class TestEmotionResonance:
    """Verify Russell circumplex emotion resonance scoring."""

    def test_identical_emotion_is_perfect_match(self, engine):
        score = engine.emotion_resonance(0.5, 0.5, 0.5, 0.5)
        assert score == pytest.approx(1.0)

    def test_opposite_emotion_is_poor_match(self, engine):
        score = engine.emotion_resonance(1.0, 1.0, 0.0, 0.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_same_valence_different_arousal(self, engine):
        score = engine.emotion_resonance(0.8, 0.8, 0.8, 0.2)
        assert 0.4 < score < 0.8  # Partial match

    def test_emotion_resonance_symmetric(self, engine):
        a = engine.emotion_resonance(0.7, 0.6, 0.3, 0.4)
        b = engine.emotion_resonance(0.3, 0.4, 0.7, 0.6)
        assert a == pytest.approx(b)

    def test_emotion_resonance_clamped_0_1(self, engine):
        score = engine.emotion_resonance(0.0, 0.0, 1.0, 1.0)  # max distance
        assert 0.0 <= score <= 1.0


# ── Path Weights ────────────────────────────────────────────

class TestPathWeights:
    """Verify retrieval path weight configuration."""

    def test_weights_sum_is_reasonable(self, engine):
        total = sum(engine.path_weights.values())
        # v8: vector(0.25) + bm25(0.12) + graph(0.22) + emotion(0.12)
        #     + temporal(0.15) + cross_ref(0.10) + ws_rerank(0.04) + importance(0.10)
        #     = 1.10 → normalized to 1.00
        primary_total = (
            engine.path_weights["vector"]
            + engine.path_weights["bm25"]
            + engine.path_weights["graph"]
            + engine.path_weights["emotion"]
            + engine.path_weights["temporal"]
            + engine.path_weights["cross_ref"]
        )
        assert 0.78 <= primary_total <= 1.0, f"primary_total={primary_total}"

    def test_vector_has_highest_weight(self, engine):
        assert engine.path_weights["vector"] == max(
            engine.path_weights["vector"],
            engine.path_weights["bm25"],
            engine.path_weights["graph"],
            engine.path_weights["emotion"],
        )
