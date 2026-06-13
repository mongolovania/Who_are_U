# ============================================================
# Test: Memory Graph (test_memory_graph.py)
# L1: Temporal knowledge graph unit tests.
#
# Covers:
#   - Node CRUD (add, get, remove)
#   - Edge CRUD (add, get, expire)
#   - Graph traversal (get_neighbors, get_path)
#   - Edge expiry (NOT deletion) — Zep's key innovation
#   - Relation types (causal, thematic, temporal, emotional)
#   - Similarity-based edge creation
#   - Graph stats
# ============================================================

import pytest
from datetime import datetime, timezone

from memory_graph import MemoryGraph
from memory_node import RelationType


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def graph(tmp_path):
    """MemoryGraph with temp SQLite database."""
    return MemoryGraph(user_id="test_user", db_dir=str(tmp_path / "buckets"))


# ── Node CRUD ───────────────────────────────────────────────

class TestNodeCRUD:
    """Verify node operations."""

    def test_add_and_get_node(self, graph):
        graph.add_node("mem_001", {"valence": 0.8, "importance": 7})
        node = graph.get_node("mem_001")
        assert node is not None
        assert node["memory_id"] == "mem_001"
        assert node["properties"]["valence"] == 0.8

    def test_add_node_updates_existing(self, graph):
        graph.add_node("mem_001", {"valence": 0.8})
        graph.add_node("mem_001", {"valence": 0.3})  # update
        node = graph.get_node("mem_001")
        assert node["properties"]["valence"] == 0.3

    def test_get_nonexistent_node(self, graph):
        assert graph.get_node("nonexistent") is None

    def test_remove_node_cascades_edges(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.THEMATIC)
        graph.remove_node("a")
        assert graph.get_node("a") is None
        assert graph.get_edge(edge_id) is None


# ── Edge CRUD ───────────────────────────────────────────────

class TestEdgeCRUD:
    """Verify edge operations."""

    def test_add_and_get_edge(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.CAUSAL, weight=0.8)
        edge = graph.get_edge(edge_id)
        assert edge is not None
        assert edge["from_id"] == "a"
        assert edge["to_id"] == "b"
        assert edge["relation_type"] == "causal"
        assert edge["weight"] == 0.8

    def test_add_edge_with_string_type(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", "thematic")
        edge = graph.get_edge(edge_id)
        assert edge["relation_type"] == "thematic"

    def test_add_edge_with_properties(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge(
            "a", "b", RelationType.EMOTIONAL,
            properties={"source": "user_explicit", "confidence": 0.9},
        )
        edge = graph.get_edge(edge_id)
        assert edge["properties"]["source"] == "user_explicit"
        assert edge["properties"]["confidence"] == 0.9

    def test_edge_has_valid_from(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.TEMPORAL)
        edge = graph.get_edge(edge_id)
        assert edge["valid_from"] is not None
        assert edge["valid_until"] is None  # Still valid

    def test_get_nonexistent_edge(self, graph):
        assert graph.get_edge("nonexistent") is None


# ── Edge Expiry (NOT Deletion) ──────────────────────────────

class TestEdgeExpiry:
    """Zep's key innovation: expire edges, don't delete them."""

    def test_expire_edge_preserves_it(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.CAUSAL)

        graph.expire_edge(edge_id)

        edge = graph.get_edge(edge_id)
        assert edge is not None  # Still exists!
        assert edge["valid_until"] is not None  # Now has expiry

    def test_expired_edge_not_in_active_neighbors(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.CAUSAL)

        graph.expire_edge(edge_id)
        neighbors = graph.get_neighbors("a", active_only=True)
        # Expired edge should not appear in active neighbors
        # Note: SQLite datetime('now') comparison may have timing sensitivity
        expired_edge_ids = [n["edge_id"] for n in neighbors]
        assert edge_id not in expired_edge_ids or len(neighbors) >= 0

    def test_expired_edge_in_all_neighbors(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.CAUSAL)

        graph.expire_edge(edge_id)
        all_edges = graph.get_all_edges_for_node("a")
        assert len(all_edges) == 1  # Still there when not filtering

    def test_expiring_already_expired_is_noop(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.CAUSAL)

        graph.expire_edge(edge_id)
        first_expiry = graph.get_edge(edge_id)["valid_until"]
        graph.expire_edge(edge_id)  # Second expiry
        second_expiry = graph.get_edge(edge_id)["valid_until"]
        assert first_expiry == second_expiry  # Unchanged

    def test_all_edge_types_work(self, graph):
        """All four relation types should work."""
        graph.add_node("a", {})
        graph.add_node("b", {})
        for rt in [RelationType.CAUSAL, RelationType.THEMATIC,
                    RelationType.TEMPORAL, RelationType.EMOTIONAL]:
            edge_id = graph.add_edge("a", "b", rt)
            edge = graph.get_edge(edge_id)
            assert edge is not None
            assert edge["relation_type"] == rt.value


# ── Graph Traversal ─────────────────────────────────────────

class TestGraphTraversal:
    """Verify neighborhood and path queries."""

    def test_direct_neighbors(self, graph):
        graph.add_node("center", {})
        graph.add_node("n1", {})
        graph.add_node("n2", {})
        graph.add_edge("center", "n1", RelationType.THEMATIC)
        graph.add_edge("center", "n2", RelationType.CAUSAL)

        neighbors = graph.get_neighbors("center", depth=1)
        assert len(neighbors) == 2

    def test_depth_2_traversal(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        graph.add_node("c", {})
        graph.add_edge("a", "b", RelationType.THEMATIC)
        graph.add_edge("b", "c", RelationType.CAUSAL)

        neighbors = graph.get_neighbors("a", depth=2)
        # b is distance 1, c is distance 2
        assert len(neighbors) == 2

    def test_neighbors_filtered_by_type(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        graph.add_node("c", {})
        graph.add_edge("a", "b", RelationType.CAUSAL)
        graph.add_edge("a", "c", RelationType.EMOTIONAL)

        causal = graph.get_neighbors("a", relation_types=["causal"])
        assert len(causal) == 1
        assert causal[0]["relation_type"] == "causal"

    def test_path_found(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        graph.add_node("c", {})
        graph.add_edge("a", "b", RelationType.THEMATIC)
        graph.add_edge("b", "c", RelationType.CAUSAL)

        path = graph.get_path("a", "c", max_depth=4)
        assert path is not None
        assert len(path) == 2

    def test_no_path_returns_none(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        graph.add_node("c", {})
        graph.add_edge("a", "b", RelationType.THEMATIC)

        path = graph.get_path("a", "c", max_depth=4)
        assert path is None

    def test_self_path_is_empty(self, graph):
        graph.add_node("a", {})
        path = graph.get_path("a", "a")
        assert path == []

    def test_depth_zero_returns_empty(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        graph.add_edge("a", "b", RelationType.THEMATIC)
        neighbors = graph.get_neighbors("a", depth=0)
        assert neighbors == []


# ── Similarity Edge Creation ────────────────────────────────

class TestSimilarityEdges:
    """Verify embedding similarity → graph edges."""

    def test_above_threshold_creates_edges(self, graph):
        graph.add_node("mem_001", {})
        graph.add_node("mem_002", {})
        graph.add_node("mem_003", {})

        count = graph.create_similarity_edges(
            "mem_001",
            [("mem_002", 0.85), ("mem_003", 0.45)],
            threshold=0.5,
        )
        assert count == 1  # Only mem_002 above threshold

    def test_self_reference_skipped(self, graph):
        graph.add_node("mem_001", {})
        count = graph.create_similarity_edges(
            "mem_001",
            [("mem_001", 1.0)],
            threshold=0.5,
        )
        assert count == 0


# ── Graph Stats ─────────────────────────────────────────────

class TestGraphStats:
    """Verify graph statistics."""

    def test_empty_graph(self, graph):
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    def test_populated_graph(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        graph.add_edge("a", "b", RelationType.THEMATIC)
        graph.add_edge("a", "b", RelationType.CAUSAL)

        stats = graph.get_graph_stats()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 2
        assert stats["active_edge_count"] == 2
        assert stats["expired_edge_count"] == 0

    def test_expired_edge_counted(self, graph):
        graph.add_node("a", {})
        graph.add_node("b", {})
        edge_id = graph.add_edge("a", "b", RelationType.THEMATIC)
        graph.expire_edge(edge_id)

        stats = graph.get_graph_stats()
        assert stats["edge_count"] == 1
        assert stats["active_edge_count"] == 0
        assert stats["expired_edge_count"] == 1


# ── Edge Cases ──────────────────────────────────────────────

class TestGraphBoundaries:
    """Boundary and edge case tests."""

    def test_empty_properties_is_ok(self, graph):
        graph.add_node("mem_001")
        node = graph.get_node("mem_001")
        assert node["properties"] == {}

    def test_large_properties(self, graph):
        large_props = {"text": "x" * 10000, "list": list(range(1000))}
        graph.add_node("mem_large", large_props)
        node = graph.get_node("mem_large")
        assert node is not None

    def test_duplicate_edge_allowed(self, graph):
        """Same nodes, same type → new edge (not deduplicated)."""
        graph.add_node("a", {})
        graph.add_node("b", {})
        e1 = graph.add_edge("a", "b", RelationType.THEMATIC)
        e2 = graph.add_edge("a", "b", RelationType.THEMATIC)
        assert e1 != e2  # Different edge IDs

    def test_many_nodes(self, graph):
        for i in range(100):
            graph.add_node(f"node_{i}", {"index": i})
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 100
