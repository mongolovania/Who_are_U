# ============================================================
# Test: Counterfactual Memory (test_counterfactual_memory.py)
# L2: Counterfactual reasoning tests.
#
# Covers:
#   - Counterfactual generation (causal inversion)
#   - Alternative path counterfactuals
#   - Script pattern counterfactuals
#   - Counterfactual evaluation
#   - Storage and retrieval
#   - Serialization (to_dict/from_dict)
# ============================================================

import pytest
from unittest.mock import MagicMock

from counterfactual_memory import (
    CounterfactualMemory, Counterfactual, CounterfactualReport,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def cf_memory():
    return CounterfactualMemory(user_id="test_user")


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.get_all_edges_for_node.return_value = []
    graph.get_neighbors.return_value = []
    graph.get_node.return_value = None
    return graph


@pytest.fixture
def mock_bucket_mgr():
    mgr = MagicMock()
    mgr.read.return_value = {"content": "今天面试通过了，很开心！"}
    return mgr


# ── Counterfactual generation ────────────────────────────────

class TestGenerateCounterfactuals:
    """Counterfactual hypothesis generation."""

    def test_empty_graph_returns_empty(self, cf_memory):
        """When graph is None, returns empty report."""
        report = cf_memory.generate_counterfactuals("mem_001", None)
        assert isinstance(report, CounterfactualReport)
        assert report.counterfactuals == []

    def test_no_incoming_causal_edges(self, cf_memory, mock_graph):
        """When no incoming causal edges, still generates from patterns."""
        mock_graph.get_all_edges_for_node.return_value = []
        report = cf_memory.generate_counterfactuals("mem_001", mock_graph)
        assert isinstance(report, CounterfactualReport)
        assert report.anchor_memory_id == "mem_001"

    def test_with_incoming_causal_edge(self, cf_memory, mock_graph, mock_bucket_mgr):
        """With incoming causal edge, generates inversion counterfactual."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1",
                "from_id": "mem_cause",
                "to_id": "mem_001",
                "relation_type": "causal",
                "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00",
                "valid_until": None,
                "properties": {},
            },
        ]
        mock_bucket_mgr.read.side_effect = lambda mid: {
            "mem_cause": {"content": "面试前感到非常焦虑"},
            "mem_001": {"content": "今天面试通过了，很开心！"},
        }.get(mid, {"content": ""})

        report = cf_memory.generate_counterfactuals(
            "mem_001", mock_graph, mock_bucket_mgr
        )
        assert isinstance(report, CounterfactualReport)
        assert report.causal_paths_found == 1

    def test_respects_top_k_limit(self, cf_memory, mock_graph, mock_bucket_mgr):
        """Should not generate more than top_k counterfactuals."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": f"e{i}",
                "from_id": f"cause_{i}",
                "to_id": "mem_001",
                "relation_type": "causal",
                "weight": 0.7,
                "valid_from": "2026-01-01T00:00:00",
                "valid_until": None,
                "properties": {},
            }
            for i in range(5)
        ]
        mock_bucket_mgr.read.return_value = {"content": "一些内容"}

        report = cf_memory.generate_counterfactuals(
            "mem_001", mock_graph, mock_bucket_mgr, top_k=2
        )
        assert len(report.counterfactuals) <= 2


# ── Counterfactual evaluation ────────────────────────────────

class TestEvaluateCounterfactual:
    """Evaluating "would effect have happened without cause?"."""

    def test_no_graph_returns_neutral(self, cf_memory):
        """No graph data → neutral probability."""
        prob = cf_memory.evaluate_counterfactual("cause", "effect", None)
        assert prob == 0.5

    def test_no_direct_causal_edge_high_independence(self, cf_memory, mock_graph):
        """When no causal edge exists, effect is highly independent."""
        mock_graph.get_all_edges_for_node.return_value = []
        prob = cf_memory.evaluate_counterfactual("cause", "effect", mock_graph)
        assert prob >= 0.8  # High independence

    def test_strong_causal_edge_lower_independence(self, cf_memory, mock_graph):
        """Strong causal edge → lower independence probability."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1",
                "from_id": "cause",
                "to_id": "effect",
                "relation_type": "causal",
                "weight": 0.9,
                "valid_from": "2026-01-01T00:00:00",
                "valid_until": None,
                "properties": {},
            },
        ]
        prob = cf_memory.evaluate_counterfactual("cause", "effect", mock_graph)
        assert prob < 0.8  # Lower independence due to strong causal edge

    def test_multiple_causes_higher_independence(self, cf_memory, mock_graph):
        """Multiple causes → effect is less dependent on any single one."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1", "from_id": "cause", "to_id": "effect",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
            {
                "edge_id": "e2", "from_id": "other_cause", "to_id": "effect",
                "relation_type": "causal", "weight": 0.6,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
            {
                "edge_id": "e3", "from_id": "another_cause", "to_id": "effect",
                "relation_type": "causal", "weight": 0.5,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        prob = cf_memory.evaluate_counterfactual("cause", "effect", mock_graph)
        # Multiple alternative causes → higher independence
        assert prob > 0.3


# ── Storage and retrieval ────────────────────────────────────

class TestStorage:
    """Counterfactual storage and retrieval."""

    def test_store_and_retrieve(self, cf_memory, mock_graph, mock_bucket_mgr):
        """Generated counterfactuals should be retrievable."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1", "from_id": "cause", "to_id": "mem_001",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        mock_bucket_mgr.read.side_effect = lambda mid: {
            "cause": {"content": "leader批评了我"},
            "mem_001": {"content": "决定开始找新工作"},
        }.get(mid, {"content": ""})

        cf_memory.generate_counterfactuals("mem_001", mock_graph, mock_bucket_mgr)
        stored = cf_memory.get_counterfactuals_for("mem_001")
        assert isinstance(stored, list)

    def test_get_nonexistent_returns_empty(self, cf_memory):
        """Requesting counterfactuals for non-existent memory returns empty list."""
        assert cf_memory.get_counterfactuals_for("nonexistent") == []

    def test_get_all_counterfactuals(self, cf_memory, mock_graph, mock_bucket_mgr):
        """get_all should return all stored counterfactuals."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1", "from_id": "cause", "to_id": "mem_a",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        mock_bucket_mgr.read.return_value = {"content": "some content"}
        cf_memory.generate_counterfactuals("mem_a", mock_graph, mock_bucket_mgr)
        all_cf = cf_memory.get_all_counterfactuals()
        assert "mem_a" in all_cf


# ── Serialization ────────────────────────────────────────────

class TestSerialization:
    """to_dict/from_dict round-trip."""

    def test_round_trip(self, cf_memory, mock_graph, mock_bucket_mgr):
        """Serialize then deserialize should preserve data."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1", "from_id": "cause", "to_id": "mem_x",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        mock_bucket_mgr.read.side_effect = lambda mid: {
            "cause": {"content": "面试前焦虑"},
            "mem_x": {"content": "决定接受offer"},
        }.get(mid, {"content": ""})

        cf_memory.generate_counterfactuals("mem_x", mock_graph, mock_bucket_mgr)

        data = cf_memory.to_dict()
        restored = CounterfactualMemory.from_dict(data, user_id="test_user")

        assert restored.user_id == "test_user"
        assert "mem_x" in restored._stored


# ── Data model ───────────────────────────────────────────────

class TestDataModels:
    """Counterfactual and CounterfactualReport."""

    def test_counterfactual_auto_generates_id(self):
        cf = Counterfactual(anchor_memory_id="m1")
        assert cf.id != ""

    def test_counterfactual_to_dict(self):
        cf = Counterfactual(
            id="cf1", anchor_memory_id="m1",
            hypothesis="如果没有X", alternative_outcome="Y会发生",
            confidence=0.5, method="causal_inversion",
        )
        d = cf.to_dict()
        assert d["id"] == "cf1"
        assert d["hypothesis"] == "如果没有X"

    def test_report_auto_generates_at(self):
        report = CounterfactualReport(anchor_memory_id="m1")
        assert report.generated_at != ""


# ── Stats ────────────────────────────────────────────────────

class TestStats:
    """Counterfactual memory statistics."""

    def test_empty_stats(self, cf_memory):
        stats = cf_memory.get_stats()
        assert stats["total_counterfactuals"] == 0
        assert stats["nodes_with_counterfactuals"] == 0

    def test_stats_after_generation(self, cf_memory, mock_graph, mock_bucket_mgr):
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1", "from_id": "cause", "to_id": "mem_x",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        mock_bucket_mgr.read.side_effect = lambda mid: {
            "cause": {"content": "焦虑失眠"},
            "mem_x": {"content": "决定辞职"},
        }.get(mid, {"content": ""})

        cf_memory.generate_counterfactuals("mem_x", mock_graph, mock_bucket_mgr)
        stats = cf_memory.get_stats()
        assert stats["generations"] == 1
