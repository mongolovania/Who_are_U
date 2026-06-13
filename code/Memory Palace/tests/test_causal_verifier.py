# ============================================================
# Test: Causal Verifier (test_causal_verifier.py)
# L2: Causal edge verification tests.
#
# Covers:
#   - Single edge verification (valid/invalid/suspicious)
#   - Temporal precedence check
#   - Coherence check
#   - Circular causality detection
#   - Isolated chain detection
#   - Subgraph verification
#   - Batch verify_all
#   - Edge weight adjustment
# ============================================================

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from causal_verifier import (
    CausalVerifier, CausalVerificationResult, SubgraphVerificationReport,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def verifier():
    return CausalVerifier(user_id="test_user")


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.get_node.return_value = None
    graph.get_neighbors.return_value = []
    graph.get_edges_by_type.return_value = []
    graph.get_all_edges_for_node.return_value = []
    return graph


@pytest.fixture
def valid_causal_edge():
    """A valid causal edge (cause before effect, good coherence)."""
    return {
        "edge_id": "edge_001",
        "from_id": "mem_cause",
        "to_id": "mem_effect",
        "relation_type": "causal",
        "weight": 0.8,
        "valid_from": "2026-05-01T10:00:00",
        "valid_until": None,
        "properties": {
            "shared_concepts": ["面试", "offer"],
            "shared_emotions": ["兴奋"],
        },
    }


@pytest.fixture
def non_causal_edge():
    """A non-causal edge (should pass through without verification)."""
    return {
        "edge_id": "edge_002",
        "from_id": "mem_a",
        "to_id": "mem_b",
        "relation_type": "thematic",
        "weight": 0.7,
        "valid_from": "2026-05-01T10:00:00",
        "valid_until": None,
        "properties": {},
    }


# ── Edge verification ────────────────────────────────────────

class TestVerifyEdge:
    """Single edge verification."""

    def test_non_causal_edge_passes_through(self, verifier, non_causal_edge):
        """Non-causal edges should always be valid."""
        result = verifier.verify_edge(non_causal_edge)
        assert result.valid is True
        assert result.confidence == 1.0
        assert result.adjusted_weight == 0.7

    def test_causal_edge_basic_verification(self, verifier, valid_causal_edge, mock_graph):
        """Basic causal edge verification without temporal data."""
        result = verifier.verify_edge(valid_causal_edge, mock_graph)
        assert isinstance(result, CausalVerificationResult)
        assert result.edge_id == "edge_001"
        assert result.original_weight == 0.8

    def test_causal_edge_result_has_all_fields(self, verifier, valid_causal_edge):
        """Result should contain all required fields."""
        result = verifier.verify_edge(valid_causal_edge)
        assert hasattr(result, 'valid')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'adjusted_weight')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'verified_at')

    def test_causal_edge_with_low_weight(self, verifier):
        """Low weight edge should still pass basic verification."""
        edge = {
            "edge_id": "edge_low",
            "from_id": "mem_cause",
            "to_id": "mem_effect",
            "relation_type": "causal",
            "weight": 0.3,
            "valid_from": "2026-05-01T10:00:00",
            "valid_until": None,
            "properties": {},
        }
        result = verifier.verify_edge(edge)
        assert result.original_weight == 0.3


# ── Temporal precedence ──────────────────────────────────────

class TestTemporalPrecedence:
    """Temporal precedence checks."""

    def test_nodes_with_known_timestamps(self, verifier, mock_graph):
        """When timestamps are available and correct."""
        mock_graph.get_node.side_effect = lambda mid: {
            "mem_cause": {
                "memory_id": "mem_cause",
                "created_at": "2026-05-01T09:00:00",
                "properties": {},
            },
            "mem_effect": {
                "memory_id": "mem_effect",
                "created_at": "2026-05-01T10:00:00",
                "properties": {},
            },
        }.get(mid)

        edge = {
            "edge_id": "edge_time",
            "from_id": "mem_cause",
            "to_id": "mem_effect",
            "relation_type": "causal",
            "weight": 0.8,
            "valid_from": "2026-05-01T10:00:00",
            "valid_until": None,
            "properties": {},
        }
        result = verifier.verify_edge(edge, mock_graph)
        # Cause before effect → should not have temporal violation
        assert "temporal_precedence_violation" not in result.issues

    def test_nodes_with_reversed_timestamps(self, verifier, mock_graph):
        """When effect timestamp is before cause timestamp."""
        mock_graph.get_node.side_effect = lambda mid: {
            "mem_cause": {
                "memory_id": "mem_cause",
                "created_at": "2026-05-01T11:00:00",  # Later!
                "properties": {},
            },
            "mem_effect": {
                "memory_id": "mem_effect",
                "created_at": "2026-05-01T10:00:00",  # Earlier!
                "properties": {},
            },
        }.get(mid)

        edge = {
            "edge_id": "edge_reversed",
            "from_id": "mem_cause",
            "to_id": "mem_effect",
            "relation_type": "causal",
            "weight": 0.8,
            "valid_from": "2026-05-01T10:00:00",
            "valid_until": None,
            "properties": {},
        }
        result = verifier.verify_edge(edge, mock_graph)
        assert "temporal_precedence_violation" in result.issues
        assert result.confidence < 1.0

    def test_grace_period_for_near_simultaneous(self, verifier, mock_graph):
        """Events within 5 minutes should pass temporal check."""
        mock_graph.get_node.side_effect = lambda mid: {
            "mem_cause": {
                "memory_id": "mem_cause",
                "created_at": "2026-05-01T10:03:00",
                "properties": {},
            },
            "mem_effect": {
                "memory_id": "mem_effect",
                "created_at": "2026-05-01T10:00:00",
                "properties": {},
            },
        }.get(mid)

        edge = {
            "edge_id": "edge_near",
            "from_id": "mem_cause",
            "to_id": "mem_effect",
            "relation_type": "causal",
            "weight": 0.8,
            "valid_from": "2026-05-01T10:00:00",
            "valid_until": None,
            "properties": {},
        }
        result = verifier.verify_edge(edge, mock_graph)
        # 3 minutes apart → within grace period → no violation
        assert "temporal_precedence_violation" not in result.issues


# ── Subgraph verification ────────────────────────────────────

class TestVerifySubgraph:
    """Subgraph-level verification."""

    def test_empty_graph_returns_zero(self, verifier, mock_graph):
        """Empty graph should produce empty report."""
        mock_graph.get_neighbors.return_value = []
        report = verifier.verify_subgraph("mem_root", mock_graph, depth=2)
        assert isinstance(report, SubgraphVerificationReport)
        assert report.total_edges_checked == 0
        assert report.root_id == "mem_root"

    def test_single_causal_edge_subgraph(self, verifier, mock_graph):
        """Subgraph with one causal edge."""
        # Mock returns: one causal edge from root to child, then empty for child
        call_count = [0]

        def neighbor_side_effect(node_id, depth=1, relation_types=None, active_only=True):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{
                    "edge_id": "e1",
                    "from_id": "mem_root",
                    "to_id": "mem_child",
                    "relation_type": "causal",
                    "weight": 0.7,
                    "valid_from": "2026-01-01T00:00:00",
                    "valid_until": None,
                    "properties": {},
                }]
            return []

        mock_graph.get_neighbors.side_effect = neighbor_side_effect

        report = verifier.verify_subgraph("mem_root", mock_graph, depth=2)
        assert report.total_edges_checked == 1
        assert report.root_id == "mem_root"


# ── Batch verification ───────────────────────────────────────

class TestVerifyAll:
    """Batch verify_all."""

    def test_verify_all_empty_graph(self, verifier, mock_graph):
        """Empty graph should return zero counts."""
        mock_graph.get_edges_by_type.return_value = []
        result = verifier.verify_all(mock_graph)
        assert result["total_causal_edges"] == 0
        assert result["valid"] == 0
        assert result["invalid"] == 0

    def test_verify_all_with_edges(self, verifier, mock_graph):
        """Graph with multiple causal edges."""
        mock_graph.get_edges_by_type.return_value = [
            {
                "edge_id": "e1", "from_id": "a", "to_id": "b",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
            {
                "edge_id": "e2", "from_id": "b", "to_id": "c",
                "relation_type": "causal", "weight": 0.6,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        result = verifier.verify_all(mock_graph)
        assert result["total_causal_edges"] == 2
        assert "average_confidence" in result

    def test_verify_all_handles_graph_error(self, verifier, mock_graph):
        """Should handle graph error gracefully."""
        mock_graph.get_edges_by_type.side_effect = Exception("DB error")
        result = verifier.verify_all(mock_graph)
        assert "error" in result


# ── Cache management ─────────────────────────────────────────

class TestCache:
    """Verification cache."""

    def test_cache_stores_results(self, verifier, valid_causal_edge):
        """Results should be cached by edge_id."""
        result = verifier.verify_edge(valid_causal_edge)
        cached = verifier.get_cached_result("edge_001")
        assert cached is not None
        assert cached.edge_id == "edge_001"

    def test_clear_cache(self, verifier, valid_causal_edge):
        """Clearing cache should remove all results."""
        verifier.verify_edge(valid_causal_edge)
        verifier.clear_cache()
        assert verifier.get_cached_result("edge_001") is None

    def test_get_cached_result_missing(self, verifier):
        """Requesting non-existent cache entry returns None."""
        assert verifier.get_cached_result("nonexistent") is None


# ── Verify edges for node ────────────────────────────────────

class TestVerifyEdgesForNode:
    """Node-specific verification."""

    def test_empty_edges(self, verifier, mock_graph):
        """Node with no edges."""
        mock_graph.get_all_edges_for_node.return_value = []
        results = verifier.verify_edges_for_node("mem_x", mock_graph)
        assert results == []

    def test_only_non_causal_edges(self, verifier, mock_graph):
        """Node with only thematic/temporal edges."""
        mock_graph.get_all_edges_for_node.return_value = [
            {
                "edge_id": "e1", "from_id": "mem_x", "to_id": "mem_y",
                "relation_type": "thematic", "weight": 0.7,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        results = verifier.verify_edges_for_node("mem_x", mock_graph, adjust_weights=False)
        assert len(results) == 0  # Only causal edges are verified


# ── Edge case: coherence scoring ─────────────────────────────

class TestCoherenceScoring:
    """Entity coherence between cause and effect."""

    def test_no_entity_data_returns_neutral(self, verifier):
        """When no entity data is available, should return neutral score."""
        score = verifier._check_coherence("a", "b", None, None)
        assert score == 0.5

    def test_shared_entities_boost_score(self, verifier, mock_graph):
        """Nodes with shared entities should have higher coherence."""
        mock_graph.get_node.side_effect = lambda mid: {
            "a": {"properties": {"concepts": ["面试", "offer"]}},
            "b": {"properties": {"concepts": ["offer", "入职"]}},
        }.get(mid)

        score = verifier._check_coherence("a", "b", mock_graph, None)
        assert score > 0.3  # Some overlap


# ── Data model ───────────────────────────────────────────────

class TestDataModels:
    """CausalVerificationResult and SubgraphVerificationReport."""

    def test_result_has_verified_at(self):
        """Result should auto-set verified_at."""
        result = CausalVerificationResult(edge_id="e1")
        assert result.verified_at != ""

    def test_result_to_dict(self, valid_causal_edge):
        """to_dict should include all fields."""
        v = CausalVerificationResult(
            edge_id="e1", from_id="a", to_id="b",
            issues=["temporal_precedence_violation"],
        )
        d = v.to_dict()
        assert d["edge_id"] == "e1"
        assert "temporal_precedence_violation" in d["issues"]

    def test_report_has_generated_at(self):
        """Report should auto-set generated_at."""
        report = SubgraphVerificationReport(root_id="r")
        assert report.generated_at != ""
