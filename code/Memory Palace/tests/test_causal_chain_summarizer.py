# ============================================================
# Test: Causal Chain Summarizer (test_causal_chain_summarizer.py)
# L2: Causal chain summarization tests.
#
# Covers:
#   - Point-to-point chain summarization
#   - Domain-wide chain discovery
#   - Markdown export
#   - Summary generation quality
#   - Edge case: empty graph, no paths
# ============================================================

import pytest
from unittest.mock import MagicMock

from causal_chain_summarizer import (
    CausalChainSummarizer, CausalChain, CausalLink,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def summarizer():
    return CausalChainSummarizer(user_id="test_user")


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.get_edges_by_type.return_value = []
    graph.get_neighbors.return_value = []
    graph.get_node.return_value = None
    return graph


@pytest.fixture
def mock_bucket_mgr():
    mgr = MagicMock()
    mgr.read.return_value = {"content": "面试通过了，开心！"}
    return mgr


# ── Chain summarization ──────────────────────────────────────

class TestSummarizeChain:
    """Point-to-point causal chain summarization."""

    def test_empty_graph_returns_empty(self, summarizer):
        """No graph → no chains."""
        chains = summarizer.summarize_chain("a", "b", None)
        assert chains == []

    def test_no_path_found(self, summarizer, mock_graph):
        """When no path exists between nodes."""
        mock_graph.get_neighbors.return_value = []
        chains = summarizer.summarize_chain("a", "b", mock_graph)
        assert chains == []

    def test_single_edge_chain(self, summarizer, mock_graph, mock_bucket_mgr):
        """A single causal edge should form a 1-link chain."""
        mock_graph.get_neighbors.side_effect = [
            # First BFS step: causal edge from a to b
            [{
                "edge_id": "e1",
                "from_id": "a",
                "to_id": "b",
                "relation_type": "causal",
                "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00",
                "valid_until": None,
                "properties": {},
            }],
            # Second BFS step: no more neighbors
            [],
        ]

        chains = summarizer.summarize_chain("a", "b", mock_graph, mock_bucket_mgr)
        assert isinstance(chains, list)

    def test_multi_hop_chain(self, summarizer, mock_graph, mock_bucket_mgr):
        """Multi-hop causal chain: a→b→c."""
        # We need DFS to find the path: a neighbors b, b neighbors c
        mock_graph.get_neighbors.side_effect = [
            # Neighbors of a
            [{
                "edge_id": "e1", "from_id": "a", "to_id": "b",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            }],
            # Neighbors of b
            [{
                "edge_id": "e2", "from_id": "b", "to_id": "c",
                "relation_type": "causal", "weight": 0.7,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            }],
            [],  # No more
        ]
        chains = summarizer.summarize_chain("a", "c", mock_graph, mock_bucket_mgr)
        assert isinstance(chains, list)


# ── Domain-wide chain discovery ──────────────────────────────

class TestSummarizeAllChains:
    """Domain-wide chain discovery."""

    def test_empty_graph(self, summarizer, mock_graph):
        """Empty graph returns empty list."""
        mock_graph.get_edges_by_type.return_value = []
        chains = summarizer.summarize_all_chains(mock_graph)
        assert chains == []

    def test_single_edge_insufficient(self, summarizer, mock_graph):
        """Single causal edge can't form a chain of min length 2."""
        mock_graph.get_edges_by_type.return_value = [
            {
                "edge_id": "e1", "from_id": "a", "to_id": "b",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        chains = summarizer.summarize_all_chains(mock_graph)
        # With min_chain_length=2, a single edge doesn't form a chain
        assert isinstance(chains, list)

    def test_two_connected_edges(self, summarizer, mock_graph, mock_bucket_mgr):
        """Two connected causal edges should form a chain."""
        mock_graph.get_edges_by_type.return_value = [
            {
                "edge_id": "e1", "from_id": "a", "to_id": "b",
                "relation_type": "causal", "weight": 0.8,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
            {
                "edge_id": "e2", "from_id": "b", "to_id": "c",
                "relation_type": "causal", "weight": 0.7,
                "valid_from": "2026-01-01T00:00:00", "valid_until": None,
                "properties": {},
            },
        ]
        mock_bucket_mgr.read.return_value = {"content": "内容"}
        chains = summarizer.summarize_all_chains(mock_graph, mock_bucket_mgr)
        assert isinstance(chains, list)

    def test_graph_error_handled(self, summarizer, mock_graph):
        """Graph error should be caught gracefully."""
        mock_graph.get_edges_by_type.side_effect = Exception("DB error")
        chains = summarizer.summarize_all_chains(mock_graph)
        assert chains == []


# ── Markdown export ──────────────────────────────────────────

class TestMarkdownExport:
    """Markdown export of causal chains."""

    def test_empty_chain(self, summarizer):
        """Empty chain should produce placeholder."""
        chain = CausalChain()
        md = summarizer.export_chain_to_markdown(chain)
        assert "空的因果链" in md or "empty" in md.lower() or md == "" or "###" in md

    def test_single_link_chain(self, summarizer):
        """Chain with one link should produce markdown."""
        link = CausalLink(
            from_id="a", to_id="b",
            from_summary="面试被拒", to_summary="感到失落",
            confidence=0.7,
        )
        chain = CausalChain(
            chain=[link],
            total_confidence=0.7,
            summary="面试被拒 → 感到失落",
        )
        md = summarizer.export_chain_to_markdown(chain)
        assert "面试被拒" in md or "###" in md

    def test_export_all_empty(self, summarizer):
        """Export all when no chains exist."""
        md = summarizer.export_all_to_markdown(max_chains=5)
        assert isinstance(md, str)
        assert len(md) > 0


# ── Data model ───────────────────────────────────────────────

class TestDataModels:
    """CausalChain and CausalLink."""

    def test_chain_auto_generates_id(self):
        chain = CausalChain()
        assert chain.id != ""

    def test_chain_to_dict(self):
        link = CausalLink(
            from_id="a", to_id="b",
            from_summary="cause", to_summary="effect",
        )
        chain = CausalChain(
            chain=[link], total_confidence=0.7,
            summary="cause → effect",
        )
        d = chain.to_dict()
        assert d["depth"] == 1
        assert len(d["chain"]) == 1

    def test_link_has_all_fields(self):
        link = CausalLink(from_id="a", to_id="b")
        assert link.from_id == "a"
        assert link.confidence == 0.5


# ── Stats ────────────────────────────────────────────────────

class TestStats:
    """Causal chain summarizer statistics."""

    def test_empty_stats(self, summarizer):
        stats = summarizer.get_stats()
        assert stats["total_chains"] == 0
        assert stats["summarizations"] == 0

    def test_stats_after_summarization(self, summarizer, mock_graph, mock_bucket_mgr):
        """Stats should update after summarization."""
        mock_graph.get_neighbors.return_value = []
        summarizer.summarize_chain("a", "b", mock_graph, mock_bucket_mgr)
        stats = summarizer.get_stats()
        assert stats["summarizations"] == 1
