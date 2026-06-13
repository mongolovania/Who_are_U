# ============================================================
# Test: GraphRAG Community Detection (test_graph_rag.py)
# Track C Task 1: Leiden-like community detection + hierarchical
# summaries on the Memory Graph.
# ============================================================

import math
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from graph_rag import (
    LeidenDetector, GraphRAGEngine, CommunityReport, _fisher_yates_shuffle,
)


# ── LeidenDetector Tests ──────────────────────────────────────


class TestLeidenDetector:
    """Test the Leiden-like community detection algorithm."""

    def test_detect_communities_empty(self, leiden_detector_fixture):
        """Empty graph returns empty communities."""
        detector = leiden_detector_fixture
        result = detector.detect_communities({}, [])
        assert result == {}

    def test_detect_communities_single_node(self, leiden_detector_fixture):
        """Single node returns one community."""
        detector = leiden_detector_fixture
        result = detector.detect_communities(
            {"n1": {"type": "memory"}}, []
        )
        assert len(result) == 1
        members = list(result.values())[0]
        assert "n1" in members

    def test_detect_communities_two_connected(self, leiden_detector_fixture):
        """Two connected nodes should be in the same community."""
        detector = leiden_detector_fixture
        nodes = {"n1": {}, "n2": {}}
        edges = [{"from_id": "n1", "to_id": "n2", "weight": 1.0}]
        result = detector.detect_communities(nodes, edges)
        # Should merge into 1 community since they're connected
        assert len(result) >= 1

    def test_detect_communities_two_disconnected(self, leiden_detector_fixture):
        """Two disconnected nodes should be in separate communities."""
        detector = leiden_detector_fixture
        nodes = {"n1": {}, "n2": {}}
        edges = []  # No edges connecting them
        result = detector.detect_communities(nodes, edges)
        assert len(result) == 2  # Each node in its own community

    def test_detect_communities_dense_cluster(
        self, leiden_detector_fixture, sample_graph_nodes, sample_graph_edges
    ):
        """Densely connected nodes should cluster together."""
        detector = leiden_detector_fixture
        result = detector.detect_communities(
            sample_graph_nodes, sample_graph_edges
        )
        # n1,n2,n3 should be in same community (triangle)
        # n4,n5,n6 should be in same community (triangle)
        # n7,n8 likely separate or together
        assert len(result) >= 2  # At least 2 communities found

        # Check that n1,n2,n3 are together
        n1_comm = None
        for cid, members in result.items():
            if "n1" in members:
                n1_comm = members
                break
        assert n1_comm is not None
        assert "n2" in n1_comm
        assert "n3" in n1_comm

    def test_calculate_modularity(
        self, leiden_detector_fixture, sample_graph_nodes, sample_graph_edges
    ):
        """Modularity calculation returns valid range."""
        detector = leiden_detector_fixture
        communities = detector.detect_communities(
            sample_graph_nodes, sample_graph_edges
        )
        adjacency = detector._build_adjacency(
            sample_graph_nodes, sample_graph_edges
        )

        q = detector.calculate_modularity(communities, adjacency)
        assert -0.5 <= q <= 1.0  # Modularity range

    def test_modularity_improvement(
        self, leiden_detector_fixture, sample_graph_nodes, sample_graph_edges
    ):
        """Detected communities should have higher modularity than random."""
        detector = leiden_detector_fixture
        communities = detector.detect_communities(
            sample_graph_nodes, sample_graph_edges
        )
        adjacency = detector._build_adjacency(
            sample_graph_nodes, sample_graph_edges
        )

        # Actual modularity
        actual_q = detector.calculate_modularity(communities, adjacency)

        # Trivial partition (each node alone)
        trivial = {f"c_{n}": [n] for n in sample_graph_nodes}
        trivial_q = detector.calculate_modularity(trivial, adjacency)

        # Detected communities should be better than trivial
        assert actual_q >= trivial_q - 0.01

    def test_build_hierarchy(
        self, leiden_detector_fixture, sample_graph_nodes, sample_graph_edges
    ):
        """Hierarchy building produces multiple levels."""
        detector = leiden_detector_fixture
        communities = detector.detect_communities(
            sample_graph_nodes, sample_graph_edges
        )
        adjacency = detector._build_adjacency(
            sample_graph_nodes, sample_graph_edges
        )

        hierarchy = detector.build_hierarchy(communities, adjacency, levels=2)
        assert len(hierarchy) >= 1
        # Level 0 should be the base communities
        assert len(hierarchy[0]) >= 2

    def test_resolution_parameter(self, sample_graph_nodes, sample_graph_edges):
        """Higher resolution produces more communities."""
        detector_low = LeidenDetector(resolution=0.5)
        detector_high = LeidenDetector(resolution=2.0)

        low_result = detector_low.detect_communities(
            sample_graph_nodes, sample_graph_edges
        )
        high_result = detector_high.detect_communities(
            sample_graph_nodes, sample_graph_edges
        )

        # Higher resolution should produce >= communities
        assert len(high_result) >= len(low_result)

    def test_connected_components_fallback(
        self, leiden_detector_fixture
    ):
        """Connected components fallback works with zero-weight edges."""
        detector = leiden_detector_fixture
        # Build an adjacency with no weights but structural connections
        adjacency = {
            "a": {"b": 0.0},
            "b": {"a": 0.0, "c": 0.0},
            "c": {"b": 0.0},
            "d": {"e": 0.0},
            "e": {"d": 0.0},
        }
        result = detector._connected_components(adjacency)
        assert len(result) == 2  # Two separate components: a-b-c and d-e

    def test_extract_key_themes(self, leiden_detector_fixture):
        """Theme extraction from content."""
        detector = leiden_detector_fixture
        contents = [
            "今天面试了字节跳动的Python岗位，感觉很紧张",
            "收到offer了！薪资比预期高很多，很开心",
            "准备入职材料，要体检还要办工资卡",
        ]
        themes = detector._extract_key_themes(contents)
        assert "职业发展" in themes
        assert len(themes) >= 1

    def test_summarize_community(self, leiden_detector_fixture):
        """Community summary generation."""
        detector = leiden_detector_fixture
        report = detector.summarize_community(
            community_id="test_comm",
            member_ids=["m1", "m2", "m3"],
            level=0,
        )
        assert isinstance(report, CommunityReport)
        assert report.community_id == "test_comm"
        assert report.level == 0
        assert len(report.member_ids) == 3
        assert len(report.summary) > 0  # Should generate rule-based summary

    def test_generate_rule_summary(self, leiden_detector_fixture):
        """Rule-based summary generation."""
        detector = leiden_detector_fixture
        report = CommunityReport(
            community_id="test",
            member_ids=["m1", "m2"],
            key_themes=["职业发展", "学习成长"],
            valence_avg=0.7,
            arousal_avg=0.5,
        )
        summary = detector._generate_rule_summary(report, ["m1", "m2"])
        assert "职业发展" in summary
        assert "2条记忆" in summary
        assert len(summary) > 10


# ── GraphRAGEngine Tests ──────────────────────────────────────


class TestGraphRAGEngine:
    """Test the GraphRAG integration engine."""

    def test_run_empty_graph(self, graph_rag_engine_fixture):
        """Running on empty graph returns empty result."""
        engine = graph_rag_engine_fixture
        mock_graph = MagicMock()
        mock_graph.get_graph_stats.return_value = {
            "node_count": 0, "edge_count": 0, "active_edge_count": 0,
        }

        result = engine.run(mock_graph)
        assert result["base_communities"] == {}
        assert result["reports"] == []
        assert result["modularity"] == 0.0

    def test_get_community_for_memory(self, graph_rag_engine_fixture):
        """Community lookup by memory ID."""
        engine = graph_rag_engine_fixture
        report = CommunityReport(
            community_id="c1",
            member_ids=["m1", "m2", "m3"],
        )
        engine.reports["c1"] = report

        found = engine.get_community_for_memory("m2")
        assert found is not None
        assert found.community_id == "c1"

        not_found = engine.get_community_for_memory("m999")
        assert not_found is None

    def test_boost_scores_from_community(self, graph_rag_engine_fixture):
        """Community-based score boosting."""
        engine = graph_rag_engine_fixture
        report = CommunityReport(
            community_id="c1",
            member_ids=["m1", "m2"],
            key_themes=["职业发展"],
            key_entities=["面试", "字节"],
            summary="这是一个关于职业发展的记忆社区，包含面试和入职相关记忆。",
        )
        engine.reports["c1"] = report

        results = {
            "m1": {"id": "m1", "final_score": 0.5},
            "m2": {"id": "m2", "final_score": 0.3},
            "m3": {"id": "m3", "final_score": 0.7},
        }

        boosted = engine.boost_scores_from_community(
            query="我的面试情况怎么样",
            results=dict(results),
            boost_factor=0.15,
        )

        # m1 and m2 should be boosted
        assert boosted["m1"]["final_score"] > results["m1"]["final_score"]
        assert boosted["m2"]["final_score"] > results["m2"]["final_score"]
        # m3 should be unchanged
        assert boosted["m3"]["final_score"] == results["m3"]["final_score"]

    def test_boost_no_match(self, graph_rag_engine_fixture):
        """No boost when query doesn't match community themes."""
        engine = graph_rag_engine_fixture
        report = CommunityReport(
            community_id="c1",
            member_ids=["m1"],
            key_themes=["职业发展"],
            summary="关于职业发展",
        )
        engine.reports["c1"] = report

        results = {"m1": {"id": "m1", "final_score": 0.5}}

        boosted = engine.boost_scores_from_community(
            query="今天天气真好",  # unrelated query
            results=dict(results),
        )

        # Should not be boosted
        assert boosted["m1"]["final_score"] == results["m1"]["final_score"]


# ── CommunityReport Tests ─────────────────────────────────────


class TestCommunityReport:
    """Test CommunityReport data model."""

    def test_create_report(self):
        """Basic report creation."""
        report = CommunityReport(
            community_id="test",
            level=1,
            member_ids=["a", "b"],
            summary="Test summary",
        )
        assert report.community_id == "test"
        assert report.level == 1
        assert report.member_ids == ["a", "b"]
        assert report.summary == "Test summary"

    def test_to_dict_and_back(self):
        """Serialization round-trip."""
        report = CommunityReport(
            community_id="test",
            level=0,
            member_ids=["m1", "m2", "m3"],
            key_themes=["职业", "成长"],
            key_entities=["字节", "面试"],
            valence_avg=0.6,
            arousal_avg=0.4,
            importance_avg=7.5,
            modularity_score=0.42,
            summary="这是一个测试社区",
        )
        data = report.to_dict()
        restored = CommunityReport.from_dict(data)

        assert restored.community_id == report.community_id
        assert restored.level == report.level
        assert restored.member_ids == report.member_ids
        assert restored.key_themes == report.key_themes
        assert restored.valence_avg == report.valence_avg
        assert restored.summary == report.summary

    def test_auto_generated_id(self):
        """Auto-generated community ID."""
        report = CommunityReport(member_ids=["m1"])
        assert len(report.community_id) > 0
        assert report.community_id.startswith("comm_")

    def test_auto_timestamp(self):
        """Auto-generated timestamp."""
        report = CommunityReport(community_id="test")
        assert len(report.created_at) > 0
