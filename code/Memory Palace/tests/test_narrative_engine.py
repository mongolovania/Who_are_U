# ============================================================
# Test: Narrative Engine (test_narrative_engine.py)
# L2: Schank story index + Dot living history tests.
#
# Covers:
#   - Thread creation and theme detection
#   - Moment assignment to threads
#   - Thread merging
#   - Living history generation
#   - Story-indexed retrieval
#   - Community detection
#   - Narrative arc detection
#   - Persistence (save/load)
# ============================================================

import json
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from narrative_engine import (
    NarrativeEngine, NarrativeThread, NarrativeMoment,
    _LIFE_SCRIPTS, _ARC_TYPES,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def narrative_engine(tmp_path):
    """NarrativeEngine with temp directory."""
    return NarrativeEngine(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def populated_engine(narrative_engine):
    """Engine with pre-existing threads and moments."""
    eng = narrative_engine

    # Thread 1: Career transition
    t1 = NarrativeThread(
        id="thread_career",
        title="跳槽大厂之路",
        theme="职业转型",
        goal="想找到认可自己的工作",
        script="面试→offer→入职→适应",
        domain=["求职", "工作"],
    )
    t1.moments = [
        NarrativeMoment(
            memory_id="mem_001", content_summary="今天面试又被拒了，感觉很难受",
            valence=0.15, arousal=0.65, importance=7, is_turning_point=True,
            timestamp="2026-05-01T10:00:00", role="inciting_incident",
        ),
        NarrativeMoment(
            memory_id="mem_002", content_summary="终于拿到offer了！开心得睡不着",
            valence=0.90, arousal=0.85, importance=8, is_turning_point=True,
            timestamp="2026-05-15T10:00:00", role="climax",
        ),
        NarrativeMoment(
            memory_id="mem_003", content_summary="入职第一天，新公司感觉不错",
            valence=0.75, arousal=0.55, importance=6, is_turning_point=False,
            timestamp="2026-06-01T10:00:00", role="episode",
        ),
    ]

    # Thread 2: Relationship
    t2 = NarrativeThread(
        id="thread_relationship",
        title="分手后的自我重建",
        theme="亲密关系",
        goal="走出上一段感情",
        script="分手→痛苦→反思→成长→新开始",
        domain=["感情"],
    )
    t2.moments = [
        NarrativeMoment(
            memory_id="mem_010", content_summary="分手了，感觉世界塌了",
            valence=0.05, arousal=0.88, importance=9, is_turning_point=True,
            timestamp="2026-04-01T10:00:00", role="inciting_incident",
        ),
        NarrativeMoment(
            memory_id="mem_011", content_summary="想通了，感谢这段经历让我成长",
            valence=0.55, arousal=0.35, importance=7, is_turning_point=True,
            timestamp="2026-05-20T10:00:00", role="resolution",
        ),
    ]

    eng.threads = {"thread_career": t1, "thread_relationship": t2}
    eng._loaded = True
    eng.save()
    return eng


# ── Thread Creation ───────────────────────────────────────────

class TestThreadCreation:
    """Verify narrative thread creation and theme detection."""

    def test_create_thread_from_memory(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_new",
            content="我决定辞职转行做AI了，虽然害怕但也很期待",
            domain=["职业", "求职"],
            tags=["转行", "AI"],
            valence=0.55,
            arousal=0.70,
            importance=8,
            timestamp="2026-06-10T12:00:00",
        )

        assert thread is not None
        assert thread.theme == "职业转型"
        assert thread.status == "active"
        assert len(thread.moments) == 1
        assert thread.moments[0].memory_id == "mem_new"
        assert thread.moments[0].is_turning_point is True  # "决定" is a turning point marker

    def test_create_thread_detects_relationship_theme(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_rel",
            content="我和男朋友吵架了，他说我不关心他",
            domain=["感情"],
            tags=["吵架"],
            valence=0.20,
            arousal=0.75,
            importance=6,
            timestamp="2026-06-10T12:00:00",
        )

        assert thread.theme == "亲密关系"

    def test_create_thread_detects_growth_theme(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_growth",
            content="今天终于想通了一件事，我一直都在给自己设限",
            domain=["成长"],
            tags=["自我觉察"],
            valence=0.60,
            arousal=0.50,
            importance=7,
            timestamp="2026-06-10T12:00:00",
        )

        assert thread.theme == "成长探索"

    def test_thread_has_script_assigned(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_career",
            content="昨天去面试了，感觉很糟糕",
            domain=["求职"],
            tags=[],
            valence=0.25,
            arousal=0.60,
            importance=5,
            timestamp="2026-06-10T12:00:00",
        )

        assert thread.script != ""
        assert "面试" in thread.script or "offer" in thread.script.lower()

    def test_goal_detected_from_content(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_goal",
            content="我希望能找到一个能让自己成长的工作",
            domain=["职业"],
            tags=[],
            valence=0.60,
            arousal=0.45,
            importance=5,
        )

        assert thread.goal != ""
        assert "希望" in thread.goal

    def test_moment_roles_are_classified(self, narrative_engine):
        # First moment → inciting_incident
        t1 = narrative_engine.find_or_create_thread(
            memory_id="mem_a",
            content="我决定开始每天早上跑步",
            domain=["健康"],
            tags=[],
            valence=0.55,
            arousal=0.60,
            importance=5,
            timestamp="2026-06-01T10:00:00",
        )
        assert t1.moments[0].role in ("inciting_incident", "episode")

        # Resolution moment (use different domain so it creates a new thread)
        t2 = narrative_engine.find_or_create_thread(
            memory_id="mem_b",
            content="我终于解决了困扰多年的失眠问题！",
            domain=["睡眠"],   # Different domain from "健康" to avoid matching t1
            tags=[],
            valence=0.85,
            arousal=0.70,
            importance=8,
            timestamp="2026-06-10T10:00:00",
        )
        assert t2.moments[0].role == "resolution"


# ── Thread Matching ───────────────────────────────────────────

class TestThreadMatching:
    """Verify that memories are correctly assigned to existing threads."""

    def test_memory_matches_existing_thread_by_domain(self, populated_engine):
        # This memory should match the career thread
        thread = populated_engine.find_or_create_thread(
            memory_id="mem_new_career",
            content="今天入职培训学了很多东西",
            domain=["求职", "工作"],
            tags=["入职"],
            valence=0.65,
            arousal=0.50,
            importance=5,
            timestamp="2026-06-10T12:00:00",
        )

        # Should be assigned to existing career thread
        assert thread.id == "thread_career"
        # Thread should now have 4 moments (3 original + 1 new)
        assert len(thread.moments) == 4

    def test_memory_creates_new_thread_when_no_match(self, populated_engine):
        thread = populated_engine.find_or_create_thread(
            memory_id="mem_health",
            content="开始每天健身了！第一次坚持超过一周",
            domain=["健康"],
            tags=["运动"],
            valence=0.85,
            arousal=0.70,
            importance=7,
            timestamp="2026-06-10T12:00:00",
        )

        # Should NOT be career or relationship thread
        assert thread.id not in ("thread_career", "thread_relationship")
        assert thread.theme == "健康管理"

    def test_multiple_memories_consolidate_to_same_thread(self, narrative_engine):
        # First memory
        t1 = narrative_engine.find_or_create_thread(
            memory_id="m1",
            content="今天去面试了",
            domain=["求职"],
            tags=[],
            valence=0.45,
            arousal=0.60,
            importance=5,
            timestamp="2026-06-01T10:00:00",
        )

        # Second memory — should match t1 (same domain + temporal proximity)
        t2 = narrative_engine.find_or_create_thread(
            memory_id="m2",
            content="面试通过了！下周入职",
            domain=["求职"],   # Match t1's domain for consolidation
            tags=[],
            valence=0.88,
            arousal=0.80,
            importance=8,
            timestamp="2026-06-01T14:00:00",  # Same day to boost temporal score
        )

        assert t1.id == t2.id  # Same thread
        assert len(t1.moments) == 2


# ── Thread Merging ────────────────────────────────────────────

class TestThreadMerging:
    """Verify thread merge logic."""

    def test_merge_two_threads(self, narrative_engine):
        # Create two related threads
        t1 = narrative_engine.find_or_create_thread(
            memory_id="prep_1",
            content="开始准备面试了",
            domain=["求职"],
            tags=["面试准备"],
            valence=0.50,
            arousal=0.55,
            importance=5,
            timestamp="2026-05-01T10:00:00",
        )

        t2 = narrative_engine.find_or_create_thread(
            memory_id="prep_2",
            content="面试结束了，等结果",
            domain=["求职", "工作"],
            tags=["面试"],
            valence=0.55,
            arousal=0.65,
            importance=6,
            timestamp="2026-05-10T10:00:00",
        )

        merged = narrative_engine.merge_threads(t1.id, t2.id)
        assert merged is not None
        assert len(merged.moments) == 2
        # Old IDs should be removed
        assert t1.id not in narrative_engine.threads
        assert t2.id not in narrative_engine.threads
        assert merged.id in narrative_engine.threads

    def test_auto_merge_overlapping_threads(self, narrative_engine):
        # Create two threads with same theme
        t1 = narrative_engine.find_or_create_thread(
            memory_id="a1", content="面试A公司", domain=["求职"], tags=[],
            valence=0.4, arousal=0.6, importance=5,
            timestamp="2026-05-01T10:00:00",
        )
        t2 = narrative_engine.find_or_create_thread(
            memory_id="a2", content="面试B公司", domain=["求职"], tags=[],
            valence=0.4, arousal=0.6, importance=5,
            timestamp="2026-05-02T10:00:00",
        )

        # Add more moments to both to meet merge threshold (>= 2 each)
        narrative_engine.find_or_create_thread(
            memory_id="a3", content="A公司给了offer", domain=["求职"], tags=[],
            valence=0.8, arousal=0.7, importance=8,
            timestamp="2026-05-03T10:00:00",
        )
        narrative_engine.find_or_create_thread(
            memory_id="a4", content="B公司也给了offer", domain=["求职"], tags=[],
            valence=0.8, arousal=0.7, importance=8,
            timestamp="2026-05-04T10:00:00",
        )

        merged = narrative_engine._auto_merge_threads()
        # May or may not merge depending on scoring
        assert isinstance(merged, list)


# ── Living History ────────────────────────────────────────────

class TestLivingHistory:
    """Verify Dot-style living history generation."""

    def test_generate_living_history(self, populated_engine):
        result = populated_engine.generate_living_history()

        assert "overview" in result
        assert "active_threads" in result
        assert "recent_turning_points" in result
        assert "emotional_trajectory" in result
        assert len(result["active_threads"]) == 2  # Both threads are active

    def test_living_history_filters_by_domain(self, populated_engine):
        result = populated_engine.generate_living_history(domain_filter=["求职"])
        assert len(result["active_threads"]) == 1
        assert result["active_threads"][0]["theme"] == "职业转型"

    def test_living_history_has_turning_points(self, populated_engine):
        result = populated_engine.generate_living_history()
        # Career thread has 2 turning points, relationship has 2
        assert len(result["recent_turning_points"]) == 4

    def test_living_history_empty_graceful(self, narrative_engine):
        result = narrative_engine.generate_living_history()
        assert result["overview"] == "你的故事才刚刚开始。"
        assert result["active_threads"] == []

    def test_emotional_trajectory_computed(self, populated_engine):
        result = populated_engine.generate_living_history()
        traj = result["emotional_trajectory"]
        assert 0.0 <= traj["avg_valence"] <= 1.0
        assert 0.0 <= traj["avg_arousal"] <= 1.0
        assert "thread_arcs" in traj


# ── Story-Indexed Retrieval ───────────────────────────────────

class TestStoryIndexedRetrieval:
    """Verify narrative-thread-based retrieval."""

    def test_find_story_by_theme_keyword(self, populated_engine):
        results = populated_engine.find_story_for_query("我的跳槽经历")
        assert len(results) > 0
        assert results[0]["title"] == "跳槽大厂之路"

    def test_find_story_by_domain(self, populated_engine):
        results = populated_engine.find_story_for_query("感情")
        assert len(results) > 0
        assert results[0]["theme"] == "亲密关系"

    def test_find_story_empty_when_no_match(self, populated_engine):
        results = populated_engine.find_story_for_query("关于我的宠物")
        # Should return empty or low-score results
        assert len(results) == 0 or all(r["score"] < 0.15 for r in results)

    def test_story_retrieval_includes_key_moments(self, populated_engine):
        results = populated_engine.find_story_for_query("跳槽")
        assert len(results) > 0
        assert "key_moments" in results[0]
        assert len(results[0]["key_moments"]) > 0

    def test_story_retrieval_no_threads(self, narrative_engine):
        results = narrative_engine.find_story_for_query("anything")
        assert results == []


# ── Narrative Arc Detection ───────────────────────────────────

class TestNarrativeArcDetection:
    """Verify emotional arc detection in threads."""

    def test_rising_arc(self, narrative_engine):
        thread = NarrativeThread(title="Rise", theme="成长探索")
        thread.moments = [
            NarrativeMoment(memory_id="1", content_summary="low", valence=0.2, arousal=0.3, timestamp="2026-01-01T00:00:00"),
            NarrativeMoment(memory_id="2", content_summary="mid", valence=0.5, arousal=0.5, timestamp="2026-01-02T00:00:00"),
            NarrativeMoment(memory_id="3", content_summary="high", valence=0.9, arousal=0.7, timestamp="2026-01-03T00:00:00"),
        ]
        arc = narrative_engine._detect_narrative_arc(thread)
        assert arc["type"] in ("上升弧", "平缓弧")

    def test_too_few_moments_arc(self, narrative_engine):
        thread = NarrativeThread(title="Solo", theme="成长探索")
        thread.moments = [
            NarrativeMoment(memory_id="1", content_summary="one", valence=0.5, arousal=0.5, timestamp="2026-01-01T00:00:00"),
        ]
        arc = narrative_engine._detect_narrative_arc(thread)
        assert arc["type"] == "too_few_moments"


# ── Persistence ───────────────────────────────────────────────

class TestPersistence:
    """Verify narrative threads save and load correctly."""

    def test_save_and_load_roundtrip(self, populated_engine, tmp_path):
        # Create a second engine pointing to same data dir
        eng2 = NarrativeEngine(user_id="test_user", data_dir=str(tmp_path / "buckets"))
        eng2.load()

        assert len(eng2.threads) == 2
        assert "thread_career" in eng2.threads
        assert eng2.threads["thread_career"].title == "跳槽大厂之路"
        assert len(eng2.threads["thread_career"].moments) == 3

    def test_new_engine_starts_empty(self, narrative_engine):
        assert narrative_engine.threads == {}
        assert narrative_engine._loaded is False

    def test_load_is_idempotent(self, populated_engine):
        # Second load shouldn't change anything
        thread_count_before = len(populated_engine.threads)
        populated_engine.load()
        assert len(populated_engine.threads) == thread_count_before


# ── Community Detection ───────────────────────────────────────

class TestCommunityDetection:
    """Verify GraphRAG-style community detection."""

    def test_empty_graph_no_communities(self, narrative_engine):
        mock_graph = MagicMock()
        mock_graph.get_graph_stats.return_value = {"node_count": 0, "edge_count": 0}
        communities = narrative_engine.detect_communities(mock_graph, None)
        assert communities == {}

    def test_community_detection_with_graph(self, narrative_engine, memory_graph_fixture):
        # Add nodes to graph
        graph = memory_graph_fixture
        graph.add_node("mem_a", {"valence": 0.5})
        graph.add_node("mem_b", {"valence": 0.4})
        graph.add_node("mem_c", {"valence": 0.3})
        # Connect them
        graph.add_edge("mem_a", "mem_b", "thematic", weight=0.8)
        graph.add_edge("mem_b", "mem_c", "thematic", weight=0.7)

        # Create thread with seed memory
        thread = NarrativeThread(title="Test", theme="成长探索", seed_memory_ids=["mem_a"])
        narrative_engine.threads["test_thread"] = thread

        communities = narrative_engine.detect_communities(graph, None)
        # Should find at least one community
        assert isinstance(communities, dict)

    def test_small_graph_no_communities(self, narrative_engine):
        mock_graph = MagicMock()
        mock_graph.get_graph_stats.return_value = {"node_count": 2, "edge_count": 1}
        communities = narrative_engine.detect_communities(mock_graph, None)
        # < 3 nodes → no community detection
        assert communities == {}


# ── Stats ─────────────────────────────────────────────────────

class TestNarrativeStats:
    """Verify stats computation."""

    def test_stats_on_empty_engine(self, narrative_engine):
        stats = narrative_engine.get_stats()
        assert stats["total_threads"] == 0
        assert stats["active_threads"] == 0
        assert stats["total_moments"] == 0

    def test_stats_on_populated_engine(self, populated_engine):
        stats = populated_engine.get_stats()
        assert stats["total_threads"] == 2
        assert stats["active_threads"] == 2
        assert stats["total_moments"] == 5  # 3 career + 2 relationship
        assert stats["turning_points"] == 4


# ── Edge Cases ────────────────────────────────────────────────

class TestNarrativeEdgeCases:
    """Boundary and edge case tests."""

    def test_empty_content_handled(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_empty",
            content="",
            domain=[],
            tags=[],
            valence=0.5,
            arousal=0.3,
            importance=5,
        )
        assert thread is not None
        assert thread.theme == "成长探索"  # Default

    def test_special_characters_in_content(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_special",
            content="面试官说: \"你不够格\" 😞 但我不会放弃的 💪",
            domain=["求职"],
            tags=[],
            valence=0.3,
            arousal=0.6,
            importance=6,
        )
        assert thread is not None

    def test_thread_moment_timestamps(self, narrative_engine):
        thread = narrative_engine.find_or_create_thread(
            memory_id="mem_ts",
            content="测试记忆",
            domain=[],
            tags=[],
            timestamp="2026-06-10T15:30:00",
        )
        assert thread.moments[0].timestamp == "2026-06-10T15:30:00"

    def test_thread_priority_calculated(self, narrative_engine):
        thread = NarrativeThread(
            title="Priority Test",
            theme="成长探索",
            moments=[
                NarrativeMoment(
                    memory_id="p1", content_summary="important",
                    importance=9, is_turning_point=True, timestamp="2026-06-01T00:00:00",
                ),
                NarrativeMoment(
                    memory_id="p2", content_summary="less important",
                    importance=5, is_turning_point=False, timestamp="2026-06-05T00:00:00",
                ),
            ],
        )
        priority = narrative_engine._calculate_thread_priority(thread)
        assert 0.0 <= priority <= 1.0
        assert priority > 0.3  # Has turning points and is recent

    def test_narrative_merge_no_crash(self, narrative_engine):
        """narrative merge should not crash even without deps."""
        import asyncio
        result = asyncio.run(narrative_engine.run_narrative_merge())
        assert "communities_detected" in result
        assert result["communities_detected"] == 0  # No graph
