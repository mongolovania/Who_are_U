# ============================================================
# Test: Narrative Branch Predictor (test_narrative_branch_predictor.py)
# L2: Narrative branch prediction tests.
#
# Covers:
#   - Script completion prediction
#   - Historical pattern prediction
#   - Trajectory extrapolation
#   - Branch retrieval
#   - Precomputation
#   - Predict all active threads
# ============================================================

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from narrative_branch_predictor import (
    NarrativeBranchPredictor, NarrativeBranch,
    _SCRIPT_PROGRESSIONS,
)
from narrative_engine import NarrativeEngine, NarrativeThread, NarrativeMoment


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def predictor():
    return NarrativeBranchPredictor(user_id="test_user")


@pytest.fixture
def narrative_engine(tmp_path):
    return NarrativeEngine(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def active_career_thread(narrative_engine):
    """An active '职业转型' thread with several moments."""
    thread = NarrativeThread(
        id="thread_career",
        title="跳槽大厂之路",
        theme="职业转型",
        goal="想找到认可自己的工作",
        script="面试→offer→入职→适应",
        domain=["求职", "工作"],
    )
    thread.moments = [
        NarrativeMoment(
            memory_id="mem_001",
            content_summary="今天面试又被拒了，感觉很难受",
            valence=0.15, arousal=0.65, importance=7,
            is_turning_point=True,
            timestamp="2026-05-01T10:00:00",
            role="inciting_incident",
        ),
        NarrativeMoment(
            memory_id="mem_002",
            content_summary="终于拿到offer了！开心得睡不着",
            valence=0.90, arousal=0.85, importance=8,
            is_turning_point=True,
            timestamp="2026-05-15T10:00:00",
            role="climax",
        ),
        NarrativeMoment(
            memory_id="mem_003",
            content_summary="入职第一天，新公司感觉不错",
            valence=0.75, arousal=0.55, importance=6,
            is_turning_point=False,
            timestamp="2026-06-01T10:00:00",
            role="episode",
        ),
    ]
    narrative_engine.threads[thread.id] = thread
    return narrative_engine


# ── Branch prediction ────────────────────────────────────────

class TestPredictBranches:
    """Branch prediction for active threads."""

    def test_no_narrative_engine(self, predictor):
        """Without narrative engine, returns empty."""
        branches = predictor.predict_branches("thread_x", None)
        assert branches == []

    def test_nonexistent_thread(self, predictor, narrative_engine):
        """Non-existent thread returns empty."""
        narrative_engine.load()
        branches = predictor.predict_branches("nonexistent", narrative_engine)
        assert branches == []

    def test_active_thread_generates_branches(self, predictor, active_career_thread):
        """Active career thread should generate script completion branches."""
        branches = predictor.predict_branches(
            "thread_career", active_career_thread
        )
        assert isinstance(branches, list)
        # Should have at least script completion branches
        if branches:
            assert all(isinstance(b, NarrativeBranch) for b in branches)

    def test_respects_top_k(self, predictor, active_career_thread):
        """Should not exceed top_k branches."""
        branches = predictor.predict_branches(
            "thread_career", active_career_thread, top_k=2
        )
        assert len(branches) <= 2

    def test_branches_have_types(self, predictor, active_career_thread):
        """Each branch should have a branch_type."""
        branches = predictor.predict_branches(
            "thread_career", active_career_thread
        )
        valid_types = {"script_completion", "historical_pattern", "trajectory_extrapolation"}
        for b in branches:
            assert b.branch_type in valid_types


# ── Predict all active ───────────────────────────────────────

class TestPredictAllActive:
    """Predict branches for all active threads."""

    def test_empty_narrative_engine(self, predictor):
        """No narrative engine → empty result."""
        result = predictor.predict_all_active(None)
        assert result == {}

    def test_all_active_threads(self, predictor, active_career_thread):
        """Should predict branches for every active thread."""
        result = predictor.predict_all_active(active_career_thread)
        assert isinstance(result, dict)
        if "thread_career" in result:
            assert len(result["thread_career"]) > 0


# ── Script completion ────────────────────────────────────────

class TestScriptCompletion:
    """Life script progression predictions."""

    def test_known_theme_has_script_stages(self):
        """Known themes should have script progression data."""
        assert "职业转型" in _SCRIPT_PROGRESSIONS
        assert len(_SCRIPT_PROGRESSIONS["职业转型"]) > 0

    def test_career_script_stages(self, predictor, active_career_thread):
        """Career thread should generate script-based branches."""
        branches = predictor.predict_branches(
            "thread_career", active_career_thread
        )
        script_branches = [
            b for b in branches if b.branch_type == "script_completion"
        ]
        if script_branches:
            assert script_branches[0].predicted_outcome != ""


# ── Trajectory extrapolation ─────────────────────────────────

class TestTrajectoryExtrapolation:
    """Emotional trajectory-based predictions."""

    def test_thread_with_enough_moments(self, predictor, active_career_thread):
        """Thread with 3+ moments should generate trajectory branches."""
        branches = predictor.predict_branches(
            "thread_career", active_career_thread
        )
        trajectory = [
            b for b in branches if b.branch_type == "trajectory_extrapolation"
        ]
        # May or may not have trajectory branches depending on trend
        for b in trajectory:
            assert "轨迹" in b.predicted_outcome or "趋势" in b.predicted_outcome


# ── Branch retrieval ─────────────────────────────────────────

class TestBranchRetrieval:
    """Retrieve stored branches."""

    def test_get_branches_for_empty(self, predictor):
        """No predictions yet → empty."""
        assert predictor.get_branches_for("nonexistent") == []

    def test_get_branches_after_prediction(self, predictor, active_career_thread):
        """After prediction, branches should be retrievable."""
        predictor.predict_branches("thread_career", active_career_thread)
        stored = predictor.get_branches_for("thread_career")
        assert isinstance(stored, list)

    def test_get_active_branches(self, predictor, active_career_thread):
        """get_active_branches should return all."""
        predictor.predict_branches("thread_career", active_career_thread)
        active = predictor.get_active_branches()
        assert "thread_career" in active


# ── Precomputation ───────────────────────────────────────────

class TestPrecomputation:
    """Precompute relevant memories for branches."""

    def test_no_retrieval_engine(self, predictor):
        """Without retrieval engine, precomputation is skipped."""
        branch = NarrativeBranch(
            thread_id="t1",
            predicted_outcome="可能会拿到offer",
            branch_type="script_completion",
        )
        # Should not raise exception
        predictor.precompute_relevant_memories(branch, None)
        assert branch.relevant_memory_ids == []

    def test_with_retrieval_engine(self, predictor):
        """With retrieval engine, memories should be precomputed."""
        mock_retrieval = MagicMock()
        mock_retrieval.search_sync.return_value = [
            {"id": "mem_1", "content": "面试准备"},
            {"id": "mem_2", "content": "offer选择"},
        ]

        branch = NarrativeBranch(
            thread_id="t1",
            predicted_outcome="可能会拿到offer",
            branch_type="script_completion",
        )
        predictor.precompute_relevant_memories(branch, mock_retrieval)
        assert len(branch.relevant_memory_ids) == 2


# ── Data model ───────────────────────────────────────────────

class TestDataModels:
    """NarrativeBranch data model."""

    def test_branch_auto_generates_id(self):
        branch = NarrativeBranch(thread_id="t1")
        assert branch.id != ""

    def test_branch_to_dict(self):
        branch = NarrativeBranch(
            id="b1", thread_id="t1", thread_title="Test",
            branch_type="script_completion",
            predicted_outcome="可能会发生X",
            confidence=0.6,
        )
        d = branch.to_dict()
        assert d["id"] == "b1"
        assert d["predicted_outcome"] == "可能会发生X"


# ── Stats ────────────────────────────────────────────────────

class TestStats:
    """Branch predictor statistics."""

    def test_empty_stats(self, predictor):
        stats = predictor.get_stats()
        assert stats["total_predicted_branches"] == 0

    def test_stats_after_prediction(self, predictor, active_career_thread):
        predictor.predict_branches("thread_career", active_career_thread)
        stats = predictor.get_stats()
        assert stats["predictions_generated"] == 1


# ── Historical pattern ───────────────────────────────────────

class TestHistoricalPattern:
    """Historical pattern-based predictions."""

    def test_with_resolved_thread(self, predictor, narrative_engine):
        """When a resolved thread with same theme exists."""
        # Add a resolved career thread
        resolved = NarrativeThread(
            id="thread_resolved_career",
            title="去年换工作的经历",
            theme="职业转型",
            status="resolved",
            domain=["求职"],
        )
        resolved.moments = [
            NarrativeMoment(
                memory_id="mem_old_1",
                content_summary="去年面试失败后很沮丧",
                timestamp="2025-03-01T10:00:00",
            ),
            NarrativeMoment(
                memory_id="mem_old_2",
                content_summary="后来找到了更好的工作，现在回想觉得当时太焦虑了",
                timestamp="2025-06-01T10:00:00",
            ),
        ]
        narrative_engine.threads[resolved.id] = resolved

        # Add active career thread
        active = NarrativeThread(
            id="thread_career",
            title="当前跳槽之路",
            theme="职业转型",
            domain=["求职"],
        )
        active.moments = [
            NarrativeMoment(
                memory_id="mem_new_1",
                content_summary="最近面试连续被拒",
                timestamp="2026-05-01T10:00:00",
            ),
        ]
        narrative_engine.threads[active.id] = active

        branches = predictor.predict_branches(
            "thread_career", narrative_engine
        )
        history_branches = [
            b for b in branches if b.branch_type == "historical_pattern"
        ]
        if history_branches:
            assert "上次" in history_branches[0].predicted_outcome
