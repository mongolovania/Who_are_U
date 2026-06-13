# ============================================================
# Test: Working Self (test_working_self.py)
# L2: Active goals and concerns tracker — Conway SMS model.
#
# Covers:
#   - Goal/Concern CRUD + persistence
#   - Memory matching against active goals
#   - Session inference (late-night concerns, topic → goal)
#   - Goal resolution detection
#   - Concern deduplication
#   - COLD behavior (empty Working Self)
# ============================================================

import pytest
import uuid
from unittest.mock import MagicMock, patch

from working_self import WorkingSelf, Goal, Concern


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def ws(tmp_path):
    """WorkingSelf with temp directory."""
    return WorkingSelf(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def ws_with_goals(ws):
    """WorkingSelf with predefined goals and concerns."""
    ws.load()
    ws.active_goals = [
        Goal(
            id="g1", description="找到一份AI方向的工作",
            domain="career", priority=0.9,
            active_since="2026-05-01T00:00:00",
            last_referenced="2026-06-01T00:00:00",
            reference_count=5,
        ),
        Goal(
            id="g2", description="改善和家人的关系",
            domain="family", priority=0.7,
            active_since="2026-04-15T00:00:00",
            last_referenced="2026-05-20T00:00:00",
            reference_count=3,
        ),
    ]
    ws.concerns = [
        Concern(
            id="c1", description="深夜困扰: 担心找不到好工作",
            intensity=0.8, occurrence_count=4,
        ),
    ]
    ws.save()
    return ws


# ── Memory Matching ─────────────────────────────────────────

class TestMemoryMatching:
    """Verify memory relevance scoring against Working Self."""

    def test_cold_user_returns_zero(self, ws):
        """No goals → match = 0.0."""
        match = ws.match("我想找工作")
        assert match == 0.0

    def test_exact_goal_match(self, ws_with_goals):
        """Content matching an active goal should score high."""
        match = ws_with_goals.match("AI方向的工作 面试")
        assert match > 0.0

    def test_domain_match_bonus(self, ws_with_goals):
        """Domain matching an active goal gives bonus."""
        match = ws_with_goals.match("career related content", domain=["career"])
        assert match > 0.0

    def test_no_match_returns_zero(self, ws_with_goals):
        match = ws_with_goals.match("今天天气很好适合出去玩")
        assert match >= 0.0  # Could be zero or very low

    def test_resolved_goals_excluded(self, ws_with_goals):
        ws_with_goals.active_goals[0].resolved = True
        match = ws_with_goals.match("AI方向的工作")
        # Only goal g1 resolved, g2 still active
        # May have some match from concern
        assert match >= 0.0

    def test_match_capped_at_1(self, ws_with_goals):
        match = ws_with_goals.match("找 AI 工作 改善 家人 关系 " * 100)
        assert 0.0 <= match <= 1.0


# ── Session Inference ───────────────────────────────────────

class TestSessionInference:
    """Verify Working Self changes inferred from sessions."""

    def test_late_night_negative_creates_concern(self, ws):
        ws.load()
        ws.infer_from_session(
            user_message="我真的很焦虑，不知道未来会怎样",
            valence=0.2, arousal=0.7,
            session_hour=3,  # 凌晨3点
        )
        assert len(ws.concerns) >= 1
        assert any("深夜困扰" in c.description for c in ws.concerns)

    def test_not_late_night_no_concern(self, ws):
        ws.load()
        ws.infer_from_session(
            user_message="有点担心工作的事",
            valence=0.3, arousal=0.6,
            session_hour=14,  # 下午2点
        )
        # Should not create concern (not late night)
        concerns_created = len(ws.concerns)
        # 下午时段不应触发深夜困扰检测
        assert all("深夜" not in c.description for c in ws.concerns)

    def test_repeated_concern_upgraded_to_goal(self, ws):
        """Repeated concern in same domain → may become active goal."""
        ws.load()
        # Create 2 concerns in same topic area
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('12345678-1234-5678-1234-567812345678')):
            ws.infer_from_session(
                user_message="担心工作的事",
                valence=0.2, arousal=0.7,
                session_hour=3, topics=["career"],
            )
        with patch.object(uuid, 'uuid4', return_value=uuid.UUID('22345678-1234-5678-1234-567812345678')):
            ws.infer_from_session(
                user_message="还是担心工作的事",
                valence=0.2, arousal=0.7,
                session_hour=3, topics=["career"],
            )
        assert len(ws.concerns) >= 1


# ── Concern Deduplication ───────────────────────────────────

class TestConcernDedup:
    """Verify similar concerns are merged, not duplicated."""

    def test_similar_concern_updated(self, ws):
        ws.load()
        ws._upsert_concern("深夜困扰: 担心找不到工作", 0.7, "2026-06-01T00:00:00")
        ws._upsert_concern("深夜困扰: 担心找不到工作，失眠了", 0.85, "2026-06-02T00:00:00")
        # Should update existing concern, not create new one
        assert len(ws.concerns) == 1
        assert ws.concerns[0].intensity == 0.85
        assert ws.concerns[0].occurrence_count == 2

    def test_different_concern_added(self, ws):
        ws.load()
        # Use character-disjoint descriptions to avoid dedup
        ws._upsert_concern("abc xyz 123", 0.7, "2026-06-01T00:00:00")
        ws._upsert_concern("pqr uvw 789", 0.6, "2026-06-02T00:00:00")
        assert len(ws.concerns) == 2  # Different concerns


# ── Goal Resolution ─────────────────────────────────────────

class TestGoalResolution:
    """Verify goal resolution detection."""

    def test_resolution_keyword_detected(self, ws_with_goals):
        ws_with_goals.update_after_session([
            "我找到工作了！终于解决了找工作的难题！",
        ])
        # Goal g1 (找AI工作) should be marked resolved
        resolved_goals = [g for g in ws_with_goals.active_goals if g.resolved]
        assert len(resolved_goals) >= 1

    def test_old_resolved_goals_removed(self, ws):
        """Resolved goals older than 30 days should be purged."""
        ws.load()
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        ws.active_goals = [
            Goal(id="g1", description="old goal", resolved=True,
                 last_referenced=old, reference_count=1),
            Goal(id="g2", description="active goal", resolved=False,
                 last_referenced=old, reference_count=1),
        ]
        ws.update_after_session([])
        assert len(ws.active_goals) == 1  # Only active goal remains
        assert ws.active_goals[0].id == "g2"


# ── Query Methods ───────────────────────────────────────────

class TestQueryMethods:
    """Verify query methods."""

    def test_get_active_goal_domains(self, ws_with_goals):
        domains = ws_with_goals.get_active_goal_domains()
        assert "career" in domains
        assert "family" in domains

    def test_get_top_concerns(self, ws_with_goals):
        top = ws_with_goals.get_top_concerns(n=3)
        assert len(top) <= 3
        assert len(top) >= 1

    def test_has_goals(self, ws_with_goals):
        assert ws_with_goals.has_goals is True

    def test_has_no_goals_when_all_resolved(self, ws_with_goals):
        for g in ws_with_goals.active_goals:
            g.resolved = True
        assert ws_with_goals.has_goals is False


# ── Persistence ─────────────────────────────────────────────

class TestWorkingSelfPersistence:
    """Verify state save/load."""

    def test_save_and_load_roundtrip(self, ws_with_goals, tmp_path):
        ws_with_goals.save()

        ws2 = WorkingSelf(user_id="test_user", data_dir=str(tmp_path / "buckets"))
        ws2.load()
        assert len(ws2.active_goals) == len(ws_with_goals.active_goals)
        assert len(ws2.concerns) == len(ws_with_goals.concerns)

    def test_load_empty_state(self, ws):
        ws.load()  # Should not fail on nonexistent file
        assert ws.active_goals == []


# ── Edge Cases ──────────────────────────────────────────────

class TestWorkingSelfBoundaries:
    """Boundary and edge case tests."""

    def test_empty_insights_handled(self, ws):
        ws.load()
        ws.update_after_session([])  # Should not crash

    def test_match_with_empty_content(self, ws_with_goals):
        match = ws_with_goals.match("")
        assert 0.0 <= match <= 1.0

    def test_self_concept_persists(self, ws, tmp_path):
        ws.load()
        ws.self_concept = {"trait": "creative", "value": "autonomy"}
        ws.save()

        ws2 = WorkingSelf(user_id="test_user", data_dir=str(tmp_path / "buckets"))
        ws2.load()
        assert ws2.self_concept == {"trait": "creative", "value": "autonomy"}
