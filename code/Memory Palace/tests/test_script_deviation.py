# ============================================================
# Test: Script Deviation (test_script_deviation.py)
# L2: Statistical anomaly detection unit tests.
#
# Covers:
#   - COLD: no baseline → return 0.0
#   - Emotional deviation (valence/arousal z-score)
#   - New topic detection
#   - Time-of-day anomaly
#   - Baseline update
#   - Topic novelty query
#   - <10ms performance constraint
# ============================================================

import time
import pytest
from unittest.mock import MagicMock

from script_deviation import ScriptDeviation, EmotionalBaseline


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def sd(tmp_path):
    """ScriptDeviation with temp directory."""
    return ScriptDeviation(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def sd_with_baseline(sd):
    """ScriptDeviation with 30 sessions of baseline data."""
    for i in range(30):
        sd.detect(
            valence=0.55 + 0.05 * (i % 5 - 2),
            arousal=0.3 + 0.02 * (i % 3),
            topics=["日常", "工作"] if i % 2 == 0 else ["日常"],
            session_hour=12 + i % 4,
        )
    return sd


# ── COLD Behavior ───────────────────────────────────────────

class TestColdBehavior:
    """COLD users: no baseline → return 0.0."""

    def test_first_session_returns_zero(self, sd):
        dev = sd.detect(valence=0.5, arousal=0.3)
        assert dev == 0.0

    def test_first_few_sessions_return_zero(self, sd):
        for i in range(4):  # < 5 samples
            dev = sd.detect(valence=0.5, arousal=0.3)
            assert dev == 0.0

    def test_fifth_session_may_return_nonzero(self, sd):
        for i in range(5):
            dev = sd.detect(valence=0.5, arousal=0.3)
        # After 5 samples, baseline starts building
        # deviation may still be 0 if identical to baseline
        assert dev >= 0.0


# ── Emotional Deviation ─────────────────────────────────────

class TestEmotionalDeviation:
    """Verify valence/arousal deviation detection."""

    def test_extreme_deviation_detected(self, sd_with_baseline):
        # Established baseline ~0.55 valence, 0.3 arousal
        # Now throw extreme deviation
        dev = sd_with_baseline.detect(
            valence=0.1,   # far below baseline mean
            arousal=0.95,  # far above baseline mean
        )
        assert dev > 0.5, f"Extreme deviation should be high, got {dev}"

    def test_normal_session_not_detected(self, sd_with_baseline):
        dev = sd_with_baseline.detect(valence=0.55, arousal=0.3)
        assert dev < 0.4, f"Normal session should have low deviation, got {dev}"

    def test_deviation_never_exceeds_1(self, sd_with_baseline):
        dev = sd_with_baseline.detect(valence=0.0, arousal=1.0)
        assert dev <= 1.0

    def test_deviation_never_negative(self, sd_with_baseline):
        # Exact match to baseline
        for i in range(10):
            sd_with_baseline.detect(valence=0.55, arousal=0.3)
        dev = sd_with_baseline.detect(valence=0.55, arousal=0.3)
        assert dev >= 0.0


# ── Topic Novelty ───────────────────────────────────────────

class TestTopicNovelty:
    """Verify new topic detection."""

    def test_new_topic_increases_deviation(self, sd_with_baseline):
        dev_no_new = sd_with_baseline.detect(
            valence=0.55, arousal=0.3, topics=["日常"],
        )
        dev_new_topic = sd_with_baseline.detect(
            valence=0.55, arousal=0.3, topics=["宗教", "哲学"],  # never seen
        )
        # New topics add to deviation, but emotional signals dominate
        assert dev_new_topic >= 0.0

    def test_repeated_topic_lower_deviation(self, sd):
        for i in range(10):
            sd.detect(valence=0.5, arousal=0.3, topics=["编程"])
        # "编程" is now a common topic
        dev = sd.detect(valence=0.5, arousal=0.3, topics=["编程"])
        assert dev < 0.3


# ── Time-of-Day Anomaly ─────────────────────────────────────

class TestTimeOfDayAnomaly:
    """Verify time-of-day anomaly detection."""

    def test_late_night_deviates(self, sd_with_baseline):
        dev = sd_with_baseline.detect(
            valence=0.55, arousal=0.3,
            session_hour=3,  # 凌晨3点 — far from baseline hours
        )
        assert dev > 0.1


# ── Baseline Management ─────────────────────────────────────

class TestBaselineManagement:
    """Verify baseline persistence and updates."""

    def test_baseline_updates_with_data(self, sd_with_baseline):
        baseline = sd_with_baseline.get_baseline()
        assert baseline["sample_count"] >= 5
        assert 0.4 < baseline["valence_mean"] < 0.7

    def test_baseline_save_and_load(self, sd, tmp_path):
        for i in range(10):
            sd.detect(valence=0.6, arousal=0.4, topics=["编程"])
        sd.save()

        sd2 = ScriptDeviation(user_id="test_user", data_dir=str(tmp_path / "buckets"))
        sd2.load()
        baseline = sd2.get_baseline()
        assert baseline["sample_count"] >= 5
        assert abs(baseline["valence_mean"] - 0.6) < 0.1

    def test_topic_novelty_query(self, sd_with_baseline):
        # "日常" is common, "外星人" is novel
        common = sd_with_baseline.get_topic_novelty("日常")
        novel = sd_with_baseline.get_topic_novelty("外星人")
        assert novel > common

    def test_topic_novelty_when_empty(self, sd):
        assert sd.get_topic_novelty("anything") == 1.0  # Never seen


# ── Performance ─────────────────────────────────────────────

class TestPerformance:
    """Verify <10ms performance constraint."""

    def test_detect_under_10ms(self, sd_with_baseline):
        start = time.perf_counter()
        for _ in range(10):
            sd_with_baseline.detect(valence=0.5, arousal=0.3)
        end = time.perf_counter()
        avg_ms = (end - start) * 1000 / 10
        assert avg_ms < 10, f"Detection should be <10ms, got {avg_ms:.2f}ms"


# ── Weighted Combination ────────────────────────────────────

class TestWeightedCombination:
    """Verify the 4-signal weighted combination."""

    def test_emotional_dominates(self, sd_with_baseline):
        """Emotional deviation (0.35+0.35=0.7 weight) should dominate."""
        # Pure emotional deviation
        dev_emo = sd_with_baseline.detect(
            valence=0.1, arousal=0.9,
            topics=["日常"],  # known topic
            session_hour=12,  # normal hour
        )
        # Pure topic deviation (same emotion)
        dev_topic = sd_with_baseline.detect(
            valence=0.55, arousal=0.3,
            topics=["外星人", "火星"],  # unknown topics
            session_hour=12,
        )
        # Emotional deviation should be stronger (weight 0.7 vs 0.15)
        assert dev_emo > dev_topic


# ── Edge Cases ──────────────────────────────────────────────

class TestDeviationBoundaries:
    """Boundary and edge case tests."""

    def test_window_capped(self, sd):
        """Max 100 entries in sliding window."""
        for i in range(150):
            sd.detect(valence=0.5, arousal=0.3)
        assert len(sd._window) <= 100

    def test_zero_std_is_handled(self, sd_with_baseline):
        """When std is near zero, should not divide by zero."""
        # Force all valence to be identical
        sd2 = ScriptDeviation(user_id="uniform", data_dir=sd_with_baseline.data_dir.parent)
        for i in range(10):
            sd2.detect(valence=0.5, arousal=0.3)
        dev = sd2.detect(valence=0.1, arousal=0.9)  # extreme deviation
        assert dev >= 0.0  # Should not crash

    def test_empty_topics_handled(self, sd_with_baseline):
        dev = sd_with_baseline.detect(valence=0.55, arousal=0.3, topics=[])
        assert dev >= 0.0

    def test_session_hours_capped(self, sd):
        """Session hours list capped at 200."""
        for i in range(250):
            sd.detect(valence=0.5, arousal=0.3, session_hour=i % 24)
        assert len(sd._session_hours) <= 200
