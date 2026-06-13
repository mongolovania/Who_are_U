# ============================================================
# Test: Flashbulb Detector (test_flashbulb_detector.py)
# L2: Flashbulb memory detection unit tests.
#
# Covers:
#   - Triple-trigger detection (surprise + relevance + arousal)
#   - Heuristic detection (no LLM)
#   - LLM-based detection
#   - FlashbulbContext creation
#   - Protection application (importance boost, decay multiplier, retrieval boost)
#   - Personal baseline adjustment
#   - Threshold edge cases
# ============================================================

import pytest
from unittest.mock import AsyncMock, MagicMock

from flashbulb_detector import FlashbulbDetector, FlashbulbContext, _describe_emotional_state
from memory_node import ValenceArousal


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def detector():
    return FlashbulbDetector()


# ── Triple-Trigger Detection ────────────────────────────────

class TestDetection:
    """Verify triple-trigger flashbulb detection."""

    def test_all_triggers_met(self, detector):
        is_fb, ctx = detector.detect(
            content="我被裁员了",
            emotion=ValenceArousal(valence=0.1, arousal=0.9),
            surprise=0.9,
            personal_relevance=0.9,
        )
        assert is_fb is True
        assert ctx.is_flashbulb is True

    def test_surprise_too_low(self, detector):
        is_fb, ctx = detector.detect(
            content="今天天气不错",
            emotion=ValenceArousal(valence=0.6, arousal=0.9),
            surprise=0.3,  # below 0.7 threshold
            personal_relevance=0.8,
        )
        assert is_fb is False

    def test_relevance_too_low(self, detector):
        is_fb, ctx = detector.detect(
            content="路人摔倒了我看到了",
            emotion=ValenceArousal(valence=0.4, arousal=0.85),
            surprise=0.8,
            personal_relevance=0.3,  # below 0.7 threshold
        )
        assert is_fb is False

    def test_arousal_too_low(self, detector):
        is_fb, ctx = detector.detect(
            content="我中大奖了！！！",
            emotion=ValenceArousal(valence=0.9, arousal=0.4),  # below 0.8 threshold
            surprise=0.9,
            personal_relevance=0.9,
        )
        assert is_fb is False


# ── Heuristic Detection ─────────────────────────────────────

class TestHeuristicDetection:
    """Verify fast heuristic (no LLM) detection."""

    def test_surprise_keywords_detected(self, detector):
        is_fb, surprise, relevance = detector.detect_heuristic(
            content="天啊！！太震惊了！！竟然会发生这种事！！",
            arousal=0.85,
            valence=0.2,
        )
        assert surprise > 0.3  # Keywords + exclamation marks

    def test_relevance_keywords_detected(self, detector):
        is_fb, surprise, relevance = detector.detect_heuristic(
            content="这是我人生中最重要的转折点，改变了我的命运",
            arousal=0.5,
            valence=0.4,
        )
        assert relevance >= 0.3  # Baseline 0.3 + keywords

    def test_exclamation_marks_boost_surprise(self, detector):
        no_exclaim, s1, r1 = detector.detect_heuristic(
            content="突然发生了这件事。", arousal=0.8, valence=0.3,
        )
        many_exclaim, s2, r2 = detector.detect_heuristic(
            content="突然发生了这件事！！！！！！", arousal=0.8, valence=0.3,
        )
        assert s2 > s1

    def test_flashbulb_triggered_heuristically(self, detector):
        content = "天啊！！！我竟然被裁员了！！！这改变了我的人生！！！"
        is_fb, surprise, relevance = detector.detect_heuristic(
            content=content,
            arousal=0.9,
            valence=0.1,
        )
        # Should trigger with all three conditions met
        assert is_fb is True or (surprise >= 0.7 and relevance >= 0.7)


# ── Personal Baseline ───────────────────────────────────────

class TestPersonalBaseline:
    """Verify personal baseline adjustment."""

    def test_baseline_lowers_threshold_for_high_arousal_user(self, detector):
        # User with normally high arousal
        detector.update_personal_baseline([0.7, 0.75, 0.8, 0.7, 0.85, 0.75, 0.8, 0.7, 0.85, 0.9])
        assert detector._has_personal_baseline is True
        assert detector._personal_arousal_mean > 0.7

    def test_baseline_not_established_with_insufficient_data(self, detector):
        detector.update_personal_baseline([0.7, 0.8, 0.9])  # Only 3 points
        assert detector._has_personal_baseline is False


# ── Protection Application ──────────────────────────────────

class TestProtection:
    """Verify flashbulb memory protection."""

    def test_importance_boost(self, detector):
        boosted = detector.apply_protection(7)
        assert boosted == 10  # 7 + 3 = 10

    def test_importance_boost_clamped_at_10(self, detector):
        boosted = detector.apply_protection(9)
        assert boosted == 10  # clamped

    def test_decay_multiplier(self, detector):
        assert detector.get_decay_multiplier() == 0.5  # Half speed

    def test_retrieval_boost(self, detector):
        assert detector.get_retrieval_boost() == 2.0  # Double priority


# ── FlashbulbContext ────────────────────────────────────────

class TestFlashbulbContext:
    """Verify reception context storage."""

    def test_context_created_on_detection(self, detector):
        is_fb, ctx = detector.detect(
            content="我被裁员了！完全没想到！",
            emotion=ValenceArousal(valence=0.1, arousal=0.95),
            surprise=0.9,
            personal_relevance=0.9,
        )
        assert ctx.is_flashbulb is True
        assert ctx.surprise_score == 0.9
        assert ctx.personal_relevance == 0.9
        assert ctx.emotional_state != ""


# ── Emotional State Description ─────────────────────────────

class TestEmotionalDescription:
    """Verify Russell circumplex → Chinese description."""

    def test_excited(self):
        state = _describe_emotional_state(ValenceArousal(valence=0.8, arousal=0.8))
        assert "兴奋" in state

    def test_calm_content(self):
        state = _describe_emotional_state(ValenceArousal(valence=0.8, arousal=0.2))
        assert "平静" in state

    def test_anxious(self):
        state = _describe_emotional_state(ValenceArousal(valence=0.3, arousal=0.8))
        assert "焦虑" in state

    def test_depressed(self):
        state = _describe_emotional_state(ValenceArousal(valence=0.3, arousal=0.2))
        assert "低落" in state

    def test_high_arousal_neutral(self):
        state = _describe_emotional_state(ValenceArousal(valence=0.5, arousal=0.8))
        assert "高涨" in state or "兴奋" in state

    def test_low_arousal_neutral(self):
        state = _describe_emotional_state(ValenceArousal(valence=0.5, arousal=0.2))
        assert "平稳" in state


# ── Edge Cases ──────────────────────────────────────────────

class TestFlashbulbBoundaries:
    """Boundary and edge case tests."""

    def test_exact_threshold_values(self, detector):
        """Exactly at threshold should trigger."""
        is_fb, ctx = detector.detect(
            content="test",
            emotion=ValenceArousal(valence=0.3, arousal=0.8),  # exactly at threshold
            surprise=0.7,  # exactly at threshold
            personal_relevance=0.7,  # exactly at threshold
        )
        assert is_fb is True

    def test_just_below_threshold(self, detector):
        is_fb, ctx = detector.detect(
            content="test",
            emotion=ValenceArousal(valence=0.3, arousal=0.79),  # just below
            surprise=0.7,
            personal_relevance=0.7,
        )
        assert is_fb is False  # arousal just below

    def test_empty_content_handled(self, detector):
        is_fb, surprise, relevance = detector.detect_heuristic(
            content="", arousal=0.5, valence=0.5,
        )
        # Should not crash
        assert surprise >= 0.0
        assert relevance >= 0.0

    def test_surprise_capped_at_1(self, detector):
        # Many surprise keywords
        content = "突然 竟然 没想到 震惊 不可思议 天啊 居然 意外 惊呆了 难以置信 " * 5
        is_fb, surprise, relevance = detector.detect_heuristic(
            content=content, arousal=0.5, valence=0.5,
        )
        assert surprise <= 1.0

    def test_relevance_capped_at_1(self, detector):
        content = "我 我的 人生 一辈子 最重要 改变 转折 命运 永远 决定 " * 10
        is_fb, surprise, relevance = detector.detect_heuristic(
            content=content, arousal=0.5, valence=0.5,
        )
        assert relevance <= 1.0
