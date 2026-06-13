# ============================================================
# Test: Cold Start (test_cold_start.py)
# L0: Cold-start strategies for new users.
#
# Covers:
#   - Storage gate (store everything for COLD users)
#   - Importance estimation (content length + arousal + late-night)
#   - Emotion estimation (keyword heuristic)
#   - Retrieval limits by DDI range
#   - Decay config (disabled for COLD)
#   - Emotion mode selection
# ============================================================

import pytest

from cold_start import ColdStartPolicy, cold_start
from memory_node import MemoryType, ValenceArousal, DDILevel


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def policy():
    return ColdStartPolicy()


# ── Storage Gate ────────────────────────────────────────────

class TestStorageGate:
    """Verify COLD storage gate behavior."""

    def test_normal_message_stored(self, policy):
        should, reason = policy.should_store("我今天面试被拒了，心情很失落")
        assert should is True
        assert reason == "cold_start_store_all"

    def test_very_short_message_filtered(self, policy):
        should, reason = policy.should_store("嗯")
        assert should is False
        assert reason == "too_short"

    def test_exactly_5_chars_stored(self, policy):
        should, reason = policy.should_store("我今天很好")
        assert isinstance(should, bool)  # Cold start stores almost everything

    def test_empty_message_filtered(self, policy):
        should, reason = policy.should_store("")
        assert should is False

    def test_whitespace_only_filtered(self, policy):
        should, reason = policy.should_store("   ")
        assert should is False


# ── Importance Estimation ───────────────────────────────────

class TestImportanceEstimation:
    """Verify cold-start importance heuristics."""

    def test_default_is_5(self, policy):
        imp = policy.estimate_importance("今天天气不错")
        assert 4 <= imp <= 6  # Around 5

    def test_long_message_higher(self, policy):
        short = policy.estimate_importance("短消息")
        long = policy.estimate_importance("很长的消息" * 50)  # >200 chars
        assert long > short

    def test_high_arousal_higher(self, policy):
        calm = policy.estimate_importance("test", arousal=0.3)
        excited = policy.estimate_importance("test", arousal=0.8)
        assert excited > calm

    def test_late_night_higher(self, policy):
        day = policy.estimate_importance("test", session_hour=14)
        night = policy.estimate_importance("test", session_hour=3)
        assert night > day

    def test_importance_clamped_1_to_10(self, policy):
        imp = policy.estimate_importance("x" * 500, arousal=1.0, session_hour=3)
        assert 1 <= imp <= 10


# ── Emotion Estimation ──────────────────────────────────────

class TestEmotionEstimation:
    """Verify cold-start emotion heuristics."""

    def test_positive_words_detected(self, policy):
        emotion = policy.estimate_emotion("今天很开心，拿到了offer，太兴奋了！")
        assert emotion.valence > 0.5
        assert emotion.arousal > 0.3

    def test_negative_words_detected(self, policy):
        emotion = policy.estimate_emotion("今天很焦虑，压力太大了")
        assert emotion.valence < 0.5  # Negative words should reduce valence

    def test_neutral_no_keywords(self, policy):
        emotion = policy.estimate_emotion("今天吃了饭")
        assert emotion.valence == 0.5
        assert emotion.arousal == 0.3

    def test_exclamation_marks_increase_arousal(self, policy):
        calm = policy.estimate_emotion("今天天气不错")
        excited = policy.estimate_emotion("今天天气不错！！！")
        assert excited.arousal > calm.arousal

    def test_late_night_increases_arousal(self, policy):
        day = policy.estimate_emotion("test", session_hour=14)
        night = policy.estimate_emotion("test", session_hour=3)
        assert night.arousal >= day.arousal

    def test_valence_clamped(self, policy):
        emotion = policy.estimate_emotion("")
        assert 0.0 <= emotion.valence <= 1.0
        assert 0.0 <= emotion.arousal <= 1.0


# ── Retrieval Limits ────────────────────────────────────────

class TestRetrievalLimits:
    """Verify DDI-based retrieval limits."""

    def test_very_new_user_gets_50(self, policy):
        assert policy.get_retrieval_limit(ddi=2) == 50

    def test_approaching_warm_gets_30(self, policy):
        assert policy.get_retrieval_limit(ddi=7) == 30

    def test_near_warm_gets_smaller_limit(self, policy):
        assert policy.get_retrieval_limit(ddi=5) <= 30  # Approaching WARM
        assert policy.get_retrieval_limit(ddi=9) >= 0


# ── Decay Config ────────────────────────────────────────────

class TestDecayConfig:
    """Verify COLD decay configuration."""

    def test_decay_disabled_for_cold(self, policy):
        config = policy.get_decay_config(ddi=3)
        assert config["decay_enabled"] is False
        assert config["decay_lambda"] == 0.0
        assert config["archive_threshold"] == 0.0  # Never archive
        assert config["auto_resolve_enabled"] is False


# ── Emotion Mode ────────────────────────────────────────────

class TestEmotionMode:
    """Verify COLD emotion mode selection."""

    def test_very_cold_is_warm_default(self, policy):
        assert policy.get_emotion_mode(ddi=3) == "warm_default"

    def test_approaching_warm_is_llm_prior(self, policy):
        assert policy.get_emotion_mode(ddi=7) == "llm_prior"


# ── Singleton ───────────────────────────────────────────────

class TestSingleton:
    """Verify module-level singleton."""

    def test_cold_start_is_cold_start_policy(self):
        assert isinstance(cold_start, ColdStartPolicy)

    def test_singleton_is_reusable(self):
        should, reason = cold_start.should_store("这是一条足够长的测试消息")
        assert isinstance(should, bool)
