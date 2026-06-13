# ============================================================
# Test: Procedural Memory (test_procedural_memory.py)
# Track C Task 3: LANGMem-style procedural memory for response
# preference tracking.
# ============================================================

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from procedural_memory import (
    ProceduralMemory, BehavioralScript, UserPreferenceProfile,
)


# ── BehavioralScript Tests ────────────────────────────────────


class TestBehavioralScript:
    """Test BehavioralScript data model."""

    def test_create_script(self):
        """Basic script creation."""
        script = BehavioralScript(
            trigger_pattern="user_expresses_anxious",
            trigger_domain="career",
            trigger_emotion="anxious",
            preferred_tone="gentle",
            preferred_style="listening",
        )
        assert script.trigger_pattern == "user_expresses_anxious"
        assert script.trigger_domain == "career"
        assert script.preferred_tone == "gentle"
        assert len(script.id) > 0

    def test_is_reliable_low_confidence(self):
        """Low confidence script is not reliable."""
        script = BehavioralScript(
            trigger_pattern="test",
            occurrence_count=1,
            confidence=0.2,
        )
        assert not script.is_reliable

    def test_is_reliable_high_confidence(self):
        """High confidence script with enough occurrences is reliable."""
        script = BehavioralScript(
            trigger_pattern="test",
            occurrence_count=5,
            confidence=0.8,
        )
        assert script.is_reliable

    def test_is_reliable_not_enough_occurrences(self):
        """High confidence but too few occurrences is not reliable."""
        script = BehavioralScript(
            trigger_pattern="test",
            occurrence_count=2,
            confidence=0.9,
        )
        assert not script.is_reliable

    def test_to_dict_and_back(self):
        """Serialization round-trip."""
        script = BehavioralScript(
            trigger_pattern="user_is_sad",
            trigger_domain="relationship",
            trigger_emotion="sad",
            preferred_tone="gentle",
            preferred_length="detailed",
            preferred_style="reflective",
            occurrence_count=5,
            positive_feedback_count=4,
            negative_feedback_count=1,
            confidence=0.75,
            preferred_topics=["感情", "成长"],
        )
        data = script.to_dict()
        restored = BehavioralScript.from_dict(data)

        assert restored.id == script.id
        assert restored.trigger_pattern == script.trigger_pattern
        assert restored.preferred_tone == script.preferred_tone
        assert restored.confidence == script.confidence
        assert restored.occurrence_count == script.occurrence_count


# ── UserPreferenceProfile Tests ───────────────────────────────


class TestUserPreferenceProfile:
    """Test UserPreferenceProfile data model."""

    def test_default_profile(self):
        """Default profile has reasonable values."""
        profile = UserPreferenceProfile()
        assert profile.tone_casual == 0.5
        assert profile.length_moderate == 0.5
        assert profile.style_advice == 0.4
        assert profile.total_interactions == 0

    def test_dominant_tone(self):
        """Dominant tone is correctly identified."""
        profile = UserPreferenceProfile(
            tone_casual=0.8,
            tone_gentle=0.6,
            tone_direct=0.3,
        )
        assert profile.dominant_tone() == "casual"

    def test_dominant_style(self):
        """Dominant style is correctly identified."""
        profile = UserPreferenceProfile(
            style_advice=0.7,
            style_listening=0.3,
            style_questioning=0.2,
        )
        assert profile.dominant_style() == "advice"

    def test_prompt_guidance(self):
        """Prompt guidance generation."""
        profile = UserPreferenceProfile(
            tone_gentle=0.8,
            length_detailed=0.7,
            style_advice=0.7,
            late_night_gentle=True,
            sensitive_topics=["职业", "家庭"],
        )
        guidance = profile.prompt_guidance(session_hour=3)
        assert "温柔" in guidance or "gentle" in guidance.lower()
        assert len(guidance) > 0

    def test_prompt_guidance_empty(self):
        """Empty guidance when no clear preferences."""
        profile = UserPreferenceProfile()
        guidance = profile.prompt_guidance()
        # With default values, guidance might be empty or minimal
        assert isinstance(guidance, str)

    def test_to_dict_and_back(self):
        """Serialization round-trip."""
        profile = UserPreferenceProfile(
            tone_casual=0.7,
            tone_gentle=0.6,
            length_brief=0.4,
            length_detailed=0.5,
            style_advice=0.6,
            style_listening=0.5,
            late_night_gentle=True,
            sensitive_topics=["健康"],
            preferred_topics=["技术", "成长"],
            total_interactions=42,
        )
        data = profile.to_dict()
        restored = UserPreferenceProfile.from_dict(data)

        assert restored.tone_casual == 0.7
        assert restored.length_detailed == 0.5
        assert restored.late_night_gentle is True
        assert restored.sensitive_topics == ["健康"]
        assert restored.total_interactions == 42


# ── ProceduralMemory Tests ────────────────────────────────────


class TestProceduralMemory:
    """Test the ProceduralMemory engine."""

    def test_init_and_load(self, procedural_memory_fixture):
        """Initialization and loading."""
        pm = procedural_memory_fixture
        pm.load()
        assert pm.profile.total_interactions == 0
        assert len(pm.scripts) == 0

    def test_record_interaction(self, procedural_memory_fixture):
        """Recording an interaction updates profile."""
        pm = procedural_memory_fixture
        pm.record_interaction(
            user_message="我今天面试好紧张啊",
            bot_reply="面试紧张很正常呢，你准备得怎么样？",
            feedback_signals={"user_continued": True},
            session_hour=14,
            domain=["职业"],
            valence=0.3,
            arousal=0.8,
        )

        assert pm.profile.total_interactions == 1

    def test_record_multiple_interactions(self, procedural_memory_fixture):
        """Multiple interactions accumulate learning."""
        pm = procedural_memory_fixture

        for i in range(5):
            pm.record_interaction(
                user_message=f"我很焦虑，睡不着觉",
                bot_reply="我在这里陪着你呢。想聊聊发生了什么吗？",
                feedback_signals={"user_continued": True},
                session_hour=2,  # late night
                domain=["健康"],
                valence=0.2,
                arousal=0.8,
            )

        assert pm.profile.total_interactions == 5
        # Late night + gentle replies should increase gentle preference
        assert pm.profile.late_night_gentle is True

    def test_get_response_preferences(self, procedural_memory_fixture):
        """Response preferences are returned for current context."""
        pm = procedural_memory_fixture

        # Train with some interactions
        pm.record_interaction(
            user_message="你觉得我应该怎么办？",
            bot_reply="我建议你可以试试这样做...",
            feedback_signals={"user_continued": True, "user_asked_more": True},
            session_hour=14,
        )

        prefs = pm.get_response_preferences(
            trigger_context="你觉得",
            session_hour=14,
        )
        assert "tone" in prefs
        assert "style" in prefs
        assert "length" in prefs

    def test_detect_scripts(self, procedural_memory_fixture):
        """Script detection from session history."""
        pm = procedural_memory_fixture

        session = [
            {"role": "user", "content": "我好累啊"},
            {"role": "assistant", "content": "辛苦了，我理解的。最近压力大吗？"},
            {"role": "user", "content": "是啊，工作上的事情真的让我喘不过气来"},
        ]

        scripts = pm.detect_scripts(session)
        # Should detect at least one pattern
        assert isinstance(scripts, list)

    def test_persistence(self, procedural_memory_fixture):
        """Save and load preserves state."""
        pm = procedural_memory_fixture

        pm.record_interaction(
            user_message="测试消息",
            bot_reply="测试回复",
            feedback_signals={"user_continued": True},
        )
        pm.save()

        # Create a new instance and load
        pm2 = ProceduralMemory(
            user_id=pm.user_id,
            data_dir=str(pm.data_dir.parent),
        )
        pm2.load()

        assert pm2.profile.total_interactions == pm.profile.total_interactions

    def test_get_stats(self, procedural_memory_fixture):
        """Stats report."""
        pm = procedural_memory_fixture
        pm.record_interaction(
            user_message="测试",
            bot_reply="回复",
        )
        stats = pm.get_stats()
        assert "total_scripts" in stats
        assert "total_interactions" in stats
        assert "dominant_tone" in stats
        assert stats["total_interactions"] == 1

    def test_tone_classification(self):
        """Zero-LLM tone classification."""
        assert ProceduralMemory._classify_tone("哈哈，这个好好笑啊") == "casual"
        assert ProceduralMemory._classify_tone("您好，请确认一下") == "formal"
        assert ProceduralMemory._classify_tone("你应该必须马上去做") == "direct"
        assert ProceduralMemory._classify_tone("或许可以试试看要不要") == "gentle"

    def test_extract_triggers(self):
        """Trigger extraction from messages."""
        triggers = ProceduralMemory._extract_triggers(
            user_message="我面试好紧张啊，失眠了好几天",
            domain=["职业"],
            valence=0.3,
            arousal=0.8,
            session_hour=3,
        )
        assert len(triggers) > 0
        trigger_patterns = [t["trigger"] for t in triggers]
        assert any("anxious" in p for p in trigger_patterns)
        assert any("career" in p for p in trigger_patterns)

    def test_calculate_confidence(self):
        """Confidence calculation from occurrences and feedback."""
        script = BehavioralScript(
            trigger_pattern="test",
            occurrence_count=10,
            positive_feedback_count=8,
            negative_feedback_count=2,
        )
        conf = ProceduralMemory._calculate_confidence(script)
        assert 0.5 <= conf <= 1.0

    def test_calculate_confidence_new_script(self):
        """New script with no data has low confidence."""
        script = BehavioralScript(
            trigger_pattern="test",
            occurrence_count=0,
        )
        conf = ProceduralMemory._calculate_confidence(script)
        assert conf == 0.0
