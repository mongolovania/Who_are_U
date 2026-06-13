# ============================================================
# Module: Procedural Memory (procedural_memory.py)
# Track C Task 3: LANGMem-style procedural memory for tracking
# user response preferences and behavioral scripts.
#
# Theoretical foundation:
#   1. LANGMem (2025) — "Long-Term Memory for AI Agents."
#      Procedural memory tracks interaction patterns, response
#      preferences, and behavioral scripts learned from repeated
#      user interactions.
#   2. Schank & Abelson (1977) — Scripts, Plans, Goals and
#      Understanding. Scripts are stereotyped sequences of
#      actions that define well-known situations.
#   3. Anderson (1983) — The Architecture of Cognition.
#      Procedural memory as production rules: IF condition
#      THEN action. Acquired through repeated practice.
#   4. Conway (2005) — Self-Memory System. Working Self
#      modulates what procedural knowledge is active.
#
# Implementation:
#   - Behavioral scripts: trigger → preferred response style
#   - User preference aggregation across sessions
#   - Zero-LLM preference extraction from interaction signals
#   - Persistence across sessions via JSON
# ============================================================

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory_palace.procedural")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class BehavioralScript:
    """
    A learned behavioral script (LANGMem / Schank style).

    IF trigger_context matches → THEN preferred response style.
    Learned from repeated user interactions.
    """
    trigger_pattern: str = ""     # What user says/does that triggers this script
    id: str = ""
    trigger_domain: str = ""      # Domain context (career, relationship, etc.)
    trigger_emotion: str = ""      # Associated emotion (anxious, happy, etc.)
    trigger_time_pattern: str = ""  # Time-of-day pattern (late_night, morning, etc.)

    # Preferred response style
    preferred_tone: str = ""       # casual | formal | gentle | direct | humorous
    preferred_length: str = ""     # brief | moderate | detailed
    preferred_style: str = ""      # advice | listening | questioning | reflective
    preferred_topics: list[str] = field(default_factory=list)

    # Learning stats
    occurrence_count: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    confidence: float = 0.0        # 0-1, how confident we are in this script
    last_triggered: str = ""
    created_at: str = ""

    # Example interactions that formed this script
    example_interactions: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"script_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_reliable(self) -> bool:
        """A script is reliable if confidence >= 0.6 and occurrence >= 3."""
        return self.confidence >= 0.6 and self.occurrence_count >= 3

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger_pattern": self.trigger_pattern,
            "trigger_domain": self.trigger_domain,
            "trigger_emotion": self.trigger_emotion,
            "trigger_time_pattern": self.trigger_time_pattern,
            "preferred_tone": self.preferred_tone,
            "preferred_length": self.preferred_length,
            "preferred_style": self.preferred_style,
            "preferred_topics": self.preferred_topics,
            "occurrence_count": self.occurrence_count,
            "positive_feedback_count": self.positive_feedback_count,
            "negative_feedback_count": self.negative_feedback_count,
            "confidence": self.confidence,
            "last_triggered": self.last_triggered,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BehavioralScript:
        return cls(
            id=data.get("id", ""),
            trigger_pattern=data.get("trigger_pattern", ""),
            trigger_domain=data.get("trigger_domain", ""),
            trigger_emotion=data.get("trigger_emotion", ""),
            trigger_time_pattern=data.get("trigger_time_pattern", ""),
            preferred_tone=data.get("preferred_tone", ""),
            preferred_length=data.get("preferred_length", ""),
            preferred_style=data.get("preferred_style", ""),
            preferred_topics=data.get("preferred_topics", []),
            occurrence_count=data.get("occurrence_count", 0),
            positive_feedback_count=data.get("positive_feedback_count", 0),
            negative_feedback_count=data.get("negative_feedback_count", 0),
            confidence=data.get("confidence", 0.0),
            last_triggered=data.get("last_triggered", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class UserPreferenceProfile:
    """
    Aggregated user preference profile learned across sessions.

    Summarizes tone, length, style, and topic preferences
    for use in system prompt customization.
    """
    # Tone preferences (0-1 scores)
    tone_casual: float = 0.5
    tone_formal: float = 0.5
    tone_gentle: float = 0.5
    tone_direct: float = 0.5
    tone_humorous: float = 0.3

    # Length preference
    length_brief: float = 0.3
    length_moderate: float = 0.5
    length_detailed: float = 0.2

    # Style preference
    style_advice: float = 0.4
    style_listening: float = 0.4
    style_questioning: float = 0.2
    style_reflective: float = 0.3

    # Time-based adjustments
    late_night_gentle: bool = True
    morning_direct: bool = False

    # Topic sensitivity (topics where user needs special handling)
    sensitive_topics: list[str] = field(default_factory=list)
    preferred_topics: list[str] = field(default_factory=list)

    # Update count
    total_interactions: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "tone_casual": self.tone_casual,
            "tone_formal": self.tone_formal,
            "tone_gentle": self.tone_gentle,
            "tone_direct": self.tone_direct,
            "tone_humorous": self.tone_humorous,
            "length_brief": self.length_brief,
            "length_moderate": self.length_moderate,
            "length_detailed": self.length_detailed,
            "style_advice": self.style_advice,
            "style_listening": self.style_listening,
            "style_questioning": self.style_questioning,
            "style_reflective": self.style_reflective,
            "late_night_gentle": self.late_night_gentle,
            "morning_direct": self.morning_direct,
            "sensitive_topics": self.sensitive_topics,
            "preferred_topics": self.preferred_topics,
            "total_interactions": self.total_interactions,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserPreferenceProfile:
        return cls(**{k: data.get(k, (
            False if k in ("late_night_gentle", "morning_direct")
            else ([] if k in ("sensitive_topics", "preferred_topics")
                  else ("" if k == "last_updated" else 0))
        )) for k in [
            "tone_casual", "tone_formal", "tone_gentle", "tone_direct",
            "tone_humorous", "length_brief", "length_moderate", "length_detailed",
            "style_advice", "style_listening", "style_questioning", "style_reflective",
            "late_night_gentle", "morning_direct", "sensitive_topics",
            "preferred_topics", "total_interactions", "last_updated",
        ]})

    def dominant_tone(self) -> str:
        """Return the dominant tone preference."""
        tones = {
            "casual": self.tone_casual,
            "formal": self.tone_formal,
            "gentle": self.tone_gentle,
            "direct": self.tone_direct,
            "humorous": self.tone_humorous,
        }
        return max(tones, key=tones.get)

    def dominant_style(self) -> str:
        """Return the dominant style preference."""
        styles = {
            "advice": self.style_advice,
            "listening": self.style_listening,
            "questioning": self.style_questioning,
            "reflective": self.style_reflective,
        }
        return max(styles, key=styles.get)

    def prompt_guidance(self, session_hour: int = 12) -> str:
        """
        Generate prompt guidance for the LLM based on learned preferences.

        Returns a string to inject into the system prompt.
        """
        parts = []

        # Tone guidance
        tone = self.dominant_tone()
        if tone == "gentle" or (self.late_night_gentle and session_hour <= 5):
            parts.append("语气要温柔")
        elif tone == "direct":
            parts.append("说话直接一些")
        elif tone == "casual":
            parts.append("保持轻松随意的语气")
        elif tone == "humorous":
            parts.append("可以适当幽默")

        # Length guidance
        if self.length_brief > self.length_detailed:
            parts.append("回复简洁")
        elif self.length_detailed > self.length_brief:
            parts.append("可以多说一些")

        # Style guidance
        style = self.dominant_style()
        if style == "advice":
            parts.append("用户喜欢直接的建议")
        elif style == "listening":
            parts.append("多听少说，以陪伴为主")
        elif style == "questioning":
            parts.append("多问问题帮助用户思考")

        # Sensitive topics
        if self.sensitive_topics:
            parts.append(f"对{','.join(self.sensitive_topics[:3])}话题要特别温和")

        if not parts:
            return ""
        return "【用户偏好】" + "；".join(parts) + "。"


# ═══════════════════════════════════════════════════════════════
# Procedural Memory Engine
# ═══════════════════════════════════════════════════════════════


class ProceduralMemory:
    """
    LANGMem-style procedural memory for response preferences.

    Learns from each user interaction:
      - What triggers certain response preferences
      - What tone/style/length the user prefers
      - What topics are sensitive
      - What time-of-day adjustments are needed
    """

    def __init__(
        self,
        user_id: str = "",
        data_dir: str = "./buckets",
    ):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        if user_id:
            self.data_dir = self.data_dir / user_id
        os.makedirs(self.data_dir, exist_ok=True)

        self.scripts: dict[str, BehavioralScript] = {}
        self.profile = UserPreferenceProfile()
        self._loaded = False

    # ── Persistence ──────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.data_dir / "procedural_memory.json"

    def load(self):
        """Load procedural memory from disk."""
        if self._loaded:
            return
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.scripts = {
                    sid: BehavioralScript.from_dict(sd)
                    for sid, sd in data.get("scripts", {}).items()
                }
                self.profile = UserPreferenceProfile.from_dict(
                    data.get("profile", {})
                )
            except Exception as e:
                logger.warning(f"Failed to load procedural memory: {e}")
        self._loaded = True

    def save(self):
        """Persist procedural memory to disk."""
        path = self._state_path()
        path.write_text(json.dumps({
            "scripts": {sid: s.to_dict() for sid, s in self.scripts.items()},
            "profile": self.profile.to_dict(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Interaction recording ────────────────────────────────

    def record_interaction(
        self,
        user_message: str,
        bot_reply: str,
        feedback_signals: dict | None = None,
        session_hour: int = 12,
        domain: list[str] | None = None,
        valence: float = 0.5,
        arousal: float = 0.3,
    ):
        """
        Record a user-bot interaction to learn preferences.

        Called after each chat turn in the async pipeline.

        Args:
            user_message: what the user said
            bot_reply: what the bot replied
            feedback_signals: {
                "user_continued": bool,       # Did user keep talking?
                "user_contradicted": bool,     # Did user disagree?
                "user_asked_more": bool,       # Did user ask for more detail?
                "user_changed_topic": bool,    # Did user abruptly change topic?
                "response_length_ok": bool,    # Was the length appropriate?
            }
            session_hour: hour of day (0-23)
            domain: active domains for this interaction
            valence: emotional valence
            arousal: emotional arousal
        """
        self.load()
        feedback = feedback_signals or {}

        # Update profile
        self.profile.total_interactions += 1

        # Learn tone preferences from feedback
        self._learn_tone_from_interaction(user_message, bot_reply, feedback)
        self._learn_length_from_interaction(user_message, bot_reply, feedback)
        self._learn_style_from_interaction(user_message, feedback)
        self._learn_time_patterns(session_hour, feedback)
        self._learn_topic_sensitivity(user_message, domain, valence, arousal)

        # Detect and update behavioral scripts
        self._detect_and_update_scripts(
            user_message, bot_reply, feedback, session_hour, domain, valence, arousal
        )

        self.profile.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()

    # ── Preference retrieval ─────────────────────────────────

    def get_response_preferences(
        self,
        trigger_context: str = "",
        session_hour: int = 12,
        domain: list[str] | None = None,
    ) -> dict:
        """
        Get active response preferences for the current context.

        Returns a dict suitable for prompt customization.
        """
        self.load()

        result = {
            "tone": self.profile.dominant_tone(),
            "length": (
                "brief" if self.profile.length_brief > self.profile.length_detailed
                else "detailed" if self.profile.length_detailed > self.profile.length_brief
                else "moderate"
            ),
            "style": self.profile.dominant_style(),
            "guidance": self.profile.prompt_guidance(session_hour),
        }

        # Check if any scripts match the current context
        matching_scripts = self._find_matching_scripts(
            trigger_context, session_hour, domain
        )

        if matching_scripts:
            # Use the most confident matching script
            best = matching_scripts[0]
            if best.preferred_tone:
                result["tone"] = best.preferred_tone
            if best.preferred_length:
                result["length"] = best.preferred_length
            if best.preferred_style:
                result["style"] = best.preferred_style

        return result

    # ── Script detection ─────────────────────────────────────

    def detect_scripts(
        self,
        session_history: list[dict],
    ) -> list[BehavioralScript]:
        """
        Extract behavioral scripts from a session's conversation history.

        Analyzes patterns in user-bot interaction to identify
        recurring scripts (Schank-style).
        """
        self.load()

        newly_detected = []

        # Look for repeated patterns across turns
        turn_patterns = self._extract_turn_patterns(session_history)

        for pattern in turn_patterns:
            existing = self._find_similar_script(pattern)
            if existing:
                existing.occurrence_count += 1
                existing.last_triggered = datetime.now(timezone.utc).isoformat()
                existing.confidence = min(1.0, existing.confidence + 0.1)
                continue

            # New script candidate
            script = BehavioralScript(
                trigger_pattern=pattern.get("trigger", ""),
                trigger_domain=pattern.get("domain", ""),
                trigger_emotion=pattern.get("emotion", ""),
                preferred_tone=pattern.get("tone", ""),
                preferred_style=pattern.get("style", ""),
                occurrence_count=1,
                confidence=0.3,
                last_triggered=datetime.now(timezone.utc).isoformat(),
            )
            self.scripts[script.id] = script
            newly_detected.append(script)

        self.save()
        return newly_detected

    # ── Private: Learning methods ────────────────────────────

    def _learn_tone_from_interaction(
        self,
        user_message: str,
        bot_reply: str,
        feedback: dict,
    ):
        """Learn tone preferences from feedback signals."""
        alpha = 0.1  # EMA learning rate

        # Detect bot's tone
        bot_tone = self._classify_tone(bot_reply)

        # Positive signals → reinforce
        if feedback.get("user_continued"):
            if bot_tone == "casual":
                self.profile.tone_casual += alpha * 0.1
            elif bot_tone == "gentle":
                self.profile.tone_gentle += alpha * 0.1
            elif bot_tone == "direct":
                self.profile.tone_direct += alpha * 0.1

        # Negative signals → decrease
        if feedback.get("user_contradicted") or feedback.get("user_changed_topic"):
            if bot_tone == "direct":
                self.profile.tone_direct -= alpha * 0.05
                self.profile.tone_gentle += alpha * 0.05

        # Detect user's own tone as preference indicator
        user_tone = self._classify_tone(user_message)
        if user_tone == "casual":
            self.profile.tone_casual += alpha * 0.05
        elif user_tone == "formal":
            self.profile.tone_formal += alpha * 0.05

        # Clamp
        for attr in ["tone_casual", "tone_formal", "tone_gentle", "tone_direct", "tone_humorous"]:
            setattr(self.profile, attr, max(0.0, min(1.0, getattr(self.profile, attr))))

    def _learn_length_from_interaction(
        self,
        user_message: str,
        bot_reply: str,
        feedback: dict,
    ):
        """Learn length preferences."""
        alpha = 0.1
        reply_len = len(bot_reply)
        user_len = len(user_message)

        if feedback.get("user_asked_more"):
            # User wanted more detail
            self.profile.length_detailed += alpha * 0.15
            self.profile.length_brief -= alpha * 0.05
        elif feedback.get("user_changed_topic") and reply_len > 200:
            # Long reply may have caused topic change
            self.profile.length_brief += alpha * 0.1
            self.profile.length_detailed -= alpha * 0.05
        elif user_len > 100 and reply_len < 50:
            # User wrote a lot, bot was brief — user might want more
            self.profile.length_detailed += alpha * 0.05

        # Clamp
        for attr in ["length_brief", "length_moderate", "length_detailed"]:
            setattr(self.profile, attr, max(0.0, min(1.0, getattr(self.profile, attr))))

    def _learn_style_from_interaction(
        self,
        user_message: str,
        feedback: dict,
    ):
        """Learn response style preferences."""
        alpha = 0.1

        # Detect if user asks for advice
        advice_keywords = ["你觉得", "怎么办", "建议", "应该", "帮我分析", "怎么看"]
        if any(kw in user_message for kw in advice_keywords):
            self.profile.style_advice += alpha * 0.1

        # Detect if user seeks listening
        vent_keywords = ["好累", "好烦", "难受", "想哭", "崩溃", "压力", "焦虑", "失眠"]
        if any(kw in user_message for kw in vent_keywords):
            self.profile.style_listening += alpha * 0.1
            self.profile.style_advice -= alpha * 0.05  # Don't advise when venting

        # Detect if user wants questions
        if user_message.endswith("?") or user_message.endswith("？"):
            self.profile.style_questioning += alpha * 0.05

        if feedback.get("user_continued"):
            self.profile.style_reflective += alpha * 0.03

        # Clamp
        for attr in ["style_advice", "style_listening", "style_questioning", "style_reflective"]:
            setattr(self.profile, attr, max(0.0, min(1.0, getattr(self.profile, attr))))

    def _learn_time_patterns(self, session_hour: int, feedback: dict):
        """Learn time-of-day preference adjustments."""
        if 0 <= session_hour <= 5:
            self.profile.late_night_gentle = True
        elif 6 <= session_hour <= 9:
            if feedback.get("user_continued"):
                self.profile.morning_direct = True

    def _learn_topic_sensitivity(
        self,
        user_message: str,
        domain: list[str] | None,
        valence: float,
        arousal: float,
    ):
        """Learn topics where user needs special handling."""
        # High-arousal negative valence → potentially sensitive
        if valence < 0.3 and arousal > 0.6:
            for d in (domain or []):
                if d and d not in self.profile.sensitive_topics:
                    self.profile.sensitive_topics.append(d)
                    self.profile.sensitive_topics = self.profile.sensitive_topics[-5:]

        # High-engagement positive → preferred topic
        if valence > 0.6 and arousal > 0.4:
            for d in (domain or []):
                if d and d not in self.profile.preferred_topics:
                    self.profile.preferred_topics.append(d)
                    self.profile.preferred_topics = self.profile.preferred_topics[-5:]

    def _detect_and_update_scripts(
        self,
        user_message: str,
        bot_reply: str,
        feedback: dict,
        session_hour: int,
        domain: list[str] | None,
        valence: float,
        arousal: float,
    ):
        """Detect trigger→response patterns and update scripts."""
        # Detect trigger patterns
        triggers = self._extract_triggers(user_message, domain, valence, arousal, session_hour)

        for trigger in triggers:
            script = self._find_similar_script(trigger)
            if script:
                script.occurrence_count += 1
                if feedback.get("user_continued"):
                    script.positive_feedback_count += 1
                if feedback.get("user_contradicted"):
                    script.negative_feedback_count += 1
                script.confidence = self._calculate_confidence(script)
                script.last_triggered = datetime.now(timezone.utc).isoformat()
            else:
                # New script candidate (low confidence initially)
                script = BehavioralScript(
                    trigger_pattern=trigger.get("trigger", ""),
                    trigger_domain=trigger.get("domain", ""),
                    trigger_emotion=trigger.get("emotion", ""),
                    trigger_time_pattern=trigger.get("time_pattern", ""),
                    occurrence_count=1,
                    positive_feedback_count=1 if feedback.get("user_continued") else 0,
                    confidence=0.2,
                    last_triggered=datetime.now(timezone.utc).isoformat(),
                )
                self.scripts[script.id] = script

    # ── Private: Pattern extraction helpers ──────────────────

    @staticmethod
    def _classify_tone(text: str) -> str:
        """Classify the tone of a text (zero-LLM heuristic)."""
        text_lower = text.lower()

        casual_markers = ["哈哈", "呢", "吧", "啊", "哦", "啦", "呀", "嘛", "hh", "lol"]
        formal_markers = ["您好", "请", "谢谢", "确认", "通知", "根据"]
        direct_markers = ["应该", "必须", "一定", "肯定", "就是", "不对"]
        gentle_markers = ["或许", "可能", "也许", "可以", "要不要", "要不要试试"]
        humorous_markers = ["笑死", "绝了", "逗", "搞笑", "哈哈哈"]

        scores = {
            "casual": sum(1 for m in casual_markers if m in text_lower),
            "formal": sum(1 for m in formal_markers if m in text_lower),
            "direct": sum(1 for m in direct_markers if m in text_lower),
            "gentle": sum(1 for m in gentle_markers if m in text_lower),
            "humorous": sum(1 for m in humorous_markers if m in text_lower),
        }

        if not any(scores.values()):
            return "casual"  # default
        return max(scores, key=scores.get)

    @staticmethod
    def _extract_triggers(
        user_message: str,
        domain: list[str] | None,
        valence: float,
        arousal: float,
        session_hour: int,
    ) -> list[dict]:
        """Extract trigger patterns from a user message."""
        triggers = []

        # Trigger: emotion-related
        emotion_markers = {
            "anxious": ["焦虑", "紧张", "担心", "害怕", "不安", "慌", "stress"],
            "sad": ["难过", "伤心", "哭", "失落", "低落", "抑郁", "崩溃"],
            "angry": ["生气", "愤怒", "烦", "气死", "受不了", "讨厌"],
            "happy": ["开心", "高兴", "兴奋", "惊喜", "好了", "成功", "顺利"],
            "confused": ["迷茫", "不知道", "不确定", "困惑", "怎么办", "纠结"],
        }

        for emotion, keywords in emotion_markers.items():
            if any(kw in user_message for kw in keywords):
                triggers.append({
                    "trigger": f"user_expresses_{emotion}",
                    "emotion": emotion,
                    "domain": domain[0] if domain else "",
                    "time_pattern": "late_night" if session_hour <= 5 else (
                        "morning" if 6 <= session_hour <= 9 else "daytime"
                    ),
                })

        # Trigger: domain-specific
        domain_markers = {
            "career": ["工作", "面试", "老板", "同事", "跳槽", "工资", "升职"],
            "relationship": ["恋爱", "分手", "约会", "男朋友", "女朋友", "感情"],
            "family": ["父母", "妈妈", "爸爸", "回家", "家庭"],
            "health": ["身体", "失眠", "焦虑", "体检", "医院", "不舒服"],
            "finance": ["钱", "工资", "贷款", "买房", "投资"],
        }

        for dom, keywords in domain_markers.items():
            if any(kw in user_message for kw in keywords):
                triggers.append({
                    "trigger": f"user_discusses_{dom}",
                    "domain": dom,
                    "emotion": "",
                    "time_pattern": "",
                })

        # If no specific trigger found, use generic
        if not triggers:
            triggers.append({
                "trigger": "general_conversation",
                "domain": domain[0] if domain else "",
                "emotion": "",
                "time_pattern": "",
            })

        return triggers

    def _find_similar_script(self, trigger: dict) -> BehavioralScript | None:
        """Find an existing script matching this trigger pattern."""
        trigger_pattern = trigger.get("trigger", "")
        trigger_domain = trigger.get("domain", "")

        for script in self.scripts.values():
            if script.trigger_pattern == trigger_pattern:
                if not trigger_domain or script.trigger_domain == trigger_domain:
                    return script
            # Fuzzy match: same domain and similar pattern
            if trigger_domain and script.trigger_domain == trigger_domain:
                pattern_words = set(trigger_pattern.split("_"))
                script_words = set(script.trigger_pattern.split("_"))
                overlap = len(pattern_words & script_words)
                if overlap >= 2:
                    return script

        return None

    @staticmethod
    def _calculate_confidence(script: BehavioralScript) -> float:
        """Calculate script confidence from occurrence and feedback."""
        if script.occurrence_count == 0:
            return 0.0

        # Base confidence from occurrences
        occ_confidence = min(1.0, script.occurrence_count / 10.0)

        # Adjust by feedback ratio
        total_feedback = script.positive_feedback_count + script.negative_feedback_count
        if total_feedback > 0:
            feedback_ratio = script.positive_feedback_count / total_feedback
            # Blend: 60% occurrence + 40% feedback
            confidence = 0.6 * occ_confidence + 0.4 * feedback_ratio
        else:
            confidence = occ_confidence * 0.7  # lower without feedback

        return round(min(1.0, confidence), 3)

    @staticmethod
    def _extract_turn_patterns(session_history: list[dict]) -> list[dict]:
        """Extract behavioral patterns from session history."""
        patterns = []

        for i, turn in enumerate(session_history):
            if i == 0:
                continue

            prev = session_history[i - 1]
            user_msg = turn.get("content", "") if turn.get("role") == "user" else ""
            prev_bot = prev.get("content", "") if prev.get("role") == "assistant" else ""

            if not user_msg or not prev_bot:
                continue

            # Pattern: bot gave advice → user continued (positive)
            if any(kw in prev_bot for kw in ["建议", "可以试试", "或许可以"]):
                if len(user_msg) > 20:  # User engaged
                    patterns.append({
                        "trigger": "bot_gives_advice",
                        "style": "advice",
                        "tone": ProceduralMemory._classify_tone(prev_bot),
                    })

            # Pattern: bot listened/reflected → user opened up more
            if any(kw in prev_bot for kw in ["理解", "辛苦了", "不容易", "我明白"]):
                if len(user_msg) > 50:  # User shared more
                    patterns.append({
                        "trigger": "bot_shows_empathy",
                        "style": "listening",
                        "tone": "gentle",
                    })

        return patterns

    def _find_matching_scripts(
        self,
        context: str,
        session_hour: int,
        domain: list[str] | None,
    ) -> list[BehavioralScript]:
        """Find scripts matching the current context, sorted by confidence."""
        matching = []

        time_pattern = (
            "late_night" if session_hour <= 5
            else "morning" if 6 <= session_hour <= 9
            else "daytime"
        )

        for script in self.scripts.values():
            if not script.is_reliable:
                continue

            score = 0.0

            # Domain match
            if domain and script.trigger_domain:
                if any(d == script.trigger_domain for d in domain):
                    score += 0.4

            # Time pattern match
            if script.trigger_time_pattern == time_pattern:
                score += 0.2

            # Context keyword match
            if context and script.trigger_pattern:
                pattern_keywords = script.trigger_pattern.replace("_", " ")
                context_lower = context.lower()
                if any(kw in context_lower for kw in pattern_keywords.split() if len(kw) >= 2):
                    score += 0.3

            if score > 0.3:
                matching.append((score, script))

        matching.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in matching]

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get procedural memory statistics."""
        self.load()
        return {
            "total_scripts": len(self.scripts),
            "reliable_scripts": sum(1 for s in self.scripts.values() if s.is_reliable),
            "total_interactions": self.profile.total_interactions,
            "dominant_tone": self.profile.dominant_tone(),
            "dominant_style": self.profile.dominant_style(),
            "sensitive_topics": self.profile.sensitive_topics,
        }
