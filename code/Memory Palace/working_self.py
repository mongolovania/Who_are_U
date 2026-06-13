# ============================================================
# Module: Working Self (working_self.py)
# L2: Active goals and concerns tracker — Conway SMS model.
# L2：活跃目标追踪 — Conway (2005) Self-Memory System
#
# The Working Self determines what memories are activated
# and what is suppressed. It contains the user's currently
# active goals, concerns, and self-concept.
# Working Self 决定了什么记忆被激活、什么被抑制。
#
# Key functions:
#   - Infer goals/concerns from session meta-signals
#   - Match memories to active goals for relevance scoring
#   - Update goals after each session
#
# For COLD users: no active goals (empty Working Self).
# For WARM+ users: infer from accumulated memory patterns.
# ============================================================

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory_palace.working_self")


@dataclass
class Goal:
    """An active goal in the Working Self."""
    id: str
    description: str
    domain: str = ""               # career, relationship, personal, etc.
    priority: float = 0.5          # 0-1
    active_since: str = ""
    last_referenced: str = ""
    reference_count: int = 0
    resolved: bool = False
    source_memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "domain": self.domain,
            "priority": self.priority,
            "active_since": self.active_since,
            "last_referenced": self.last_referenced,
            "reference_count": self.reference_count,
            "resolved": self.resolved,
            "source_memory_ids": self.source_memory_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Goal:
        return cls(**{k: data.get(k, "" if k in ("id", "description", "domain", "active_since", "last_referenced") else ([] if k == "source_memory_ids" else 0)) for k in [
            "id", "description", "domain", "priority", "active_since",
            "last_referenced", "reference_count", "resolved", "source_memory_ids",
        ]})


@dataclass
class Concern:
    """An active concern/worry in the Working Self."""
    id: str
    description: str
    intensity: float = 0.5         # 0-1, how much this bothers the user
    first_noted: str = ""
    last_noted: str = ""
    occurrence_count: int = 1
    source_memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "intensity": self.intensity, "first_noted": self.first_noted,
            "last_noted": self.last_noted, "occurrence_count": self.occurrence_count,
            "source_memory_ids": self.source_memory_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Concern:
        return cls(**{k: data.get(k, "" if k in ("id", "description", "first_noted", "last_noted") else ([] if k == "source_memory_ids" else 0)) for k in [
            "id", "description", "intensity", "first_noted",
            "last_noted", "occurrence_count", "source_memory_ids",
        ]})


class WorkingSelf:
    """
    Conway (2005) Self-Memory System — Working Self.

    The Working Self is a control mechanism that modulates
    memory access based on the currently active self-concept,
    goals, and concerns.

    COLD behavior: Empty Working Self. All memories treated equally.
    WARM+: Goals and concerns inferred from accumulated sessions.
    """

    def __init__(self, user_id: str = "", data_dir: str = "./buckets"):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        if user_id:
            self.data_dir = self.data_dir / user_id
        os.makedirs(self.data_dir, exist_ok=True)

        self.active_goals: list[Goal] = []
        self.concerns: list[Concern] = []
        self.self_concept: dict = {}  # key traits, values, identity statements
        self._loaded = False

        # Session context (set at start of each session)
        self._current_session_hour: int = 12
        self._current_session_topics: list[str] = []

    # ── Persistence ────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.data_dir / "working_self.json"

    def load(self):
        """Load Working Self state from disk."""
        if self._loaded:
            return
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.active_goals = [Goal.from_dict(g) for g in data.get("goals", [])]
                self.concerns = [Concern.from_dict(c) for c in data.get("concerns", [])]
                self.self_concept = data.get("self_concept", {})
            except Exception as e:
                logger.warning(f"Failed to load Working Self: {e}")
        self._loaded = True

    def save(self):
        """Persist Working Self to disk."""
        path = self._state_path()
        path.write_text(json.dumps({
            "goals": [g.to_dict() for g in self.active_goals],
            "concerns": [c.to_dict() for c in self.concerns],
            "self_concept": self.self_concept,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Session context ────────────────────────────────────

    def set_session_context(self, hour: int, topics: list[str]):
        """Set the current session's context for goal matching."""
        self._current_session_hour = hour
        self._current_session_topics = topics

    # ── Goal matching ──────────────────────────────────────

    def match(self, content: str, domain: list[str] | None = None) -> float:
        """
        Calculate how relevant a memory is to the Working Self (0-1).
        计算记忆与 Working Self 的相关度。

        Higher match = memory is more likely to surface in breath().
        Used by retrieval_engine for WS re-ranking.
        """
        self.load()

        if not self.active_goals and not self.concerns:
            return 0.0  # COLD: no Working Self yet

        scores = []

        # Match against active goals
        for goal in self.active_goals:
            if goal.resolved:
                continue
            # Simple keyword overlap
            goal_words = set(goal.description)
            content_words = set(content)
            overlap = len(goal_words & content_words) / max(len(goal_words), 1)
            scores.append(overlap * goal.priority)

        # Match against concerns
        for concern in self.concerns:
            concern_words = set(concern.description)
            content_words = set(content)
            overlap = len(concern_words & content_words) / max(len(concern_words), 1)
            scores.append(overlap * concern.intensity * 0.5)

        # Domain match bonus
        if domain:
            for goal in self.active_goals:
                if goal.domain and goal.domain in domain:
                    scores.append(0.3 * goal.priority)

        if not scores:
            return 0.0

        return min(1.0, sum(scores) / len(scores))

    # ── Infer from session ─────────────────────────────────

    def infer_from_session(
        self,
        user_message: str,
        valence: float,
        arousal: float,
        session_hour: int = 12,
        topics: list[str] | None = None,
    ):
        """
        Infer Working Self changes from a single session.
        从单次会话推断 Working Self 的变化。

        Heuristic approach (no LLM for COLD users):
          - Late-night + high arousal + negative valence → possible concern
          - Topic repetition → possible active goal
          - Self-reference keywords → self-concept update
        """
        self.load()
        now = datetime.now(timezone.utc).isoformat()

        # Detect potential concerns (late-night negative emotional sessions)
        if session_hour <= 5 and valence < 0.4 and arousal > 0.5:
            self._upsert_concern(
                description=f"深夜困扰: {user_message[:80]}",
                intensity=arousal,
                now=now,
            )

        # Detect potential goals from topic patterns
        if topics:
            for topic in topics:
                # Check if this topic has appeared in existing goals
                existing = [g for g in self.active_goals if topic in g.description]
                if not existing:
                    # Check if topic has appeared in concerns
                    related = [c for c in self.concerns if topic in c.description]
                    if related and len(related) >= 2:
                        # Repeated concern → may indicate an active goal
                        import uuid
                        self.active_goals.append(Goal(
                            id=uuid.uuid4().hex[:8],
                            description=f"处理{topic}相关的问题",
                            domain=topic,
                            priority=0.6,
                            active_since=now,
                            last_referenced=now,
                            reference_count=1,
                        ))

        self.save()

    def _upsert_concern(self, description: str, intensity: float, now: str):
        """Add or update a concern."""
        import uuid
        # Check for similar existing concern
        for c in self.concerns:
            # Simple overlap check
            c_words = set(c.description)
            d_words = set(description)
            if len(c_words & d_words) / max(len(c_words | d_words), 1) > 0.3:
                c.intensity = max(c.intensity, intensity)
                c.last_noted = now
                c.occurrence_count += 1
                return

        self.concerns.append(Concern(
            id=uuid.uuid4().hex[:8],
            description=description,
            intensity=intensity,
            first_noted=now,
            last_noted=now,
            occurrence_count=1,
        ))

    # ── Update after session ───────────────────────────────

    def update_after_session(self, insights: list[str]):
        """
        Update Working Self after a conversation session.
        Called by memory_orchestrator.dream().

        Insights are key takeaways from the session that may
        affect goals, concerns, or self-concept.
        """
        self.load()
        now = datetime.now(timezone.utc).isoformat()

        for insight in insights:
            # Check if insight relates to existing goals
            for goal in self.active_goals:
                if any(word in goal.description for word in insight[:20]):
                    goal.last_referenced = now
                    goal.reference_count += 1

            # Check if insight signals goal resolution
            resolution_keywords = {"解决了", "放下了", "想通了", "不再", "完成", "做到了"}
            if any(kw in insight for kw in resolution_keywords):
                for goal in self.active_goals:
                    if any(word in goal.description for word in insight[:20]):
                        goal.resolved = True

        # Remove resolved goals older than 30 days
        cutoff = datetime.now(timezone.utc).timestamp() - 30 * 86400
        self.active_goals = [
            g for g in self.active_goals
            if not g.resolved
            or (datetime.fromisoformat(g.last_referenced).timestamp() > cutoff)
        ]

        self.save()

    # ── Query ──────────────────────────────────────────────

    def get_active_goal_domains(self) -> list[str]:
        """Get unique domains of active goals."""
        self.load()
        return list({g.domain for g in self.active_goals if g.domain and not g.resolved})

    def get_top_concerns(self, n: int = 3) -> list[Concern]:
        """Get top N concerns by intensity."""
        self.load()
        return sorted(self.concerns, key=lambda c: c.intensity * c.occurrence_count, reverse=True)[:n]

    @property
    def has_goals(self) -> bool:
        return len([g for g in self.active_goals if not g.resolved]) > 0
