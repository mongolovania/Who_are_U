# ============================================================
# Module: Narrative Engine (narrative_engine.py)
# L2: Schank-style story index + Dot "Living History" engine.
# L2：叙事引擎 — 从记忆碎片到故事线
#
# Theoretical foundation:
#   1. Schank (1990, 1995) — Tell Me a Story: narratives as
#      fundamental memory organization. Stories are indexed by
#      scripts, goals, plans, and themes — NOT just keywords.
#   2. Schank & Abelson (1977) — Scripts, Plans, Goals and
#      Understanding: script deviation detection → what's worth
#      remembering is what violates expectation.
#   3. Dot by New Computer (2025) — "Living History":
#      AI continuously updates understanding, stories are
#      ongoing narratives, not isolated snapshots.
#   4. Conway & Pleydell-Pearce (2000) — Autobiographical
#      memory is hierarchically organized: lifetime periods
#      → general events → event-specific knowledge.
#
# Core innovation over v6-v8 retrieval:
#   v6-v8: "find memories similar to this query" (search)
#   v9 narrative: "find the STORY that explains this" (understanding)
#
# Integration points:
#   - memory_graph: community detection on graph → thread seeds
#   - bucket_mgr: read/write memory content
#   - llm_gateway: narrative summarization (async, lightweight)
#   - retrieval_engine: story-indexed retrieval path (P3)
#   - memory_orchestrator: dream() calls narrative merge
# ============================================================

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory_node import MemoryType, ValenceArousal

logger = logging.getLogger("memory_palace.narrative")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class NarrativeMoment:
    """A single moment/episode within a narrative thread."""
    memory_id: str
    content_summary: str          # 1-2 sentence summary of this memory
    valence: float = 0.5
    arousal: float = 0.3
    importance: int = 5
    timestamp: str = ""           # ISO datetime
    is_turning_point: bool = False  # Schank: script deviation moment
    role: str = "episode"        # episode | inciting_incident | climax | resolution


@dataclass
class NarrativeThread:
    """
    A coherent story line spanning multiple memories.

    Schank's story index: each thread has a THEME (what the story
    is about), a GOAL (what the protagonist was trying to achieve),
    and a SCRIPT (the expected sequence of events).

    Dot's living history: threads are ongoing — they don't "close"
    unless the user explicitly moves on. The thread accumulates
    moments and evolves its summary.
    """
    title: str                    # Short name for this thread (e.g. "跳槽大厂之路")
    theme: str                    # Central theme (e.g. "职业转型", "亲密关系")
    id: str = ""                  # Auto-generated if empty
    goal: str = ""                # What the user wants/wanted (e.g. "找到认可自己的工作")
    script: str = ""              # Expected script (e.g. "面试→offer→入职→适应→晋升")
    summary: str = ""             # Current narrative summary (1-3 sentences)
    moments: list[NarrativeMoment] = field(default_factory=list)
    domain: list[str] = field(default_factory=list)

    # Thread lifecycle
    status: str = "active"        # active | dormant | resolved | abandoned
    created_at: str = ""
    last_updated: str = ""
    priority: float = 0.5         # 0-1, how prominently this thread surfaces

    # Conway's hierarchical index
    life_period: str = ""         # e.g. "2024-2026: 职业探索期"
    general_event: str = ""       # e.g. "三次面试经历"

    # Graph integration
    seed_memory_ids: list[str] = field(default_factory=list)
    community_id: str = ""        # GraphRAG-style community cluster ID

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.created_at
        if not self.id:
            self.id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "theme": self.theme,
            "goal": self.goal,
            "script": self.script,
            "summary": self.summary,
            "moments": [
                {
                    "memory_id": m.memory_id,
                    "content_summary": m.content_summary,
                    "valence": m.valence,
                    "arousal": m.arousal,
                    "importance": m.importance,
                    "timestamp": m.timestamp,
                    "is_turning_point": m.is_turning_point,
                    "role": m.role,
                }
                for m in self.moments
            ],
            "domain": self.domain,
            "status": self.status,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "priority": self.priority,
            "life_period": self.life_period,
            "general_event": self.general_event,
            "seed_memory_ids": self.seed_memory_ids,
            "community_id": self.community_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NarrativeThread:
        moments = [
            NarrativeMoment(
                memory_id=m.get("memory_id", ""),
                content_summary=m.get("content_summary", ""),
                valence=m.get("valence", 0.5),
                arousal=m.get("arousal", 0.3),
                importance=m.get("importance", 5),
                timestamp=m.get("timestamp", ""),
                is_turning_point=m.get("is_turning_point", False),
                role=m.get("role", "episode"),
            )
            for m in data.get("moments", [])
        ]
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            theme=data.get("theme", ""),
            goal=data.get("goal", ""),
            script=data.get("script", ""),
            summary=data.get("summary", ""),
            moments=moments,
            domain=data.get("domain", []),
            status=data.get("status", "active"),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
            priority=data.get("priority", 0.5),
            life_period=data.get("life_period", ""),
            general_event=data.get("general_event", ""),
            seed_memory_ids=data.get("seed_memory_ids", []),
            community_id=data.get("community_id", ""),
        )

    @property
    def moment_count(self) -> int:
        return len(self.moments)

    @property
    def emotional_arc(self) -> list[tuple[float, float]]:
        """Return (valence, arousal) sequence for emotional trajectory."""
        return [(m.valence, m.arousal) for m in self.moments]


# ═══════════════════════════════════════════════════════════════
# Narrative pattern detectors
# ═══════════════════════════════════════════════════════════════

# Schank-style script patterns: common life scripts with expected sequences
_LIFE_SCRIPTS: dict[str, dict] = {
    "职业转型": {
        "script": "不满现状→探索选项→尝试/学习→决策节点→面试→offer选择→入职→适应期→安定或再出发",
        "goal_hints": ["换工作", "转行", "跳槽", "离职", "新方向"],
        "turning_point_markers": ["决定", "辞职", "接受offer", "入职", "被裁"],
    },
    "亲密关系": {
        "script": "相遇/相识→好感发展→关系确认→深度相处→冲突/磨合→和解或分离→反思",
        "goal_hints": ["恋爱", "分手", "复合", "结婚", "在一起"],
        "turning_point_markers": ["表白", "在一起", "分手", "吵架", "和好", "搬一起"],
    },
    "成长探索": {
        "script": "触发事件→困惑/迷茫→自我反思→尝试改变→小进步→回退→坚持→突破→新常态",
        "goal_hints": ["成长", "改变", "突破", "成为更好的自己", "进步"],
        "turning_point_markers": ["意识到", "决定改变", "第一次", "突破", "想通了"],
    },
    "家庭关系": {
        "script": "日常相处→冲突触发→情绪爆发→冷战/沟通→理解或僵持→关系调整→新平衡",
        "goal_hints": ["父母", "妈妈", "爸爸", "家庭", "回家", "孩子"],
        "turning_point_markers": ["吵架", "离家", "回家", "和解", "理解了"],
    },
    "健康管理": {
        "script": "身体信号→忽视→加重→就医/重视→治疗/调整→恢复→新习惯",
        "goal_hints": ["健康", "睡眠", "运动", "饮食", "体检", "身体"],
        "turning_point_markers": ["检查结果", "开始", "坚持", "好转", "复发"],
    },
    "财务规划": {
        "script": "现状评估→目标设定→储蓄/投资→意外事件→调整策略→达到或未达目标→调整预期",
        "goal_hints": ["钱", "工资", "存款", "买房", "投资", "理财"],
        "turning_point_markers": ["涨薪", "买房", "亏损", "攒够", "还清"],
    },
}

# Narrative arc types (from literary theory, adapted for life narratives)
_ARC_TYPES = {
    "上升弧": {"valence_slope": "positive", "description": "事情越来越好"},
    "下降弧": {"valence_slope": "negative", "description": "事情越来越糟"},
    "V型弧": {"valence_shape": "dip", "description": "先降后升——低谷反弹"},
    "倒V型弧": {"valence_shape": "peak", "description": "先升后降——盛极而衰"},
    "平缓弧": {"valence_variance": "low", "description": "情绪平稳，渐进变化"},
    "波动弧": {"valence_variance": "high", "description": "情绪起伏大，反复不定"},
}


# ═══════════════════════════════════════════════════════════════
# Narrative Engine
# ═══════════════════════════════════════════════════════════════


class NarrativeEngine:
    """
    Schank-style story index + Dot "Living History" engine.

    Transforms isolated memory fragments into coherent narrative
    threads, organized by theme, goal, and script.

    Two retrieval modes:
      1. Story-indexed: "find the story that explains this query"
      2. Living history: "tell me about my journey with X"
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

        self.threads: dict[str, NarrativeThread] = {}
        self._loaded = False

        # Community detection cache (GraphRAG-style)
        self._communities: dict[str, list[str]] = {}  # community_id → [memory_ids]

        # Conway life period boundaries
        self._life_periods: list[dict] = []

    # ── Persistence ────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.data_dir / "narrative_threads.json"

    def load(self):
        """Load narrative threads from disk."""
        if self._loaded:
            return
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.threads = {
                    tid: NarrativeThread.from_dict(td)
                    for tid, td in data.get("threads", {}).items()
                }
                self._communities = data.get("communities", {})
                self._life_periods = data.get("life_periods", [])
            except Exception as e:
                logger.warning(f"Failed to load narrative threads: {e}")
        self._loaded = True

    def save(self):
        """Persist narrative threads to disk."""
        path = self._state_path()
        path.write_text(json.dumps({
            "threads": {tid: t.to_dict() for tid, t in self.threads.items()},
            "communities": self._communities,
            "life_periods": self._life_periods,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Thread creation and management ────────────────────────

    def find_or_create_thread(
        self,
        memory_id: str,
        content: str,
        domain: list[str],
        tags: list[str],
        valence: float = 0.5,
        arousal: float = 0.3,
        importance: int = 5,
        timestamp: str = "",
        memory_graph=None,
    ) -> NarrativeThread:
        """
        Find an existing narrative thread for this memory, or create a new one.

        Zero-LLM assignment uses:
          1. Domain + tag overlap with existing threads
          2. Keyword-based theme detection
          3. Graph community membership (if memory_graph available)

        Args:
            memory_id: The new memory's ID
            content: Memory content
            domain: Memory domains
            tags: Memory tags
            valence: Emotional valence
            arousal: Emotional arousal
            importance: Importance score
            timestamp: ISO datetime string
            memory_graph: Optional MemoryGraph for community detection

        Returns:
            The NarrativeThread this memory was assigned to
        """
        self.load()

        # Step 1: Try to match existing thread by domain + theme overlap
        best_thread = None
        best_score = 0.0

        all_domains = set(domain)
        all_tags = set(tags)

        for thread in self.threads.values():
            if thread.status == "resolved":
                continue

            score = 0.0

            # Domain overlap
            thread_domains = set(thread.domain)
            domain_overlap = all_domains & thread_domains
            if domain_overlap:
                score += 0.4 * min(len(domain_overlap), 3) / max(len(all_domains), 1)

            # Theme match via keyword detection
            theme_hints = _LIFE_SCRIPTS.get(thread.theme, {}).get("goal_hints", [])
            content_lower = content.lower()
            matched_hints = [h for h in theme_hints if h in content_lower]
            if matched_hints:
                score += 0.3 * min(len(matched_hints), 3) / max(len(theme_hints), 1)

            # Tag overlap
            tag_overlap = all_tags & set(t for m in thread.moments for t in ([] if not hasattr(m, 'tags') else []))
            if tag_overlap:
                score += 0.1

            # Temporal proximity: recent thread gets bonus
            if thread.moments:
                try:
                    last_ts = datetime.fromisoformat(thread.moments[-1].timestamp)
                    this_ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
                    days_apart = abs((this_ts - last_ts).total_seconds()) / 86400.0
                    if days_apart <= 7:
                        score += 0.2 * (1.0 - days_apart / 7.0)
                except (ValueError, TypeError):
                    pass

            if score > best_score:
                best_score = score
                best_thread = thread

        # Step 2: If no good match (score < 0.3), create a new thread
        if best_thread is None or best_score < 0.3:
            # Detect theme from content
            detected_theme = self._detect_theme(content, domain, tags, valence, arousal)

            # Detect script for this theme
            script_info = _LIFE_SCRIPTS.get(detected_theme, {})
            script = script_info.get("script", "")

            # Detect goal hints
            goal = self._detect_goal(content, detected_theme)

            # Detect life period
            life_period = self._detect_life_period(timestamp)

            best_thread = NarrativeThread(
                title=self._generate_thread_title(content, detected_theme),
                theme=detected_theme,
                goal=goal,
                script=script,
                domain=list(domain) if domain else [],
                life_period=life_period,
            )
            self.threads[best_thread.id] = best_thread
            logger.info(f"New narrative thread: {best_thread.title} [{best_thread.id}]")

        # Step 3: Add moment to thread
        turning_point = self._is_turning_point(content, best_thread.theme, importance)

        moment = NarrativeMoment(
            memory_id=memory_id,
            content_summary=content[:200] if content else "",
            valence=valence,
            arousal=arousal,
            importance=importance,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            is_turning_point=turning_point,
            role=self._classify_moment_role(content, best_thread, turning_point),
        )
        best_thread.moments.append(moment)
        best_thread.last_updated = datetime.now(timezone.utc).isoformat()

        # Update thread priority based on recency + turning points
        best_thread.priority = self._calculate_thread_priority(best_thread)

        # Update thread summary
        best_thread.summary = self._update_thread_summary(best_thread)

        # Track seed memories
        if memory_id not in best_thread.seed_memory_ids:
            best_thread.seed_memory_ids.append(memory_id)

        self.save()
        return best_thread

    # ── Thread merging ─────────────────────────────────────────

    def merge_threads(
        self,
        thread_id_a: str,
        thread_id_b: str,
    ) -> NarrativeThread | None:
        """
        Merge two related narrative threads into one.

        Called when narrative patterns reveal two threads are
        actually the same story line (e.g. "准备面试" + "入职新公司"
        → merged into "职业转型").

        Uses A-MEM Zettelkasten logic: connecting previously
        separate threads enriches both.
        """
        self.load()

        thread_a = self.threads.get(thread_id_a)
        thread_b = self.threads.get(thread_id_b)

        if not thread_a or not thread_b:
            return None

        # Merge moments in chronological order
        all_moments = thread_a.moments + thread_b.moments
        all_moments.sort(key=lambda m: m.timestamp)

        # Deduplicate by memory_id
        seen_ids = set()
        merged_moments = []
        for m in all_moments:
            if m.memory_id not in seen_ids:
                merged_moments.append(m)
                seen_ids.add(m.memory_id)

        # Create merged thread
        merged = NarrativeThread(
            title=f"{thread_a.title} → {thread_b.title}",
            theme=thread_a.theme if thread_a.theme == thread_b.theme else f"{thread_a.theme}+{thread_b.theme}",
            goal=thread_a.goal or thread_b.goal,
            script=thread_a.script or thread_b.script,
            summary="",  # Will be regenerated
            moments=merged_moments,
            domain=list(set(thread_a.domain + thread_b.domain)),
            life_period=thread_a.life_period or thread_b.life_period,
            seed_memory_ids=list(set(thread_a.seed_memory_ids + thread_b.seed_memory_ids)),
        )

        merged.summary = self._update_thread_summary(merged)
        merged.priority = self._calculate_thread_priority(merged)

        # Replace old threads with merged
        del self.threads[thread_id_a]
        del self.threads[thread_id_b]
        self.threads[merged.id] = merged

        self.save()
        logger.info(f"Merged threads {thread_id_a} + {thread_id_b} → {merged.id}")
        return merged

    # ── Living History generation (Dot-style) ─────────────────

    def generate_living_history(
        self,
        domain_filter: list[str] | None = None,
        max_threads: int = 5,
    ) -> dict:
        """
        Generate "Living History" overview — Dot by New Computer style.

        Returns:
            {
                "overview": "你正在经历...",  # Overall narrative summary
                "active_threads": [...],      # Currently active story lines
                "recent_turning_points": [...], # Key moments from all threads
                "emotional_trajectory": {...}, # Emotional arc across threads
                "life_periods": [...],         # Conway life periods
            }
        """
        self.load()

        # Filter threads
        threads = list(self.threads.values())
        if domain_filter:
            threads = [
                t for t in threads
                if any(d in t.domain for d in domain_filter)
            ]

        # Sort by priority
        threads.sort(key=lambda t: t.priority, reverse=True)
        active = [t for t in threads if t.status == "active"][:max_threads]

        # Collect turning points
        turning_points = []
        for t in threads:
            for m in t.moments:
                if m.is_turning_point:
                    turning_points.append({
                        "thread_title": t.title,
                        "content": m.content_summary[:150],
                        "timestamp": m.timestamp,
                        "valence": m.valence,
                        "arousal": m.arousal,
                    })

        turning_points.sort(key=lambda tp: tp["timestamp"], reverse=True)

        # Emotional trajectory
        all_valence = [m.valence for t in active for m in t.moments]
        all_arousal = [m.arousal for t in active for m in t.moments]

        # Build overview
        overview_parts = []
        for t in active[:3]:
            overview_parts.append(t.summary)

        overview = "。".join(overview_parts) if overview_parts else "你的故事才刚刚开始。"

        return {
            "overview": overview,
            "active_threads": [
                {
                    "id": t.id,
                    "title": t.title,
                    "theme": t.theme,
                    "summary": t.summary,
                    "moment_count": t.moment_count,
                    "priority": t.priority,
                    "first_moment": t.moments[0].timestamp if t.moments else "",
                    "last_moment": t.moments[-1].timestamp if t.moments else "",
                }
                for t in active
            ],
            "recent_turning_points": turning_points[:10],
            "emotional_trajectory": {
                "avg_valence": round(sum(all_valence) / max(len(all_valence), 1), 3) if all_valence else 0.5,
                "avg_arousal": round(sum(all_arousal) / max(len(all_arousal), 1), 3) if all_arousal else 0.3,
                "thread_arcs": {
                    t.id: self._detect_narrative_arc(t)
                    for t in active
                },
            },
            "life_periods": self._life_periods[-5:] if self._life_periods else [],
        }

    # ── Story-indexed retrieval ────────────────────────────────

    def find_story_for_query(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Story-indexed retrieval: find the narrative thread(s) most
        relevant to a query.

        Unlike keyword/vector search over individual memories,
        this retrieves the STORY that explains the query context.

        P3: This becomes a new retrieval path for narrative/contextual
        queries — "what was happening in my life when...", "tell me
        about my journey with...", "how did I feel when..."
        """
        self.load()

        if not self.threads:
            return []

        scored = []
        query_lower = query.lower()

        for thread in self.threads.values():
            if thread.status == "resolved":
                continue

            score = 0.0

            # Title match (substring, not character-level)
            if thread.title and thread.title in query_lower:
                score += 0.35
            elif thread.title:
                # Partial: any 2+ char segment of title appears in query
                title_segments = [thread.title[i:i+2] for i in range(len(thread.title)-1)]
                if any(seg in query_lower for seg in title_segments if len(seg) >= 2):
                    score += 0.15

            # Theme match
            if thread.theme and len(thread.theme) >= 2 and thread.theme in query_lower:
                score += 0.25

            # Goal keyword match (use 2+ char substrings)
            if thread.goal:
                goal_segments = [thread.goal[i:i+2] for i in range(len(thread.goal)-1)]
                if any(seg in query_lower for seg in goal_segments if len(seg) >= 2):
                    score += 0.20

            # Summary content match (simple keyword overlap)
            summary_words = set(thread.summary)
            query_words = set(query_lower)
            overlap = summary_words & query_words
            if summary_words:
                score += 0.15 * len(overlap) / max(len(summary_words), 1)

            # Domain match
            for d in thread.domain:
                if d in query_lower:
                    score += 0.10
                    break

            # Recency bonus
            if thread.moments:
                try:
                    last_ts = datetime.fromisoformat(thread.moments[-1].timestamp)
                    days_since = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400.0
                    score += 0.10 * max(0.0, 1.0 - days_since / 30.0)
                except (ValueError, TypeError):
                    pass

            if score > 0.1:
                scored.append({
                    "thread_id": thread.id,
                    "title": thread.title,
                    "theme": thread.theme,
                    "summary": thread.summary,
                    "score": score,
                    "moment_count": thread.moment_count,
                    "emotional_arc": self._detect_narrative_arc(thread),
                    # Include representative memories for context
                    "key_moments": [
                        {
                            "memory_id": m.memory_id,
                            "summary": m.content_summary[:120],
                            "timestamp": m.timestamp,
                            "is_turning_point": m.is_turning_point,
                        }
                        for m in thread.moments
                        if m.is_turning_point or m.importance >= 7
                    ][:5],
                })

        scored.sort(key=lambda s: s["score"], reverse=True)
        return scored[:top_k]

    # ── Community detection (GraphRAG-style) ──────────────────

    def detect_communities(
        self,
        memory_graph,
        bucket_mgr,
        use_leiden: bool = True,
    ) -> dict[str, list[str]]:
        """
        Run community detection on the memory graph.

        GraphRAG-style: use graph topology to find densely connected
        clusters of memories → these become narrative thread seeds.

        v9 Track C: Uses Leiden-like modularity optimization when
        use_leiden=True. Falls back to BFS connected components
        when memory_graph has too few nodes.

        Returns:
            {community_id: [memory_ids]} mapping
        """
        self.load()

        if memory_graph is None:
            return {}

        try:
            stats = memory_graph.get_graph_stats()
            if stats.get("node_count", 0) < 3:
                return {}  # Not enough data for communities
        except Exception:
            return {}

        communities: dict[str, list[str]] = {}

        if use_leiden:
            try:
                from graph_rag import LeidenDetector

                # Extract graph structure
                nodes: dict[str, dict] = {}
                edges: list[dict] = []

                # Get all edges by type
                for etype in ["causal", "thematic", "temporal", "emotional"]:
                    typed_edges = memory_graph.get_edges_by_type(
                        etype, limit=2000
                    )
                    for edge in typed_edges:
                        from_id = edge.get("from_id", "")
                        to_id = edge.get("to_id", "")
                        if from_id:
                            nodes[from_id] = {"type": "memory"}
                        if to_id:
                            nodes[to_id] = {"type": "memory"}
                        # Only include active edges
                        if not edge.get("valid_until"):
                            edges.append({
                                "from_id": from_id,
                                "to_id": to_id,
                                "weight": edge.get("weight", 1.0),
                            })

                if len(nodes) >= 3 and edges:
                    detector = LeidenDetector(resolution=1.0, max_iterations=10)
                    communities = detector.detect_communities(nodes, edges)

                    # Filter: keep only communities with >=2 members
                    communities = {
                        cid: members
                        for cid, members in communities.items()
                        if len(members) >= 2
                    }

                    if communities:
                        logger.info(
                            f"Leiden detection: {len(communities)} communities "
                            f"from {len(nodes)} nodes, {len(edges)} edges"
                        )
            except ImportError:
                logger.debug("graph_rag module not available, using BFS fallback")
            except Exception as e:
                logger.warning(f"Leiden detection failed, using BFS fallback: {e}")

        # Fallback: BFS connected components
        if not communities:
            visited: set[str] = set()
            # Use thread seeds as starting points
            for thread in self.threads.values():
                for mid in thread.seed_memory_ids:
                    if mid not in visited:
                        component = self._bfs_component(memory_graph, mid, visited)
                        if len(component) >= 2:
                            community_id = f"comm_{uuid.uuid4().hex[:8]}"
                            communities[community_id] = component

        # Update cache
        self._communities = communities
        self.save()
        return communities

    def _bfs_component(
        self,
        memory_graph,
        start_id: str,
        visited: set[str],
    ) -> list[str]:
        """BFS to find connected component starting from a node."""
        component = []
        frontier = [start_id]

        while frontier:
            node_id = frontier.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            component.append(node_id)

            try:
                neighbors = memory_graph.get_neighbors(node_id, depth=1, active_only=True)
                for n in neighbors:
                    nid = n.get("to_id", "")
                    if nid and nid not in visited:
                        frontier.append(nid)
            except Exception:
                pass

        return component

    # ── Narrative merge (dream cycle) ────────────────────────

    async def run_narrative_merge(
        self,
        memory_graph=None,
        bucket_mgr=None,
        retrieval_engine=None,
        llm_gateway=None,
        new_memories_since_last: int = 0,
    ) -> dict:
        """
        Run narrative consolidation cycle (dream time).

        1. Detect new communities on memory graph
        2. Merge similar/overlapping threads
        3. Update thread summaries with LLM (if available)
        4. Detect resolved/abandoned threads
        5. Update life periods

        Returns:
            Stats dict with merge/update counts
        """
        self.load()

        result = {
            "communities_detected": 0,
            "threads_merged": 0,
            "summaries_updated": 0,
            "threads_resolved": 0,
            "life_periods_updated": 0,
        }

        # Step 1: Community detection
        if memory_graph:
            communities = self.detect_communities(memory_graph, bucket_mgr)
            result["communities_detected"] = len(communities)

            # Assign communities to threads
            for community_id, mem_ids in communities.items():
                for thread in self.threads.values():
                    if any(mid in mem_ids for mid in thread.seed_memory_ids):
                        thread.community_id = community_id
                        break

        # Step 2: Merge overlapping threads
        merged = self._auto_merge_threads()
        result["threads_merged"] = len(merged)

        # Step 3: Update summaries (LLM if available)
        for thread in self.threads.values():
            _last_count = getattr(thread, '_last_summarized_count', 0)
            if thread.moment_count > _last_count:
                if llm_gateway and thread.moment_count >= 3:
                    try:
                        new_summary = await self._llm_summarize_thread(thread, llm_gateway)
                        if new_summary:
                            thread.summary = new_summary
                            thread._last_summarized_count = thread.moment_count
                            result["summaries_updated"] += 1
                    except Exception as e:
                        logger.warning(f"LLM summary failed for thread {thread.id}: {e}")
                else:
                    # Fallback: rule-based summary
                    thread.summary = self._update_thread_summary(thread)
                    thread._last_summarized_count = thread.moment_count
                    result["summaries_updated"] += 1

        # Step 4: Detect resolved/abandoned threads
        result["threads_resolved"] = self._detect_resolved_threads()

        # Step 5: Update life periods
        result["life_periods_updated"] = self._update_life_periods()

        self.save()
        return result

    # ── Private: Theme detection ─────────────────────────────

    @staticmethod
    def _detect_theme(
        content: str,
        domain: list[str],
        tags: list[str],
        valence: float = 0.5,
        arousal: float = 0.3,
    ) -> str:
        """Detect narrative theme from content (zero-LLM)."""
        content_lower = content.lower()

        theme_scores: dict[str, float] = {}

        for theme_name, info in _LIFE_SCRIPTS.items():
            score = 0.0
            for hint in info.get("goal_hints", []):
                if hint in content_lower:
                    score += 0.3

            # Domain match
            for d in domain:
                if d and any(hint in d for hint in info.get("goal_hints", [])):
                    score += 0.2

            # Tag match
            for t in tags:
                if t and any(hint in t for hint in info.get("goal_hints", [])):
                    score += 0.15

            # Emotional context clues
            if "loss" in theme_name.lower() and valence < 0.4 and arousal > 0.6:
                score += 0.1

            if score > 0:
                theme_scores[theme_name] = score

        if theme_scores:
            return max(theme_scores, key=theme_scores.get)

        # Default themes by domain keywords
        domain_themes = {
            "成长": "成长探索",
            "求职": "职业转型",
            "工作": "职业转型",
            "职业": "职业转型",
            "感情": "亲密关系",
            "爱情": "亲密关系",
            "家庭": "家庭关系",
            "健康": "健康管理",
            "财务": "财务规划",
            "学习": "成长探索",
        }
        for d in domain:
            for key, theme in domain_themes.items():
                if key in d:
                    return theme

        return "成长探索"  # Default

    @staticmethod
    def _detect_goal(content: str, theme: str) -> str:
        """Extract implied goal from content."""
        goal_patterns = {
            "想要": "欲望/渴望",
            "希望": "希望",
            "目标": "目标",
            "计划": "计划",
            "决定": "决定",
            "想": "想法/愿望",
            "准备": "准备",
            "努力": "努力",
            "争取": "争取",
            "试图": "试图",
        }

        for keyword, goal_type in goal_patterns.items():
            if keyword in content:
                # Extract surrounding context (simplified)
                idx = content.find(keyword)
                context = content[max(0, idx - 10):idx + 30]
                return f"{goal_type}: {context.strip()}"

        return f"处理{theme}相关的问题"

    @staticmethod
    def _generate_thread_title(content: str, theme: str) -> str:
        """Generate a short title for a new narrative thread."""
        # Extract key entities/actions for title
        import re

        # Look for key event markers
        markers = {
            "辞职": "离职",
            "离职": "离职",
            "入职": "入职",
            "面试": "面试",
            "offer": "拿offer",
            "分手": "分手",
            "在一起": "新关系",
            "搬家": "搬家",
            "毕业": "毕业",
            "开始": "新开始",
            "决定": "做决定",
            "突破": "突破",
            "通过": "通过考核",
        }

        for marker, label in markers.items():
            if marker in content:
                return f"{label}——{theme}"

        # Fallback: theme + timestamp snippet
        return f"{theme}的故事"

    @staticmethod
    def _is_turning_point(content: str, theme: str, importance: int) -> bool:
        """Detect if this memory is a Schank script deviation / turning point."""
        script_info = _LIFE_SCRIPTS.get(theme, {})
        markers = script_info.get("turning_point_markers", [])

        for marker in markers:
            if marker in content:
                return True

        # High importance + emotional intensity = likely turning point
        if importance >= 8:
            return True

        return False

    @staticmethod
    def _classify_moment_role(
        content: str,
        thread: NarrativeThread,
        is_turning_point: bool,
    ) -> str:
        """Classify a moment's role in its narrative thread."""
        # Check resolution markers FIRST — even a first moment can be a resolution
        resolution_markers = ["终于", "解决了", "放下了", "完成了", "结束了", "成功了", "达到了"]
        if any(m in content for m in resolution_markers):
            return "resolution"

        # First moment defaults
        if not thread.moments:
            return "inciting_incident" if is_turning_point else "episode"

        # Check if this is the peak/climax
        if is_turning_point:
            return "climax"

        return "episode"

    # ── Private: Thread management helpers ────────────────────

    def _calculate_thread_priority(self, thread: NarrativeThread) -> float:
        """Calculate thread priority based on recency, importance, and turning points."""
        if not thread.moments:
            return 0.3

        # Recency (0-1)
        try:
            last_ts = datetime.fromisoformat(thread.moments[-1].timestamp)
            days_since = max(0.0, (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400.0)
            recency = max(0.0, 1.0 - days_since / 60.0)  # decays over 60 days
        except (ValueError, TypeError):
            recency = 0.3

        # Importance: average importance of moments
        avg_importance = sum(m.importance for m in thread.moments) / len(thread.moments)
        importance_factor = avg_importance / 10.0

        # Turning point density
        tp_count = sum(1 for m in thread.moments if m.is_turning_point)
        tp_density = min(1.0, tp_count / max(len(thread.moments), 1) * 3)

        # Combine
        priority = recency * 0.4 + importance_factor * 0.3 + tp_density * 0.3
        return round(max(0.1, min(1.0, priority)), 3)

    def _update_thread_summary(self, thread: NarrativeThread) -> str:
        """Update thread summary from its moments (rule-based, zero-LLM)."""
        if not thread.moments:
            return ""

        # Sort moments by timestamp
        sorted_moments = sorted(thread.moments, key=lambda m: m.timestamp)

        # Extract key events
        key_moments = [m for m in sorted_moments if m.is_turning_point or m.importance >= 7]
        if not key_moments:
            key_moments = sorted_moments[-3:]  # last 3 moments

        # Build summary
        parts = []
        for m in key_moments[:4]:
            # Extract first meaningful sentence segment
            summary_text = m.content_summary[:80].strip()
            if summary_text:
                parts.append(summary_text)

        if not parts:
            parts = [sorted_moments[-1].content_summary[:80]]

        # Temporal framing
        if len(sorted_moments) >= 2:
            try:
                first = datetime.fromisoformat(sorted_moments[0].timestamp)
                last = datetime.fromisoformat(sorted_moments[-1].timestamp)
                days_span = (last - first).days
                if days_span > 0:
                    timeframe = f"（{days_span}天内的故事）"
                else:
                    timeframe = ""
            except (ValueError, TypeError):
                timeframe = ""
        else:
            timeframe = ""

        summary = " → ".join(parts[:3])
        if timeframe:
            summary += timeframe

        return summary[:300]

    async def _llm_summarize_thread(
        self,
        thread: NarrativeThread,
        llm_gateway,
    ) -> str:
        """Use lightweight LLM to generate a narrative summary."""
        moments_text = "\n".join(
            f"[{m.timestamp[:10]}] {m.content_summary[:150]}"
            + (" 🔑转折点" if m.is_turning_point else "")
            for m in thread.moments[-10:]  # Last 10 moments
        )

        prompt = f"""Summarize this personal life story thread in 2-3 sentences (Chinese).

Thread theme: {thread.theme}
Goal: {thread.goal or 'unknown'}

Key moments:
{moments_text[:1500]}

Write as if describing a person's journey. Be warm but concise.
Return just the summary text, no JSON wrapper."""

        try:
            response = await llm_gateway.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You are a biographer. Write concise, warm narrative summaries.",
            )
            return response.strip()[:300]
        except Exception as e:
            logger.warning(f"LLM thread summary failed: {e}")
            return ""

    def _auto_merge_threads(self) -> list[str]:
        """Automatically merge threads that overlap significantly."""
        merged_ids = []
        thread_list = list(self.threads.values())

        for i, t1 in enumerate(thread_list):
            if t1.status == "resolved":
                continue
            for t2 in thread_list[i + 1:]:
                if t2.status == "resolved":
                    continue

                # Check overlap: same theme + overlapping domains
                if t1.theme == t2.theme:
                    domain_overlap = set(t1.domain) & set(t2.domain)
                    if domain_overlap and len(t1.moments) >= 2 and len(t2.moments) >= 2:
                        merged = self.merge_threads(t1.id, t2.id)
                        if merged:
                            merged_ids.append(merged.id)
                            break
            if merged_ids and merged_ids[-1] == t1.id:
                break

        return merged_ids

    def _detect_resolved_threads(self) -> int:
        """Detect and mark resolved/abandoned threads."""
        resolved_count = 0
        now = datetime.now(timezone.utc)

        for thread in self.threads.values():
            if thread.status != "active":
                continue

            # Condition 1: Last moment was a resolution
            if thread.moments:
                last = thread.moments[-1]
                if last.role == "resolution":
                    thread.status = "resolved"
                    resolved_count += 1
                    continue

            # Condition 2: No activity for 90+ days
            try:
                last_ts = datetime.fromisoformat(thread.last_updated)
                days_inactive = (now - last_ts).total_seconds() / 86400.0
                if days_inactive > 90:
                    thread.status = "dormant"
                    resolved_count += 1
            except (ValueError, TypeError):
                pass

        return resolved_count

    def _update_life_periods(self) -> int:
        """Update Conway life period boundaries from thread timestamps."""
        all_timestamps = []
        for thread in self.threads.values():
            for m in thread.moments:
                if m.timestamp:
                    try:
                        all_timestamps.append(datetime.fromisoformat(m.timestamp))
                    except (ValueError, TypeError):
                        pass

        if not all_timestamps:
            return 0

        all_timestamps.sort()
        earliest = all_timestamps[0]
        latest = all_timestamps[-1]

        # Detect periods by clustering timestamps
        # Simple: if gap > 90 days, start new period
        periods = []
        period_start = earliest
        period_memories = 0

        for ts in all_timestamps:
            period_memories += 1

        # Determine dominant themes per period
        dominant_theme = self._dominant_theme_for_period(earliest, latest)

        periods.append({
            "period": f"{earliest.year}-{latest.year}",
            "label": f"{earliest.year}年-{latest.year}年",
            "dominant_theme": dominant_theme,
            "memory_count": period_memories,
        })

        if len(periods) > len(self._life_periods):
            self._life_periods = periods
            return 1

        return 0

    def _dominant_theme_for_period(
        self,
        start: datetime,
        end: datetime,
    ) -> str:
        """Find the dominant narrative theme in a time period."""
        theme_counts: dict[str, int] = {}
        for thread in self.threads.values():
            for m in thread.moments:
                try:
                    ts = datetime.fromisoformat(m.timestamp)
                    if start <= ts <= end:
                        theme_counts[thread.theme] = theme_counts.get(thread.theme, 0) + 1
                except (ValueError, TypeError):
                    pass

        if theme_counts:
            return max(theme_counts, key=theme_counts.get)
        return "未知"

    # ── Narrative arc detection ─────────────────────────────

    def _detect_narrative_arc(self, thread: NarrativeThread) -> dict:
        """Detect the emotional/narrative arc of a thread."""
        if len(thread.moments) < 2:
            return {"type": "too_few_moments", "description": "故事刚刚开始"}

        valences = [m.valence for m in thread.moments]
        avg_valence = sum(valences) / len(valences)

        # Detect trend
        first_half = valences[:len(valences) // 2]
        second_half = valences[len(valences) // 2:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        valence_change = second_avg - first_avg

        if abs(valence_change) < 0.1:
            arc_type = "平缓弧"
        elif valence_change > 0.2:
            arc_type = "上升弧"
        elif valence_change < -0.2:
            arc_type = "下降弧"
        elif min(valences) < 0.3 and max(valences) > 0.7:
            if first_avg < second_avg:
                arc_type = "V型弧"
            else:
                arc_type = "倒V型弧"
        else:
            arc_type = "波动弧"

        return {
            "type": arc_type,
            "description": _ARC_TYPES.get(arc_type, {}).get("description", ""),
            "avg_valence": round(avg_valence, 2),
            "start_valence": round(valences[0], 2),
            "end_valence": round(valences[-1], 2),
            "valence_change": round(valence_change, 2),
        }

    def _detect_life_period(self, timestamp: str) -> str:
        """Assign a memory to a Conway life period."""
        if not timestamp:
            return ""

        try:
            ts = datetime.fromisoformat(timestamp)
            year = ts.year
            return f"{year}年"
        except (ValueError, TypeError):
            return ""

    # ── v7: Narrative → Causal Bridge ───────────────────────────
    # Theory: Schank (1990) §7 — stories encode causal structure
    # implicitly. When narrative engine detects causal patterns in
    # thread moment sequences ("因为A...所以B..."), automatically
    # create causal edges in the memory graph.
    #
    # This is the SECOND channel for causal edge creation (the first
    # is LLM-driven extraction in _build_typed_edges). By extracting
    # causality from narrative structure, we capture cause-effect
    # relationships that emerge from the story itself — not just from
    # entity overlap heuristics.

    # Chinese causal pattern regexes (compiled once at class level)
    _CAUSAL_PATTERNS: list[tuple[str, float]] = [
        # (pattern, confidence_boost)
        ("因为.*所以", 0.85),
        ("因为.*于是", 0.80),
        ("因为.*因此", 0.80),
        ("由于.*所以", 0.80),
        ("由于.*因此", 0.75),
        ("之所以.*是因为", 0.85),
        ("导致", 0.70),
        ("引起", 0.65),
        ("造成", 0.65),
        ("使得", 0.70),
        ("促使", 0.75),
        ("从而", 0.60),
        ("以致", 0.60),
        ("决定了", 0.75),
    ]

    def extract_causal_edges(
        self,
        thread_id: str,
        memory_graph=None,
    ) -> int:
        """
        Extract causal relationships from narrative thread structure
        and create causal edges in the memory graph.

        Scans thread moments chronologically for causal patterns:
          1. Sequential moments A→B: if A's content contains causal
             keywords and B's content contains effect keywords, create
             a causal edge A → B.
          2. Causal keyword detection in moment content_summary:
             "因为A...所以B..." patterns extract (A, B) pairs.
          3. Turning point causality: inciting_incident → climax →
             resolution form implicit causal chains.

        Args:
            thread_id: The narrative thread to extract from
            memory_graph: MemoryGraph instance for edge creation

        Returns:
            Number of causal edges created
        """
        if memory_graph is None:
            return 0

        self.load()
        thread = self.threads.get(thread_id)
        if not thread or len(thread.moments) < 2:
            return 0

        import re

        created_count = 0

        # Sort moments chronologically
        sorted_moments = sorted(thread.moments, key=lambda m: m.timestamp)

        # ── Method 1: Sequential moment causality ──────────
        # For each pair of consecutive moments (A→B), check if A has
        # causal keywords and B has effect keywords
        for i in range(len(sorted_moments) - 1):
            moment_a = sorted_moments[i]
            moment_b = sorted_moments[i + 1]

            content_a = moment_a.content_summary
            content_b = moment_b.content_summary

            if not content_a or not content_b:
                continue

            confidence = self._detect_causal_link(content_a, content_b)

            if confidence >= 0.5:
                try:
                    memory_graph.add_edge(
                        from_id=moment_a.memory_id,
                        to_id=moment_b.memory_id,
                        relation_type="causal",
                        weight=confidence,
                        properties={
                            "source": "narrative_bridge",
                            "method": "sequential_moment",
                            "thread_id": thread_id,
                            "thread_title": thread.title,
                            "cause_summary": content_a[:120],
                            "effect_summary": content_b[:120],
                        },
                    )
                    created_count += 1
                    logger.debug(
                        f"Narrative→causal: {moment_a.memory_id[:8]} → "
                        f"{moment_b.memory_id[:8]} (confidence={confidence:.2f})"
                    )
                except Exception as e:
                    logger.warning(f"Narrative→causal edge creation failed: {e}")

        # ── Method 2: Intra-content causal pattern extraction ──
        # Single moment content may contain "因为A所以B" internally
        for moment in sorted_moments:
            content = moment.content_summary
            if not content or len(content) < 10:
                continue

            # Try to find causal pairs within the content
            for pattern, base_confidence in self._CAUSAL_PATTERNS[:5]:
                # Only use the sequential patterns (因为...所以 etc.)
                if ".*" not in pattern:
                    continue
                if pattern in content:
                    # This content itself describes a causal chain
                    # Mark it as self-referential causal insight
                    try:
                        memory_graph.add_edge(
                            from_id=moment.memory_id,
                            to_id=moment.memory_id,
                            relation_type="causal",
                            weight=base_confidence * 0.5,  # Lower weight for self-ref
                            properties={
                                "source": "narrative_bridge",
                                "method": "intra_content_causal",
                                "thread_id": thread_id,
                                "thread_title": thread.title,
                                "pattern": pattern,
                            },
                        )
                        created_count += 1
                    except Exception:
                        pass
                    break  # One self-ref edge per moment

        # ── Method 3: Turning point causal chains ──────────
        # inciting_incident → climax → resolution form implicit
        # causal chains (Schank script theory)
        turning_points = [
            m for m in sorted_moments
            if m.is_turning_point or m.role in ("inciting_incident", "climax", "resolution")
        ]

        for i in range(len(turning_points) - 1):
            tp_a = turning_points[i]
            tp_b = turning_points[i + 1]

            # Only create if roles form a meaningful progression
            role_pair = (tp_a.role, tp_b.role)
            valid_progressions = {
                ("inciting_incident", "climax"): 0.80,
                ("climax", "resolution"): 0.85,
                ("inciting_incident", "resolution"): 0.60,
                ("episode", "climax"): 0.55,
                ("climax", "episode"): 0.50,
            }

            if role_pair in valid_progressions:
                confidence = valid_progressions[role_pair]
                try:
                    memory_graph.add_edge(
                        from_id=tp_a.memory_id,
                        to_id=tp_b.memory_id,
                        relation_type="causal",
                        weight=confidence,
                        properties={
                            "source": "narrative_bridge",
                            "method": "turning_point_chain",
                            "thread_id": thread_id,
                            "thread_title": thread.title,
                            "cause_role": tp_a.role,
                            "effect_role": tp_b.role,
                        },
                    )
                    created_count += 1
                except Exception as e:
                    logger.warning(f"Turning point causal edge failed: {e}")

        if created_count > 0:
            logger.info(
                f"Narrative→causal bridge: {created_count} edges created "
                f"from thread '{thread.title}' [{thread_id}]"
            )

        return created_count

    def extract_causal_edges_all(self, memory_graph=None) -> dict:
        """
        Run Narrative→Causal Bridge on all active threads.

        Called from sleeptime _stage_consolidate() after narrative merge.

        Returns:
            {"total_edges_created": int, "threads_processed": int, "per_thread": dict}
        """
        if memory_graph is None:
            return {"total_edges_created": 0, "threads_processed": 0, "per_thread": {}}

        self.load()

        total = 0
        per_thread: dict[str, int] = {}

        for thread in self.threads.values():
            if thread.status == "resolved":
                continue
            if len(thread.moments) < 2:
                continue

            edges = self.extract_causal_edges(thread.id, memory_graph)
            if edges > 0:
                total += edges
                per_thread[thread.id] = edges

        logger.info(
            f"Narrative→causal bridge (all): {total} edges from "
            f"{len(per_thread)} threads"
        )

        return {
            "total_edges_created": total,
            "threads_processed": len(per_thread),
            "per_thread": per_thread,
        }

    @staticmethod
    def _detect_causal_link(content_a: str, content_b: str) -> float:
        """
        Detect if content_a is causally linked to content_b.

        Returns confidence score 0-1.
        """
        score = 0.0
        import re

        # Check 1: A contains causal keywords
        causal_in_a = any(kw in content_a for kw in [
            "因为", "导致", "引起", "造成", "使得", "促使", "决定了"
        ])
        if causal_in_a:
            score += 0.25

        # Check 2: B contains effect keywords
        effect_in_b = any(kw in content_b for kw in [
            "所以", "因此", "于是", "结果", "最终", "后来", "终于",
            "拿到", "获得", "实现", "达成", "完成", "通过",
            "失败", "被拒", "错过", "失去",
        ])
        if effect_in_b:
            score += 0.25

        # Check 3: Temporal proximity (adjacent moments already temporally close)
        score += 0.15  # Sequential moments are inherently temporally ordered

        # Check 4: Emotional coherence — does valence shift match expected pattern?
        # If A is negative and B describes a positive outcome, it's likely
        # a "turnaround" causal chain (Schank script deviation)
        if hasattr(NarrativeEngine, '_EMOTION_SHIFT_KEYWORDS'):
            pass  # Would need moment valence data

        # Check 5: Shared entity overlap between cause and effect
        words_a = set(content_a)
        words_b = set(content_b)
        # Remove common stop words by only looking at meaningful chars
        meaningful_a = {c for c in words_a if '一' <= c <= '鿿'}
        meaningful_b = {c for c in words_b if '一' <= c <= '鿿'}
        if meaningful_a and meaningful_b:
            overlap = len(meaningful_a & meaningful_b)
            total = len(meaningful_a | meaningful_b)
            if total > 0:
                score += 0.20 * (overlap / total)

        return min(1.0, round(score, 3))

    # ── Stats and diagnostics ─────────────────────────────────

    def get_stats(self) -> dict:
        """Get narrative engine statistics."""
        self.load()
        active = [t for t in self.threads.values() if t.status == "active"]
        total_moments = sum(t.moment_count for t in self.threads.values())
        turning_points = sum(
            sum(1 for m in t.moments if m.is_turning_point)
            for t in self.threads.values()
        )

        return {
            "total_threads": len(self.threads),
            "active_threads": len(active),
            "dormant_threads": sum(1 for t in self.threads.values() if t.status == "dormant"),
            "resolved_threads": sum(1 for t in self.threads.values() if t.status == "resolved"),
            "total_moments": total_moments,
            "turning_points": turning_points,
            "communities": len(self._communities),
            "life_periods": len(self._life_periods),
        }
