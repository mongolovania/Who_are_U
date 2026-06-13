# ============================================================
# Module: Narrative Branch Predictor (narrative_branch_predictor.py)
# L2: Future narrative branch prediction — "what might happen next?"
# L2：叙事分支预测器 — 前瞻性记忆·"接下来可能发生什么？"
#
# Theoretical foundation:
#   1. Schank (1990). Tell Me a Story. §7 — "Story creation is the
#      core of understanding. Future AI must not only understand past
#      stories but create 'possible future stories'." Narrative
#      intelligence is not just retrospective but prospective.
#   2. Dot by New Computer (2025). "Living History" — memory is not
#      only retrospective; AI can imagine the next chapter of your
#      life based on current narrative threads.
#   3. Conway & Pleydell-Pearce (2000). — Autobiographical memory
#      hierarchy includes "future self" projections as part of the
#      Working Self's goal system.
#   4. Foster (2017). — Forward replay in hippocampus: the brain
#      doesn't just replay past experiences; it anticipates future
#      paths. Forward replay during sleep is cognitive preparation.
#
# Design §12.6:
#   - Predict possible future branches from active narrative threads
#   - Three prediction modes: script completion, historical pattern,
#     trajectory extrapolation
#   - Precompute relevant memories for each branch
#
# Integration points:
#   - memory_orchestrator.dream(): predict_branches in PRECOMPUTE stage
#   - narrative_engine: active threads + life scripts
#   - memory_graph: graph traversal for historical pattern matching
#   - retrieval_engine: precompute relevant memories for each branch
# ============================================================

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory_palace.branch_predictor")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class NarrativeBranch:
    """A predicted future branch from a narrative thread."""
    id: str = ""
    thread_id: str = ""
    thread_title: str = ""

    # What kind of prediction
    branch_type: str = "script_completion"
    # script_completion | historical_pattern | trajectory_extrapolation

    # The prediction
    predicted_outcome: str = ""     # "可能会拿到offer并开始新工作"
    confidence: float = 0.5         # 0-1
    timeframe: str = ""             # "2周内" | "1个月内" | "未知"

    # Evidence
    evidence_memory_ids: list[str] = field(default_factory=list)
    similar_thread_id: str = ""     # For historical patterns
    script_stage: str = ""          # Current stage in life script

    # Precomputed context for quick retrieval when this branch materializes
    precomputed_context: str = ""   # Relevant memory summaries
    relevant_memory_ids: list[str] = field(default_factory=list)

    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "thread_title": self.thread_title,
            "branch_type": self.branch_type,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "evidence_memory_ids": self.evidence_memory_ids,
            "similar_thread_id": self.similar_thread_id,
            "script_stage": self.script_stage,
            "relevant_memory_ids": self.relevant_memory_ids,
            "created_at": self.created_at,
        }


# Life script stage progression maps
# Maps current script stage → possible next stages with probabilities
_SCRIPT_PROGRESSIONS: dict[str, list[tuple[str, float, str]]] = {
    "职业转型": [
        ("开始面试", 0.70, "1-4周"),
        ("收到offer", 0.50, "1-8周"),
        ("决定入职", 0.40, "2-12周"),
        ("适应新环境", 0.60, "入职后2-4周"),
        ("重新评估职业方向", 0.35, "入职后3-6个月"),
        ("寻求晋升/新挑战", 0.30, "入职后6-18个月"),
    ],
    "亲密关系": [
        ("关系进入稳定期", 0.65, "1-3个月"),
        ("面对第一次重大冲突", 0.55, "2-6个月"),
        ("考虑下一步（同居/婚姻）", 0.35, "6-18个月"),
        ("重新评估关系", 0.30, "6-24个月"),
    ],
    "成长探索": [
        ("经历小突破", 0.60, "1-4周"),
        ("遇到瓶颈和回退", 0.50, "2-8周"),
        ("找到新的方向", 0.35, "1-6个月"),
        ("形成新习惯/新常态", 0.45, "2-6个月"),
    ],
    "家庭关系": [
        ("沟通改善", 0.55, "1-4周"),
        ("重新理解家人", 0.40, "1-6个月"),
        ("建立新边界", 0.35, "1-3个月"),
        ("关系进入新平衡", 0.50, "2-6个月"),
    ],
    "健康管理": [
        ("养成新习惯", 0.50, "2-8周"),
        ("看到初步效果", 0.45, "4-12周"),
        ("遇到瓶颈期", 0.40, "1-3个月"),
        ("调整策略", 0.35, "1-3个月"),
    ],
    "财务规划": [
        ("达到第一个里程碑", 0.45, "3-12个月"),
        ("遇到意外支出", 0.40, "随机"),
        ("调整财务策略", 0.35, "1-3个月"),
        ("实现主要目标", 0.25, "6-24个月"),
    ],
}


# ═══════════════════════════════════════════════════════════════
# Narrative Branch Predictor
# ═══════════════════════════════════════════════════════════════


class NarrativeBranchPredictor:
    """
    Predict possible future narrative branches from active threads.

    Three prediction modes:
      1. Script completion: what happens next in this life script?
      2. Historical pattern: what happened in a similar past thread?
      3. Trajectory extrapolation: where is the emotional arc heading?
    """

    def __init__(self, user_id: str = ""):
        self.user_id = user_id
        self._branches: dict[str, list[NarrativeBranch]] = {}  # thread_id → branches
        self._prediction_count: int = 0

        # Config
        self.max_branches_per_thread: int = 3
        self.min_confidence: float = 0.2
        self.context_window_days: int = 90  # Look back for historical patterns

    # ── Prediction ─────────────────────────────────────────────

    def predict_branches(
        self,
        thread_id: str,
        narrative_engine,
        graph=None,
        retrieval_engine=None,
        top_k: int = 3,
    ) -> list[NarrativeBranch]:
        """
        Predict possible future branches for a narrative thread.

        Args:
            thread_id: The narrative thread to predict from
            narrative_engine: NarrativeEngine instance
            graph: MemoryGraph for historical pattern search
            retrieval_engine: RetrievalEngine for precomputation
            top_k: Max branches to predict

        Returns:
            List of predicted branches
        """
        if narrative_engine is None:
            return []

        narrative_engine.load()
        thread = narrative_engine.threads.get(thread_id)
        if not thread or thread.status != "active":
            return []

        branches: list[NarrativeBranch] = []

        # ── Method 1: Script completion ──
        script_branches = self._predict_from_script(thread)
        branches.extend(script_branches)

        # ── Method 2: Historical pattern ──
        if graph and len(branches) < top_k:
            history_branches = self._predict_from_history(
                thread, narrative_engine, graph
            )
            branches.extend(history_branches)

        # ── Method 3: Trajectory extrapolation ──
        if len(branches) < top_k:
            trajectory_branches = self._predict_from_trajectory(thread)
            branches.extend(trajectory_branches)

        # Filter and rank
        branches = [b for b in branches if b.confidence >= self.min_confidence]
        branches.sort(key=lambda b: b.confidence, reverse=True)
        selected = branches[:top_k]

        # Precompute relevant memories for each branch
        if retrieval_engine:
            for branch in selected:
                self.precompute_relevant_memories(branch, retrieval_engine)

        self._branches[thread_id] = selected
        self._prediction_count += 1

        logger.info(
            f"Predicted {len(selected)} branches for thread "
            f"'{thread.title}' [{thread_id[:8]}]: "
            + ", ".join(
                f"{b.predicted_outcome[:40]} ({b.branch_type}, {b.confidence:.0%})"
                for b in selected
            )
        )

        return selected

    def predict_all_active(
        self,
        narrative_engine,
        graph=None,
        retrieval_engine=None,
    ) -> dict[str, list[NarrativeBranch]]:
        """
        Predict branches for all active threads.

        Called from sleeptime PRECOMPUTE stage.
        """
        if narrative_engine is None:
            return {}

        narrative_engine.load()
        result: dict[str, list[NarrativeBranch]] = {}

        for thread in narrative_engine.threads.values():
            if thread.status != "active":
                continue

            branches = self.predict_branches(
                thread.id, narrative_engine, graph, retrieval_engine
            )
            if branches:
                result[thread.id] = branches

        logger.info(
            f"Predicted branches for {len(result)} active threads, "
            f"{sum(len(b) for b in result.values())} total branches"
        )

        return result

    # ── Precomputation ─────────────────────────────────────────

    def precompute_relevant_memories(
        self,
        branch: NarrativeBranch,
        retrieval_engine,
    ):
        """
        Precompute relevant memories for a predicted branch.

        When the user actually reaches this branch, these memories
        can be injected immediately without search latency.
        """
        if retrieval_engine is None:
            return

        # Build a query from the predicted outcome
        query = f"关于 {branch.predicted_outcome}"

        try:
            import asyncio

            # Use retrieval engine to find relevant memories
            # (synchronous fallback if async not available)
            if hasattr(retrieval_engine, 'search_sync'):
                results = retrieval_engine.search_sync(query, top_k=5)
            elif hasattr(retrieval_engine, 'search'):
                result = retrieval_engine.search(query=query, top_k=5)
                if asyncio.iscoroutine(result):
                    # Can't await in sync context — skip precomputation
                    return
                results = result
            else:
                return

            if isinstance(results, list):
                branch.relevant_memory_ids = [
                    r.get("id", r.get("memory_id", ""))
                    for r in results[:5]
                    if isinstance(r, dict)
                ]

                # Build precomputed context summary
                summaries = []
                for r in results[:3]:
                    if isinstance(r, dict):
                        summaries.append(
                            r.get("content", r.get("name", ""))[:100]
                        )
                branch.precomputed_context = " | ".join(summaries)

        except Exception as e:
            logger.debug(f"Precomputation for branch {branch.id} failed: {e}")

    # ── Retrieval ──────────────────────────────────────────────

    def get_branches_for(self, thread_id: str) -> list[NarrativeBranch]:
        """Get previously predicted branches for a thread."""
        return self._branches.get(thread_id, [])

    def get_active_branches(self) -> dict[str, list[NarrativeBranch]]:
        """Get all active predicted branches."""
        return dict(self._branches)

    # ── Private: Script completion prediction ──────────────────

    def _predict_from_script(
        self,
        thread,
    ) -> list[NarrativeBranch]:
        """
        Predict based on life script progression:
        "Given where you are in the 'career transition' script,
        what typically happens next?"
        """
        branches: list[NarrativeBranch] = []
        theme = thread.theme
        script_stages = _SCRIPT_PROGRESSIONS.get(theme, [])

        if not script_stages:
            return branches

        # Determine current script stage from latest moments
        current_stage = self._infer_script_stage(thread, theme)

        # Find next stages after current
        candidates = []
        for label, prob, timeframe in script_stages:
            # Simple: first few stages are most relevant
            candidates.append((label, prob, timeframe))

        # Sort by probability and pick top
        candidates.sort(key=lambda x: x[1], reverse=True)

        for label, prob, timeframe in candidates[:self.max_branches_per_thread]:
            branch = NarrativeBranch(
                thread_id=thread.id,
                thread_title=thread.title,
                branch_type="script_completion",
                predicted_outcome=f"在「{theme}」这个阶段，接下来可能会{label}",
                confidence=prob,
                timeframe=timeframe,
                script_stage=current_stage,
                evidence_memory_ids=[
                    m.memory_id for m in thread.moments[-3:]
                ] if thread.moments else [],
            )
            branches.append(branch)

        return branches

    def _infer_script_stage(self, thread, theme: str) -> str:
        """Infer the current stage in a life script from thread moments."""
        if not thread.moments:
            return "开始阶段"

        # Map keywords to script stages
        stage_keywords = {
            "职业转型": {
                "不满": "不满现状",
                "面试": "面试中",
                "offer": "拿到offer",
                "入职": "刚入职",
                "跳槽": "探索选项",
                "辞职": "决策节点",
            },
            "亲密关系": {
                "认识": "相识阶段",
                "好感": "好感发展",
                "在一起": "关系确认",
                "吵架": "冲突磨合",
                "分手": "分离",
            },
        }

        keywords = stage_keywords.get(theme, {})
        latest_content = thread.moments[-1].content_summary if thread.moments else ""

        for keyword, stage in keywords.items():
            if keyword in latest_content:
                return stage

        # Default: infer from moment count
        if len(thread.moments) <= 2:
            return "开始阶段"
        elif len(thread.moments) <= 5:
            return "进行中"

        return "深入阶段"

    # ── Private: Historical pattern prediction ──────────────────

    def _predict_from_history(
        self,
        thread,
        narrative_engine,
        graph,
    ) -> list[NarrativeBranch]:
        """
        Predict based on similar past threads:
        "Last time you went through something similar, X happened.
        This time might follow a similar pattern."
        """
        branches: list[NarrativeBranch] = []

        # Find resolved threads with the same theme
        similar_threads = [
            t for t in narrative_engine.threads.values()
            if t.id != thread.id
            and t.theme == thread.theme
            and t.status in ("resolved", "dormant")
        ]

        for similar in similar_threads[:2]:
            # Extract the outcome from the resolved thread
            outcome = ""
            if similar.moments:
                # The last moment's summary is the outcome
                outcome = similar.moments[-1].content_summary[:100]

            if outcome:
                branch = NarrativeBranch(
                    thread_id=thread.id,
                    thread_title=thread.title,
                    branch_type="historical_pattern",
                    predicted_outcome=f"上次你经历类似的「{thread.theme}」时，最终{outcome}。这次可能会走类似的路，也可能不同。",
                    confidence=0.45,
                    timeframe="未知",
                    similar_thread_id=similar.id,
                    evidence_memory_ids=[
                        m.memory_id for m in similar.moments[-3:]
                    ] if similar.moments else [],
                )
                branches.append(branch)

        return branches

    # ── Private: Trajectory extrapolation ──────────────────────

    def _predict_from_trajectory(
        self,
        thread,
    ) -> list[NarrativeBranch]:
        """
        Predict based on emotional trajectory extrapolation:
        "Your valence has been trending upward for 3 weeks —
        this positive momentum may continue."
        """
        if len(thread.moments) < 3:
            return []

        branches: list[NarrativeBranch] = []

        # Compute emotional trajectory
        valences = [m.valence for m in thread.moments]
        arousals = [m.arousal for m in thread.moments]
        importance_scores = [m.importance for m in thread.moments]

        # Recent trend (last 3 vs all)
        recent_val = valences[-3:]
        all_val = valences
        recent_avg = sum(recent_val) / len(recent_val)
        all_avg = sum(all_val) / len(all_val)
        trend = recent_avg - all_avg

        # Turning point frequency
        tp_count = sum(1 for m in thread.moments if m.is_turning_point)
        tp_frequency = tp_count / max(len(thread.moments), 1)

        if abs(trend) > 0.1:
            direction = "上升" if trend > 0 else "下降"
            direction_label = "好转" if trend > 0 else "低落"

            branch = NarrativeBranch(
                thread_id=thread.id,
                thread_title=thread.title,
                branch_type="trajectory_extrapolation",
                predicted_outcome=(
                    f"你的情绪轨迹在过去几周呈{direction}趋势。"
                    f"如果这个趋势延续，情况可能会继续{direction_label}。"
                    + ("可能会有新的转折点。" if tp_frequency > 0.3 else "")
                ),
                confidence=min(0.55, 0.3 + abs(trend) * 1.5),
                timeframe="2-4周",
                evidence_memory_ids=[
                    m.memory_id for m in thread.moments[-5:]
                ] if thread.moments else [],
            )
            branches.append(branch)

        # High turning point frequency → expect more change
        if tp_frequency > 0.3:
            branch = NarrativeBranch(
                thread_id=thread.id,
                thread_title=thread.title,
                branch_type="trajectory_extrapolation",
                predicted_outcome=(
                    f"你最近经历了{tp_count}个转折点——变化可能还没有结束。"
                    "接下来几周可能会有新的重要事件发生。"
                ),
                confidence=min(0.50, tp_frequency * 1.2),
                timeframe="1-4周",
                evidence_memory_ids=[
                    m.memory_id for m in thread.moments if m.is_turning_point
                ][:5] if thread.moments else [],
            )
            branches.append(branch)

        return branches

    # ── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get branch predictor statistics."""
        total_branches = sum(len(b) for b in self._branches.values())
        type_counts = {}
        for branches in self._branches.values():
            for b in branches:
                type_counts[b.branch_type] = type_counts.get(b.branch_type, 0) + 1

        return {
            "total_predicted_branches": total_branches,
            "threads_with_branches": len(self._branches),
            "predictions_generated": self._prediction_count,
            "avg_confidence": round(
                sum(
                    b.confidence
                    for branches in self._branches.values()
                    for b in branches
                ) / max(total_branches, 1),
                3,
            ),
            "branch_types": type_counts,
        }
