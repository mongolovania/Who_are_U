# ============================================================
# Module: Counterfactual Memory (counterfactual_memory.py)
# L2: Counterfactual reasoning on causal graph — Pearl Ladder L3.
# L2：反事实推理 — 在因果图上进行"如果当初..."推理
#
# Theoretical foundation:
#   1. Pearl & Mackenzie (2018). The Book of Why: The New Science of
#      Cause and Effect. Basic Books. — Causal Ladder L3 (Counterfactuals):
#      "If X had not happened, would Y still have happened?"
#   2. Pearl (2009). Causality: Models, Reasoning, and Inference.
#      Cambridge University Press. — Structural Causal Models (SCM)
#      and the do-calculus for counterfactual inference.
#   3. CausalRAG (ACL 2025). — Causal graph constraints for retrieval;
#      counterfactual reasoning improves decision understanding by
#      enabling "what if" exploration.
#   4. Causal Cartographer (2025). arXiv:2505.14396. — Graph RAG agent
#      + counterfactual inference agent for constrained what-if reasoning.
#
# Design §12.4:
#   - Generate counterfactuals for decision/event nodes
#   - Evaluate counterfactual probability using graph structure heuristics
#   - Store counterfactuals as structured metadata
#
# Integration points:
#   - memory_orchestrator.dream(): counterfactual generation in EVOLVE stage
#   - memory_graph: uses causal + temporal edges for alternative path search
#   - memory_orchestrator.chat(): inject counterfactual insights on request
# ============================================================

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory_palace.counterfactual")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class Counterfactual:
    """A single counterfactual hypothesis."""
    id: str = ""
    anchor_memory_id: str = ""         # The event being questioned
    hypothesis: str = ""               # "如果没有X..."
    alternative_outcome: str = ""      # Hypothetical alternative result
    confidence: float = 0.3            # How plausible this counterfactual is (0-1)
    method: str = ""                   # graph_search | pattern_match | causal_inversion
    evidence: list[str] = field(default_factory=list)  # Memory IDs supporting this
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "anchor_memory_id": self.anchor_memory_id,
            "hypothesis": self.hypothesis,
            "alternative_outcome": self.alternative_outcome,
            "confidence": self.confidence,
            "method": self.method,
            "evidence": self.evidence,
            "created_at": self.created_at,
        }


@dataclass
class CounterfactualReport:
    """Report for counterfactual generation on a node."""
    anchor_memory_id: str
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    causal_paths_found: int = 0
    alternatives_explored: int = 0
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Counterfactual Memory Engine
# ═══════════════════════════════════════════════════════════════


class CounterfactualMemory:
    """
    Counterfactual reasoning on the memory causal graph.

    Enables "what if" exploration:
      - "If I hadn't quit my job, would I be happier?"
      - "If I had accepted that offer, where would I be now?"
      - "What if I had spoken up during that meeting?"

    Uses three methods:
      1. Graph search: find alternative causal paths that bypass the anchor
      2. Pattern match: use known life scripts to suggest plausible alternatives
      3. Causal inversion: for each incoming causal edge, ask "what if cause didn't happen?"
    """

    def __init__(self, user_id: str = ""):
        self.user_id = user_id
        self._stored: dict[str, list[Counterfactual]] = {}  # memory_id → counterfactuals
        self._generation_count: int = 0

        # Config
        self.max_counterfactuals_per_node: int = 3
        self.min_confidence: float = 0.1
        self.search_depth: int = 3

    # ── Generation ─────────────────────────────────────────────

    def generate_counterfactuals(
        self,
        memory_id: str,
        graph,
        bucket_mgr=None,
        top_k: int = 3,
    ) -> CounterfactualReport:
        """
        Generate counterfactual hypotheses for a memory node.

        For a given decision/event, explore what might have happened
        if the causal chain had been different.

        Args:
            memory_id: The anchor memory (typically a decision or milestone)
            graph: MemoryGraph instance
            bucket_mgr: BucketManager for content lookup
            top_k: Max counterfactuals to generate

        Returns:
            CounterfactualReport with generated hypotheses
        """
        report = CounterfactualReport(anchor_memory_id=memory_id)
        counterfactuals: list[Counterfactual] = []

        if graph is None:
            return report

        # ── Method 1: Causal inversion ──
        # For each incoming causal edge to this node, ask:
        # "What if this cause hadn't happened?"
        incoming_causal = self._get_incoming_causal_edges(memory_id, graph)
        report.causal_paths_found = len(incoming_causal)

        for edge in incoming_causal[:3]:
            cause_id = edge.get("from_id", "")
            if not cause_id:
                continue

            cf = self._generate_inversion_counterfactual(
                cause_id, memory_id, edge, graph, bucket_mgr
            )
            if cf and cf.confidence >= self.min_confidence:
                counterfactuals.append(cf)

        # ── Method 2: Alternative path search ──
        # Find temporal/thematic neighbors that are NOT on the causal
        # path → these represent "roads not taken"
        alt_counterfactuals = self._generate_alternative_path_counterfactuals(
            memory_id, graph, bucket_mgr, max(0, top_k - len(counterfactuals))
        )
        counterfactuals.extend(alt_counterfactuals)
        report.alternatives_explored = len(alt_counterfactuals) + len(incoming_causal)

        # ── Method 3: Life script pattern ──
        # Use known life scripts to suggest plausible alternative outcomes
        if len(counterfactuals) < top_k:
            script_cfs = self._generate_script_pattern_counterfactuals(
                memory_id, graph, bucket_mgr, top_k - len(counterfactuals)
            )
            counterfactuals.extend(script_cfs)

        # Deduplicate and sort by confidence
        seen = set()
        unique: list[Counterfactual] = []
        for cf in sorted(counterfactuals, key=lambda c: c.confidence, reverse=True):
            key = (cf.hypothesis[:60],)
            if key not in seen:
                seen.add(key)
                unique.append(cf)

        report.counterfactuals = unique[:top_k]

        # Store for later retrieval
        self._stored[memory_id] = report.counterfactuals
        self._generation_count += 1

        logger.debug(
            f"Generated {len(report.counterfactuals)} counterfactuals "
            f"for {memory_id[:12]} (from {report.causal_paths_found} causal paths, "
            f"{report.alternatives_explored} alternatives)"
        )

        return report

    # ── Evaluation ─────────────────────────────────────────────

    def evaluate_counterfactual(
        self,
        cause_id: str,
        effect_id: str,
        graph,
    ) -> float:
        """
        Estimate probability that the effect would have occurred
        WITHOUT the cause.

        Uses graph structure heuristics:
          1. How many other paths lead to the effect? (more = less dependent)
          2. How strong is the causal edge? (stronger = more dependent)
          3. Are there temporal neighbors suggesting independent causation?

        Returns:
            Probability (0-1) that effect would STILL have happened
            without the cause. Higher = the cause was less critical.
        """
        if graph is None:
            return 0.5

        # Get the causal edge
        edges = graph.get_all_edges_for_node(effect_id)
        causal_edges = [e for e in edges if e.get("relation_type") == "causal"
                        and e.get("from_id") == cause_id]

        if not causal_edges:
            # No direct causal edge → effect is independent
            return 0.9

        edge = causal_edges[0]
        edge_weight = edge.get("weight", 0.5)

        # Count alternative paths to the effect
        other_causes = [
            e for e in edges
            if e.get("relation_type") == "causal"
            and e.get("to_id") == effect_id
            and e.get("from_id") != cause_id
        ]
        alt_cause_count = len(other_causes)

        # Count temporal neighbors (could be independent causes)
        temporal_edges = [
            e for e in edges
            if e.get("relation_type") == "temporal"
            and e.get("to_id") == effect_id
        ]
        temporal_count = len(temporal_edges)

        # Compute independence probability
        # More alternative causes + temporal neighbors → more independent
        independence = 0.3  # Base: some chance effect would happen anyway

        independence += 0.15 * min(alt_cause_count, 3)  # Up to +0.45
        independence += 0.05 * min(temporal_count, 4)    # Up to +0.20
        independence -= 0.2 * edge_weight                 # Strong edge = less independent

        return round(max(0.05, min(0.95, independence)), 3)

    # ── Retrieval ──────────────────────────────────────────────

    def get_counterfactuals_for(self, memory_id: str) -> list[Counterfactual]:
        """Get previously generated counterfactuals for a memory."""
        return self._stored.get(memory_id, [])

    def get_all_counterfactuals(self) -> dict[str, list[Counterfactual]]:
        """Get all stored counterfactuals."""
        return dict(self._stored)

    # ── Private: Generation methods ────────────────────────────

    def _generate_inversion_counterfactual(
        self,
        cause_id: str,
        effect_id: str,
        edge: dict,
        graph,
        bucket_mgr,
    ) -> Counterfactual | None:
        """Generate counterfactual by inverting a causal edge."""
        cause_content = self._get_content(cause_id, bucket_mgr)
        effect_content = self._get_content(effect_id, bucket_mgr)

        if not cause_content or not effect_content:
            return None

        # Estimate how likely the effect would have happened without this cause
        independence = self.evaluate_counterfactual(cause_id, effect_id, graph)

        # Build hypothesis
        cause_summary = cause_content[:60]
        effect_summary = effect_content[:60]

        hypothesis = f"如果没有「{cause_summary}」"
        if independence > 0.7:
            alternative = f"「{effect_summary}」很可能仍然会发生（独立概率 {independence:.0%}）"
            confidence = independence
        elif independence > 0.4:
            alternative = f"「{effect_summary}」可能会推迟或减弱，但不一定会消失"
            confidence = 0.5
        else:
            alternative = f"「{effect_summary}」可能不会发生，或者以完全不同的方式发生"
            confidence = 1.0 - independence

        return Counterfactual(
            anchor_memory_id=effect_id,
            hypothesis=hypothesis,
            alternative_outcome=alternative,
            confidence=round(confidence, 3),
            method="causal_inversion",
            evidence=[cause_id, effect_id],
        )

    def _generate_alternative_path_counterfactuals(
        self,
        memory_id: str,
        graph,
        bucket_mgr,
        count: int,
    ) -> list[Counterfactual]:
        """
        Generate counterfactuals by searching for alternative paths —
        "roads not taken" from the same temporal/thematic context.
        """
        if count <= 0:
            return []

        cfs: list[Counterfactual] = []

        # Find all edges connected to this node
        all_edges = graph.get_all_edges_for_node(memory_id)

        # Find thematic neighbors (similar topics, different outcomes)
        thematic = [e for e in all_edges if e.get("relation_type") == "thematic"]
        temporal = [e for e in all_edges if e.get("relation_type") == "temporal"]

        # Look at thematic neighbors to find "what others did in similar situation"
        for edge in thematic[:count]:
            neighbor_id = (
                edge.get("to_id") if edge.get("from_id") == memory_id
                else edge.get("from_id")
            )
            if not neighbor_id:
                continue

            neighbor_content = self._get_content(neighbor_id, bucket_mgr)
            if not neighbor_content:
                continue

            # Check if this neighbor represents a different outcome
            anchor_content = self._get_content(memory_id, bucket_mgr)
            if anchor_content and self._different_outcome(anchor_content, neighbor_content):
                cf = Counterfactual(
                    anchor_memory_id=memory_id,
                    hypothesis="如果我选择了另一条路...",
                    alternative_outcome=f"可能会像「{neighbor_content[:80]}」这样",
                    confidence=0.35,
                    method="graph_search",
                    evidence=[neighbor_id],
                )
                cfs.append(cf)

        # Look at temporal neighbors for near-miss alternatives
        for edge in temporal[:max(0, count - len(cfs))]:
            neighbor_id = (
                edge.get("to_id") if edge.get("from_id") == memory_id
                else edge.get("from_id")
            )
            if not neighbor_id:
                continue

            neighbor_content = self._get_content(neighbor_id, bucket_mgr)
            if not neighbor_content:
                continue

            properties = edge.get("properties", {})
            days_apart = properties.get("days_apart", 999)

            if days_apart <= 3:  # Close in time = more likely alternative
                cf = Counterfactual(
                    anchor_memory_id=memory_id,
                    hypothesis="如果那天选择了不同的行动...",
                    alternative_outcome=f"同期的「{neighbor_content[:80]}」暗示了另一种可能",
                    confidence=0.25,
                    method="graph_search",
                    evidence=[neighbor_id],
                )
                cfs.append(cf)

        return cfs

    def _generate_script_pattern_counterfactuals(
        self,
        memory_id: str,
        graph,
        bucket_mgr,
        count: int,
    ) -> list[Counterfactual]:
        """
        Generate counterfactuals using known life script patterns.
        Schank: life scripts encode expected outcomes — suggesting
        alternatives that fit the script but weren't taken.
        """
        if count <= 0:
            return []

        content = self._get_content(memory_id, bucket_mgr)
        if not content:
            return []

        cfs: list[Counterfactual] = []

        # Life script patterns with alternative outcomes
        script_alternatives = {
            "辞职": [
                ("如果没有辞职", "可能会在原来的岗位上逐渐适应，或者等到更好的机会再走"),
            ],
            "接受": [
                ("如果当时拒绝了", "可能会遇到更适合的机会，但也可能错过这次成长"),
            ],
            "拒绝": [
                ("如果当时接受了", "生活可能会走向完全不同的方向"),
            ],
            "分手": [
                ("如果没有分开", "关系可能会继续消耗，也可能会慢慢修复"),
            ],
            "入职": [
                ("如果选择了另一家公司", "职业发展轨迹可能会完全不同"),
            ],
            "面试": [
                ("如果那次面试通过了", "人生可能会进入一个完全不同的阶段"),
                ("如果没有去面试", "可能会在现有轨道上继续，等待下一个机会"),
            ],
            "放弃": [
                ("如果再坚持一下", "可能会突破瓶颈，但也可能只是徒劳"),
            ],
            "决定": [
                ("如果当时犹豫了", "可能会错过时机，或者避免一次冲动"),
            ],
        }

        for keyword, alternatives in script_alternatives.items():
            if keyword in content:
                for hypothesis, outcome in alternatives[:count]:
                    cf = Counterfactual(
                        anchor_memory_id=memory_id,
                        hypothesis=hypothesis,
                        alternative_outcome=outcome,
                        confidence=0.30,
                        method="pattern_match",
                        evidence=[],
                    )
                    cfs.append(cf)
                break  # One keyword match per memory

        return cfs

    # ── Private: Helpers ───────────────────────────────────────

    @staticmethod
    def _get_content(memory_id: str, bucket_mgr) -> str:
        """Get content text for a memory."""
        if bucket_mgr is None:
            return ""
        try:
            import asyncio
            if hasattr(bucket_mgr, 'read'):
                result = bucket_mgr.read(memory_id)
                if asyncio.iscoroutine(result):
                    return ""
                if isinstance(result, dict):
                    return result.get("content", "")
                return str(result) if result else ""
        except Exception:
            pass
        return ""

    @staticmethod
    def _different_outcome(content_a: str, content_b: str) -> bool:
        """Check if two content strings describe different outcomes."""
        positive_markers = ["成功", "通过", "拿下", "开心", "幸福", "满意", "拿到", "获得"]
        negative_markers = ["失败", "被拒", "错过", "难过", "后悔", "失去", "放弃"]

        a_pos = any(m in content_a for m in positive_markers)
        a_neg = any(m in content_a for m in negative_markers)
        b_pos = any(m in content_b for m in positive_markers)
        b_neg = any(m in content_b for m in negative_markers)

        # Different valence = different outcome
        return (a_pos and b_neg) or (a_neg and b_pos)

    @staticmethod
    def _get_incoming_causal_edges(memory_id: str, graph) -> list[dict]:
        """Get all incoming causal edges to a node."""
        if graph is None:
            return []
        edges = graph.get_all_edges_for_node(memory_id)
        return [
            e for e in edges
            if e.get("relation_type") == "causal"
            and e.get("to_id") == memory_id
        ]

    # ── Persistence ────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize all stored counterfactuals."""
        return {
            mid: [cf.to_dict() for cf in cfs]
            for mid, cfs in self._stored.items()
        }

    @classmethod
    def from_dict(cls, data: dict, user_id: str = "") -> CounterfactualMemory:
        """Deserialize from dict."""
        instance = cls(user_id=user_id)
        for mid, cf_list in data.items():
            instance._stored[mid] = [
                Counterfactual(
                    id=cf.get("id", ""),
                    anchor_memory_id=cf.get("anchor_memory_id", mid),
                    hypothesis=cf.get("hypothesis", ""),
                    alternative_outcome=cf.get("alternative_outcome", ""),
                    confidence=cf.get("confidence", 0.3),
                    method=cf.get("method", ""),
                    evidence=cf.get("evidence", []),
                    created_at=cf.get("created_at", ""),
                )
                for cf in cf_list
            ]
        return instance

    def get_stats(self) -> dict:
        """Get counterfactual memory statistics."""
        total = sum(len(cfs) for cfs in self._stored.values())
        methods = {}
        for cfs in self._stored.values():
            for cf in cfs:
                methods[cf.method] = methods.get(cf.method, 0) + 1

        return {
            "total_counterfactuals": total,
            "nodes_with_counterfactuals": len(self._stored),
            "generations": self._generation_count,
            "avg_confidence": round(
                sum(cf.confidence for cfs in self._stored.values() for cf in cfs) / max(total, 1), 3
            ),
            "methods": methods,
        }
