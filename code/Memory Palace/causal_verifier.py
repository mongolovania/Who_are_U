# ============================================================
# Module: Causal Verifier (causal_verifier.py)
# L2: Causal edge coherence verification — Pearl Ladder L1→L2 bridge.
# L2：因果边验证器 — 排除伪因果关系·提升因果边质量
#
# Theoretical foundation:
#   1. Pearl (2009). Causality: Models, Reasoning, and Inference.
#      Cambridge University Press. — Causal Ladder: Association(L1)
#      → Intervention(L2) → Counterfactuals(L3). This module bridges
#      L1→L2 by verifying that L1 associations meet temporal and
#      coherence criteria before being treated as causal.
#   2. Pearl & Mackenzie (2018). The Book of Why. Basic Books. —
#      Three rungs of the causal ladder, operationalized as:
#        L1: see (association → causal edge creation)
#        L2: do (intervention → this module verifies coherence)
#        L3: imagine (counterfactuals → counterfactual_memory.py)
#   3. CausalRAG (ACL 2025). — Causal graph constraints improve
#      retrieval quality; verification is pre-requisite for
#      reliable causal retrieval.
#   4. Causal Cartographer (2025). arXiv:2505.14396. — Graph RAG
#      agent + counterfactual agent; causal network as constrained
#      inference substrate.
#
# Design §12.2:
#   - Verify temporal precedence (cause before effect)
#   - Verify coherence (shared entities/concepts)
#   - Detect circular causality (A→B→C→A cycles)
#   - Detect isolated causal chains (no supporting evidence)
#   - Batch verification for sleeptime CONSOLIDATE stage
#
# Integration points:
#   - memory_orchestrator._async_hold_pipeline(): verify new edges
#   - sleeptime_compute._stage_consolidate(): batch verify all
#   - memory_graph: read/write edge properties (weight adjustment)
# ============================================================

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory_palace.causal_verifier")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class CausalVerificationResult:
    """Result of verifying a single causal edge."""
    edge_id: str
    from_id: str = ""
    to_id: str = ""
    valid: bool = True
    confidence: float = 1.0        # Adjusted confidence after verification (0-1)
    original_weight: float = 1.0   # Original edge weight before verification
    adjusted_weight: float = 1.0   # Post-verification weight (may be reduced)
    issues: list[str] = field(default_factory=list)
    verified_at: str = ""

    def __post_init__(self):
        if not self.verified_at:
            self.verified_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "valid": self.valid,
            "confidence": self.confidence,
            "original_weight": self.original_weight,
            "adjusted_weight": self.adjusted_weight,
            "issues": self.issues,
            "verified_at": self.verified_at,
        }


@dataclass
class SubgraphVerificationReport:
    """Report for verifying all causal edges reachable from a node."""
    root_id: str
    total_edges_checked: int = 0
    valid_edges: int = 0
    suspicious_edges: int = 0     # Valid but low confidence
    invalid_edges: int = 0        # Failed verification
    circular_chains: list[list[str]] = field(default_factory=list)
    isolated_chains: list[list[str]] = field(default_factory=list)
    results: list[CausalVerificationResult] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Chinese causal pattern detectors (Schank 1990 narrative→causal)
# ═══════════════════════════════════════════════════════════════

# Causal keywords for coherence verification
_CAUSAL_KEYWORDS_CN: list[str] = [
    "因为", "所以", "因此", "于是", "导致", "引起", "造成",
    "使得", "促使", "从而", "以致", "之所以", "是由于",
    "决定", "选择", "放弃", "接受", "拒绝",  # Action→outcome markers
]

# Result/effect keywords — indicate an outcome of a prior cause
_EFFECT_KEYWORDS_CN: list[str] = [
    "结果", "最终", "后来", "于是", "所以", "因此", "终于",
    "拿到", "获得", "实现", "达成", "完成", "通过",
    "失败", "被拒", "错过", "失去",
]

# Emotion→action causality chains (Schank script theory)
_EMOTION_ACTION_PAIRS: list[tuple[str, str]] = [
    ("焦虑", "准备"),
    ("压抑", "离职"),
    ("兴奋", "接受"),
    ("愤怒", "拒绝"),
    ("迷茫", "探索"),
    ("后悔", "改变"),
    ("期待", "尝试"),
    ("害怕", "逃避"),
]


# ═══════════════════════════════════════════════════════════════
# Causal Verifier
# ═══════════════════════════════════════════════════════════════


class CausalVerifier:
    """
    Verify causal edges in the memory graph for coherence.

    Three-stage verification:
      1. Temporal precedence: cause node must precede effect node
      2. Coherence: shared entities/concepts between cause and effect
      3. Structural: no circular causality, no isolated single edges
    """

    def __init__(self, user_id: str = ""):
        self.user_id = user_id
        self._verification_count: int = 0
        self._cache: dict[str, CausalVerificationResult] = {}

        # Thresholds
        self.temporal_grace_minutes: int = 5    # Allow small time reversals
        self.min_coherence_score: float = 0.2   # Minimum shared entity score
        self.suspicious_threshold: float = 0.5   # Below this → suspicious
        self.invalid_threshold: float = 0.25     # Below this → invalid

    # ── Single edge verification ──────────────────────────────

    def verify_edge(
        self,
        edge: dict,
        graph=None,
        bucket_mgr=None,
    ) -> CausalVerificationResult:
        """
        Verify a single causal edge for validity.

        Args:
            edge: Edge dict from memory_graph (must have edge_id, from_id, to_id,
                  relation_type, weight, properties)
            graph: MemoryGraph instance (optional, for node lookup)
            bucket_mgr: BucketManager instance (optional, for content lookup)

        Returns:
            CausalVerificationResult with validity and adjusted confidence
        """
        edge_id = edge.get("edge_id", "")
        from_id = edge.get("from_id", "")
        to_id = edge.get("to_id", "")
        original_weight = edge.get("weight", 1.0)

        # Only verify causal edges
        if edge.get("relation_type") != "causal":
            result = CausalVerificationResult(
                edge_id=edge_id, from_id=from_id, to_id=to_id,
                valid=True, confidence=1.0,
                original_weight=original_weight, adjusted_weight=original_weight,
            )
            self._cache[edge_id] = result
            return result

        result = CausalVerificationResult(
            edge_id=edge_id, from_id=from_id, to_id=to_id,
            original_weight=original_weight,
        )
        issues: list[str] = []
        penalty: float = 0.0  # Cumulative penalty on confidence

        # ── Check 1: Temporal precedence ──
        temporal_ok = self._check_temporal_precedence(
            from_id, to_id, graph, bucket_mgr
        )
        if not temporal_ok:
            issues.append("temporal_precedence_violation")
            penalty += 0.4

        # ── Check 2: Coherence (shared entities/concepts) ──
        coherence_score = self._check_coherence(from_id, to_id, graph, bucket_mgr)
        if coherence_score < self.min_coherence_score:
            issues.append(f"low_coherence({coherence_score:.2f})")
            penalty += 0.3
        elif coherence_score < 0.4:
            issues.append(f"moderate_coherence({coherence_score:.2f})")
            penalty += 0.1

        # ── Check 3: Emotion→Action plausibility ──
        emotion_action_ok = self._check_emotion_action_plausibility(
            from_id, to_id, graph, bucket_mgr
        )
        if emotion_action_ok is False:  # Explicitly implausible
            issues.append("implausible_emotion_action_chain")
            penalty += 0.2
        elif emotion_action_ok is True:  # Confirmed plausible
            penalty -= 0.1  # Bonus for plausible chain

        # ── Compute final confidence ──
        confidence = max(0.05, 1.0 - penalty)
        adjusted_weight = original_weight * confidence

        # ── Classification ──
        if confidence < self.invalid_threshold:
            result.valid = False
        result.confidence = confidence
        result.adjusted_weight = adjusted_weight
        result.issues = issues

        self._cache[edge_id] = result
        self._verification_count += 1

        if issues:
            logger.debug(
                f"Causal edge {edge_id} verified: valid={result.valid}, "
                f"confidence={confidence:.3f}, issues={issues}"
            )

        return result

    # ── Subgraph verification ─────────────────────────────────

    def verify_subgraph(
        self,
        root_id: str,
        graph,
        bucket_mgr=None,
        depth: int = 3,
    ) -> SubgraphVerificationReport:
        """
        Verify all causal edges reachable from a root node up to depth.

        Also detects structural issues:
          - Circular causality: A → B → C → A
          - Isolated causal chains: A → B with no other connections

        Args:
            root_id: Starting memory node ID
            graph: MemoryGraph instance
            bucket_mgr: BucketManager instance (optional)
            depth: How many hops to traverse

        Returns:
            SubgraphVerificationReport
        """
        report = SubgraphVerificationReport(root_id=root_id)

        # Collect all causal edges in the subgraph
        causal_edges = self._collect_causal_edges(root_id, graph, depth)
        report.total_edges_checked = len(causal_edges)

        # Verify each edge
        for edge in causal_edges:
            result = self.verify_edge(edge, graph, bucket_mgr)
            report.results.append(result)

            if result.valid:
                if result.confidence < self.suspicious_threshold:
                    report.suspicious_edges += 1
                else:
                    report.valid_edges += 1
            else:
                report.invalid_edges += 1

        # ── Structural checks ──

        # Detect circular causality
        cycles = self._detect_circular_causality(root_id, graph, depth)
        report.circular_chains = cycles

        # Detect isolated causal chains
        isolated = self._detect_isolated_chains(causal_edges, graph)
        report.isolated_chains = isolated

        logger.info(
            f"Subgraph verification [{root_id}]: "
            f"{report.valid_edges} valid, {report.suspicious_edges} suspicious, "
            f"{report.invalid_edges} invalid, {len(cycles)} cycles, "
            f"{len(isolated)} isolated chains"
        )

        return report

    # ── Batch verification ────────────────────────────────────

    def verify_all(self, graph, bucket_mgr=None) -> dict:
        """
        Batch-verify all causal edges in the graph.

        Called during sleeptime CONSOLIDATE stage for full-graph
        causal edge quality audit.

        Returns:
            {
                "total_causal_edges": int,
                "valid": int,
                "suspicious": int,
                "invalid": int,
                "circular_chains": int,
                "isolated_chains": int,
                "average_confidence": float,
            }
        """
        try:
            all_edges = graph.get_edges_by_type("causal", limit=1000)
        except Exception as e:
            logger.warning(f"Failed to fetch causal edges: {e}")
            return {"total_causal_edges": 0, "error": str(e)}

        if not all_edges:
            return {
                "total_causal_edges": 0,
                "valid": 0, "suspicious": 0, "invalid": 0,
                "circular_chains": 0, "isolated_chains": 0,
                "average_confidence": 0.0,
            }

        results: list[CausalVerificationResult] = []
        for edge in all_edges:
            result = self.verify_edge(edge, graph, bucket_mgr)
            results.append(result)

        valid = sum(1 for r in results if r.valid and r.confidence >= self.suspicious_threshold)
        suspicious = sum(1 for r in results if r.valid and r.confidence < self.suspicious_threshold)
        invalid = sum(1 for r in results if not r.valid)
        avg_conf = sum(r.confidence for r in results) / max(len(results), 1)

        # Collect all unique nodes for structural checks
        all_nodes = set()
        for edge in all_edges:
            all_nodes.add(edge.get("from_id", ""))
            all_nodes.add(edge.get("to_id", ""))

        # Detect cycles across all nodes (sample: check from each node with >2 edges)
        all_cycles = 0
        all_isolated = 0
        for node_id in list(all_nodes)[:50]:  # Cap at 50 to avoid O(n²) explosion
            sub_report = self.verify_subgraph(node_id, graph, bucket_mgr, depth=4)
            all_cycles += len(sub_report.circular_chains)
            all_isolated += len(sub_report.isolated_chains)

        summary = {
            "total_causal_edges": len(all_edges),
            "valid": valid,
            "suspicious": suspicious,
            "invalid": invalid,
            "circular_chains": all_cycles,
            "isolated_chains": all_isolated,
            "average_confidence": round(avg_conf, 3),
        }

        logger.info(f"Causal verification complete: {json.dumps(summary)}")
        return summary

    # ── Verify edges for a specific node ──────────────────────

    def verify_edges_for_node(
        self,
        memory_id: str,
        graph,
        bucket_mgr=None,
        adjust_weights: bool = True,
    ) -> list[CausalVerificationResult]:
        """
        Verify all causal edges connected to a specific node.

        Called from _async_hold_pipeline after creating new edges.
        If adjust_weights=True, updates edge weights in the graph
        based on verification results.

        Returns list of verification results.
        """
        edges = graph.get_all_edges_for_node(memory_id)
        causal_edges = [e for e in edges if e.get("relation_type") == "causal"]

        results: list[CausalVerificationResult] = []
        for edge in causal_edges:
            result = self.verify_edge(edge, graph, bucket_mgr)
            results.append(result)

            # Adjust edge weight in graph if needed
            if adjust_weights and result.adjusted_weight != result.original_weight:
                try:
                    # Update weight via expire + re-add (preserve history)
                    edge_id = edge.get("edge_id", "")
                    if edge_id:
                        graph.expire_edge(edge_id)
                        graph.add_edge(
                            from_id=result.from_id,
                            to_id=result.to_id,
                            relation_type="causal",
                            weight=result.adjusted_weight,
                            properties={
                                **(edge.get("properties", {})),
                                "verified": True,
                                "verification_confidence": result.confidence,
                                "original_edge_id": edge_id,
                            },
                        )
                except Exception as e:
                    logger.warning(f"Failed to update edge weight for {edge_id}: {e}")

        logger.debug(
            f"Verified {len(results)} causal edges for node {memory_id}: "
            f"{sum(1 for r in results if r.valid)} valid, "
            f"{sum(1 for r in results if not r.valid)} invalid"
        )

        return results

    # ── Private: Temporal precedence check ────────────────────

    def _check_temporal_precedence(
        self,
        from_id: str,
        to_id: str,
        graph,
        bucket_mgr,
    ) -> bool:
        """
        Check that the cause (from_id) precedes the effect (to_id) in time.

        Looks up node creation timestamps from graph nodes or bucket metadata.
        Returns True if temporal order is correct (cause before effect),
        or if timestamps are unavailable (give benefit of doubt).
        """
        from_ts = self._get_node_timestamp(from_id, graph, bucket_mgr)
        to_ts = self._get_node_timestamp(to_id, graph, bucket_mgr)

        if from_ts is None or to_ts is None:
            # Cannot determine: give benefit of doubt
            return True

        # Allow small time reversal (grace period) for near-simultaneous events
        diff_seconds = (to_ts - from_ts).total_seconds()
        if diff_seconds < -self.temporal_grace_minutes * 60:
            logger.debug(
                f"Temporal precedence violation: {from_id} ({from_ts.isoformat()}) "
                f"after {to_id} ({to_ts.isoformat()})"
            )
            return False

        return True

    def _get_node_timestamp(
        self,
        memory_id: str,
        graph,
        bucket_mgr,
    ) -> datetime | None:
        """Extract creation timestamp from graph node or bucket metadata."""
        # Try graph node first
        if graph:
            try:
                node = graph.get_node(memory_id)
                if node and node.get("created_at"):
                    return datetime.fromisoformat(node["created_at"])
                if node and node.get("properties", {}).get("created"):
                    return datetime.fromisoformat(node["properties"]["created"])
            except (ValueError, TypeError, Exception):
                pass

        # Try bucket metadata
        if bucket_mgr:
            try:
                import asyncio
                # bucket_mgr may be sync or async — handle both
                meta = bucket_mgr.get_metadata(memory_id)
                if asyncio.iscoroutine(meta):
                    # Can't await in sync context — skip
                    pass
                elif meta and meta.get("created"):
                    return datetime.fromisoformat(meta["created"])
            except (ValueError, TypeError, Exception):
                pass

        return None

    # ── Private: Coherence check ──────────────────────────────

    def _check_coherence(
        self,
        from_id: str,
        to_id: str,
        graph,
        bucket_mgr,
    ) -> float:
        """
        Check entity/concept coherence between cause and effect.

        Returns a score 0-1 where:
          1.0 = strong shared entities
          0.5 = moderate thematic overlap
          0.0 = no detectable connection

        Uses graph node properties (entity lists) and causal keyword detection
        in content summaries.
        """
        from_entities = self._get_node_entities(from_id, graph, bucket_mgr)
        to_entities = self._get_node_entities(to_id, graph, bucket_mgr)

        if not from_entities and not to_entities:
            # No entity data available — neutral score
            return 0.5

        if not from_entities or not to_entities:
            return 0.3

        # Jaccard-like overlap
        intersection = len(from_entities & to_entities)
        union = len(from_entities | to_entities)

        if union == 0:
            return 0.5

        jaccard = intersection / union

        # Boost: check for causal keyword patterns
        causal_boost = 0.0
        from_content = self._get_node_content(from_id, bucket_mgr)
        to_content = self._get_node_content(to_id, bucket_mgr)

        if from_content and to_content:
            from_has_causal = any(kw in from_content for kw in _CAUSAL_KEYWORDS_CN)
            to_has_effect = any(kw in to_content for kw in _EFFECT_KEYWORDS_CN)
            if from_has_causal and to_has_effect:
                causal_boost = 0.2
            elif from_has_causal or to_has_effect:
                causal_boost = 0.1

        return min(1.0, jaccard + causal_boost)

    def _get_node_entities(
        self,
        memory_id: str,
        graph,
        bucket_mgr,
    ) -> set[str]:
        """Extract entity set from a memory node."""
        entities: set[str] = set()

        # From graph node properties
        if graph:
            try:
                node = graph.get_node(memory_id)
                if node:
                    props = node.get("properties", {})
                    for key in ("entities", "concepts", "persons", "emotion_markers"):
                        vals = props.get(key, [])
                        if isinstance(vals, list):
                            entities.update(vals)
            except Exception:
                pass

        # From bucket content tags
        if bucket_mgr:
            try:
                import asyncio
                meta = bucket_mgr.get_metadata(memory_id)
                if asyncio.iscoroutine(meta):
                    pass
                elif meta:
                    tags = meta.get("tags", [])
                    if isinstance(tags, list):
                        entities.update(tags)
                    domain = meta.get("domain", [])
                    if isinstance(domain, list):
                        entities.update(domain)
            except Exception:
                pass

        return entities

    def _get_node_content(self, memory_id: str, bucket_mgr) -> str:
        """Get content text for a memory node."""
        if bucket_mgr is None:
            return ""
        try:
            import asyncio
            # Try read_content (synchronous variant if available)
            if hasattr(bucket_mgr, 'read_content'):
                result = bucket_mgr.read_content(memory_id)
                if asyncio.iscoroutine(result):
                    return ""
                return result or ""
            if hasattr(bucket_mgr, 'read'):
                result = bucket_mgr.read(memory_id)
                if asyncio.iscoroutine(result):
                    return ""
                node = result
                if isinstance(node, dict):
                    return node.get("content", "")
                return str(node) if node else ""
        except Exception:
            pass
        return ""

    # ── Private: Emotion→Action plausibility ──────────────────

    def _check_emotion_action_plausibility(
        self,
        from_id: str,
        to_id: str,
        graph,
        bucket_mgr,
    ) -> bool | None:
        """
        Check if an emotion→action causal chain is plausible.

        Returns:
          True  = explicitly plausible (emotion→action pair matches known pattern)
          False = explicitly implausible (contradicts known patterns)
          None  = not an emotion→action edge, or insufficient data
        """
        from_content = self._get_node_content(from_id, bucket_mgr)
        to_content = self._get_node_content(to_id, bucket_mgr)

        if not from_content or not to_content:
            return None

        # Detect if this is an emotion→action edge
        from_emotion = None
        to_action = None

        for emotion, action in _EMOTION_ACTION_PAIRS:
            if emotion in from_content:
                from_emotion = emotion
                break

        for _, action in _EMOTION_ACTION_PAIRS:
            if action in to_content:
                to_action = action
                break

        if from_emotion is None or to_action is None:
            return None  # Not an emotion→action edge

        # Check if this specific pair is plausible
        for emotion, action in _EMOTION_ACTION_PAIRS:
            if emotion == from_emotion and action == to_action:
                return True

        return False

    # ── Private: Structural checks ────────────────────────────

    def _collect_causal_edges(
        self,
        root_id: str,
        graph,
        depth: int,
    ) -> list[dict]:
        """Collect all causal edges reachable from root_id within depth."""
        if graph is None:
            return []

        all_edges: list[dict] = []
        seen_edges: set[str] = set()
        frontier = [root_id]
        visited_nodes: set[str] = {root_id}

        for _ in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                neighbors = graph.get_neighbors(
                    node_id, depth=1, relation_types=["causal"], active_only=True
                )
                for edge in neighbors:
                    eid = edge.get("edge_id", "")
                    if eid and eid not in seen_edges:
                        seen_edges.add(eid)
                        all_edges.append(edge)

                    # Traverse both directions
                    to_node = edge.get("to_id", "")
                    from_node = edge.get("from_id", "")
                    for nid in (to_node, from_node):
                        if nid and nid not in visited_nodes:
                            visited_nodes.add(nid)
                            next_frontier.append(nid)

            frontier = next_frontier
            if not frontier:
                break

        return all_edges

    def _detect_circular_causality(
        self,
        root_id: str,
        graph,
        max_depth: int,
    ) -> list[list[str]]:
        """
        Detect circular causality: A → B → C → A.

        Uses DFS with path tracking to find cycles in the causal subgraph.
        """
        if graph is None:
            return []

        cycles: list[list[str]] = []
        visited_path: list[str] = []
        in_path: set[str] = set()

        def dfs(node_id: str, depth: int):
            if depth > max_depth:
                return
            if node_id in in_path:
                # Found a cycle
                cycle_start = visited_path.index(node_id)
                cycle = visited_path[cycle_start:] + [node_id]
                if len(cycle) >= 3 and len(cycle) <= 10:  # Minimum 3, max sanity cap
                    cycles.append(cycle)
                return

            visited_path.append(node_id)
            in_path.add(node_id)

            # Get outgoing causal edges
            try:
                neighbors = graph.get_neighbors(
                    node_id, depth=1, relation_types=["causal"], active_only=True
                )
                for edge in neighbors[:10]:  # Cap edges per node
                    to_id = edge.get("to_id", "")
                    if to_id and to_id != node_id:
                        dfs(to_id, depth + 1)
            except Exception:
                pass

            visited_path.pop()
            in_path.discard(node_id)

        dfs(root_id, 0)
        return cycles

    def _detect_isolated_chains(
        self,
        causal_edges: list[dict],
        graph,
    ) -> list[list[str]]:
        """
        Detect isolated causal chains: A → B where neither A nor B
        have any other graph connections beyond each other.
        """
        if graph is None or len(causal_edges) < 2:
            return []

        # Build degree map
        degree: dict[str, int] = {}
        for edge in causal_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            degree[from_id] = degree.get(from_id, 0) + 1
            degree[to_id] = degree.get(to_id, 0) + 1

        # Find nodes with only 1 causal connection
        isolated_nodes = {nid for nid, deg in degree.items() if deg == 1}

        # Group into chains
        isolated_chains: list[list[str]] = []
        seen: set[str] = set()

        for edge in causal_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            if from_id in isolated_nodes and to_id in isolated_nodes:
                pair = tuple(sorted([from_id, to_id]))
                if pair not in seen:
                    seen.add(pair)
                    isolated_chains.append(list(pair))

        return isolated_chains

    # ── Cache management ──────────────────────────────────────

    def clear_cache(self):
        """Clear verification cache."""
        self._cache.clear()
        logger.debug("Verification cache cleared")

    def get_cached_result(self, edge_id: str) -> CausalVerificationResult | None:
        """Get a previously cached verification result."""
        return self._cache.get(edge_id)
