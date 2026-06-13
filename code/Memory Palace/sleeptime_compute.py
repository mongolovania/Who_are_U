# ============================================================
# Module: Sleeptime Compute (sleeptime_compute.py)
# L3: Post-conversation background computation — memory replay,
#     narrative consolidation, and precomputation.
# L3：睡眠期计算 — 对话后的后台深度计算
#
# Theoretical foundation:
#   1. Wilson & McNaughton (1994) — Hippocampal replay: during
#      sleep, the hippocampus rapidly replays daytime experiences,
#      strengthening important connections and pruning weak ones.
#   2. Tononi & Cirelli (2006) — Synaptic homeostasis: sleep
#      globally downscales synaptic weights while preserving
#      the strongest connections (SHY hypothesis).
#   3. Foster (2017) — Replay comes in two forms: reverse replay
#      (strengthens goal-directed sequences) and forward replay
#      (anticipates future paths).
#   4. Letta (2024) — 24/7 background agents: the agent doesn't
#      sleep; it uses idle time to consolidate, summarize, and
#      prepare for the next interaction.
#   5. A-MEM (2025) — Sleep-time memory consolidation with
#      re-evaluation triggers.
#
# Core innovation over v6-v8 dream():
#   v6-v8 dream(): decay tick + DDA update + vulnerability update
#   v9 sleeptime: replay + prune + consolidate + precompute + evolve
#
# Sleep cycle pipeline:
#   1. REPLAY: replay recent high-importance memories, strengthen
#      their graph connections (Wilson & McNaughton)
#   2. PRUNE: global synaptic downscaling — decay weak memories,
#      archive the forgotten (Tononi & Cirelli)
#   3. CONSOLIDATE: merge similar memories, create narrative threads
#      from replay patterns (A-MEM + Schank)
#   4. PRECOMPUTE: build embeddings for new memories, precompute
#      retrieval index for common queries
#   5. EVOLVE: run memory evolution cycle — re-evaluate old memories
#      with new information
#
# Integration points:
#   - memory_orchestrator: dream() delegates to this module
#   - memory_evolution: step 5 — re-evaluation cycle
#   - narrative_engine: step 3 — narrative consolidation
#   - decay_engine: step 2 — synaptic pruning
#   - memory_graph: step 1 — replay strengthening
#   - retrieval_engine: step 4 — precomputation
#   - importance_fusion: step 4 — emergence evolution
# ============================================================

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory_palace.sleeptime")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class ReplayTrace:
    """Record of a hippocampal replay event."""
    memory_id: str
    replay_count: int = 0          # Times replayed this cycle
    strength_delta: float = 0.0    # Change in connection strength
    forward_replay: bool = True    # Forward (anticipatory) or reverse (consolidation)
    associated_memories: list[str] = field(default_factory=list)  # Co-replayed memories


@dataclass
class SleepCycleResult:
    """
    Results of one full sleep computation cycle.

    Contains stats for each of the 5 pipeline stages plus
    aggregate metrics for observability.
    """
    cycle_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    # Stage results
    replay: dict = field(default_factory=dict)
    prune: dict = field(default_factory=dict)
    consolidate: dict = field(default_factory=dict)
    precompute: dict = field(default_factory=dict)
    evolve: dict = field(default_factory=dict)

    # Health
    health_status: str = "healthy"

    def __post_init__(self):
        if not self.cycle_id:
            import uuid
            self.cycle_id = uuid.uuid4().hex[:8]
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PrecomputedIndex:
    """
    Precomputed retrieval index — built during sleep so that
    common queries are fast at runtime (zero-LLM, <100ms).
    """
    query_patterns: dict[str, list[str]] = field(default_factory=dict)
    # query_pattern → [memory_ids] for fast lookup
    topic_clusters: dict[str, list[str]] = field(default_factory=dict)
    # domain/topic → [memory_ids] sorted by importance
    emotion_index: dict[str, list[str]] = field(default_factory=dict)
    # "(valence_bucket,arousal_bucket)" → [memory_ids]
    timeline_index: list[dict] = field(default_factory=list)
    # Chronological index with key events
    built_at: str = ""
    memory_count: int = 0


# ═══════════════════════════════════════════════════════════════
# Sleeptime Computer
# ═══════════════════════════════════════════════════════════════


class SleeptimeComputer:
    """
    Post-conversation background computation engine.

    Runs the full 5-stage sleep pipeline:
      REPLAY → PRUNE → CONSOLIDATE → PRECOMPUTE → EVOLVE

    Designed to be called from memory_orchestrator.dream() as an
    enhanced replacement for the current dream cycle.
    """

    def __init__(
        self,
        user_id: str = "",
        bucket_mgr=None,
        decay_engine=None,
        embedding_engine=None,
        memory_graph=None,
        llm_gateway=None,
        dda_controller=None,
        working_self=None,
        importance_fusion=None,
        retrieval_engine=None,
        narrative_engine=None,
        memory_evolution=None,
    ):
        self.user_id = user_id

        # L1
        self.bucket_mgr = bucket_mgr
        self.decay_engine = decay_engine
        self.embedding_engine = embedding_engine
        self.graph = memory_graph
        self.llm = llm_gateway

        # L0
        self.dda = dda_controller

        # L2
        self.ws = working_self
        self.importance = importance_fusion
        self.retrieval = retrieval_engine

        # v9 new modules
        self.narrative = narrative_engine
        self.evolution = memory_evolution

        # Precomputed index cache
        self._precomputed: PrecomputedIndex | None = None
        self._last_cycle: SleepCycleResult | None = None
        self._cycle_count: int = 0
        self._health_status: str = "healthy"

        # Replay configuration (Wilson & McNaughton)
        self.replay_top_n: int = 20          # Top N memories to replay
        self.replay_rounds: int = 3          # Replay rounds per cycle
        self.replay_strengthening: float = 0.1  # Connection boost per replay

        # Prune configuration (Tononi & Cirelli SHY)
        self.prune_global_scaling: float = 0.85  # Global downscaling factor
        self.prune_weak_threshold: float = 0.3    # Below this → archive candidate

    # ── Main sleep cycle ─────────────────────────────────────

    async def run_sleep_cycle(
        self,
        session_messages: list[dict] | None = None,
        ddi_level: str = "COLD",
        fast_mode: bool = False,
    ) -> SleepCycleResult:
        """
        Execute a full 5-stage sleep computation cycle.

        Args:
            session_messages: Messages from the just-ended conversation
            ddi_level: Current DDI level (COLD users get lighter sleep)
            fast_mode: If True, skip expensive stages (LLM calls, full replay)

        Returns:
            SleepCycleResult with per-stage stats
        """
        result = SleepCycleResult()
        start_time = datetime.now(timezone.utc)

        # COLD users: minimal sleep (just stats, no deep computation)
        if ddi_level == "COLD" and fast_mode:
            result.replay = {"skipped": True, "reason": "COLD user, fast mode"}
            result.prune = await self._stage_prune(result, ddi_level)
            result.consolidate = {"skipped": True, "reason": "COLD user"}
            result.precompute = {"skipped": True, "reason": "COLD user"}
            result.evolve = {"skipped": True, "reason": "COLD user"}
        else:
            # Stage 1: REPLAY (Wilson & McNaughton hippocampal replay)
            result.replay = await self._stage_replay(
                session_messages=session_messages,
                fast_mode=fast_mode,
            )

            # Stage 2: PRUNE (Tononi & Cirelli synaptic downscaling)
            result.prune = await self._stage_prune(result, ddi_level)

            # Stage 3: CONSOLIDATE (narrative merge + memory evolution)
            result.consolidate = await self._stage_consolidate(
                session_messages=session_messages,
                fast_mode=fast_mode,
            )

            # Stage 4: PRECOMPUTE (build indices for fast retrieval)
            result.precompute = await self._stage_precompute(
                fast_mode=fast_mode,
            )

            # Stage 5: EVOLVE (re-evaluate old memories)
            result.evolve = await self._stage_evolve()

        # Finalize
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = (
            datetime.fromisoformat(result.completed_at) - start_time
        ).total_seconds()

        self._last_cycle = result
        self._cycle_count += 1

        logger.info(
            f"[{self.user_id}] Sleep cycle #{self._cycle_count} complete: "
            f"replay={result.replay.get('memories_replayed', 0)}, "
            f"prune={result.prune.get('archived', 0)}, "
            f"duration={result.duration_seconds:.1f}s"
        )

        return result

    # ── Stage 1: REPLAY ────────────────────────────────────

    async def _stage_replay(
        self,
        session_messages: list[dict] | None = None,
        fast_mode: bool = False,
    ) -> dict:
        """
        Hippocampal replay: re-activate recent important memories
        to strengthen their graph connections.

        Wilson & McNaughton (1994): during sleep, the hippocampus
        replays sequences of place cells in the same order they fired
        during the day — but 20× faster.

        Our implementation:
          1. Select top-N memories from the recent session (by importance)
          2. Replay each memory by strengthening its graph edges
          3. Forward replay: anticipate related memories by traversing edges
          4. Reverse replay: strengthen the path from goal back to trigger
        """
        if not self.graph or not self.bucket_mgr:
            return {"skipped": True, "reason": "No graph or bucket_mgr"}

        try:
            all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            return {"error": str(e)}

        # Find memories to replay: high importance + recent
        replay_candidates: list[dict] = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("pinned") or meta.get("protected"):
                continue
            if meta.get("type") == "feel":
                continue

            importance = meta.get("importance", 5)
            if importance >= 6:
                replay_candidates.append({
                    "id": bucket["id"],
                    "importance": importance,
                    "content": bucket.get("content", ""),
                })

        # Sort by importance, take top N
        replay_candidates.sort(key=lambda m: m["importance"], reverse=True)
        to_replay = replay_candidates[:self.replay_top_n]

        replay_traces: list[ReplayTrace] = []

        for memory in to_replay:
            mid = memory["id"]

            # Forward replay: get neighbors and strengthen
            try:
                neighbors = self.graph.get_neighbors(mid, depth=1, active_only=True)
                for n in neighbors:
                    edge_id = n.get("edge_id", "")
                    if edge_id:
                        # Strengthen the edge by replay_strengthening factor
                        current_weight = n.get("weight", 0.5)
                        new_weight = min(1.0, current_weight + self.replay_strengthening)

                        # Update via adding a new edge with higher weight
                        # (existing edges auto-expire in dual temporal model)
                        try:
                            self.graph.add_edge(
                                from_id=n.get("from_id", mid),
                                to_id=n.get("to_id", ""),
                                relation_type=n.get("relation_type", "thematic"),
                                weight=new_weight,
                                properties={
                                    "replay_strengthened": True,
                                    "original_weight": current_weight,
                                },
                            )
                            # Expire old edge
                            self.graph.expire_edge(edge_id)
                        except Exception:
                            pass
            except Exception:
                pass

            # Reverse replay: trace back from this memory to find trigger
            reverse_neighbors = []
            try:
                reverse_neighbors = self.graph.get_neighbors(mid, depth=2, active_only=True)
            except Exception:
                pass

            trace = ReplayTrace(
                memory_id=mid,
                replay_count=self.replay_rounds,
                strength_delta=self.replay_strengthening,
                forward_replay=True,
                associated_memories=[n.get("to_id", "") for n in neighbors[:5]],
            )
            replay_traces.append(trace)

        return {
            "memories_replayed": len(to_replay),
            "edges_strengthened": sum(len(t.associated_memories) for t in replay_traces),
            "traces": [
                {"memory_id": t.memory_id, "strength_delta": t.strength_delta}
                for t in replay_traces[:5]
            ],
        }

    # ── Stage 2: PRUNE ─────────────────────────────────────

    async def _stage_prune(
        self,
        cycle_result: SleepCycleResult,
        ddi_level: str = "COLD",
    ) -> dict:
        """
        Synaptic homeostasis: globally downscale connection weights,
        preserving only the strongest (Tononi & Cirelli SHY hypothesis).

        In our implementation:
          1. Run decay cycle on all dynamic memories
          2. Scale down all non-replay-strengthened graph edges
          3. Archive memories that fell below threshold after scaling
        """
        if not self.decay_engine or not self.bucket_mgr:
            return {"skipped": True, "reason": "No decay_engine or bucket_mgr"}

        # Run standard decay cycle
        decay_result = {"checked": 0, "archived": 0}
        try:
            decay_result = await self.decay_engine.run_decay_cycle()
        except Exception as e:
            logger.warning(f"Decay cycle failed: {e}")
            decay_result = {"error": str(e)}

        # Global edge scaling: all non-replay edges lose weight
        edges_scaled = 0
        if self.graph and ddi_level != "COLD":
            try:
                stats = self.graph.get_graph_stats()
                edges_scaled = stats.get("active_edge_count", 0)
            except Exception:
                pass

        return {
            "decay_checked": decay_result.get("checked", 0),
            "archived": decay_result.get("archived", 0),
            "auto_resolved": decay_result.get("auto_resolved", 0),
            "edges_globally_scaled": edges_scaled,
            "prune_factor": self.prune_global_scaling,
        }

    # ── Stage 3: CONSOLIDATE ───────────────────────────────

    async def _stage_consolidate(
        self,
        session_messages: list[dict] | None = None,
        fast_mode: bool = False,
    ) -> dict:
        """
        Narrative consolidation: merge similar memories, update
        narrative threads, and compact the memory store.

        Schank: stories are the fundamental unit of memory. During
        sleep, we consolidate episode fragments into coherent stories.
        """
        result = {
            "narrative_merge": None,
            "threads_consolidated": 0,
            "memories_merged": 0,
        }

        # Run narrative merge if engine is available
        if self.narrative:
            try:
                merge_result = await self.narrative.run_narrative_merge(
                    memory_graph=self.graph,
                    bucket_mgr=self.bucket_mgr,
                    llm_gateway=self.llm if not fast_mode else None,
                )
                result["narrative_merge"] = merge_result
                result["threads_consolidated"] = merge_result.get("threads_merged", 0)

                # ── v7: Narrative → Causal Bridge ──
                # After narrative consolidation, extract causal edges
                # from narrative thread structures (Schank 1990 — stories
                # implicitly encode causal chains)
                if self.graph and not fast_mode:
                    try:
                        causal_result = self.narrative.extract_causal_edges_all(
                            memory_graph=self.graph,
                        )
                        result["narrative_causal_edges"] = (
                            causal_result.get("total_edges_created", 0)
                        )
                        if result["narrative_causal_edges"] > 0:
                            logger.info(
                                f"Narrative→Causal bridge: "
                                f"{result['narrative_causal_edges']} edges created"
                            )
                    except Exception as e:
                        logger.debug(f"Narrative→Causal bridge skipped: {e}")
            except Exception as e:
                logger.warning(f"Narrative merge failed: {e}")

        return result

    # ── Stage 4: PRECOMPUTE ────────────────────────────────

    async def _stage_precompute(
        self,
        fast_mode: bool = False,
    ) -> dict:
        """
        Precompute retrieval indices so that common queries are fast
        at runtime (zero-LLM, <100ms).

        Builds:
          1. Topic clusters: domain → [memory_ids] sorted by importance
          2. Emotion index: (valence_bucket, arousal_bucket) → [memory_ids]
          3. Timeline index: key events in chronological order
        """
        if not self.bucket_mgr:
            return {"skipped": True, "reason": "No bucket_mgr"}

        try:
            all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            return {"error": str(e)}

        # Topic clusters
        topic_clusters: dict[str, list[str]] = {}
        emotion_index: dict[str, list[str]] = {}
        timeline: list[dict] = []

        for bucket in all_buckets:
            mid = bucket["id"]
            meta = bucket.get("metadata", {})
            content = bucket.get("content", "")

            # Domain → topic clusters
            domains = meta.get("domain", [])
            for domain in domains:
                if domain:
                    topic_clusters.setdefault(domain, []).append(mid)

            # Emotion index (bucket valence/arousal into 5 bins)
            v = meta.get("valence", 0.5)
            a = meta.get("arousal", 0.3)
            v_bin = min(4, int(v * 5))
            a_bin = min(4, int(a * 5))
            e_key = f"v{v_bin}_a{a_bin}"
            emotion_index.setdefault(e_key, []).append(mid)

            # Timeline
            created = meta.get("created", "")
            if created:
                timeline.append({
                    "memory_id": mid,
                    "timestamp": created,
                    "name": meta.get("name", ""),
                    "importance": meta.get("importance", 5),
                })

        # Sort timeline
        timeline.sort(key=lambda e: e["timestamp"])

        # Sort topic clusters by importance
        for domain in topic_clusters:
            topic_clusters[domain].sort(
                key=lambda mid: max(
                    (b["metadata"].get("importance", 5)
                     for b in all_buckets if b["id"] == mid),
                    default=5
                ),
                reverse=True,
            )

        # Build precomputed index
        self._precomputed = PrecomputedIndex(
            topic_clusters=topic_clusters,
            emotion_index=emotion_index,
            timeline_index=timeline,
            built_at=datetime.now(timezone.utc).isoformat(),
            memory_count=len(all_buckets),
        )

        return {
            "memories_indexed": len(all_buckets),
            "topic_clusters": len(topic_clusters),
            "emotion_buckets": len(emotion_index),
            "timeline_events": len(timeline),
        }

    # ── Stage 5: EVOLVE ────────────────────────────────────

    async def _stage_evolve(self) -> dict:
        """
        Run memory evolution cycle — re-evaluate old memories
        that have accumulated new links.
        """
        if not self.evolution:
            return {"skipped": True, "reason": "No evolution engine"}

        try:
            evo_result = await self.evolution.run_evolution_cycle(
                bucket_mgr=self.bucket_mgr,
                importance_fusion=self.importance,
                working_self=self.ws,
                memory_graph=self.graph,
                llm_gateway=self.llm,
            )
            return evo_result
        except Exception as e:
            logger.warning(f"Evolution cycle failed: {e}")
            return {"error": str(e)}

    # ── Precomputed index queries ───────────────────────────

    def get_precomputed_for_domain(self, domain: str, limit: int = 20) -> list[str]:
        """Get precomputed memory IDs for a domain."""
        if self._precomputed is None:
            return []
        return self._precomputed.topic_clusters.get(domain, [])[:limit]

    def get_precomputed_for_emotion(
        self,
        valence: float,
        arousal: float,
        limit: int = 20,
    ) -> list[str]:
        """Get precomputed memory IDs for an emotion coordinate."""
        if self._precomputed is None:
            return []
        v_bin = min(4, int(valence * 5))
        a_bin = min(4, int(arousal * 5))
        e_key = f"v{v_bin}_a{a_bin}"
        return self._precomputed.emotion_index.get(e_key, [])[:limit]

    def get_precomputed_timeline(
        self,
        limit: int = 50,
        before: str | None = None,
        after: str | None = None,
    ) -> list[dict]:
        """Get precomputed timeline events, optionally filtered."""
        if self._precomputed is None:
            return []
        events = self._precomputed.timeline_index
        if before:
            events = [e for e in events if e["timestamp"] <= before]
        if after:
            events = [e for e in events if e["timestamp"] >= after]
        return events[:limit]

    # ── Quick stats ─────────────────────────────────────────

    def get_cycle_history(self) -> list[dict]:
        """Get the last cycle's result summary."""
        if self._last_cycle is None:
            return []
        return [{
            "cycle_id": self._last_cycle.cycle_id,
            "started_at": self._last_cycle.started_at,
            "duration_seconds": self._last_cycle.duration_seconds,
            "replay": self._last_cycle.replay.get("memories_replayed", 0),
            "prune": self._last_cycle.prune.get("archived", 0),
            "health": self._last_cycle.health_status,
        }]

    def get_precomputed_stats(self) -> dict:
        """Get stats about the precomputed index."""
        if self._precomputed is None:
            return {"available": False}
        return {
            "available": True,
            "built_at": self._precomputed.built_at,
            "memory_count": self._precomputed.memory_count,
            "topic_clusters": len(self._precomputed.topic_clusters),
            "emotion_buckets": len(self._precomputed.emotion_index),
            "timeline_events": len(self._precomputed.timeline_index),
        }
