# ============================================================
# Module: Memory Load Monitor (memory_load_monitor.py)
# L0: Adaptive sleep cycle triggering — load-based, not fixed interval.
# L0：记忆负载监控器 — 自适应睡眠触发·基于负载而非固定间隔
#
# Theoretical foundation:
#   1. McClelland, McNaughton & O'Reilly (1995). "Why there are
#      complementary learning systems in the hippocampus and
#      neocortex." Psychological Review, 102(3), 419-457. —
#      Hippocampus (fast learning, high interference) and neocortex
#      (slow learning, low interference) as complementary systems.
#      Memory Palace L1 = hippocampus, L2/L3 = neocortex.
#   2. Diekelmann & Born (2010). "The memory function of sleep."
#      Nature Reviews Neuroscience, 11, 114-126. —
#      SWS promotes hippocampus→cortex transfer; REM promotes
#      emotional memory integration. Sleep intensity should match
#      consolidation demand.
#   3. Tononi & Cirelli (2006). "Sleep function and synaptic
#      homeostasis." Sleep Medicine Reviews, 10, 49-62. —
#      SHY hypothesis: sleep globally downscales synaptic weights.
#      The amount of downscaling needed depends on the amount of
#      new learning (memory load).
#
# Design §12.7:
#   - Compute memory load from 5 metrics
#   - Recommend sleep cycle (should_run, urgency, stages, intensity)
#   - Track sleep history for trend analysis
#
# Integration points:
#   - memory_orchestrator.dream(): check load before running sleep
#   - sleeptime_compute: use recommended intensity for adaptive stages
#   - dda_controller: DDI level influences sleep recommendations
# ============================================================

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory_palace.load_monitor")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class MemoryLoad:
    """Current memory system load metrics."""
    # Core metrics
    new_memories_since_last_sleep: int = 0
    avg_importance_since_last_sleep: float = 0.0
    edge_density: float = 0.0               # Edges per node
    emotional_volatility: float = 0.0        # Variance in recent valence/arousal
    time_since_last_sleep_hours: float = 0.0

    # Derived
    load_score: float = 0.0                  # Composite load score (0-1)
    consolidation_need: float = 0.0          # How badly consolidation is needed (0-1)
    computed_at: str = ""

    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "new_memories_since_last_sleep": self.new_memories_since_last_sleep,
            "avg_importance_since_last_sleep": self.avg_importance_since_last_sleep,
            "edge_density": self.edge_density,
            "emotional_volatility": self.emotional_volatility,
            "time_since_last_sleep_hours": self.time_since_last_sleep_hours,
            "load_score": self.load_score,
            "consolidation_need": self.consolidation_need,
            "computed_at": self.computed_at,
        }


@dataclass
class SleepRecommendation:
    """Recommendation for whether/how to run a sleep cycle."""
    should_sleep: bool = False
    urgency: float = 0.0                    # 0-1
    recommended_stages: list[str] = field(default_factory=list)
    # Subset of: REPLAY, PRUNE, CONSOLIDATE, PRECOMPUTE, EVOLVE
    recommended_intensity: str = "normal"   # light | normal | deep
    reason: str = ""
    load: MemoryLoad | None = None
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "should_sleep": self.should_sleep,
            "urgency": self.urgency,
            "recommended_stages": self.recommended_stages,
            "recommended_intensity": self.recommended_intensity,
            "reason": self.reason,
            "load": self.load.to_dict() if self.load else None,
            "generated_at": self.generated_at,
        }


# ═══════════════════════════════════════════════════════════════
# Memory Load Monitor
# ═══════════════════════════════════════════════════════════════


class MemoryLoadMonitor:
    """
    Monitor memory system load and recommend adaptive sleep cycles.

    Replaces fixed-interval sleep triggering (e.g., "run dream every
    24 hours") with load-adaptive triggering based on McClelland's
    complementary learning systems theory and Diekelmann & Born's
    sleep-dependent consolidation model.

    Key insight: COLD users have minimal load → light/infrequent sleep.
    RICH users with high emotional volatility → deep/frequent sleep.
    """

    def __init__(self, user_id: str = "", data_dir: str = "./buckets"):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        if user_id:
            self.data_dir = self.data_dir / user_id
        os.makedirs(self.data_dir, exist_ok=True)

        # State
        self._last_sleep_at: str = ""
        self._sleep_history: list[dict] = []
        self._recommendation_count: int = 0
        self._loaded: bool = False

        # Thresholds (configurable)
        self.high_load_threshold: float = 0.6
        self.critical_load_threshold: float = 0.8
        self.min_hours_between_sleep: float = 1.0   # Don't sleep more than once/hour
        self.max_hours_without_sleep: float = 48.0    # Force sleep after this long
        self.edge_density_threshold: float = 3.0      # Edges/node → consolidation needed
        self.emotional_volatility_threshold: float = 0.15  # High valence variance → sleep

    # ── Load computation ──────────────────────────────────────

    def compute_load(
        self,
        bucket_mgr=None,
        graph=None,
        dda_level: str = "COLD",
    ) -> MemoryLoad:
        """
        Compute current memory system load.

        Args:
            bucket_mgr: BucketManager for memory counts
            graph: MemoryGraph for edge density
            dda_level: Current DDI level for context

        Returns:
            MemoryLoad with all metrics
        """
        load = MemoryLoad()

        # ── Metric 1: New memories since last sleep ──
        load.new_memories_since_last_sleep = self._count_new_memories(bucket_mgr)

        # ── Metric 2: Average importance of new memories ──
        load.avg_importance_since_last_sleep = self._avg_new_importance(bucket_mgr)

        # ── Metric 3: Edge density ──
        if graph:
            try:
                stats = graph.get_graph_stats()
                node_count = stats.get("node_count", 0)
                edge_count = stats.get("active_edge_count", 0)
                load.edge_density = (
                    edge_count / max(node_count, 1)
                    if node_count > 0 else 0.0
                )
            except Exception as e:
                logger.debug(f"Edge density calculation failed: {e}")

        # ── Metric 4: Emotional volatility ──
        load.emotional_volatility = self._compute_emotional_volatility(bucket_mgr)

        # ── Metric 5: Time since last sleep ──
        load.time_since_last_sleep_hours = self._hours_since_last_sleep()

        # ── Composite load score ──
        # Weighted combination of normalized metrics
        new_mem_score = min(1.0, load.new_memories_since_last_sleep / 50.0)
        importance_score = load.avg_importance_since_last_sleep / 10.0
        density_score = min(1.0, load.edge_density / 10.0)
        volatility_score = min(1.0, load.emotional_volatility / 0.3)
        time_score = min(1.0, load.time_since_last_sleep_hours / 24.0)

        load.load_score = round(
            new_mem_score * 0.30 +
            importance_score * 0.20 +
            density_score * 0.20 +
            volatility_score * 0.20 +
            time_score * 0.10,
            3,
        )

        # Consolidation need: weighted toward density + emotional volatility
        # (McClelland: consolidation needed when hippocampus has encoded much new info)
        load.consolidation_need = round(
            new_mem_score * 0.25 +
            importance_score * 0.15 +
            density_score * 0.35 +         # Edge density = high connectivity = need consolidation
            volatility_score * 0.25,        # Emotional volatility = need REM-like processing
            3,
        )

        logger.debug(
            f"[{self.user_id}] Memory load: score={load.load_score:.3f}, "
            f"consolidation={load.consolidation_need:.3f}, "
            f"new={load.new_memories_since_last_sleep}, "
            f"density={load.edge_density:.2f}, "
            f"volatility={load.emotional_volatility:.3f}"
        )

        return load

    # ── Sleep recommendation ──────────────────────────────────

    def recommend_sleep_cycle(
        self,
        load: MemoryLoad,
        dda_level: str = "COLD",
    ) -> SleepRecommendation:
        """
        Recommend whether and how to run a sleep cycle.

        Adapts to DDA level:
          - COLD: only sleep when forced (time-based)
          - WARM: sleep when consolidation need is moderate
          - HOT: sleep when any metric is elevated
          - RICH: sleep proactively, deep mode when high load

        Returns:
            SleepRecommendation with should_sleep, urgency, stages, intensity
        """
        # ── Guard: don't sleep too frequently ──
        if load.time_since_last_sleep_hours < self.min_hours_between_sleep:
            return SleepRecommendation(
                should_sleep=False,
                urgency=0.0,
                reason=f"too_soon ({load.time_since_last_sleep_hours:.1f}h since last sleep)",
                load=load,
            )

        # ── DDA-adaptive decision ──
        dda_thresholds = {
            "COLD": (0.70, 0.8),    # (load_threshold, consolidation_threshold)
            "WARM": (0.45, 0.5),
            "HOT": (0.30, 0.35),
            "RICH": (0.20, 0.25),
        }
        load_threshold, consol_threshold = dda_thresholds.get(dda_level, (0.50, 0.55))

        should_sleep = False
        urgency = 0.0
        reasons: list[str] = []

        # Check 1: Consolidation need
        if load.consolidation_need > consol_threshold:
            should_sleep = True
            urgency = max(urgency, load.consolidation_need)
            reasons.append(f"consolidation_needed({load.consolidation_need:.2f})")

        # Check 2: High edge density (needs synaptic downscaling)
        if load.edge_density > self.edge_density_threshold:
            should_sleep = True
            urgency = max(urgency, min(1.0, load.edge_density / 10.0))
            reasons.append(f"high_edge_density({load.edge_density:.1f})")

        # Check 3: High emotional volatility (needs REM-like processing)
        if load.emotional_volatility > self.emotional_volatility_threshold:
            should_sleep = True
            urgency = max(urgency, min(1.0, load.emotional_volatility / 0.3))
            reasons.append(f"emotional_volatility({load.emotional_volatility:.3f})")

        # Check 4: Long time without sleep (safety net)
        if load.time_since_last_sleep_hours > self.max_hours_without_sleep:
            should_sleep = True
            urgency = max(urgency, 0.9)
            reasons.append(f"forced({load.time_since_last_sleep_hours:.0f}h)")

        # Check 5: Overall load score
        if load.load_score > load_threshold:
            should_sleep = True
            urgency = max(urgency, load.load_score)
            reasons.append(f"high_load({load.load_score:.2f})")

        # ── Determine stages and intensity ──
        if not should_sleep:
            recommended_stages = []
            intensity = "light"
        elif urgency > self.critical_load_threshold:
            recommended_stages = ["REPLAY", "PRUNE", "CONSOLIDATE", "PRECOMPUTE", "EVOLVE"]
            intensity = "deep"
        elif urgency > self.high_load_threshold:
            recommended_stages = ["REPLAY", "PRUNE", "CONSOLIDATE", "PRECOMPUTE"]
            intensity = "normal"
        else:
            recommended_stages = ["REPLAY", "PRUNE", "CONSOLIDATE"]
            intensity = "light"

        # COLD users: always light sleep, skip LLM stages
        if dda_level == "COLD":
            recommended_stages = ["PRUNE"]
            intensity = "light"

        recommendation = SleepRecommendation(
            should_sleep=should_sleep,
            urgency=round(urgency, 3),
            recommended_stages=recommended_stages,
            recommended_intensity=intensity,
            reason=", ".join(reasons) if reasons else "load_normal",
            load=load,
        )

        self._recommendation_count += 1

        logger.info(
            f"[{self.user_id}] Sleep recommendation: "
            f"should={should_sleep}, urgency={urgency:.3f}, "
            f"intensity={intensity}, stages={recommended_stages}, "
            f"reason={recommendation.reason}"
        )

        return recommendation

    # ── Sleep tracking ────────────────────────────────────────

    def record_sleep_complete(self, result: dict):
        """
        Record that a sleep cycle completed.

        Updates internal state and sleep history for trend analysis.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._last_sleep_at = now

        entry = {
            "completed_at": now,
            "cycle_id": result.get("cycle_id", ""),
            "duration_seconds": result.get("duration_seconds", 0),
            "stages_completed": result.get("stages_completed", []),
            "memories_processed": result.get("memories_processed", 0),
        }
        self._sleep_history.append(entry)

        # Keep last 30 sleep cycles
        if len(self._sleep_history) > 30:
            self._sleep_history = self._sleep_history[-30:]

        self.save()

        logger.info(
            f"[{self.user_id}] Sleep recorded: {entry['duration_seconds']:.1f}s, "
            f"{entry['memories_processed']} memories processed"
        )

    # ── Private: Metric computation ───────────────────────────

    def _count_new_memories(self, bucket_mgr) -> int:
        """Count memories created since last sleep."""
        if bucket_mgr is None:
            return 0

        since = self._last_sleep_at
        if not since:
            return 0

        try:
            import asyncio

            if hasattr(bucket_mgr, 'list_all'):
                result = bucket_mgr.list_all(include_archive=False)
                if asyncio.iscoroutine(result):
                    return 0
                all_buckets = result
                count = 0
                for b in all_buckets:
                    created = (b.get("metadata", {}) if isinstance(b, dict) else {}).get(
                        "created", ""
                    )
                    if created and created > since:
                        count += 1
                return count
        except Exception:
            pass

        return 0

    def _avg_new_importance(self, bucket_mgr) -> float:
        """Compute average importance of memories since last sleep."""
        if bucket_mgr is None:
            return 0.0

        since = self._last_sleep_at
        if not since:
            return 0.0

        try:
            import asyncio

            if hasattr(bucket_mgr, 'list_all'):
                result = bucket_mgr.list_all(include_archive=False)
                if asyncio.iscoroutine(result):
                    return 0.0
                all_buckets = result
                scores = []
                for b in all_buckets:
                    meta = (b.get("metadata", {}) if isinstance(b, dict) else {})
                    created = meta.get("created", "")
                    if created and created > since:
                        imp = meta.get("importance", 5)
                        if isinstance(imp, (int, float)):
                            scores.append(float(imp))
                return sum(scores) / max(len(scores), 1) if scores else 0.0
        except Exception:
            pass

        return 0.0

    def _compute_emotional_volatility(self, bucket_mgr) -> float:
        """
        Compute emotional volatility from recent memories.

        High variance in valence/arousal → emotional turbulence →
        higher need for REM-like sleep processing (Diekelmann & Born).
        """
        if bucket_mgr is None:
            return 0.0

        try:
            import asyncio

            if hasattr(bucket_mgr, 'list_all'):
                result = bucket_mgr.list_all(include_archive=False)
                if asyncio.iscoroutine(result):
                    return 0.0
                all_buckets = result
                valences = []
                for b in all_buckets[-20:]:  # Last 20 memories
                    meta = (b.get("metadata", {}) if isinstance(b, dict) else {})
                    v = meta.get("valence", 0.5)
                    if isinstance(v, (int, float)):
                        valences.append(float(v))

                if len(valences) < 2:
                    return 0.0

                mean = sum(valences) / len(valences)
                variance = sum((v - mean) ** 2 for v in valences) / len(valences)
                return round(math.sqrt(variance), 3)
        except Exception:
            pass

        return 0.0

    def _hours_since_last_sleep(self) -> float:
        """Hours since last sleep cycle completed."""
        if not self._last_sleep_at:
            return 999.0  # Never slept → high pseudo-value

        try:
            last = datetime.fromisoformat(self._last_sleep_at)
            now = datetime.now(timezone.utc)
            return round((now - last).total_seconds() / 3600.0, 2)
        except (ValueError, TypeError):
            return 999.0

    # ── Persistence ────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.data_dir / "memory_load_monitor.json"

    def load(self):
        """Load monitor state from disk."""
        if self._loaded:
            return

        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._last_sleep_at = data.get("last_sleep_at", "")
                self._sleep_history = data.get("sleep_history", [])
            except Exception as e:
                logger.warning(f"Failed to load monitor state: {e}")

        self._loaded = True

    def save(self):
        """Persist monitor state to disk."""
        path = self._state_path()
        path.write_text(json.dumps({
            "last_sleep_at": self._last_sleep_at,
            "sleep_history": self._sleep_history[-30:],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get load monitor statistics."""
        self.load()

        avg_duration = 0.0
        if self._sleep_history:
            durations = [
                s.get("duration_seconds", 0)
                for s in self._sleep_history
            ]
            avg_duration = sum(durations) / max(len(durations), 1)

        return {
            "recommendations_generated": self._recommendation_count,
            "last_sleep_at": self._last_sleep_at,
            "hours_since_last_sleep": self._hours_since_last_sleep(),
            "total_sleep_cycles": len(self._sleep_history),
            "avg_sleep_duration_seconds": round(avg_duration, 1),
            "sleep_history_7d": len([
                s for s in self._sleep_history
                if s.get("completed_at", "") > ""
            ]),
        }
