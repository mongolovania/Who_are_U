# ============================================================
# Module: Learnable Path Weights (learnable_weights.py)
# Track C Task 5: MemLong-style online learning of retrieval
# path weights from user engagement feedback.
#
# Theoretical foundation:
#   1. MemLong (2025) — "MemLong: Long-Term Memory Augmented LLMs."
#      Uses learnable retrieval weights that adapt over time
#      based on retrieval quality feedback. EMA-based weight
#      updates with exploration-exploitation balance.
#   2. Sutton & Barto (2018) — Reinforcement Learning: An
#      Introduction. ε-greedy exploration, multi-armed bandit
#      for path weight optimization.
#   3. Auer, Cesa-Bianchi & Fischer (2002), Machine Learning —
#      "Finite-time Analysis of the Multiarmed Bandit Problem."
#      UCB1-inspired confidence bounds for path selection.
#
# Implementation:
#   - EMA weight adaptation: w_new = α * w_observed + (1-α) * w_old
#   - Implicit feedback extraction from user behavior
#   - ε-greedy exploration with decaying epsilon
#   - Weight regularization toward base weights (avoid overfitting)
#   - Per-query-category weight specialization
# ============================================================

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory_palace.learnable_weights")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class FeedbackSignal:
    """A single feedback signal for weight learning."""
    result_id: str               # Memory ID that was presented
    query: str = ""              # The query that produced this result
    query_category: str = ""     # emotional | causal | temporal | factual | cross_reference
    path_contributions: dict[str, float] = field(default_factory=dict)
    # Which path contributed how much to this result's final score
    engaged: bool = False        # Did the user engage with this result?
    referenced: bool = False     # Did the user reference this memory in reply?
    ignored: bool = True         # Did the user ignore this result? (auto-computed)
    timestamp: str = ""

    def __post_init__(self):
        # Auto-compute ignored from engaged/referenced
        self.ignored = not self.engaged and not self.referenced


@dataclass
class PathWeightState:
    """Learned weights for a single retrieval path."""
    path_name: str
    base_weight: float           # Original hardcoded weight
    learned_weight: float        # Current learned weight
    observation_count: int = 0   # Number of feedback signals received
    success_count: int = 0       # Number of positive engagements
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "path_name": self.path_name,
            "base_weight": self.base_weight,
            "learned_weight": self.learned_weight,
            "observation_count": self.observation_count,
            "success_count": self.success_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PathWeightState:
        return cls(
            path_name=data.get("path_name", ""),
            base_weight=data.get("base_weight", 0.0),
            learned_weight=data.get("learned_weight", data.get("base_weight", 0.0)),
            observation_count=data.get("observation_count", 0),
            success_count=data.get("success_count", 0),
            last_updated=data.get("last_updated", ""),
        )

    @property
    def success_rate(self) -> float:
        if self.observation_count == 0:
            return 0.5  # neutral prior
        return self.success_count / self.observation_count


# ═══════════════════════════════════════════════════════════════
# Learnable Path Weights Engine
# ═══════════════════════════════════════════════════════════════


class LearnablePathWeights:
    """
    MemLong-style online weight learning for retrieval path fusion.

    Adapts path weights based on implicit user feedback:
      - User referencing a memory → that path gets boosted
      - User ignoring results → those paths get slightly penalized
      - Query category specialization → different weights per query type

    Uses EMA (Exponential Moving Average) for smooth adaptation
    and ε-greedy exploration to avoid local optima.
    """

    def __init__(
        self,
        base_weights: dict[str, float],
        learning_rate: float = 0.1,
        regularization_strength: float = 0.01,
        epsilon: float = 0.05,
        epsilon_decay: float = 0.999,
        min_weight: float = 0.02,
        user_id: str = "",
        data_dir: str = "./buckets",
    ):
        """
        Args:
            base_weights: initial path weights (from retrieval_engine)
            learning_rate: EMA alpha for weight updates (0-1)
            regularization_strength: pull toward base weights (L2)
            epsilon: exploration probability (ε-greedy)
            epsilon_decay: multiplicative epsilon decay per update
            min_weight: minimum weight for any path (avoid zeroing out)
            user_id: user identifier for persistence
            data_dir: data directory for persistence
        """
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_weight = min_weight

        # Per-category weight states
        self._global_state: dict[str, PathWeightState] = {}
        self._category_states: dict[str, dict[str, PathWeightState]] = {}
        # category → {path_name: PathWeightState}

        # Initialize from base weights
        for path_name, weight in base_weights.items():
            self._global_state[path_name] = PathWeightState(
                path_name=path_name,
                base_weight=weight,
                learned_weight=weight,
            )

        # Feedback buffer
        self._feedback_buffer: list[FeedbackSignal] = []
        self._total_updates: int = 0

        # Persistence
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        if user_id:
            self.data_dir = self.data_dir / user_id
        os.makedirs(self.data_dir, exist_ok=True)

    # ── Persistence ──────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.data_dir / "learnable_weights.json"

    def save(self):
        """Persist learned weights to disk."""
        path = self._state_path()
        data = {
            "global_state": {
                pn: ps.to_dict() for pn, ps in self._global_state.items()
            },
            "category_states": {
                cat: {pn: ps.to_dict() for pn, ps in states.items()}
                for cat, states in self._category_states.items()
            },
            "total_updates": self._total_updates,
            "epsilon": self.epsilon,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self):
        """Load learned weights from disk."""
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._global_state = {
                    pn: PathWeightState.from_dict(ps)
                    for pn, ps in data.get("global_state", {}).items()
                }
                self._category_states = {}
                for cat, states in data.get("category_states", {}).items():
                    self._category_states[cat] = {
                        pn: PathWeightState.from_dict(ps)
                        for pn, ps in states.items()
                    }
                self._total_updates = data.get("total_updates", 0)
                self.epsilon = data.get("epsilon", self.epsilon)
            except Exception as e:
                logger.warning(f"Failed to load learnable weights: {e}")

    # ── Weight retrieval ─────────────────────────────────────

    def get_weights(self, query_category: str = "factual") -> dict[str, float]:
        """
        Get current path weights, optionally specialized for query category.

        With probability ε, returns slightly randomized weights (exploration).
        Otherwise returns learned weights (exploitation).

        Args:
            query_category: emotional | causal | temporal | factual | cross_reference

        Returns:
            {path_name: weight} dict, normalized to sum to 1.0
        """
        # ε-greedy exploration
        if random.random() < self.epsilon:
            return self._get_exploration_weights(query_category)

        return self._get_exploitation_weights(query_category)

    def get_base_weights(self) -> dict[str, float]:
        """Get the original base weights (no learning applied)."""
        weights = {
            pn: ps.base_weight
            for pn, ps in self._global_state.items()
        }
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def _get_exploitation_weights(self, query_category: str) -> dict[str, float]:
        """Get learned weights for a query category."""
        # Start with global weights
        weights = {
            pn: ps.learned_weight
            for pn, ps in self._global_state.items()
        }

        # Blend with category-specific if available
        if query_category in self._category_states:
            cat_states = self._category_states[query_category]
            # Blend: 70% category + 30% global for paths with enough data
            for pn, cs in cat_states.items():
                if pn in weights:
                    if cs.observation_count >= 10:
                        weights[pn] = 0.7 * cs.learned_weight + 0.3 * weights[pn]
                    elif cs.observation_count >= 3:
                        weights[pn] = 0.4 * cs.learned_weight + 0.6 * weights[pn]
                    # else: use global (not enough data)

        # Apply regularization toward base
        for pn in weights:
            base = self._global_state[pn].base_weight
            weights[pn] = (
                weights[pn] * (1 - self.regularization_strength)
                + base * self.regularization_strength
            )

        # Ensure minimum weight
        for pn in weights:
            weights[pn] = max(self.min_weight, weights[pn])

        # Normalize
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def _get_exploration_weights(self, query_category: str) -> dict[str, float]:
        """Generate exploration weights with added noise."""
        base = self._get_exploitation_weights(query_category)

        # Add Gaussian noise to each weight
        noisy = {}
        for pn, w in base.items():
            noise = random.gauss(0, 0.1)  # std=0.1
            noisy[pn] = max(self.min_weight, w + noise)

        # Normalize
        total = sum(noisy.values())
        if total > 0:
            return {k: v / total for k, v in noisy.items()}
        return noisy

    # ── Feedback recording ───────────────────────────────────

    def record_feedback(
        self,
        result_id: str,
        path_contributions: dict[str, float],
        engaged: bool = False,
        referenced: bool = False,
        query: str = "",
        query_category: str = "factual",
    ):
        """
        Record user feedback for a retrieved result.

        Called after each retrieval to track which paths
        produced engaging results.

        Args:
            result_id: memory ID
            path_contributions: {path_name: contribution_score}
            engaged: did user click/expand/engage?
            referenced: did user reference this memory in chat?
            query: original query
            query_category: inferred query type
        """
        signal = FeedbackSignal(
            result_id=result_id,
            query=query,
            query_category=query_category,
            path_contributions=dict(path_contributions),
            engaged=engaged,
            referenced=referenced,
            ignored=not engaged and not referenced,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._feedback_buffer.append(signal)

        # Batch process every 10 signals
        if len(self._feedback_buffer) >= 10:
            self._process_feedback_buffer()

    def _process_feedback_buffer(self):
        """Process accumulated feedback signals."""
        if not self._feedback_buffer:
            return

        # Group by query category
        by_category: dict[str, list[FeedbackSignal]] = {}
        for signal in self._feedback_buffer:
            cat = signal.query_category or "factual"
            by_category.setdefault(cat, []).append(signal)

        # Update global weights from all signals
        self._update_from_signals(self._global_state, self._feedback_buffer)

        # Update category-specific weights
        for cat, signals in by_category.items():
            if cat not in self._category_states:
                self._init_category_state(cat)
            self._update_from_signals(self._category_states[cat], signals)

        # Clear buffer
        self._feedback_buffer.clear()
        self._total_updates += 1

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def _update_from_signals(
        self,
        state: dict[str, PathWeightState],
        signals: list[FeedbackSignal],
    ):
        """Update path weights from feedback signals."""
        if not signals:
            return

        # Count successes per path
        path_successes: dict[str, int] = {}
        path_observations: dict[str, int] = {}

        for signal in signals:
            contributions = signal.path_contributions
            total_contribution = sum(contributions.values()) or 1.0

            for path_name, contribution in contributions.items():
                if path_name not in state:
                    continue

                path_observations[path_name] = (
                    path_observations.get(path_name, 0) + 1
                )

                if signal.engaged or signal.referenced:
                    # Weight success by contribution proportion
                    prop = contribution / total_contribution
                    path_successes[path_name] = (
                        path_successes.get(path_name, 0) + prop
                    )

        # Update each path's weight using EMA
        for path_name, ps in state.items():
            obs = path_observations.get(path_name, 0)
            succ = path_successes.get(path_name, 0.0)

            if obs == 0:
                continue

            ps.observation_count += obs
            ps.success_count += succ

            # Observed weight: proportion of successes
            observed_weight = succ / obs

            # EMA update: w_new = α * w_observed + (1-α) * w_old
            # But we scale learning_rate by observation ratio
            # so paths with more observations get more updates
            effective_lr = self.learning_rate * min(1.0, obs / 20.0)
            ps.learned_weight = (
                effective_lr * observed_weight
                + (1 - effective_lr) * ps.learned_weight
            )

            # Clamp
            ps.learned_weight = max(self.min_weight, min(1.0, ps.learned_weight))
            ps.last_updated = datetime.now(timezone.utc).isoformat()

    def _init_category_state(self, category: str):
        """Initialize per-category weight state from global."""
        self._category_states[category] = {
            pn: PathWeightState(
                path_name=pn,
                base_weight=ps.base_weight,
                learned_weight=ps.base_weight,
            )
            for pn, ps in self._global_state.items()
        }

    # ── Explicit feedback (user ratings, etc.) ────────────────

    def apply_explicit_feedback(
        self,
        path_ratings: dict[str, float],
        query_category: str = "",
    ):
        """
        Apply explicit user feedback (e.g., "this path is good/bad").

        Args:
            path_ratings: {path_name: rating} where rating 0-1
                          0 = bad, 0.5 = neutral, 1 = good
            query_category: optional category for specialization
        """
        target_state = self._global_state
        if query_category and query_category in self._category_states:
            target_state = self._category_states[query_category]

        for path_name, rating in path_ratings.items():
            if path_name in target_state:
                ps = target_state[path_name]
                ps.observation_count += 1
                ps.success_count += rating

                # Direct update (higher impact than implicit)
                effective_lr = self.learning_rate * 2.0  # 2x for explicit
                ps.learned_weight = (
                    effective_lr * rating
                    + (1 - effective_lr) * ps.learned_weight
                )
                ps.last_updated = datetime.now(timezone.utc).isoformat()

    # ── Weight statistics ────────────────────────────────────

    def get_weight_deltas(self) -> dict[str, float]:
        """
        Get weight deltas: how much each path has shifted from base.

        Positive delta = path is more useful than expected.
        Negative delta = path is less useful than expected.
        """
        deltas = {}
        for pn, ps in self._global_state.items():
            deltas[pn] = round(ps.learned_weight - ps.base_weight, 4)
        return deltas

    def get_stats(self) -> dict:
        """Get weight learning statistics."""
        return {
            "total_updates": self._total_updates,
            "epsilon": round(self.epsilon, 4),
            "weight_deltas": self.get_weight_deltas(),
            "path_stats": {
                pn: {
                    "base": round(ps.base_weight, 4),
                    "learned": round(ps.learned_weight, 4),
                    "observations": ps.observation_count,
                    "success_rate": round(ps.success_rate, 3),
                }
                for pn, ps in self._global_state.items()
            },
            "categories_tracked": len(self._category_states),
        }

    def reset_to_base(self):
        """Reset all learned weights to base weights."""
        for ps in self._global_state.values():
            ps.learned_weight = ps.base_weight
            ps.observation_count = 0
            ps.success_count = 0

        self._category_states.clear()
        self._feedback_buffer.clear()
        self.epsilon = 0.05
        logger.info("Learnable weights reset to base")
