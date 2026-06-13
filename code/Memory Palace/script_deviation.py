# ============================================================
# Module: Script Deviation (script_deviation.py)
# L2: Statistical anomaly detection for daily "scripts".
# L2：脚本偏离检测 — 基于 Schank (1982) 动态记忆理论
#
# Detects when the user's "daily script" deviates from baseline.
# This is the FIRST LINE of the storage gate — cheap, statistical,
# O(1) computation, <10ms.
# 检测用户的"日常脚本"发生偏离，作为存储门禁的第一道防线。
#
# Method (pure statistics, no LLM):
#   1. Sliding window (30-day) emotional mean/variance
#   2. Topic model: detect new topics appearing
#   3. Session frequency/time-of-day anomaly
#
# IMPORTANT: Statistical deviation ≠ importance.
#   Deviation flags "attention" (this is unusual),
#   not "importance" (this matters). Importance is judged
#   by ImportanceFusion further down the pipeline.
#
# Privacy: Per-user sliding window, zero cross-user data.
# ============================================================

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory_palace.script_deviation")


@dataclass
class EmotionalBaseline:
    """30-day sliding window emotional statistics."""
    valence_mean: float = 0.5
    valence_std: float = 0.15
    arousal_mean: float = 0.3
    arousal_std: float = 0.15
    sample_count: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "valence_mean": self.valence_mean,
            "valence_std": self.valence_std,
            "arousal_mean": self.arousal_mean,
            "arousal_std": self.arousal_std,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmotionalBaseline:
        return cls(**{k: data.get(k, 0.0) for k in [
            "valence_mean", "valence_std", "arousal_mean", "arousal_std",
            "sample_count", "last_updated",
        ]})


class ScriptDeviation:
    """
    Schank (1982) script deviation detector.

    Tracks the user's "typical" emotional and behavioral patterns
    over a 30-day sliding window. Flags deviations for attention.

    For COLD users (no baseline yet): returns neutral (0.0 deviation).
    For WARM+ users: compares current session against 30-day baseline.
    """

    def __init__(self, user_id: str = "", data_dir: str = "./buckets"):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        if user_id:
            self.data_dir = self.data_dir / user_id
        os.makedirs(self.data_dir, exist_ok=True)

        # Sliding window: last 30 days of emotional snapshots
        self._window: deque[dict] = deque(maxlen=100)  # ~3-4 sessions/day × 30 days
        self._baseline = EmotionalBaseline()
        self._topic_history: dict[str, int] = {}  # topic → occurrence count
        self._session_hours: list[int] = []       # hour-of-day of each session
        self._loaded = False

    # ── Persistence ────────────────────────────────────────

    def _baseline_path(self) -> Path:
        return self.data_dir / "script_baseline.json"

    def _window_path(self) -> Path:
        return self.data_dir / "script_window.jsonl"

    def load(self):
        """Load baseline and window from disk."""
        if self._loaded:
            return

        # Load baseline
        bp = self._baseline_path()
        if bp.exists():
            try:
                data = json.loads(bp.read_text(encoding="utf-8"))
                self._baseline = EmotionalBaseline.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load script baseline: {e}")

        # Load window
        wp = self._window_path()
        if wp.exists():
            try:
                for line in wp.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        self._window.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Failed to load script window: {e}")

        self._loaded = True

    def save(self):
        """Persist baseline and window to disk."""
        bp = self._baseline_path()
        bp.write_text(json.dumps(self._baseline.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        wp = self._window_path()
        with open(wp, "w", encoding="utf-8") as f:
            for entry in self._window:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Core detection ─────────────────────────────────────

    def detect(
        self,
        valence: float,
        arousal: float,
        topics: list[str] | None = None,
        session_hour: int = 12,
    ) -> float:
        """
        Detect script deviation for current session.
        检测当前会话的偏离程度。

        Returns 0.0 (no deviation) to 1.0 (extreme deviation).

        For COLD users (no baseline): always returns 0.0
        (we don't know what "normal" is yet).
        """
        self.load()

        # Record this session
        entry = {
            "valence": valence,
            "arousal": arousal,
            "topics": topics or [],
            "hour": session_hour,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._window.append(entry)

        if topics:
            for t in topics:
                self._topic_history[t] = self._topic_history.get(t, 0) + 1

        self._session_hours.append(session_hour)
        if len(self._session_hours) > 200:
            self._session_hours = self._session_hours[-200:]

        # COLD: no baseline yet
        if self._baseline.sample_count < 5:
            self._update_baseline()
            self.save()
            return 0.0

        # Calculate deviation score
        dev_scores = []

        # 1. Emotional deviation (valence)
        valence_z = abs(valence - self._baseline.valence_mean) / max(self._baseline.valence_std, 0.01)
        dev_scores.append(min(1.0, valence_z / 3.0))  # 3 std = max deviation

        # 2. Emotional deviation (arousal)
        arousal_z = abs(arousal - self._baseline.arousal_mean) / max(self._baseline.arousal_std, 0.01)
        dev_scores.append(min(1.0, arousal_z / 3.0))

        # 3. New topic detection
        if topics:
            known_topics = set(self._topic_history.keys())
            new_topic_ratio = sum(1 for t in topics if t not in known_topics) / len(topics)
            dev_scores.append(new_topic_ratio)

        # 4. Time-of-day anomaly
        if len(self._session_hours) >= 5:
            hour_mean = sum(self._session_hours) / len(self._session_hours)
            hour_std = max(1.0, (sum((h - hour_mean) ** 2 for h in self._session_hours) / len(self._session_hours)) ** 0.5)
            hour_z = abs(session_hour - hour_mean) / hour_std
            dev_scores.append(min(1.0, hour_z / 3.0))

        # Weighted combination: emotional dominates
        weights = [0.35, 0.35, 0.15, 0.15]  # valence, arousal, topic, hour
        if len(dev_scores) < len(weights):
            weights = weights[:len(dev_scores)]

        deviation = sum(s * w for s, w in zip(dev_scores, weights)) / sum(weights[:len(dev_scores)])

        # Update baseline with new data
        self._update_baseline()
        self.save()

        return round(min(1.0, deviation), 4)

    # ── Baseline update ────────────────────────────────────

    def _update_baseline(self):
        """Recalculate 30-day emotional baseline from window."""
        if not self._window:
            return

        valences = [e["valence"] for e in self._window]
        arousals = [e["arousal"] for e in self._window]
        n = len(valences)

        if n < 2:
            self._baseline = EmotionalBaseline(
                valence_mean=valences[0] if valences else 0.5,
                valence_std=0.15,
                arousal_mean=arousals[0] if arousals else 0.3,
                arousal_std=0.15,
                sample_count=n,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            return

        mean_v = sum(valences) / n
        mean_a = sum(arousals) / n
        std_v = (sum((v - mean_v) ** 2 for v in valences) / (n - 1)) ** 0.5
        std_a = (sum((a - mean_a) ** 2 for a in arousals) / (n - 1)) ** 0.5

        self._baseline = EmotionalBaseline(
            valence_mean=round(mean_v, 4),
            valence_std=round(max(std_v, 0.01), 4),
            arousal_mean=round(mean_a, 4),
            arousal_std=round(max(std_a, 0.01), 4),
            sample_count=n,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    # ── Query ──────────────────────────────────────────────

    def get_baseline(self) -> dict:
        """Get current emotional baseline."""
        self.load()
        return self._baseline.to_dict()

    def get_topic_novelty(self, topic: str) -> float:
        """
        How novel is this topic? 0=common, 1=never seen before.
        """
        total = sum(self._topic_history.values())
        if total == 0:
            return 1.0
        count = self._topic_history.get(topic, 0)
        # Inverse frequency
        return 1.0 - (count / total)
