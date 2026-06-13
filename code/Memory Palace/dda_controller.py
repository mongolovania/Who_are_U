# ============================================================
# Module: DDA Controller (dda_controller.py)
# L0: Data-Density-Adaptive controller.
# L0：数据密度自适应控制器
#
# Calculates Data Density Index (DDI) from user statistics,
# maps it to COLD/WARM/HOT/RICH levels, and selects the
# corresponding strategy from the strategy matrix.
# 计算数据密度指数，映射到四级，选择对应策略。
#
# Core formula (Design §2.1):
#   DDI = weighted_sum([
#       total_sessions * 0.20,
#       sessions_per_week * 0.25,
#       avg_session_duration * 0.15,
#       avg_session_depth * 0.15,
#       days_since_first_use * 0.10,
#       session_regularity * 0.10,
#       time_of_day_pattern * 0.05,
#   ])
#
# Privacy constraint (宪章级):
#   ZERO cross-user data flow. All models use ONLY:
#   (a) This user's own data
#   (b) LLM training corpus general knowledge
#   (c) Published academic research conclusions
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

from memory_node import DDILevel, DDAStrategy, STRATEGY_MATRIX

logger = logging.getLogger("memory_palace.dda")


@dataclass
class UserStats:
    """Per-user statistics for DDI calculation."""
    user_id: str = ""
    total_sessions: int = 0
    sessions_this_week: int = 0
    avg_session_duration_minutes: float = 0.0
    avg_session_depth: float = 0.0         # 0-1, composite of emotion intensity × topic depth
    days_since_first_use: float = 0.0
    session_regularity: float = 1.0         # 0-1, 1=perfectly regular
    time_of_day_pattern_score: float = 0.0  # 0-1, higher = late-night sessions (higher urgency)

    # Derived
    sessions_per_week: float = 0.0

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "total_sessions": self.total_sessions,
            "sessions_per_week": self.sessions_per_week,
            "avg_session_duration_minutes": self.avg_session_duration_minutes,
            "avg_session_depth": self.avg_session_depth,
            "days_since_first_use": self.days_since_first_use,
            "session_regularity": self.session_regularity,
            "time_of_day_pattern_score": self.time_of_day_pattern_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserStats:
        return cls(**{k: data.get(k, 0.0 if k != "user_id" else "") for k in [
            "user_id", "total_sessions", "sessions_this_week", "avg_session_duration_minutes",
            "avg_session_depth", "days_since_first_use", "session_regularity",
            "time_of_day_pattern_score", "sessions_per_week",
        ]})


class DDAController:
    """
    L0: Data-Density-Adaptive Controller.

    Responsibilities:
      1. Track per-user statistics (sessions, frequency, depth, regularity)
      2. Calculate DDI score
      3. Map DDI → COLD/WARM/HOT/RICH
      4. Return the corresponding DDAStrategy from the strategy matrix
      5. Persist user stats to disk for cross-session continuity

    Design principle: Model complexity = min(personal_data_amount, v6_model_requirement)
                     — Vapnik Structural Risk Minimization
    """

    def __init__(self, stats_dir: str = "./buckets"):
        self.stats_dir = Path(stats_dir)
        os.makedirs(self.stats_dir, exist_ok=True)

        # DDI thresholds (Design §2.2)
        self.thresholds: dict[DDILevel, tuple[float, float]] = {
            DDILevel.COLD: (0, 10),
            DDILevel.WARM: (10, 50),
            DDILevel.HOT:  (50, 200),
            DDILevel.RICH: (200, float("inf")),
        }

    # ── Stats persistence ──────────────────────────────────

    def _stats_path(self, user_id: str) -> Path:
        return self.stats_dir / user_id / "user_stats.json"

    def load_stats(self, user_id: str) -> UserStats:
        """Load user stats from disk. Returns empty stats for new users."""
        path = self._stats_path(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return UserStats.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load stats for {user_id}: {e}")
        return UserStats(user_id=user_id)

    def save_stats(self, stats: UserStats):
        """Persist user stats to disk."""
        path = self._stats_path(stats.user_id)
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Stats update ───────────────────────────────────────

    def update_after_session(
        self,
        stats: UserStats,
        session_duration_minutes: float,
        session_depth: float,        # 0-1
        session_start_hour: int,     # 0-23
    ) -> UserStats:
        """
        Update user stats after each conversation session.
        每次对话结束后更新统计。

        Called by: memory_orchestrator.dream() — async path
        """
        now = datetime.now(timezone.utc)

        # Track first-use timestamp
        if stats.total_sessions == 0:
            self._set_first_use_timestamp(stats.user_id, now)

        first_use = self._get_first_use_timestamp(stats.user_id, now)
        stats.days_since_first_use = max(0, (now - first_use).total_seconds() / 86400)

        # Session counts
        stats.total_sessions += 1
        stats.sessions_this_week = self._count_sessions_this_week(stats.user_id)

        # Rolling averages (exponential moving average)
        alpha = 0.3  # weight for new value
        stats.avg_session_duration_minutes = (
            (1 - alpha) * stats.avg_session_duration_minutes
            + alpha * session_duration_minutes
        )
        stats.avg_session_depth = (
            (1 - alpha) * stats.avg_session_depth
            + alpha * session_depth
        )

        # Sessions per week
        if stats.days_since_first_use > 0:
            weeks = max(1, stats.days_since_first_use / 7)
            stats.sessions_per_week = stats.total_sessions / weeks
        else:
            stats.sessions_per_week = 1.0

        # Session regularity (1 - normalized std of intervals)
        stats.session_regularity = self._calculate_regularity(stats.user_id)

        # Time-of-day pattern (late-night = higher urgency)
        if 0 <= session_start_hour <= 5:
            stats.time_of_day_pattern_score = 0.8  # 凌晨·高紧迫度
        elif 22 <= session_start_hour <= 23:
            stats.time_of_day_pattern_score = 0.5  # 深夜
        else:
            stats.time_of_day_pattern_score = 0.1  # 白天

        return stats

    # ── DDI Calculation ────────────────────────────────────

    def calculate_ddi(self, stats: UserStats) -> float:
        """
        Calculate Data Density Index (Design §2.1).

        Weights tuned for the "你谁啊" use case:
          - sessions_per_week (0.25): usage frequency is strongest signal
          - total_sessions (0.20): cumulative data volume
          - session_depth (0.15): quality of engagement
          - session_duration (0.15): length of engagement
          - days_since_first_use (0.10): history length (capped at 365 days)
          - session_regularity (0.10): consistency
          - time_of_day_pattern (0.05): urgency signal
        """
        # Normalize days_since_first_use: 0~365 → 0~30 score
        days_norm = min(stats.days_since_first_use, 365) / 365 * 30

        # Normalize session duration: 0~60min → 0~30 score
        duration_norm = min(stats.avg_session_duration_minutes, 60) / 60 * 30

        # Normalize sessions per week: 0~14 → 0~30 score
        spw_norm = min(stats.sessions_per_week, 14) / 14 * 30

        # Normalize total sessions: 0~500 → 0~30 score
        total_norm = min(stats.total_sessions, 500) / 500 * 30

        ddi = (
            total_norm * 0.20
            + spw_norm * 0.25
            + duration_norm * 0.15
            + stats.avg_session_depth * 30 * 0.15
            + days_norm * 0.10
            + stats.session_regularity * 30 * 0.10
            + stats.time_of_day_pattern_score * 30 * 0.05
        )

        return round(ddi, 2)

    def get_level(self, ddi: float) -> DDILevel:
        """Map DDI score to level."""
        for level, (low, high) in self.thresholds.items():
            if low <= ddi < high or (high == float("inf") and ddi >= low):
                return level
        return DDILevel.COLD

    def get_strategy(self, ddi: float) -> DDAStrategy:
        """Get the DDA strategy for a given DDI score."""
        level = self.get_level(ddi)
        return STRATEGY_MATRIX[level]

    def get_strategy_for_user(self, user_id: str) -> tuple[DDILevel, float, DDAStrategy]:
        """
        One-stop: load stats → calculate DDI → get strategy.
        Called by memory_orchestrator at the start of each session.
        """
        stats = self.load_stats(user_id)
        ddi = self.calculate_ddi(stats)
        level = self.get_level(ddi)
        strategy = STRATEGY_MATRIX[level]
        logger.info(f"[{user_id}] DDI={ddi} → {level.value} "
                     f"(sessions={stats.total_sessions}, spw={stats.sessions_per_week:.1f})")
        return level, ddi, strategy

    # ── Internal helpers ───────────────────────────────────

    def _set_first_use_timestamp(self, user_id: str, dt: datetime):
        path = self.stats_dir / user_id / ".first_use"
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(dt.isoformat())

    def _get_first_use_timestamp(self, user_id: str, fallback: datetime) -> datetime:
        path = self.stats_dir / user_id / ".first_use"
        if path.exists():
            try:
                return datetime.fromisoformat(path.read_text().strip())
            except (ValueError, TypeError):
                pass
        return fallback

    def _count_sessions_this_week(self, user_id: str) -> int:
        """Count sessions in the last 7 days from session log."""
        log_path = self.stats_dir / user_id / "session_log.jsonl"
        if not log_path.exists():
            return 1
        cutoff = (datetime.now(timezone.utc).timestamp() - 7 * 86400)
        count = 0
        try:
            for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp", 0)
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts).timestamp()
                    if ts >= cutoff:
                        count += 1
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception:
            return 1
        return max(1, count)

    def _calculate_regularity(self, user_id: str) -> float:
        """
        Calculate session interval regularity.
        1.0 = perfectly regular intervals, 0.0 = completely random.
        """
        log_path = self.stats_dir / user_id / "session_log.jsonl"
        if not log_path.exists():
            return 1.0  # new user = assume regular

        timestamps = []
        try:
            for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp", 0)
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts).timestamp()
                    timestamps.append(ts)
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception:
            return 1.0

        if len(timestamps) < 3:
            return 1.0

        timestamps.sort()
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        if not intervals or max(intervals) == 0:
            return 1.0

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 1.0

        # Coefficient of variation: std/mean → regularity = 1 - normalized_cv
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean_interval
        # cv=0 → regularity=1, cv=2+ → regularity≈0
        regularity = max(0.0, min(1.0, 1.0 - cv / 2.0))
        return regularity

    def log_session(self, user_id: str, stats: UserStats):
        """Append a session entry to the user's session log."""
        log_path = self.stats_dir / user_id / "session_log.jsonl"
        os.makedirs(log_path.parent, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sessions": stats.total_sessions,
            "ddi": self.calculate_ddi(stats),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
