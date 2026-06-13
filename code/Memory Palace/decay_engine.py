# ============================================================
# Module: Memory Decay Engine (decay_engine.py)
# 模块：记忆衰减引擎
#
# Simulates human forgetting curve; auto-decays inactive memories and archives them.
# 模拟人类遗忘曲线，自动衰减不活跃记忆并归档。
#
# Core formula (improved Ebbinghaus + emotion coordinates):
# 核心公式（改进版艾宾浩斯遗忘曲线 + 情感坐标）：
#   Score = Importance × (activation_count^0.3) × e^(-λ×days) × emotion_weight
#
# Emotion weight (continuous coordinate, not discrete labels):
# 情感权重（基于连续坐标而非离散列举）：
#   emotion_weight = base + (arousal × arousal_boost)
#   Higher arousal → higher emotion weight → slower decay
#   唤醒度越高 → 情感权重越大 → 记忆衰减越慢
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================

import math
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("ombre_brain.decay")


class DecayEngine:
    """
    Memory decay engine — periodically scans all dynamic buckets,
    calculates decay scores, auto-archives low-activity buckets
    to simulate natural forgetting.
    记忆衰减引擎 —— 定期扫描所有动态桶，
    计算衰减得分，将低活跃桶自动归档，模拟自然遗忘。
    """

    def __init__(self, config: dict, bucket_mgr, user_id: str = ""):
        # --- Load decay parameters / 加载衰减参数 ---
        decay_cfg = config.get("decay", {})
        self.decay_lambda = decay_cfg.get("lambda", 0.05)
        self.threshold = decay_cfg.get("threshold", 0.3)
        self.check_interval = decay_cfg.get("check_interval_hours", 24)
        self.user_id = user_id

        # --- Emotion weight params (continuous arousal coordinate) ---
        # --- 情感权重参数（基于连续 arousal 坐标）---
        emotion_cfg = decay_cfg.get("emotion_weights", {})
        self.emotion_base = emotion_cfg.get("base", 1.0)
        self.arousal_boost = emotion_cfg.get("arousal_boost", 0.8)

        self.bucket_mgr = bucket_mgr

        # --- Background task control / 后台任务控制 ---
        self._task: asyncio.Task | None = None
        self._running = False

        # --- Life Tick: health monitoring (v6 plan) / 心跳健康监控 ---
        self._last_tick = 0.0
        self._tick_timeout = 300          # 5 minutes without tick = dead
        self._cycle_count = 0
        self._error_count = 0
        self._health_status = "healthy"   # healthy | degraded | dead
        self._life_tick_interval = 60 * 60  # Health check interval (1h)

        # ── DDA integration (v6 L0→L1 coupling) ──────────
        self._ddi_level = "COLD"
        self._strategy_applied = False

    # ── DDA adaptive lambda ─────────────────────────────

    def apply_dda_strategy(self, strategy) -> None:
        """
        Apply DDA strategy settings to decay engine.
        将 DDA 策略应用到衰减引擎参数。

        Called by memory_orchestrator when DDI level changes.
        COLD users: no decay (protect early memories)
        WARM+: decay enabled with adaptive lambda
        """
        from memory_node import DDAStrategy
        if isinstance(strategy, DDAStrategy):
            self.decay_enabled = strategy.decay_enabled
            if strategy.decay_enabled:
                self.decay_lambda = strategy.decay_lambda
            else:
                self.decay_lambda = 0.0
            self._strategy_applied = True

    def set_ddi_level(self, level: str) -> None:
        """
        Set DDI level for adaptive decay.
        COLD → no decay, WARM → global λ, HOT → personal λ, RICH → fully adaptive.
        """
        self._ddi_level = level
        if level == "COLD":
            self.decay_enabled = False
            self.decay_lambda = 0.0
            self.threshold = 0.0  # never archive
        elif level == "WARM":
            self.decay_enabled = True
            self.decay_lambda = 0.05
            self.threshold = 0.3
        elif level == "HOT":
            self.decay_enabled = True
            self.decay_lambda = 0.05
            self.threshold = 0.25
        elif level == "RICH":
            self.decay_enabled = True
            self.decay_lambda = 0.05
            self.threshold = 0.2

    @property
    def is_running(self) -> bool:
        """Whether the decay engine is running in the background.
        衰减引擎是否正在后台运行。"""
        return self._running

    # ---------------------------------------------------------
    # Core: calculate decay score for a single bucket
    # 核心：计算单个桶的衰减得分
    #
    # Higher score = more vivid memory; below threshold → archive
    # 得分越高 = 记忆越鲜活，低于阈值则归档
    # Permanent buckets never decay / 固化桶永远不衰减
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Freshness bonus: continuous exponential decay
    # 新鲜度加成：连续指数衰减
    # bonus = 1.0 + 1.0 × e^(-t/36), t in hours
    # t=0 → 2.0×, t≈25h(半衰) → 1.5×, t≈72h → ≈1.14×, t→∞ → 1.0×
    # ---------------------------------------------------------
    @staticmethod
    def _calc_time_weight(days_since: float) -> float:
        """
        Freshness bonus multiplier: 1.0 + e^(-t/36), t in hours.
        新鲜度加成乘数：刚存入×2.0，~36小时半衰，72小时后趋近×1.0。
        """
        hours = days_since * 24.0
        return 1.0 + 1.0 * math.exp(-hours / 36.0)

    def calculate_score(self, metadata: dict) -> float:
        """
        Calculate current activity score for a memory bucket.
        计算一个记忆桶的当前活跃度得分。

        New model: short-term vs long-term weight separation.
        新模型：短期/长期权重分离。
        - Short-term (≤3 days): time_weight dominates, emotion amplifies
        - Long-term (>3 days): emotion_weight dominates, time decays to floor
        短期（≤3天）：时间权重主导，情感放大
        长期（>3天）：情感权重主导，时间衰减到底线
        """
        if not isinstance(metadata, dict):
            return 0.0

        # --- Pinned/protected buckets: never decay, importance locked to 10 ---
        if metadata.get("pinned") or metadata.get("protected"):
            return 999.0

        # --- Permanent buckets never decay ---
        if metadata.get("type") == "permanent":
            return 999.0

        # --- Feel buckets: never decay, fixed moderate score ---
        if metadata.get("type") == "feel":
            return 50.0

        importance = max(1, min(10, int(metadata.get("importance", 5))))
        activation_count = max(1.0, float(metadata.get("activation_count", 1)))

        # --- Days since last activation ---
        last_active_str = metadata.get("last_active", metadata.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days_since = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days_since = 30

        # --- Emotion weight ---
        try:
            arousal = max(0.0, min(1.0, float(metadata.get("arousal", 0.3))))
        except (ValueError, TypeError):
            arousal = 0.3
        emotion_weight = self.emotion_base + arousal * self.arousal_boost

        # --- Time weight ---
        time_weight = self._calc_time_weight(days_since)

        # --- Short-term vs Long-term weight separation ---
        # 短期（≤3天）：time_weight 占 70%，emotion 占 30%
        # 长期（>3天）：emotion 占 70%，time_weight 占 30%
        if days_since <= 3.0:
            # Short-term: time dominates, emotion amplifies
            combined_weight = time_weight * 0.7 + emotion_weight * 0.3
        else:
            # Long-term: emotion dominates, time provides baseline
            combined_weight = emotion_weight * 0.7 + time_weight * 0.3

        # --- Base score ---
        base_score = (
            importance
            * (activation_count ** 0.3)
            * math.exp(-self.decay_lambda * days_since)
            * combined_weight
        )

        # --- Weight pool modifiers ---
        # resolved + digested (has feel) → accelerated fade: ×0.02
        # resolved only → ×0.05
        # 已处理+已消化（写过feel）→ 加速淡化：×0.02
        # 仅已处理 → ×0.05
        resolved = metadata.get("resolved", False)
        digested = metadata.get("digested", False)  # set when feel is written for this memory
        if resolved and digested:
            resolved_factor = 0.02
        elif resolved:
            resolved_factor = 0.05
        else:
            resolved_factor = 1.0
        urgency_boost = 1.5 if (arousal > 0.7 and not resolved) else 1.0

        return round(base_score * resolved_factor * urgency_boost, 4)

    # ---------------------------------------------------------
    # Execute one decay cycle
    # 执行一轮衰减周期
    # Scan all dynamic buckets → score → archive those below threshold
    # 扫描所有动态桶 → 算分 → 低于阈值的归档
    # ---------------------------------------------------------
    async def run_decay_cycle(self) -> dict:
        """
        Execute one decay cycle: iterate dynamic buckets, archive those
        scoring below threshold.
        执行一轮衰减：遍历动态桶，归档得分低于阈值的桶。

        Returns stats: {"checked": N, "archived": N, "lowest_score": X}
        """
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for decay / 衰减周期列桶失败: {e}")
            return {"checked": 0, "archived": 0, "lowest_score": 0, "error": str(e)}

        checked = 0
        archived = 0
        auto_resolved = 0
        lowest_score = float("inf")

        for bucket in buckets:
            meta = bucket.get("metadata", {})

            # Skip permanent / pinned / protected / feel buckets
            # 跳过固化桶、钉选/保护桶和 feel 桶
            if meta.get("type") in ("permanent", "feel") or meta.get("pinned") or meta.get("protected"):
                continue

            checked += 1

            # --- Auto-resolve: imp≤4 + >30 days old + not resolved → auto resolve ---
            # --- 自动结案：重要度≤4 + 超过30天 + 未解决 → 自动 resolve ---
            if not meta.get("resolved", False):
                imp = int(meta.get("importance", 5))
                last_active_str = meta.get("last_active", meta.get("created", ""))
                try:
                    last_active = datetime.fromisoformat(str(last_active_str))
                    days_since = (datetime.now() - last_active).total_seconds() / 86400
                except (ValueError, TypeError):
                    days_since = 999
                if imp <= 4 and days_since > 30:
                    try:
                        await self.bucket_mgr.update(bucket["id"], resolved=True)
                        meta["resolved"] = True  # refresh local meta so resolved_factor applies this cycle
                        auto_resolved += 1
                        logger.info(
                            f"Auto-resolved / 自动结案: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(imp={imp}, days={days_since:.0f})"
                        )
                    except Exception as e:
                        logger.warning(f"Auto-resolve failed / 自动结案失败: {e}")

            try:
                score = self.calculate_score(meta)
            except Exception as e:
                logger.warning(
                    f"Score calculation failed for {bucket.get('id', '?')} / "
                    f"计算得分失败: {e}"
                )
                continue

            lowest_score = min(lowest_score, score)

            # --- Below threshold → archive (simulate forgetting) ---
            # --- 低于阈值 → 归档（模拟遗忘）---
            if score < self.threshold:
                try:
                    success = await self.bucket_mgr.archive(bucket["id"])
                    if success:
                        archived += 1
                        logger.info(
                            f"Decay archived / 衰减归档: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(score={score:.4f}, threshold={self.threshold})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Archive failed for {bucket.get('id', '?')} / "
                        f"归档失败: {e}"
                    )

        result = {
            "checked": checked,
            "archived": archived,
            "auto_resolved": auto_resolved,
            "lowest_score": lowest_score if checked > 0 else 0,
        }
        logger.info(f"Decay cycle complete / 衰减周期完成: {result}")
        return result

    # ---------------------------------------------------------
    # Background decay task management
    # 后台衰减任务管理
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        """
        Ensure the decay engine is started (lazy init on first call).
        确保衰减引擎已启动（懒加载，首次调用时启动）。
        """
        if not self._running:
            await self.start()

    async def start(self) -> None:
        """Start the background decay loop.
        启动后台衰减循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"Decay engine started, interval: {self.check_interval}h / "
            f"衰减引擎已启动，检查间隔: {self.check_interval} 小时"
        )

    async def stop(self) -> None:
        """Stop the background decay loop.
        停止后台衰减循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Decay engine stopped / 衰减引擎已停止")

    async def _background_loop(self) -> None:
        """Background loop with Life Tick health monitoring (v6 plan).
        后台循环体：执行衰减 → 健康检查 → 睡眠 → 重复。"""
        import time as _time
        while self._running:
            try:
                # Run decay cycle with 2-min timeout per cycle
                await asyncio.wait_for(
                    self.run_decay_cycle(),
                    timeout=120
                )
                self._last_tick = _time.time()
                self._cycle_count += 1
                self._error_count = 0
                self._health_status = "healthy"
            except asyncio.TimeoutError:
                logger.warning(f"[{self.user_id or 'global'}] Decay cycle timed out, skipping...")
                self._error_count += 1
                if self._error_count > 5:
                    self._health_status = "degraded"
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.user_id or 'global'}] Decay cycle error: {e}")
                self._error_count += 1
                if self._error_count > 5:
                    self._health_status = "degraded"

            # Health check: reset if stuck for too long
            elapsed = _time.time() - self._last_tick if self._last_tick else 0
            if elapsed > self._tick_timeout:
                logger.error(f"[{self.user_id or 'global'}] Decay engine appears dead, resetting...")
                self._error_count = 0
                self._last_tick = _time.time()
                self._health_status = "healthy"

            # Wait for next cycle
            try:
                await asyncio.sleep(self.check_interval * 3600)
            except asyncio.CancelledError:
                break
