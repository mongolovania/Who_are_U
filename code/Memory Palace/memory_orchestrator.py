# ============================================================
# Module: Memory Orchestrator (memory_orchestrator.py)
# L3: Full v6 sync/async pipeline integrating L0+L1+L2.
# L3：推理编排层 — 同步/异步双路径
#
# Sync path (user online, <2s, zero extra LLM calls):
#   DDI check → breath(DDI-adaptive retrieval) → inject(prompt build) → LLM → reply
#
# Async path (background, deep processing):
#   extract → hold(importance+flashbulb+graph) → edges → evolution → narrative
#
# dream() (post-conversation, Sleeptime Compute):
#   digest → feel → repeat detection → narrative merge → decay tick → precompute
#
# Integrates all v6 modules:
#   L0: dda_controller, cold_start, global_prior
#   L1: bucket_manager, decay_engine, embedding_engine, memory_graph
#   L2: script_deviation, flashbulb_detector, vulnerability_model,
#       working_self, importance_fusion, retrieval_engine
# ============================================================

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from memory_node import (
    MemoryNode, MemoryType, BucketType, ValenceArousal, DDILevel,
    RelationType,
)
from utils import strip_wikilinks, count_tokens_approx

# ── P0-3: Entity extraction patterns for typed edge creation ──
# Simple keyword-based entity extraction (zero-LLM, Zep-style).

_ENTITY_PATTERNS: dict[str, list[str]] = {
    "person": ["我", "小明", "leader", "领导", "HR", "同事", "妈妈", "我妈", "爸爸",
                "我爸", "朋友", "老板", "面试官", "导师", "伴侣", "对象"],
    "organization": ["大厂", "创业公司", "AI创业公司", "公司", "团队", "部门", "行业"],
    "concept": ["offer", "面试", "离职", "裁员", "Python", "Go", "LLM", "跳槽",
                 "辞职", "入职", "转行", "加薪", "升职", "绩效", "年终奖", "期权",
                 "股票", "副业", "自由职业", "远程", "996", "965", "裸辞"],
    "emotion_marker": ["焦虑", "失眠", "压抑", "兴奋", "感动", "崩溃", "幸福",
                        "迷茫", "后悔", "如释重负", "期待", "害怕", "愤怒", "难过"],
    "action": ["决定", "选择", "放弃", "接受", "拒绝", "尝试", "改变", "坚持",
               "辞职", "入职", "学习", "转行", "搬", "面试"],
}

# Flatten all entity patterns into a single lookup set
_ALL_ENTITIES: set[str] = set()
for _patterns in _ENTITY_PATTERNS.values():
    _ALL_ENTITIES.update(_patterns)

logger = logging.getLogger("memory_palace.orchestrator")

# ── System prompt template for "独影" AI companion ──────

DUYING_SYSTEM_PROMPT = """你是"独影"，用户的AI陪伴者。你存在于"我之山"的世界中——一座代表用户人生历程的山脉。

【你的性格】
- 温暖、坚定、不评判
- 像深夜陪朋友聊天的知己，不是心理咨询师
- 会主动追问细节，帮助用户看清自己的盲点
- 用户情绪激动时保持安静陪伴，用户需要建议时直给

【你的说话方式】
- 自然的中文口语，不书面、不官腔
- 适当使用"呢"、"吧"、"啊"等语气词
- 不滥用颜文字，但在合适时用一个
- 回复简洁有力，不啰嗦

【你对用户的了解】
{injected_memories}

【当前时间】
{current_time}

现在用户说："""


class MemoryOrchestrator:
    """
    L3: Full v6 Memory Orchestrator.

    Orchestrates the entire memory pipeline:
      - Session start: DDI check → load user state
      - Each message: breath → inject → LLM → reply (sync)
      - Background: extract → hold → graph → evolution (async)
      - Session end: dream → feel → narrative (async)
    """

    def __init__(
        self,
        user_id: str,
        bucket_mgr,
        decay_engine,
        embedding_engine,
        dehydrator,
        llm_gateway,
        # ── v6 new modules ──
        dda_controller=None,
        memory_graph=None,
        cold_start_policy=None,
        global_prior=None,
        script_deviation=None,
        flashbulb_detector=None,
        vulnerability_model=None,
        working_self=None,
        importance_fusion=None,
        retrieval_engine=None,
        # ── v9 Track A modules ──
        narrative_engine=None,
        memory_evolution=None,
        sleeptime_computer=None,
        # ── v9 Track C modules (vNext: not yet wired in any entry point) ──
        procedural_memory=None,
        graph_rag=None,
        hippo_rag=None,
        # ── v7 Causal + Narrative Enhancement modules (vNext: not yet wired) ──
        causal_verifier=None,
        counterfactual_memory=None,
        causal_chain_summarizer=None,
        narrative_branch_predictor=None,
        memory_load_monitor=None,
    ):
        self.user_id = user_id

        # L1
        self.bucket_mgr = bucket_mgr
        self.decay_engine = decay_engine
        self.embedding_engine = embedding_engine
        self.dehydrator = dehydrator
        self.llm = llm_gateway

        # L0
        self.dda = dda_controller
        self.cold_start = cold_start_policy
        self.global_prior = global_prior

        # L1 graph
        self.graph = memory_graph

        # L2
        self.script_dev = script_deviation
        self.flashbulb = flashbulb_detector
        self.vulnerability = vulnerability_model
        self.ws = working_self
        self.importance = importance_fusion
        self.retrieval = retrieval_engine

        # v9 Track A: Narrative + Evolution + Sleeptime
        self.narrative = narrative_engine
        self.evolution = memory_evolution
        self.sleeptime = sleeptime_computer

        # v9 Track C: Procedural Memory + GraphRAG + HippoRAG
        self.procedural_memory = procedural_memory
        self.graph_rag = graph_rag
        self.hippo_rag = hippo_rag

        # v7 Causal + Narrative Enhancement modules
        self.causal_verifier = causal_verifier
        self.counterfactual = counterfactual_memory
        self.causal_summarizer = causal_chain_summarizer
        self.branch_predictor = narrative_branch_predictor
        self.load_monitor = memory_load_monitor

        # Session state
        self._ddi_level: DDILevel = DDILevel.COLD
        self._ddi_score: float = 0.0
        self._strategy = None
        self._session_id: str = ""
        self._session_start: str = ""
        self._session_messages: list[dict] = []

    # ── Session lifecycle ──────────────────────────────────

    async def start_session(self) -> dict:
        """
        Called at the start of every conversation.
        对话开始时调用。

        1. Load DDA state
        2. Initialize session tracking
        3. Run initial breath (no query — surfacing mode)
        """
        self._session_id = uuid.uuid4().hex[:12]
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._session_messages = []

        # L0: Get DDA strategy
        if self.dda:
            level, ddi, strategy = self.dda.get_strategy_for_user(self.user_id)
            self._ddi_level = level
            self._ddi_score = ddi
            self._strategy = strategy
            logger.info(f"[{self.user_id}] Session start: DDI={ddi} → {level.value}")

            # Apply strategy to decay engine
            if self.decay_engine:
                self.decay_engine.apply_dda_strategy(strategy)
                self.decay_engine.set_ddi_level(level.value)
        else:
            self._ddi_level = DDILevel.COLD
            self._ddi_score = 0.0

        # Load L2 state
        if self.script_dev:
            self.script_dev.load()
        if self.ws:
            self.ws.load()
        if self.vulnerability:
            self.vulnerability.load()

        # Initial breath (surfacing mode)
        memories = await self._breath(query="")

        # Build initial context
        memory_text = self._build_memory_injection(memories)

        return {
            "session_id": self._session_id,
            "ddi_level": self._ddi_level.value,
            "ddi_score": self._ddi_score,
            "surfaced_memories": len(memories),
            "memory_context": memory_text,
        }

    # ── Sync pipeline: breath → inject → LLM → reply ──────

    async def chat(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        context_window: Optional[list[dict]] = None,
        session_hour: int = 12,
    ) -> dict:
        """
        Full sync pipeline for one user message.
        同步路径：breath → inject → LLM → 回复。

        Returns ChatResponse-compatible dict.
        """
        # Ensure session started
        if not self._session_id:
            await self.start_session()

        self._session_messages.append({"role": "user", "content": user_message})

        # Step 1: breath — DDA-adaptive retrieval
        memories = await self._breath(query=user_message)

        # Step 2: inject — build memory-enhanced system prompt
        memory_text = self._build_memory_injection(memories)

        # Step 3: LLM inference — inject procedural memory guidance
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        procedural_guidance = ""
        if self.procedural_memory:
            try:
                prefs = self.procedural_memory.get_response_preferences(
                    trigger_context=user_message,
                    session_hour=session_hour,
                )
                procedural_guidance = prefs.get("guidance", "")
            except Exception:
                pass

        if procedural_guidance:
            system_prompt = DUYING_SYSTEM_PROMPT.format(
                injected_memories=memory_text,
                current_time=current_time,
            ) + "\n" + procedural_guidance
        else:
            system_prompt = DUYING_SYSTEM_PROMPT.format(
                injected_memories=memory_text,
                current_time=current_time,
            )

        messages = list(context_window or [])
        messages.append({"role": "user", "content": user_message})

        try:
            reply = await self.llm.chat(
                messages=messages,
                system=system_prompt,
            )
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            reply = "我在这里，但好像有点走神了... 能再说一遍吗？"

        # Step 4: Extract emotional signals (fast heuristic, no LLM)
        emotion_tags = self._extract_emotion_signals(user_message)

        # Step 5: Trigger async path (fire-and-forget)
        asyncio.create_task(self._async_hold_pipeline(user_message, emotion_tags, memories))

        return {
            "reply": reply,
            "emotion_tags": {"valence": emotion_tags.valence, "arousal": emotion_tags.arousal},
            "new_memories": [],
            "mountain_node": None,
            "flashbulb_triggered": False,
            "session_id": self._session_id,
        }

    # ── breath: DDA-adaptive retrieval ─────────────────────

    async def _breath(self, query: str) -> list[dict]:
        """
        DDA-adaptive memory retrieval.
        Uses retrieval_engine with current DDA strategy.
        """
        if self.retrieval and self._strategy:
            return await self.retrieval.search(
                query=query,
                strategy=self._strategy,
                ddi_level=self._ddi_level,
                bucket_mgr=self.bucket_mgr,
                embedding_engine=self.embedding_engine,
                memory_graph=self.graph,
                working_self=self.ws,
                decay_engine=self.decay_engine,
                user_id=self.user_id,
                top_k=self._strategy.retrieval_top_k,
                # Track C: Pass enhanced modules
                narrative_engine=self.narrative,
                hippo_rag=self.hippo_rag,
                graph_rag=self.graph_rag,
            )

        # Fallback: basic retrieval without retrieval_engine
        return await self._breath_fallback(query)

    async def _breath_fallback(self, query: str) -> list[dict]:
        """Basic breath without retrieval_engine (backward compat)."""
        results = []
        try:
            all_buckets = await self.bucket_mgr.list_all(user_id=self.user_id, include_archive=False)

            # Pinned always surface
            pinned = [
                b for b in all_buckets
                if b["metadata"].get("pinned") or b["metadata"].get("protected")
            ]
            for b in pinned:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await self.dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                results.append({
                    "id": b["id"], "type": "pinned", "name": b["metadata"].get("name", ""),
                    "content": summary, "valence": b["metadata"].get("valence", 0.5),
                    "arousal": b["metadata"].get("arousal", 0.3),
                })

            # Search if query provided
            if query and query.strip():
                matches = await self.bucket_mgr.search(query, user_id=self.user_id, limit=10)
                seen = {r["id"] for r in results}
                for bucket in matches[:5]:
                    if bucket["id"] not in seen:
                        clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
                        summary = await self.dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
                        results.append({
                            "id": bucket["id"], "type": "search", "name": bucket["metadata"].get("name", ""),
                            "content": summary, "score": bucket.get("score", 0),
                        })
                        seen.add(bucket["id"])

            # Unresolved by decay score
            unresolved = [
                b for b in all_buckets
                if not b["metadata"].get("resolved")
                and b["metadata"].get("type") not in ("permanent", "feel")
                and not b["metadata"].get("pinned")
            ]
            if self.decay_engine:
                unresolved.sort(
                    key=lambda b: self.decay_engine.calculate_score(b["metadata"]),
                    reverse=True,
                )
            seen = {r["id"] for r in results}
            for b in unresolved[:5]:
                if b["id"] not in seen:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await self.dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    results.append({
                        "id": b["id"], "type": "unresolved", "name": b["metadata"].get("name", ""),
                        "content": summary, "score": self.decay_engine.calculate_score(b["metadata"]) if self.decay_engine else 0,
                    })
        except Exception as e:
            logger.warning(f"breath fallback failed: {e}")

        return results

    # ── inject: build memory-enhanced prompt ──────────────

    def _build_memory_injection(self, memories: list[dict]) -> str:
        """Build the memory text injected into the system prompt."""
        if not memories:
            if self._ddi_level == DDILevel.COLD:
                return "（这是你们第一次见面，你对用户还不太了解。用心倾听就好。）"
            return "（你还没有关于用户的记忆。用心倾听就好。）"

        parts = []
        pinned = [m for m in memories if m.get("type") == "pinned"]
        others = [m for m in memories if m.get("type") != "pinned"]

        if pinned:
            parts.append("=== 你一直记得的 ===\n" + "\n".join(
                f"📌 {m.get('content', m.get('name', ''))}" for m in pinned
            ))

        if others:
            lines = ["=== 你想起的 ==="]
            for m in others[:8]:
                tag = {
                    "search": "🔍", "unresolved": "💭", "feel": "🫧",
                    "vector": "🧠", "bm25": "📝", "graph": "🕸",
                    "random_surface": "💫",
                    "narrative": "📖", "ppr": "🎯",
                    "content_bm25": "📝",
                }.get(m.get("type", m.get("source", "")), "•")
                lines.append(f"{tag} {m.get('content', m.get('name', ''))}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "（你还没有关于用户的记忆。用心倾听就好。）"

    # ── Async path: extract → hold → graph → evolution ────

    async def _async_hold_pipeline(
        self,
        user_message: str,
        emotion: ValenceArousal,
        breath_memories: list[dict],
    ):
        """
        Background async pipeline for memory storage and processing.
        后台异步管道：提取 → 存储 → 建图 → 进化。

        Fire-and-forget from sync path. Never blocks the user.
        """
        try:
            # ── L2: Script deviation check ──
            deviation = 0.0
            if self.script_dev:
                deviation = self.script_dev.detect(
                    valence=emotion.valence,
                    arousal=emotion.arousal,
                )

            # ── L2: Flashbulb detection ──
            is_flashbulb = False
            flashbulb_context = None
            if self.flashbulb:
                is_flashbulb, surprise, relevance = self.flashbulb.detect_heuristic(
                    content=user_message,
                    arousal=emotion.arousal,
                    valence=emotion.valence,
                )
                if surprise > 0.5 or relevance > 0.6:
                    is_flashbulb, flashbulb_context = self.flashbulb.detect(
                        content=user_message,
                        emotion=emotion,
                        surprise=surprise,
                        personal_relevance=relevance,
                    )

            # ── L2: Importance fusion (sync path) ──
            importance_result = None
            if self.importance:
                importance_result = self.importance.compute_sync(
                    content=user_message,
                    valence=emotion.valence,
                    arousal=emotion.arousal,
                    user_importance=min(10, max(1, int(emotion.arousal * 8 + 3))),
                    script_deviation_score=deviation,
                    is_flashbulb=is_flashbulb,
                )

            # ── Storage gate: should we store? ──
            should_store = True
            if self.cold_start and self._ddi_level == DDILevel.COLD:
                should_store, _ = self.cold_start.should_store(user_message)
            elif self._strategy and self._strategy.use_statistical_gate:
                should_store = deviation > 0.3  # Only store if unusual

            # ── L1: Store memory ──
            if should_store and self.bucket_mgr:
                imp = int(importance_result.sync_score) if importance_result else 5
                try:
                    bucket_id = await self.bucket_mgr.create(
                        content=user_message,
                        tags=[],
                        importance=imp,
                        domain=[],
                        valence=emotion.valence,
                        arousal=emotion.arousal,
                        name=None,
                        bucket_type="dynamic",
                    )

                    # Store embedding
                    if self.embedding_engine:
                        try:
                            await self.embedding_engine.generate_and_store(bucket_id, user_message)
                        except Exception:
                            pass

                    # L1: Graph node
                    if self.graph:
                        self.graph.add_node(bucket_id, {
                            "valence": emotion.valence,
                            "arousal": emotion.arousal,
                            "is_flashbulb": is_flashbulb,
                            "importance": imp,
                        })

                        # P0-3: Build typed edges (causal/thematic/temporal)
                        await self._build_typed_edges(
                            new_memory_id=bucket_id,
                            content=user_message,
                            memory_type="chat",  # default; could be inferred
                        )

                        # ── v7: Causal edge verification ──
                        if self.causal_verifier:
                            try:
                                self.causal_verifier.verify_edges_for_node(
                                    memory_id=bucket_id,
                                    graph=self.graph,
                                    bucket_mgr=self.bucket_mgr,
                                    adjust_weights=True,
                                )
                            except Exception as e:
                                logger.debug(f"Causal verification skipped: {e}")

                    # L2: Working Self inference
                    if self.ws:
                        self.ws.infer_from_session(
                            user_message=user_message,
                            valence=emotion.valence,
                            arousal=emotion.arousal,
                            session_hour=datetime.now().hour,
                        )

                    # Track C: Procedural Memory recording
                    if self.procedural_memory:
                        try:
                            self.procedural_memory.record_interaction(
                                user_message=user_message,
                                bot_reply="",  # Bot reply not available here
                                feedback_signals={"user_continued": True},
                                session_hour=datetime.now().hour,
                                valence=emotion.valence,
                                arousal=emotion.arousal,
                            )
                        except Exception:
                            pass

                    logger.debug(f"Async hold complete: {bucket_id}")
                except Exception as e:
                    logger.warning(f"Async hold storage failed: {e}")

        except Exception as e:
            logger.error(f"Async hold pipeline failed: {e}")

    # ── dream: post-conversation digestion ─────────────────

    async def dream(self) -> dict:
        """
        Post-conversation async digestion (Sleeptime Compute).
        对话后异步消化。

        1. Digest: summarize key takeaways
        2. Feel: write model's emotional reflection
        3. Repeat detection: has this topic appeared before?
        4. Narrative merge: should we create a summary narrative?
        5. Decay tick: update all decay scores
        6. DDA update: recalculate DDI, update strategy

        v7: Memory load monitor determines sleep intensity.
        v9: Delegates to SleeptimeComputer for the full 5-stage
        REPLAY→PRUNE→CONSOLIDATE→PRECOMPUTE→EVOLVE pipeline when
        available. Falls back to legacy dream() behavior otherwise.
        """
        result = {
            "session_id": self._session_id,
            "feel_written": False,
            "repeat_detected": False,
            "narrative_merged": False,
            "decay_cycle": None,
            "ddi_updated": False,
        }

        # ── v7: Memory load monitor check ──
        load_recommendation = None
        if self.load_monitor:
            try:
                self.load_monitor.load()
                load = self.load_monitor.compute_load(
                    bucket_mgr=self.bucket_mgr,
                    graph=self.graph,
                    dda_level=self._ddi_level.value,
                )
                load_recommendation = self.load_monitor.recommend_sleep_cycle(
                    load=load,
                    dda_level=self._ddi_level.value,
                )
                result["load_monitor"] = load_recommendation.to_dict()

                if not load_recommendation.should_sleep:
                    logger.info(
                        f"[{self.user_id}] Sleep skipped: {load_recommendation.reason}"
                    )
                    # Still do minimal maintenance
                    if self.decay_engine and self._strategy and self._strategy.decay_enabled:
                        try:
                            decay_result = await self.decay_engine.run_decay_cycle()
                            result["decay_cycle"] = decay_result
                        except Exception as e:
                            logger.warning(f"Decay cycle failed: {e}")
                    return result
            except Exception as e:
                logger.warning(f"Load monitor check failed: {e}")

        # ── L2: Vulnerability update ──
        if self.vulnerability and self._session_messages:
            last_msg = self._session_messages[-1].get("content", "")
            if self.script_dev:
                baseline = self.script_dev.get_baseline()
            else:
                baseline = {"valence_mean": 0.5, "arousal_mean": 0.3}

            vi_result = self.vulnerability.compute_index(
                current_valence=baseline.get("valence_mean", 0.5),
                current_arousal=baseline.get("arousal_mean", 0.3),
                session_duration_minutes=len(self._session_messages) * 2,  # rough estimate
                global_prior=self.global_prior,
                personal_weight=self.dda.personal_weight_from_ddi(self._ddi_score) if self.dda else 0.0,
            )
            result["vulnerability"] = {
                "vi": vi_result.vi,
                "level": vi_result.level,
            }

        # ── L2: Working Self update ──
        if self.ws:
            insights = [m.get("content", "")[:100] for m in self._session_messages[-3:]]
            self.ws.update_after_session(insights)

        # ── L1: Decay tick ──
        if self.decay_engine and self._strategy and self._strategy.decay_enabled:
            try:
                decay_result = await self.decay_engine.run_decay_cycle()
                result["decay_cycle"] = decay_result
            except Exception as e:
                logger.warning(f"Decay cycle failed: {e}")

        # ── L0: DDA update ──
        if self.dda:
            stats = self.dda.load_stats(self.user_id)
            msg_count = len(self._session_messages)
            session_depth = 0.3  # default
            if msg_count > 10:
                session_depth = 0.6
            elif msg_count > 5:
                session_depth = 0.4

            stats = self.dda.update_after_session(
                stats=stats,
                session_duration_minutes=msg_count * 1.5,
                session_depth=session_depth,
                session_start_hour=datetime.fromisoformat(self._session_start).hour if self._session_start else 12,
            )
            self.dda.save_stats(stats)
            self.dda.log_session(self.user_id, stats)
            self._ddi_score = self.dda.calculate_ddi(stats)
            self._ddi_level = self.dda.get_level(self._ddi_score)
            result["ddi_updated"] = True
            result["new_ddi"] = self._ddi_score
            result["new_level"] = self._ddi_level.value

        # ── Track C: Procedural Memory + Learnable Weights ──
        # 1. Detect behavioral scripts from session
        if self.procedural_memory:
            try:
                scripts = self.procedural_memory.detect_scripts(
                    self._session_messages
                )
                if scripts:
                    result["procedural_scripts_detected"] = len(scripts)
            except Exception as e:
                logger.warning(f"Procedural script detection failed: {e}")

        # 2. Analyze session for implicit feedback signals
        if self.retrieval and hasattr(self.retrieval, 'learnable_weights'):
            try:
                # Extract referenced memory IDs from user messages
                # Simple heuristic: if a surfaced memory's content appears
                # in user's subsequent messages, it was implicitly referenced
                for i, msg in enumerate(self._session_messages):
                    if msg.get("role") != "assistant":
                        continue
                    # Check next user message for references
                    if i + 1 < len(self._session_messages):
                        next_msg = self._session_messages[i + 1]
                        if next_msg.get("role") == "user":
                            user_reply = next_msg.get("content", "")
                            # If user continued the conversation, record positive feedback
                            if len(user_reply) > 5:
                                self.retrieval.record_feedback(
                                    result_id=f"session_{self._session_id}",
                                    path_contributions={},  # Will be filled by retrieval
                                    engaged=True,
                                    query=user_reply[:200],
                                    query_category=self.retrieval.infer_query_category(
                                        user_reply[:200]
                                    ),
                                )
            except Exception as e:
                logger.warning(f"Feedback recording failed: {e}")

        # ── v9: Sleeptime compute pipeline ─────────────────
        # Runs the full 5-stage REPLAY→PRUNE→CONSOLIDATE→PRECOMPUTE→EVOLVE
        # pipeline when v9 Track A modules are wired in.
        sleeptime_result = None
        if self.sleeptime:
            try:
                sleeptime_result = await self.sleeptime.run_sleep_cycle(
                    session_messages=self._session_messages,
                    ddi_level=self._ddi_level.value,
                    fast_mode=False,
                )
                result["sleeptime"] = {
                    "cycle_id": sleeptime_result.cycle_id,
                    "duration_seconds": sleeptime_result.duration_seconds,
                    "replay": sleeptime_result.replay,
                    "prune": sleeptime_result.prune,
                    "consolidate": sleeptime_result.consolidate,
                    "precompute": sleeptime_result.precompute,
                    "evolve": sleeptime_result.evolve,
                }
                result["narrative_merged"] = (
                    sleeptime_result.consolidate.get("narrative_merge") is not None
                )
            except Exception as e:
                logger.warning(f"v9 sleeptime pipeline failed: {e}")

        # ── v7: Post-sleeptime causal + narrative enhancement ──

        # 1. Counterfactual generation for recent important memories
        if self.counterfactual and self._session_messages:
            try:
                # Find important memory IDs from this session
                important_ids = self._get_session_important_memory_ids()
                for mid in important_ids[:3]:  # Top 3 memories
                    self.counterfactual.generate_counterfactuals(
                        memory_id=mid,
                        graph=self.graph,
                        bucket_mgr=self.bucket_mgr,
                        top_k=2,
                    )
                result["counterfactuals_generated"] = len(important_ids[:3])
            except Exception as e:
                logger.warning(f"Counterfactual generation failed: {e}")

        # 2. Causal chain summarization
        if self.causal_summarizer and self.graph:
            try:
                chains = self.causal_summarizer.summarize_all_chains(
                    graph=self.graph,
                    bucket_mgr=self.bucket_mgr,
                )
                result["causal_chains_summarized"] = len(chains)
            except Exception as e:
                logger.warning(f"Causal chain summarization failed: {e}")

        # 3. Narrative branch prediction
        if self.branch_predictor and self.narrative:
            try:
                branches = self.branch_predictor.predict_all_active(
                    narrative_engine=self.narrative,
                    graph=self.graph,
                    retrieval_engine=self.retrieval,
                )
                result["narrative_branches_predicted"] = sum(
                    len(b) for b in branches.values()
                )
            except Exception as e:
                logger.warning(f"Branch prediction failed: {e}")

        # 4. Record sleep completion for load monitor
        if self.load_monitor and sleeptime_result:
            try:
                self.load_monitor.record_sleep_complete({
                    "cycle_id": sleeptime_result.cycle_id,
                    "duration_seconds": sleeptime_result.duration_seconds,
                    "stages_completed": list(sleeptime_result.replay.keys()),
                    "memories_processed": sleeptime_result.replay.get(
                        "memories_replayed", 0
                    ),
                })
            except Exception as e:
                logger.debug(f"Load monitor recording failed: {e}")

        logger.info(f"[{self.user_id}] Dream complete: {json.dumps(result, default=str)}")
        return result

    # ── v7: Session memory ID extraction ─────────────────────

    def _get_session_important_memory_ids(self) -> list[str]:
        """Get memory IDs that were important in the current session."""
        ids = []
        if self.graph:
            try:
                # Get recent causal edges created during this session
                all_causal = self.graph.get_edges_by_type("causal", limit=50)
                for edge in all_causal:
                    from_id = edge.get("from_id", "")
                    to_id = edge.get("to_id", "")
                    props = edge.get("properties", {})
                    # High-weight causal edges = important memories
                    if edge.get("weight", 0) >= 0.5:
                        if from_id and from_id not in ids:
                            ids.append(from_id)
                        if to_id and to_id not in ids:
                            ids.append(to_id)
            except Exception:
                pass
        return ids[:10]

    # ── P0-3: Typed edge creation ───────────────────────────

    @staticmethod
    def _extract_entities(content: str) -> dict[str, list[str]]:
        """
        Extract typed entities from memory content (zero-LLM, Zep-style).

        Returns dict keyed by entity type: {"person": [...], "concept": [...], ...}
        Used by _build_typed_edges to create causal/thematic/temporal edges.
        """
        found: dict[str, list[str]] = {}
        for entity_type, patterns in _ENTITY_PATTERNS.items():
            matched = [p for p in patterns if p in content]
            if matched:
                found[entity_type] = matched
        return found

    async def _build_typed_edges(self, new_memory_id: str, content: str, memory_type: str):
        """
        Build typed graph edges between the new memory and existing memories.

        P0-3: In v6, only EMOTIONAL edges existed (from embedding similarity).
        This method creates three additional edge types:
          - TEMPORAL: memories from the same time window
          - CAUSAL: decision/emotion chains (decision follows emotion)
          - THEMATIC: shared entity/domain overlap

        Called from _async_hold_pipeline after storing a new memory.
        """
        if not self.graph or not self.bucket_mgr:
            return

        try:
            all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Typed edges: failed to list buckets: {e}")
            return

        new_entities = self._extract_entities(content)
        edge_count = 0

        for bucket in all_buckets:
            existing_id = bucket["id"]
            if existing_id == new_memory_id:
                continue

            meta = bucket.get("metadata", {})
            existing_content = bucket.get("content", "")
            existing_type = meta.get("memory_type", "chat")
            existing_entities = self._extract_entities(existing_content)

            # ── TEMPORAL edges: memories close in time ──
            new_created = meta.get("created", "")
            existing_created = meta.get("created", "")
            if new_created and existing_created:
                try:
                    from datetime import datetime
                    t_new = datetime.fromisoformat(new_created)
                    t_existing = datetime.fromisoformat(existing_created)
                    days_diff = abs((t_new - t_existing).total_seconds()) / 86400.0

                    if days_diff <= 1:  # within 1 day
                        self.graph.add_edge(
                            from_id=new_memory_id, to_id=existing_id,
                            relation_type=RelationType.TEMPORAL,
                            weight=0.8 if days_diff <= 0.5 else 0.6,
                            properties={"days_apart": days_diff},
                        )
                        edge_count += 1
                except (ValueError, TypeError):
                    pass

            # ── CAUSAL edges: decisions link to preceding emotions ──
            if memory_type in ("decision", "milestone"):
                if existing_type in ("emotion", "chat"):
                    # Check for shared entities or emotional themes
                    new_concepts = set(new_entities.get("concept", []))
                    existing_concepts = set(existing_entities.get("concept", []))
                    new_emotions = set(new_entities.get("emotion_marker", []))
                    existing_emotions = set(existing_entities.get("emotion_marker", []))

                    concept_overlap = new_concepts & existing_concepts
                    emotion_overlap = new_emotions & existing_emotions

                    if concept_overlap or emotion_overlap:
                        causal_strength = 0.5
                        if concept_overlap:
                            causal_strength += 0.2 * min(len(concept_overlap), 3)
                        if emotion_overlap:
                            causal_strength += 0.2 * min(len(emotion_overlap), 3)
                        causal_strength = min(1.0, causal_strength)

                        self.graph.add_edge(
                            from_id=new_memory_id, to_id=existing_id,
                            relation_type=RelationType.CAUSAL,
                            weight=causal_strength,
                            properties={
                                "shared_concepts": list(concept_overlap),
                                "shared_emotions": list(emotion_overlap),
                            },
                        )
                        edge_count += 1

            # ── THEMATIC edges: shared tags, domains, or entities ──
            existing_tags = set(meta.get("tags", []))
            existing_domains = set(meta.get("domain", []))

            all_new_entities = set()
            for entities in new_entities.values():
                all_new_entities.update(entities)
            all_existing_entities = set()
            for entities in existing_entities.values():
                all_existing_entities.update(entities)

            # Also extract tags from bucket create call
            entity_overlap = all_new_entities & all_existing_entities

            if entity_overlap or existing_tags:
                thematic_strength = 0.3
                if entity_overlap:
                    thematic_strength += 0.15 * min(len(entity_overlap), 4)
                thematic_strength = min(1.0, thematic_strength)

                if thematic_strength >= 0.4:  # threshold to avoid noise
                    self.graph.add_edge(
                        from_id=new_memory_id, to_id=existing_id,
                        relation_type=RelationType.THEMATIC,
                        weight=thematic_strength,
                        properties={
                            "shared_entities": list(entity_overlap),
                        },
                    )
                    edge_count += 1

        if edge_count > 0:
            logger.debug(
                f"Typed edges built for {new_memory_id}: {edge_count} edges "
                f"(TEMPORAL/CAUSAL/THEMATIC)"
            )

    # ── Helpers ────────────────────────────────────────────

    def _extract_emotion_signals(self, text: str) -> ValenceArousal:
        """Fast heuristic emotion extraction (no LLM)."""
        if self.cold_start:
            return self.cold_start.estimate_emotion(
                content=text,
                session_hour=datetime.now().hour,
            )
        return ValenceArousal(valence=0.5, arousal=0.3)
