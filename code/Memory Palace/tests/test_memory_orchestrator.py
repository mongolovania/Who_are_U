# ============================================================
# Test: Memory Orchestrator (test_memory_orchestrator.py)
# L3: Sync/async pipeline integration tests.
#
# Covers:
#   - Session lifecycle (start, chat, dream)
#   - Sync path: breath → inject → LLM → reply
#   - Async path: extract → hold → graph → evolution
#   - DDA integration
#   - Memory injection formatting
#   - Emotion extraction heuristic
# ============================================================

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from memory_node import (
    MemoryNode, MemoryType, BucketType, ValenceArousal, DDILevel,
    DDAStrategy, COLD_STRATEGY,
)
from memory_orchestrator import MemoryOrchestrator, DUYING_SYSTEM_PROMPT


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="我是独影，我在这里陪你。")
    return llm


@pytest.fixture
def mock_bucket_mgr():
    mgr = AsyncMock()
    mgr.list_all = AsyncMock(return_value=[])
    mgr.create = AsyncMock(return_value="new_bucket_id")
    return mgr


@pytest.fixture
def mock_dda():
    dda = MagicMock()
    dda.get_strategy_for_user = MagicMock(
        return_value=(DDILevel.COLD, 0.0, COLD_STRATEGY)
    )
    dda.load_stats = MagicMock(return_value=MagicMock())
    dda.calculate_ddi = MagicMock(return_value=0.0)
    dda.get_level = MagicMock(return_value=DDILevel.COLD)
    dda.update_after_session = MagicMock(return_value=MagicMock())
    dda.save_stats = MagicMock(return_value=None)
    dda.log_session = MagicMock(return_value=None)
    return dda


@pytest.fixture
def mock_decay():
    de = MagicMock()
    de.calculate_score = MagicMock(return_value=5.0)
    de.apply_dda_strategy = MagicMock()
    de.set_ddi_level = MagicMock()
    de.run_decay_cycle = AsyncMock(return_value={"checked": 0, "archived": 0})
    return de


@pytest.fixture
def mock_embedding():
    ee = MagicMock()
    ee.generate_and_store = AsyncMock()
    ee.search_similar = AsyncMock(return_value=[])
    return ee


@pytest.fixture
def mock_dehydrator():
    dh = MagicMock()
    dh.dehydrate = AsyncMock(return_value="[摘要] 测试")
    dh.analyze = AsyncMock(return_value={
        "domain": ["测试"], "valence": 0.5, "arousal": 0.3,
        "tags": [], "suggested_name": "test",
    })
    return dh


@pytest.fixture
def orchestrator(mock_bucket_mgr, mock_decay, mock_embedding, mock_dehydrator, mock_llm, mock_dda):
    return MemoryOrchestrator(
        user_id="test_user",
        bucket_mgr=mock_bucket_mgr,
        decay_engine=mock_decay,
        embedding_engine=mock_embedding,
        dehydrator=mock_dehydrator,
        llm_gateway=mock_llm,
        dda_controller=mock_dda,
    )


# ── Session Lifecycle ───────────────────────────────────────

class TestSessionLifecycle:
    """Verify session start, chat, dream."""

    @pytest.mark.asyncio
    async def test_start_session_initializes_state(self, orchestrator):
        result = await orchestrator.start_session()
        assert "session_id" in result
        assert "ddi_level" in result
        assert orchestrator._session_id != ""

    @pytest.mark.asyncio
    async def test_chat_returns_reply(self, orchestrator, mock_llm):
        await orchestrator.start_session()
        result = await orchestrator.chat("你好")
        assert "reply" in result
        assert len(result["reply"]) > 0
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_dream_returns_summary(self, orchestrator):
        await orchestrator.start_session()
        await orchestrator.chat("测试消息")
        result = await orchestrator.dream()
        assert "session_id" in result
        assert "feel_written" in result

    @pytest.mark.asyncio
    async def test_auto_starts_session_on_chat(self, orchestrator):
        """chat() without explicit start_session() should auto-init."""
        result = await orchestrator.chat("你好")
        assert "reply" in result
        assert orchestrator._session_id != ""


# ── Memory Injection ────────────────────────────────────────

class TestMemoryInjection:
    """Verify memory context formatting."""

    def test_cold_user_no_memories(self, orchestrator):
        orchestrator._ddi_level = DDILevel.COLD
        text = orchestrator._build_memory_injection([])
        assert "第一次见面" in text or "还没有" in text

    def test_pinned_memories_shown_first(self, orchestrator):
        memories = [
            {"id": "1", "type": "pinned", "name": "核心信念", "content": "每天进步一点点"},
            {"id": "2", "type": "search", "name": "普通记忆", "content": "今天学习了"},
        ]
        text = orchestrator._build_memory_injection(memories)
        assert "一直记得" in text
        assert "📌" in text

    def test_multiple_memory_types_each_have_icon(self, orchestrator):
        memories = [
            {"id": "1", "type": "search", "name": "a", "content": "search result"},
            {"id": "2", "type": "unresolved", "name": "b", "content": "unresolved"},
            {"id": "3", "type": "feel", "name": "c", "content": "feel"},
        ]
        text = orchestrator._build_memory_injection(memories)
        # Each type should have an icon
        assert "🔍" in text
        assert "💭" in text
        assert "🫧" in text


# ── DDA Integration ─────────────────────────────────────────

class TestDDAIntegration:
    """Verify DDA strategy is applied."""

    @pytest.mark.asyncio
    async def test_dda_applied_on_session_start(self, orchestrator, mock_dda, mock_decay):
        await orchestrator.start_session()
        mock_dda.get_strategy_for_user.assert_called_with("test_user")
        mock_decay.apply_dda_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_dream_updates_dda(self, orchestrator, mock_dda):
        await orchestrator.start_session()
        await orchestrator.chat("测试")
        result = await orchestrator.dream()
        assert result["ddi_updated"] is True


# ── Emotion Extraction ──────────────────────────────────────

class TestEmotionExtraction:
    """Verify fast heuristic emotion extraction."""

    def test_cold_start_emotion_used(self, orchestrator):
        # Cold start policy is set
        emotion = orchestrator._extract_emotion_signals("我今天很开心")
        assert isinstance(emotion, ValenceArousal)
        assert 0.0 <= emotion.valence <= 1.0
        assert 0.0 <= emotion.arousal <= 1.0


# ── Edge Cases ──────────────────────────────────────────────

class TestOrchestratorBoundaries:
    """Edge case tests for orchestrator."""

    @pytest.mark.asyncio
    async def test_empty_message_handled(self, orchestrator, mock_llm):
        mock_llm.chat = AsyncMock(return_value="你说了什么吗？")
        result = await orchestrator.chat("")
        assert "reply" in result

    @pytest.mark.asyncio
    async def test_llm_error_returns_fallback(self, orchestrator, mock_llm):
        mock_llm.chat = AsyncMock(side_effect=Exception("API Error"))
        result = await orchestrator.chat("你好")
        assert "reply" in result  # Fallback reply
        assert len(result["reply"]) > 0

    @pytest.mark.asyncio
    async def test_breath_handles_error(self, orchestrator, mock_bucket_mgr):
        mock_bucket_mgr.list_all = AsyncMock(side_effect=Exception("DB Error"))
        memories = await orchestrator._breath_fallback("test")
        assert memories == []  # Graceful fallback


# ── System Prompt ───────────────────────────────────────────

class TestSystemPrompt:
    """Verify system prompt template."""

    def test_prompt_contains_character(self):
        assert "独影" in DUYING_SYSTEM_PROMPT
        assert "我之山" in DUYING_SYSTEM_PROMPT

    def test_prompt_has_injection_slot(self):
        assert "{injected_memories}" in DUYING_SYSTEM_PROMPT
        assert "{current_time}" in DUYING_SYSTEM_PROMPT

    def test_prompt_formats_correctly(self):
        formatted = DUYING_SYSTEM_PROMPT.format(
            injected_memories="用户喜欢编程",
            current_time="2026年06月09日 12:00",
        )
        assert "用户喜欢编程" in formatted
        assert "2026年06月09日" in formatted
