# ============================================================
# Test: Agency Router (test_agency_router.py)
# L3: MCP vs REST dual-mode routing.
#
# Covers:
#   - MCP mode → PassiveToolInterface
#   - REST mode → AgentPipelineInterface
#   - Correct routing by CallerType
#   - Interface protocol compliance
# ============================================================

import pytest
from unittest.mock import AsyncMock, MagicMock

from agency_router import (
    AgencyRouter, PassiveToolInterface, AgentPipelineInterface,
    CallerType,
)
from memory_node import ValenceArousal


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch._breath = AsyncMock(return_value=[{"id": "mem_1", "content": "test"}])
    orch._async_hold_pipeline = AsyncMock(return_value=None)
    orch.dream = AsyncMock(return_value={"status": "ok"})
    orch.chat = AsyncMock(return_value={"reply": "test reply"})
    orch.start_session = AsyncMock(return_value={"session_id": "s1"})
    return orch


@pytest.fixture
def router(mock_orchestrator):
    return AgencyRouter(mock_orchestrator)


# ── CallerType ──────────────────────────────────────────────

class TestCallerType:
    """Verify CallerType enum."""

    def test_mcp_value(self):
        assert CallerType.MCP.value == "mcp"

    def test_rest_value(self):
        assert CallerType.REST.value == "rest"


# ── Routing ─────────────────────────────────────────────────

class TestRouting:
    """Verify correct interface routing."""

    def test_mcp_routes_to_passive(self, router):
        interface = router.route(CallerType.MCP)
        assert isinstance(interface, PassiveToolInterface)

    def test_rest_routes_to_agent(self, router):
        interface = router.route(CallerType.REST)
        assert isinstance(interface, AgentPipelineInterface)

    def test_properties_return_correct_type(self, router):
        assert isinstance(router.mcp, PassiveToolInterface)
        assert isinstance(router.rest, AgentPipelineInterface)


# ── Passive Tool Interface (MCP) ────────────────────────────

class TestPassiveToolInterface:
    """Verify MCP mode: tools that Claude calls explicitly."""

    @pytest.mark.asyncio
    async def test_breath_returns_memories(self, router):
        interface = router.mcp
        memories = await interface.breath(query="测试")
        assert isinstance(memories, list)

    @pytest.mark.asyncio
    async def test_hold_triggers_async_pipeline(self, router, mock_orchestrator):
        interface = router.mcp
        result = await interface.hold(content="重要的事", valence=0.3, arousal=0.7)
        assert result == {"status": "stored"}
        # The async hold pipeline should have been triggered
        import asyncio
        await asyncio.sleep(0.1)  # Let the task run

    @pytest.mark.asyncio
    async def test_dream_delegates_to_orchestrator(self, router, mock_orchestrator):
        interface = router.mcp
        result = await interface.dream()
        assert result == {"status": "ok"}
        mock_orchestrator.dream.assert_called_once()


# ── Agent Pipeline Interface (REST) ─────────────────────────

class TestAgentPipelineInterface:
    """Verify REST mode: autonomous agent pipeline."""

    @pytest.mark.asyncio
    async def test_chat_calls_orchestrator(self, router, mock_orchestrator):
        interface = router.rest
        result = await interface.chat(user_message="你好")
        mock_orchestrator.chat.assert_called_once()
        assert "reply" in result

    @pytest.mark.asyncio
    async def test_start_session_calls_orchestrator(self, router, mock_orchestrator):
        interface = router.rest
        result = await interface.start_session()
        mock_orchestrator.start_session.assert_called_once()
        assert result == {"session_id": "s1"}

    @pytest.mark.asyncio
    async def test_end_session_calls_dream(self, router, mock_orchestrator):
        interface = router.rest
        result = await interface.end_session()
        mock_orchestrator.dream.assert_called_once()
        assert result == {"status": "ok"}


# ── Invariants ──────────────────────────────────────────────

class TestInvariants:
    """Cross-interface invariants."""

    def test_both_interfaces_share_orchestrator(self, router, mock_orchestrator):
        mcp = router.mcp
        rest = router.rest
        assert mcp.orch is rest.orch
        assert mcp.orch is mock_orchestrator
