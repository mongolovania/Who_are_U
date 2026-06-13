# ============================================================
# Module: Agency Router (agency_router.py)
# L3: MCP vs REST dual-mode routing.
# L3：MCP/REST 双模式路由
#
# Design §5.2:
#   MCP mode (for Claude): passive tool interface
#     - breath as a tool Claude can call
#     - Claude decides when to retrieve and store
#     - Memory Palace makes NO autonomous decisions
#
#   REST mode (for Flutter App): Agent pipeline
#     - App calls /api/chat → Memory Palace auto breath/hold/dream
#     - Memory Palace acts as the Agent
#     - Orchestrator automatically manages memory lifecycle
# ============================================================

from __future__ import annotations

from enum import Enum
from typing import Protocol

from memory_node import ValenceArousal


class CallerType(str, Enum):
    """Who is calling the Memory Palace?"""
    MCP = "mcp"        # Claude Desktop or other MCP client
    REST = "rest"      # Flutter App via REST API


class AgencyInterface(Protocol):
    """Protocol that both MCP and REST interfaces must satisfy."""

    async def breath(self, query: str = "", **kwargs) -> list[dict]:
        """Retrieve/surface memories."""
        ...

    async def hold(self, content: str, **kwargs) -> dict:
        """Store a memory."""
        ...

    async def dream(self) -> dict:
        """Post-conversation digestion."""
        ...


class PassiveToolInterface:
    """
    MCP mode: Passive tools that Claude calls explicitly.

    Memory Palace is a tool provider, not an autonomous agent.
    Claude decides WHEN to call breath/hold/dream.
    Memory Palace never acts without being called.
    """

    def __init__(self, orchestrator):
        self.orch = orchestrator

    async def breath(self, query: str = "", **kwargs) -> list[dict]:
        """Called by Claude at session start or when searching."""
        memories = await self.orch._breath(query=query)
        return memories

    async def hold(self, content: str, **kwargs) -> dict:
        """Called by Claude when something is worth remembering."""
        emotion = ValenceArousal(
            valence=kwargs.get("valence", 0.5),
            arousal=kwargs.get("arousal", 0.3),
        )
        # Fire async hold pipeline
        import asyncio
        asyncio.create_task(self.orch._async_hold_pipeline(content, emotion, []))
        return {"status": "stored"}

    async def dream(self) -> dict:
        """Called by Claude at session end for reflection."""
        return await self.orch.dream()


class AgentPipelineInterface:
    """
    REST mode: Autonomous agent pipeline for Flutter App.

    Memory Palace acts AS the agent. It automatically:
      - breaths at session start
      - holds after each message
      - dreams at session end
    The Flutter App just sends messages and gets replies.
    """

    def __init__(self, orchestrator):
        self.orch = orchestrator

    async def chat(
        self,
        user_message: str,
        context_window: list[dict] | None = None,
    ) -> dict:
        """
        One-shot chat: breath → inject → LLM → reply.
        Memory lifecycle is fully automated.
        """
        return await self.orch.chat(
            user_message=user_message,
            context_window=context_window,
        )

    async def start_session(self) -> dict:
        """Initialize session with automatic breath."""
        return await self.orch.start_session()

    async def end_session(self) -> dict:
        """End session with automatic dream."""
        return await self.orch.dream()


class AgencyRouter:
    """
    Routes to the correct interface based on caller type.

    Usage:
        router = AgencyRouter(orchestrator)
        interface = router.route(CallerType.REST)
        reply = await interface.chat("我今天...")
    """

    def __init__(self, orchestrator):
        self._mcp = PassiveToolInterface(orchestrator)
        self._rest = AgentPipelineInterface(orchestrator)

    def route(self, caller: CallerType) -> AgencyInterface:
        if caller == CallerType.MCP:
            return self._mcp
        return self._rest

    @property
    def mcp(self) -> PassiveToolInterface:
        return self._mcp

    @property
    def rest(self) -> AgentPipelineInterface:
        return self._rest
