# ============================================================
# Module: Pydantic Models (models.py)
# Request/response models for Memory Palace REST API.
# ============================================================

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, description="User message content")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID for continuation")
    context_window: list[dict] = Field(default_factory=list, description="Recent conversation turns")


class ChatResponse(BaseModel):
    reply: str
    emotion_tags: dict = Field(default_factory=lambda: {"valence": 0.5, "arousal": 0.3})
    new_memories: list[str] = Field(default_factory=list)
    mountain_node: Optional[dict] = None
    flashbulb_triggered: bool = False
    agency_action: Optional[str] = None


# ── Memory CRUD ─────────────────────────────────────────

class HoldRequest(BaseModel):
    content: str = Field(..., min_length=1)
    tags: str = ""
    importance: int = Field(5, ge=1, le=10)
    pinned: bool = False
    feel: bool = False
    source_bucket: str = ""
    valence: float = Field(-1, ge=-1, le=1)
    arousal: float = Field(-1, ge=-1, le=1)


class GrowRequest(BaseModel):
    content: str = Field(..., min_length=1)


class TraceRequest(BaseModel):
    name: str = ""
    domain: str = ""
    valence: float = -1
    arousal: float = -1
    importance: int = -1
    tags: str = ""
    resolved: int = -1
    pinned: int = -1
    digested: int = -1
    content: str = ""
    delete: bool = False


# ── Breath / Search ─────────────────────────────────────

class BreathRequest(BaseModel):
    query: str = ""
    max_tokens: int = Field(10000, ge=1, le=20000)
    domain: str = ""
    valence: float = Field(-1, ge=-1, le=1)
    arousal: float = Field(-1, ge=-1, le=1)
    max_results: int = Field(20, ge=1, le=50)
    importance_min: int = -1


# ── Auth ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


# ── Stats ───────────────────────────────────────────────

class MemoryStats(BaseModel):
    total_memories: int
    dynamic_count: int
    permanent_count: int
    feel_count: int
    archive_count: int
    decision_count: int
    emotion_curve: list[dict] = Field(default_factory=list)


class PulseResponse(BaseModel):
    status: str
    decay_engine: str
    stats: MemoryStats
    ddi_level: str = "COLD"


# ── Sync ────────────────────────────────────────────────

class SyncPushRequest(BaseModel):
    objects: list[dict] = Field(..., description="Encrypted sync objects")


class SyncPullResponse(BaseModel):
    objects: list[dict]
    server_version: int


# ── Consent ─────────────────────────────────────────────

class ConsentUpdate(BaseModel):
    analytics: bool = False
    aggregation: bool = False
    model_training: bool = False
