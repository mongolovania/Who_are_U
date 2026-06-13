# ============================================================
# Memory Palace — FastAPI Application Entry Point (app.py)
#
# Mounts:
#   - MCP server (original Ombre Brain tools, user_id aware)
#   - REST API (for Flutter App)
#
# Run:
#   python app.py                   # stdio mode (MCP only)
#   OMBRE_TRANSPORT=http python app.py  # HTTP with REST API
# ============================================================

from __future__ import annotations

import os
import sys
import logging
from contextlib import asynccontextmanager

# Ensure same-directory modules importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils import load_config, setup_logging

# ── Load config ─────────────────────────────────────────
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("memory_palace")

# ── Initialize core components ──────────────────────────
from embedding_engine import EmbeddingEngine
from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from llm_gateway import LLMGateway
from namespace_manager import NamespaceManager
from auth_service import AuthService
from memory_orchestrator import MemoryOrchestrator

# Shared components (stateless, thread-safe)
llm_gateway = LLMGateway(config=config)
namespace_mgr = NamespaceManager(config.get("buckets_dir", "./buckets"))
dehydrator = Dehydrator(config)
auth_service = AuthService()

# Per-user component factory
def _make_components(user_id: str) -> dict:
    """Create per-user components (embedding, bucket mgr, decay engine)."""
    paths = namespace_mgr.resolve(user_id)

    embedding_engine = EmbeddingEngine(config, user_id=user_id)

    # Override buckets_dir for this user
    user_config = {**config, "buckets_dir": paths["buckets_dir"]}
    bucket_mgr = BucketManager(user_config, embedding_engine=embedding_engine, user_id=user_id)

    # Decay engine per user (uses user-specific bucket manager)
    decay_engine = DecayEngine(user_config, bucket_mgr, user_id=user_id)

    return {
        "bucket_mgr": bucket_mgr,
        "decay_engine": decay_engine,
        "embedding_engine": embedding_engine,
        "paths": paths,
    }


# ── Orchester factory ───────────────────────────────────
def get_orchestrator(user_id: str) -> MemoryOrchestrator:
    comps = _make_components(user_id)
    return MemoryOrchestrator(
        bucket_mgr=comps["bucket_mgr"],
        decay_engine=comps["decay_engine"],
        dehydrator=dehydrator,
        embedding_engine=comps["embedding_engine"],
        llm_gateway=llm_gateway,
    )


# ── FastAPI app ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    logger.info("Memory Palace starting...")
    yield
    logger.info("Memory Palace shutting down...")


app = FastAPI(
    title="Memory Palace",
    description="Cognitive Narrative Memory Engine for 你谁啊",
    version="9.0.0",
    lifespan=lifespan,
)

# CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ─────────────────────────────────────
def get_current_user(request: Request) -> str:
    """Extract and validate JWT from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing or invalid Authorization header")
    token = auth_header[7:]
    user_id = auth_service.verify_token(token)
    if not user_id:
        raise ValueError("Invalid or expired token")
    return user_id["sub"] if isinstance(user_id, dict) else user_id


# ── Health check ────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "memory_palace", "version": "9.0.0"}


# ── Mount REST API router ───────────────────────────────
from api_router import router as api_router
app.include_router(api_router, prefix="/api/v1")


# ── Main (stdio / HTTP) ──────────────────────────────────
if __name__ == "__main__":
    transport = os.environ.get("OMBRE_TRANSPORT", config.get("transport", "stdio"))

    if transport == "stdio":
        # Track B: Use V2 MCP server for stdio mode
        logger.info("Starting Memory Palace V2 in stdio (MCP) mode")
        import asyncio
        from mcp_server import mcp as mcp_v2
        asyncio.run(mcp_v2.run(transport="stdio"))
    elif transport in ("sse", "streamable-http"):
        # HTTP with REST API + MCP tools
        import uvicorn
        port = int(os.environ.get("OMBRE_PORT", "8000"))
        logger.info(f"Starting HTTP server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Default: REST API HTTP mode
        import uvicorn
        port = int(os.environ.get("OMBRE_PORT", "8000"))
        logger.info(f"Starting HTTP server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
