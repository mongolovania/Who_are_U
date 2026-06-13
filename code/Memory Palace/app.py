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
from retrieval_engine import RetrievalEngine
from memory_graph import MemoryGraph
from importance_fusion import ImportanceFusion
from flashbulb_detector import FlashbulbDetector
from script_deviation import ScriptDeviation
from vulnerability_model import VulnerabilityModel
from working_self import WorkingSelf
from dda_controller import DDAController
from cold_start import ColdStartPolicy
from global_prior import GlobalPrior
from narrative_engine import NarrativeEngine
from memory_evolution import MemoryEvolution
from sleeptime_compute import SleeptimeComputer
from procedural_memory import ProceduralMemory
from graph_rag import GraphRAG
from hippo_rag import HippoRAG
from learnable_weights import LearnableWeights
from causal_verifier import CausalVerifier
from counterfactual_memory import CounterfactualMemory
from causal_chain_summarizer import CausalChainSummarizer
from narrative_branch_predictor import NarrativeBranchPredictor
from memory_load_monitor import MemoryLoadMonitor

# Shared components (stateless, thread-safe)
llm_gateway = LLMGateway(config=config)
namespace_mgr = NamespaceManager(config.get("buckets_dir", "./buckets"))
dehydrator = Dehydrator(config)
auth_service = AuthService()
global_prior = GlobalPrior()

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


# ── Orchestrator factory ─────────────────────────────────
def _make_orchestrator(user_id: str) -> MemoryOrchestrator:
    """Create a fully-wired MemoryOrchestrator for a user.

    Wires all 22 optional modules (DDA+graph+L2+Track C+v7 causal),
    per WBS 2.8.1: 工厂函数注入完整 v9 模块.
    """
    paths = namespace_mgr.resolve(user_id)
    user_config = {**config, "buckets_dir": paths["buckets_dir"]}

    # ── L1: Storage ──
    embedding_engine = EmbeddingEngine(config, user_id=user_id)
    bucket_mgr = BucketManager(user_config, embedding_engine=embedding_engine, user_id=user_id)
    decay_engine = DecayEngine(user_config, bucket_mgr, user_id=user_id)

    # ── L0: DDA Adaptive ──
    dda = DDAController(stats_dir=paths["buckets_dir"])
    cold_start = ColdStartPolicy()

    # ── L1: Graph ──
    graph = MemoryGraph(user_id=user_id, db_dir=paths["buckets_dir"])

    # ── L2: Intelligence (6 modules) ──
    retrieval = RetrievalEngine()
    importance = ImportanceFusion()
    flashbulb = FlashbulbDetector()
    script_dev = ScriptDeviation(user_id=user_id, data_dir=paths["buckets_dir"])
    vulnerability = VulnerabilityModel(user_id=user_id, data_dir=paths["buckets_dir"])
    ws = WorkingSelf(user_id=user_id, data_dir=paths["buckets_dir"])

    # ── v9 Track A: Advanced ──
    narrative = NarrativeEngine(user_id=user_id, data_dir=paths["buckets_dir"])
    evolution = MemoryEvolution(user_id=user_id, data_dir=paths["buckets_dir"])
    sleeptime = SleeptimeComputer(
        user_id=user_id,
        bucket_mgr=bucket_mgr,
        decay_engine=decay_engine,
        memory_graph=graph,
        narrative_engine=narrative,
        memory_evolution=evolution,
        embedding_engine=embedding_engine,
    )

    # ── Track C: SOTA (4 modules) ──
    procedural = ProceduralMemory(user_id=user_id, data_dir=paths["buckets_dir"])
    graph_rag_engine = GraphRAG(resolution=1.0)
    hippo_rag_engine = HippoRAG(alpha=0.85)
    learnable = LearnableWeights(
        base_weights=retrieval.path_weights.copy(),
        user_id=user_id,
        data_dir=paths["buckets_dir"],
    )
    retrieval.learnable_weights = learnable

    # ── v7: Causal Reasoning (6 modules) ──
    causal_v = CausalVerifier(user_id=user_id)
    counterfactual = CounterfactualMemory(user_id=user_id)
    causal_summarizer = CausalChainSummarizer(user_id=user_id, llm_gateway=llm_gateway)
    branch_predictor = NarrativeBranchPredictor(user_id=user_id)
    load_monitor = MemoryLoadMonitor(user_id=user_id, data_dir=paths["buckets_dir"])

    # ── Assemble ──
    return MemoryOrchestrator(
        user_id=user_id,
        bucket_mgr=bucket_mgr,
        decay_engine=decay_engine,
        embedding_engine=embedding_engine,
        dehydrator=dehydrator,
        llm_gateway=llm_gateway,
        dda_controller=dda,
        memory_graph=graph,
        cold_start_policy=cold_start,
        global_prior=global_prior,
        script_deviation=script_dev,
        flashbulb_detector=flashbulb,
        vulnerability_model=vulnerability,
        working_self=ws,
        importance_fusion=importance,
        retrieval_engine=retrieval,
        narrative_engine=narrative,
        memory_evolution=evolution,
        sleeptime_computer=sleeptime,
        procedural_memory=procedural,
        graph_rag=graph_rag_engine,
        hippo_rag=hippo_rag_engine,
        causal_verifier=causal_v,
        counterfactual_memory=counterfactual,
        causal_chain_summarizer=causal_summarizer,
        narrative_branch_predictor=branch_predictor,
        memory_load_monitor=load_monitor,
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
    version="0.1.0",
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
    return {"status": "ok", "service": "who_are_u", "version": "0.1.0", "mp_version": "0.9.1"}


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
