# ============================================================
# Memory Palace V2 — MCP Server (mcp_server.py)
# Track B: Production MCP stdio JSON-RPC interface
#
# Exposes the V2 MemoryPalace orchestrator and retrieval engine
# as MCP tools for Claude and other MCP clients.
# 暴露 V2 MemoryPalace 编排器和检索引擎作为 MCP 工具。
#
# Run:
#   python mcp_server.py                    # stdio mode (default)
#   OMBRE_TRANSPORT=streamable-http python mcp_server.py  # HTTP
#
# Tools:
#   memory_search  — DDA-adaptive multi-path retrieval
#   memory_store   — Store memory with importance fusion
#   memory_status  — DDA level, bucket counts, decay status
#   memory_graph   — Query typed memory graph neighbors
#   memory_dream   — Trigger post-session digestion
#   memory_evolve  — Trigger memory evolution + zettelkasten links
# ============================================================

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from embedding_engine import EmbeddingEngine
from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from llm_gateway import LLMGateway
from namespace_manager import NamespaceManager
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
from utils import load_config, setup_logging, strip_wikilinks, count_tokens_approx

# ── Config & logging ─────────────────────────────────────
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("memory_palace.mcp")

# ── Shared components (stateless) ────────────────────────
llm_gateway = LLMGateway(config=config)
namespace_mgr = NamespaceManager(config.get("buckets_dir", "./buckets"))
dehydrator = Dehydrator(config)
global_prior = GlobalPrior()

# ── Create MCP server ────────────────────────────────────
mcp = FastMCP(
    "Memory Palace V2",
    host="0.0.0.0",
    port=int(os.environ.get("OMBRE_PORT", "8000")),
)


# ── Per-user component factory ──────────────────────────
def _make_orchestrator(user_id: str) -> MemoryOrchestrator:
    """Create a fully-wired MemoryOrchestrator for a user."""
    paths = namespace_mgr.resolve(user_id)
    user_config = {**config, "buckets_dir": paths["buckets_dir"]}

    # L1: Storage
    embedding_engine = EmbeddingEngine(config, user_id=user_id)
    bucket_mgr = BucketManager(user_config, embedding_engine=embedding_engine, user_id=user_id)
    decay_engine = DecayEngine(user_config, bucket_mgr, user_id=user_id)

    # L0: DDA
    dda = DDAController(user_id=user_id, data_dir=paths["buckets_dir"])

    # L1: Graph
    graph = MemoryGraph(user_id=user_id, data_dir=paths["buckets_dir"])

    # L2: Intelligence
    retrieval = RetrievalEngine()
    importance = ImportanceFusion()
    flashbulb = FlashbulbDetector()
    script_dev = ScriptDeviation(user_id=user_id, data_dir=paths["buckets_dir"])
    vulnerability = VulnerabilityModel(user_id=user_id, data_dir=paths["buckets_dir"])
    ws = WorkingSelf(user_id=user_id, data_dir=paths["buckets_dir"])
    cold_start = ColdStartPolicy()

    # v9 Track A
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
    )


# ═══════════════════════════════════════════════════════════
# Tool 1: memory_search — DDA-adaptive retrieval
# ═══════════════════════════════════════════════════════════

@mcp.tool()
async def memory_search(
    query: str,
    user_id: str = "default",
    max_results: int = 20,
    domain: str = "",
    valence: float = -1.0,
    arousal: float = -1.0,
    importance_min: int = -1,
) -> str:
    """
    DDA-adaptive multi-path memory retrieval. Searches all user memories
    using the current DDA strategy (COLD→WARM→HOT→RICH). Combines vector
    embedding search, BM25 keyword matching, typed graph traversal, and
    emotional resonance scoring.

    Args:
        query: Search query text
        user_id: User namespace (default="default")
        max_results: Max results to return (1-50)
        domain: Comma-separated domain filter (e.g., "职业,情感")
        valence: Emotion valence filter 0-1 (-1=ignore)
        arousal: Emotion arousal filter 0-1 (-1=ignore)
        importance_min: Minimum importance threshold (1-10, -1=ignore)
    """
    if not query or not query.strip():
        return "请提供搜索关键词。"

    max_results = max(1, min(50, max_results))
    orch = _make_orchestrator(user_id)

    try:
        # Ensure session started for DDA state
        await orch.start_session()

        domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
        q_valence = valence if 0 <= valence <= 1 else None
        q_arousal = arousal if 0 <= arousal <= 1 else None

        results = await orch.retrieval.search(
            query=query,
            strategy=orch._strategy,
            ddi_level=orch._ddi_level,
            bucket_mgr=orch.bucket_mgr,
            embedding_engine=orch.embedding_engine,
            memory_graph=orch.graph,
            working_self=orch.ws,
            decay_engine=orch.decay_engine,
            user_id=user_id,
            top_k=max_results,
        )

        if not results:
            return "未找到相关记忆。"

        lines = [f"=== 记忆检索结果 (DDI: {orch._ddi_level.value}) ==="]
        for i, r in enumerate(results[:max_results], 1):
            content = r.get("content", "")[:300]
            source = r.get("source", "unknown")
            score = r.get("final_score", 0)
            mem_type = r.get("memory_type", "chat")
            imp = r.get("importance", 5)
            lines.append(
                f"\n{i}. [{mem_type}][imp:{imp}][score:{score:.3f}][{source}]\n"
                f"   {content}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"memory_search failed: {e}")
        return f"检索失败: {e}"


# ═══════════════════════════════════════════════════════════
# Tool 2: memory_store — Store with importance fusion
# ═══════════════════════════════════════════════════════════

@mcp.tool()
async def memory_store(
    content: str,
    user_id: str = "default",
    memory_type: str = "chat",
    importance: int = 5,
    tags: str = "",
    valence: float = 0.5,
    arousal: float = 0.3,
    pinned: bool = False,
) -> str:
    """
    Store a new memory with automatic importance fusion scoring.
    Uses flashbulb detection, script deviation, and cold-start gating
    to determine storage priority.

    Args:
        content: Memory content text
        user_id: User namespace
        memory_type: Type of memory (chat/decision/emotion/milestone)
        importance: User-assigned importance (1-10)
        tags: Comma-separated tags
        valence: Emotional valence (0=negative, 1=positive)
        arousal: Emotional arousal (0=calm, 1=excited)
        pinned: Whether to pin this memory permanently
    """
    if not content or not content.strip():
        return "内容为空，无法存储。"

    orch = _make_orchestrator(user_id)
    await orch.decay_engine.ensure_started()

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # Auto-tagging via dehydrator
    try:
        analysis = await dehydrator.analyze(content)
        domain = analysis.get("domain", ["未分类"])
        auto_valence = analysis.get("valence", valence)
        auto_arousal = analysis.get("arousal", arousal)
        auto_tags = analysis.get("tags", [])
        suggested_name = analysis.get("suggested_name", "")
    except Exception:
        domain = ["未分类"]
        auto_valence = valence
        auto_arousal = arousal
        auto_tags = []
        suggested_name = ""

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    try:
        bucket_id = await orch.bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=importance,
            domain=domain,
            valence=auto_valence,
            arousal=auto_arousal,
            name=suggested_name or None,
            bucket_type="permanent" if pinned else "dynamic",
            pinned=pinned,
        )

        # Generate embedding
        if orch.embedding_engine:
            await orch.embedding_engine.generate_and_store(bucket_id, content)

        # Graph node
        if orch.graph:
            orch.graph.add_node(bucket_id, {
                "valence": auto_valence,
                "arousal": auto_arousal,
                "importance": importance,
                "memory_type": memory_type,
                "pinned": pinned,
            })

        action = "📌钉选" if pinned else "新建"
        return f"{action}→{bucket_id} {','.join(domain)} V{auto_valence:.1f}/A{auto_arousal:.1f}"
    except Exception as e:
        logger.error(f"memory_store failed: {e}")
        return f"存储失败: {e}"


# ═══════════════════════════════════════════════════════════
# Tool 3: memory_status — System status
# ═══════════════════════════════════════════════════════════

@mcp.tool()
async def memory_status(user_id: str = "default") -> str:
    """
    Report Memory Palace system status: DDA level, bucket counts,
    decay engine state, storage size, and top unresolved memories.

    Args:
        user_id: User namespace
    """
    orch = _make_orchestrator(user_id)

    try:
        stats = await orch.bucket_mgr.get_stats()
        decay_running = orch.decay_engine.is_running if orch.decay_engine else False
        emb_enabled = orch.embedding_engine.enabled if orch.embedding_engine else False
        emb_count = orch.embedding_engine._backend.count() if orch.embedding_engine and orch.embedding_engine._backend else 0

        total = stats["permanent_count"] + stats["dynamic_count"] + stats["feel_count"]
        domains_str = ", ".join(
            f"{d}({c})" for d, c in sorted(
                stats.get("domains", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:8]
        ) if stats.get("domains") else "无"

        status_lines = [
            f"=== Memory Palace V2 系统状态 ===",
            f"用户: {user_id}",
            f"总记忆桶: {total} (固{stats['permanent_count']} 动{stats['dynamic_count']} 感{stats['feel_count']} 档{stats['archive_count']})",
            f"向量索引: {'启用' if emb_enabled else '禁用'} ({emb_count} embeddings)",
            f"衰减引擎: {'运行中' if decay_running else '已停止'}",
            f"存储大小: {stats['total_size_kb']:.1f} KB",
            f"主题域: {domains_str}",
        ]

        # Top unresolved by decay score
        all_buckets = await orch.bucket_mgr.list_all(include_archive=False)
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved")
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned")
        ]
        if unresolved and orch.decay_engine:
            unresolved.sort(
                key=lambda b: orch.decay_engine.calculate_score(b["metadata"]),
                reverse=True,
            )
            status_lines.append("\n=== 权重最高的未解决记忆 ===")
            for b in unresolved[:5]:
                meta = b["metadata"]
                score = orch.decay_engine.calculate_score(meta)
                status_lines.append(
                    f"  [{meta.get('name', b['id'])}] "
                    f"重要:{meta.get('importance', '?')} "
                    f"权重:{score:.2f}"
                )

        return "\n".join(status_lines)
    except Exception as e:
        logger.error(f"memory_status failed: {e}")
        return f"获取状态失败: {e}"


# ═══════════════════════════════════════════════════════════
# Tool 4: memory_graph — Query typed memory graph
# ═══════════════════════════════════════════════════════════

@mcp.tool()
async def memory_graph_query(
    memory_id: str,
    user_id: str = "default",
    depth: int = 2,
    relation_types: str = "",
) -> str:
    """
    Query the typed memory graph for neighbors of a specific memory.
    Traverses temporal, causal, thematic, and emotional edges.

    Args:
        memory_id: The bucket ID to query neighbors for
        user_id: User namespace
        depth: Traversal depth (1-3)
        relation_types: Filter by edge types (comma-separated: temporal,causal,thematic,emotional)
                        Empty = all types
    """
    if not memory_id or not memory_id.strip():
        return "请提供 memory_id。"

    depth = max(1, min(3, depth))
    orch = _make_orchestrator(user_id)

    if not orch.graph:
        return "记忆图谱未初始化。"

    try:
        rel_types = [r.strip() for r in relation_types.split(",") if r.strip()] or None
        neighbors = orch.graph.get_neighbors(
            memory_id,
            depth=depth,
            relation_types=rel_types,
            active_only=True,
        )

        if not neighbors:
            return f"未找到与 {memory_id} 相关的记忆。"

        lines = [f"=== 记忆图谱 (depth={depth}) ==="]
        for n in neighbors[:20]:
            nid = n.get("to_id", n.get("from_id", "?"))
            rel = n.get("relation_type", "unknown")
            weight = n.get("weight", 0.0)
            lines.append(f"  [{rel}](w:{weight:.2f}) → {nid}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"memory_graph_query failed: {e}")
        return f"图谱查询失败: {e}"


# ═══════════════════════════════════════════════════════════
# Tool 5: memory_dream — Post-session digestion
# ═══════════════════════════════════════════════════════════

@mcp.tool()
async def memory_dream(user_id: str = "default") -> str:
    """
    Trigger post-conversation digestion (dream/sleeptime compute).
    Runs decay cycle, DDA update, vulnerability assessment, and
    narrative consolidation if v9 Track A modules are available.

    Args:
        user_id: User namespace
    """
    orch = _make_orchestrator(user_id)

    try:
        # Ensure session started
        if not orch._session_id:
            await orch.start_session()

        result = await orch.dream()

        lines = [
            "=== Dreaming 完成 ===",
            f"会话ID: {result.get('session_id', '?')}",
            f"DDI更新: {'是' if result.get('ddi_updated') else '否'}",
            f"衰减周期: {result.get('decay_cycle', '无')}",
        ]

        if result.get("vulnerability"):
            vi = result["vulnerability"]
            lines.append(f"脆弱性指数: {vi.get('vi', '?')} ({vi.get('level', '?')})")

        if result.get("sleeptime"):
            st = result["sleeptime"]
            lines.append(f"Sleeptime: {st.get('duration_seconds', 0):.1f}s")
            for stage in ["replay", "prune", "consolidate", "precompute", "evolve"]:
                if st.get(stage):
                    lines.append(f"  {stage}: {st[stage]}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"memory_dream failed: {e}")
        return f"Dreaming 失败: {e}"


# ═══════════════════════════════════════════════════════════
# Tool 6: memory_evolve — Trigger memory evolution
# ═══════════════════════════════════════════════════════════

@mcp.tool()
async def memory_evolve(
    memory_id: str = "",
    user_id: str = "default",
) -> str:
    """
    Trigger memory evolution: create zettelkasten links between a memory
    and related memories. If memory_id is empty, evolve all memories
    that meet the evolution threshold.

    Args:
        memory_id: Specific bucket ID to evolve (empty = all eligible)
        user_id: User namespace
    """
    orch = _make_orchestrator(user_id)

    if not orch.evolution:
        return "Memory Evolution 模块未初始化。"

    try:
        if memory_id and memory_id.strip():
            result = await orch.evolution.evolve_single(memory_id)
            return f"已进化记忆: {memory_id}\n链接: {len(result.get('links', []))} 条"
        else:
            result = await orch.evolution.evolve_all()
            evolved = result.get("evolved", 0)
            links = result.get("total_links", 0)
            return f"批量进化完成: {evolved} 条记忆, {links} 条新链接"
    except Exception as e:
        logger.error(f"memory_evolve failed: {e}")
        return f"进化失败: {e}"


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    transport = os.environ.get("OMBRE_TRANSPORT", config.get("transport", "stdio"))
    logger.info(f"Memory Palace V2 MCP starting | transport: {transport}")

    if transport in ("sse", "streamable-http"):
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        port = int(os.environ.get("OMBRE_PORT", "8000"))
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS middleware enabled, listening on port {port}")
        uvicorn.run(_app, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
