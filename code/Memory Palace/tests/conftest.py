# ============================================================
# Shared test fixtures — isolated temp environment for all tests
# 共享测试 fixtures —— 为所有测试提供隔离的临时环境
#
# IMPORTANT: All tests run against a temp directory.
# Your real /data or local buckets are NEVER touched.
# 重要：所有测试在临时目录运行，绝不触碰真实记忆数据。
# ============================================================

import os
import sys
import math
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def test_config(tmp_path):
    """
    Minimal config pointing to a temp directory.
    Uses spec-correct scoring weights (after B-05, B-06, B-07 fixes).
    """
    buckets_dir = str(tmp_path / "buckets")
    os.makedirs(os.path.join(buckets_dir, "permanent"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "dynamic"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "archive"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "feel"), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "merge_threshold": 75,
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        # Spec-correct weights (post B-05/B-06/B-07 fix)
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 1.5,   # spec: 1.5 (was 2.5 in buggy code)
            "importance": 1.0,
            "content_weight": 1.0,   # spec: 1.0 (was 3.0 in buggy code)
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": os.environ.get("OMBRE_API_KEY", "test-key"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash-lite",
        },
        "embedding": {
            "api_key": os.environ.get("OMBRE_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-embedding-001",
            "enabled": False,
        },
    }


@pytest.fixture
def buggy_config(tmp_path):
    """
    Config using the PRE-FIX (buggy) scoring weights.
    Used in regression tests to document the old broken behaviour.
    """
    buckets_dir = str(tmp_path / "buckets")
    for d in ["permanent", "dynamic", "archive", "feel"]:
        os.makedirs(os.path.join(buckets_dir, d), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "merge_threshold": 75,
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        # Buggy weights (before B-05/B-06/B-07 fixes)
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 2.5,   # B-06: was too high
            "importance": 1.0,
            "content_weight": 3.0,   # B-07: was too high
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": "",
            "base_url": "https://example.com",
            "model": "test-model",
        },
        "embedding": {"enabled": False, "api_key": ""},
    }


@pytest.fixture
def bucket_mgr(test_config):
    from bucket_manager import BucketManager
    return BucketManager(test_config)


@pytest.fixture
def decay_eng(test_config, bucket_mgr):
    from decay_engine import DecayEngine
    return DecayEngine(test_config, bucket_mgr)


@pytest.fixture
def mock_dehydrator():
    """
    Mock Dehydrator that returns deterministic results without any API calls.
    Suitable for integration tests that do not test LLM behaviour.
    """
    dh = MagicMock()

    async def fake_dehydrate(content, meta=None):
        return f"[摘要] {content[:60]}"

    async def fake_analyze(content):
        return {
            "domain": ["学习"],
            "valence": 0.7,
            "arousal": 0.5,
            "tags": ["测试"],
            "suggested_name": "测试记忆",
        }

    async def fake_merge(old, new):
        return old + "\n---合并---\n" + new

    async def fake_digest(content):
        return [
            {
                "name": "条目一",
                "content": content[:100],
                "domain": ["日常"],
                "valence": 0.6,
                "arousal": 0.4,
                "tags": ["测试"],
                "importance": 5,
            }
        ]

    dh.dehydrate = AsyncMock(side_effect=fake_dehydrate)
    dh.analyze = AsyncMock(side_effect=fake_analyze)
    dh.merge = AsyncMock(side_effect=fake_merge)
    dh.digest = AsyncMock(side_effect=fake_digest)
    dh.api_available = True
    return dh


@pytest.fixture
def mock_embedding_engine():
    """Mock EmbeddingEngine that returns empty results — no network calls."""
    ee = MagicMock()
    ee.enabled = False
    ee.generate_and_store = AsyncMock(return_value=None)
    ee.search_similar = AsyncMock(return_value=[])
    ee.delete_embedding = AsyncMock(return_value=True)
    ee.get_embedding = AsyncMock(return_value=None)
    return ee


async def _write_bucket_file(bucket_mgr, content, **kwargs):
    """
    Helper: create a bucket and optionally patch its frontmatter fields.
    Accepts extra kwargs like created/last_active/resolved/digested/pinned.
    Returns bucket_id.
    """
    import frontmatter as fm

    direct_fields = {
        k: kwargs.pop(k) for k in list(kwargs.keys())
        if k in ("created", "last_active", "resolved", "digested", "activation_count")
    }

    bid = await bucket_mgr.create(content=content, **kwargs)

    if direct_fields:
        fpath = bucket_mgr._find_bucket_file(bid)
        post = fm.load(fpath)
        for k, v in direct_fields.items():
            post[k] = v
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))

    return bid


# ═══════════════════════════════════════════════════════════════
# v6 Memory Palace module fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def dda_ctrl(tmp_path):
    """DDAController with temp directory."""
    from dda_controller import DDAController
    return DDAController(stats_dir=str(tmp_path / "buckets"))


@pytest.fixture
def cold_start_policy():
    """ColdStartPolicy singleton."""
    from cold_start import ColdStartPolicy
    return ColdStartPolicy()


@pytest.fixture
def global_prior_singleton():
    """GlobalPrior singleton."""
    from global_prior import GlobalPrior
    return GlobalPrior()


@pytest.fixture
def memory_graph_fixture(tmp_path):
    """MemoryGraph with temp SQLite database."""
    from memory_graph import MemoryGraph
    return MemoryGraph(user_id="test_user", db_dir=str(tmp_path / "buckets"))


@pytest.fixture
def flashbulb_detector_fixture():
    """FlashbulbDetector instance."""
    from flashbulb_detector import FlashbulbDetector
    return FlashbulbDetector()


@pytest.fixture
def vulnerability_model_fixture(tmp_path):
    """VulnerabilityModel with temp directory."""
    from vulnerability_model import VulnerabilityModel
    return VulnerabilityModel(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def importance_fusion_fixture():
    """ImportanceFusion instance."""
    from importance_fusion import ImportanceFusion
    return ImportanceFusion()


@pytest.fixture
def retrieval_engine_fixture():
    """RetrievalEngine instance."""
    from retrieval_engine import RetrievalEngine
    return RetrievalEngine()


@pytest.fixture
def script_deviation_fixture(tmp_path):
    """ScriptDeviation with temp directory."""
    from script_deviation import ScriptDeviation
    return ScriptDeviation(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def working_self_fixture(tmp_path):
    """WorkingSelf with temp directory."""
    from working_self import WorkingSelf
    return WorkingSelf(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def mock_llm_gateway():
    """Mock LLM gateway for tests."""
    gw = MagicMock()
    gw.chat = AsyncMock(return_value="mock reply")
    gw.chat_with_json = AsyncMock(return_value='{"result": "ok"}')
    return gw


@pytest.fixture
def mock_orchestrator_deps(mock_llm_gateway, tmp_path):
    """All mocked dependencies for MemoryOrchestrator."""
    from memory_orchestrator import MemoryOrchestrator
    from memory_node import COLD_STRATEGY, DDILevel, DDAStrategy

    bm = AsyncMock()
    bm.list_all = AsyncMock(return_value=[])
    bm.create = AsyncMock(return_value="bid_001")
    bm.search = AsyncMock(return_value=[])

    de = MagicMock()
    de.calculate_score = MagicMock(return_value=5.0)
    de.apply_dda_strategy = MagicMock()
    de.set_ddi_level = MagicMock()
    de.run_decay_cycle = AsyncMock(return_value={"checked": 0, "archived": 0})

    ee = MagicMock()
    ee.generate_and_store = AsyncMock()
    ee.search_similar = AsyncMock(return_value=[])

    dh = MagicMock()
    dh.dehydrate = AsyncMock(return_value="[摘要]")
    dh.analyze = AsyncMock(return_value={"domain": [], "valence": 0.5, "arousal": 0.3})

    dda = MagicMock()
    dda.get_strategy_for_user = MagicMock(return_value=(DDILevel.COLD, 0.0, COLD_STRATEGY))
    dda.load_stats = MagicMock(return_value=MagicMock())
    dda.calculate_ddi = MagicMock(return_value=0.0)
    dda.get_level = MagicMock(return_value=DDILevel.COLD)
    dda.update_after_session = MagicMock(return_value=MagicMock())
    dda.save_stats = MagicMock()
    dda.log_session = MagicMock()

    return {
        "user_id": "test_user",
        "bucket_mgr": bm,
        "decay_engine": de,
        "embedding_engine": ee,
        "dehydrator": dh,
        "llm_gateway": mock_llm_gateway,
        "dda_controller": dda,
    }


# ═══════════════════════════════════════════════════════════════
# v9 Track A module fixtures (Narrative + Evolution + Sleeptime)
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def narrative_engine_fixture(tmp_path):
    """NarrativeEngine with temp directory."""
    from narrative_engine import NarrativeEngine
    return NarrativeEngine(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def memory_evolution_fixture(tmp_path):
    """MemoryEvolution with temp directory."""
    from memory_evolution import MemoryEvolution
    return MemoryEvolution(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def sleeptime_computer_fixture(tmp_path):
    """SleeptimeComputer with mocked dependencies."""
    from sleeptime_compute import SleeptimeComputer

    bm = AsyncMock()
    bm.list_all = AsyncMock(return_value=[])

    de = MagicMock()
    de.calculate_score = MagicMock(return_value=5.0)
    de.run_decay_cycle = AsyncMock(return_value={"checked": 0, "archived": 0, "auto_resolved": 0})

    g = MagicMock()
    g.get_neighbors = MagicMock(return_value=[])
    g.add_edge = MagicMock()
    g.expire_edge = MagicMock()
    g.get_graph_stats = MagicMock(return_value={"node_count": 0, "edge_count": 0, "active_edge_count": 0})

    ne = MagicMock()
    ne.run_narrative_merge = AsyncMock(return_value={
        "communities_detected": 0, "threads_merged": 0,
        "summaries_updated": 0, "threads_resolved": 0,
        "life_periods_updated": 0,
    })

    ev = MagicMock()
    ev.run_evolution_cycle = AsyncMock(return_value={
        "memories_scanned": 0, "re_evaluated": 0,
        "ws_re_ranked": 0, "emergences_detected": 0,
    })

    return SleeptimeComputer(
        user_id="test_user",
        bucket_mgr=bm,
        decay_engine=de,
        memory_graph=g,
        narrative_engine=ne,
        memory_evolution=ev,
    )


# ═══════════════════════════════════════════════════════════════
# v9 Track C module fixtures (GraphRAG + HippoRAG + Procedural + Learnable)
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def leiden_detector_fixture():
    """LeidenDetector instance."""
    from graph_rag import LeidenDetector
    return LeidenDetector(resolution=1.0, max_iterations=10)


@pytest.fixture
def graph_rag_engine_fixture():
    """GraphRAGEngine instance."""
    from graph_rag import GraphRAGEngine
    return GraphRAGEngine(resolution=1.0)


@pytest.fixture
def sample_graph_edges():
    """Sample graph edges for community detection testing."""
    return [
        {"from_id": "n1", "to_id": "n2", "weight": 1.0},
        {"from_id": "n1", "to_id": "n3", "weight": 0.8},
        {"from_id": "n2", "to_id": "n3", "weight": 0.9},
        {"from_id": "n4", "to_id": "n5", "weight": 1.0},
        {"from_id": "n4", "to_id": "n6", "weight": 0.7},
        {"from_id": "n5", "to_id": "n6", "weight": 0.8},
        {"from_id": "n3", "to_id": "n7", "weight": 0.5},
        {"from_id": "n7", "to_id": "n8", "weight": 0.6},
    ]


@pytest.fixture
def sample_graph_nodes():
    """Sample graph nodes for community detection testing."""
    return {
        f"n{i}": {"type": "memory"} for i in range(1, 9)
    }


@pytest.fixture
def ppr_engine_fixture():
    """PersonalizedPageRank instance."""
    from hippo_rag import PersonalizedPageRank
    return PersonalizedPageRank(alpha=0.85, max_iterations=50)


@pytest.fixture
def hippo_rag_retriever_fixture():
    """HippoRAGRetriever instance."""
    from hippo_rag import HippoRAGRetriever
    return HippoRAGRetriever(alpha=0.85)


@pytest.fixture
def procedural_memory_fixture(tmp_path):
    """ProceduralMemory instance with temp directory."""
    from procedural_memory import ProceduralMemory
    return ProceduralMemory(user_id="test_user", data_dir=str(tmp_path / "buckets"))


@pytest.fixture
def learnable_weights_fixture():
    """LearnablePathWeights instance with sample base weights."""
    from learnable_weights import LearnablePathWeights
    base_weights = {
        "vector": 0.22,
        "bm25": 0.10,
        "graph": 0.18,
        "emotion": 0.10,
        "temporal": 0.12,
        "cross_ref": 0.08,
        "narrative": 0.08,
        "ppr": 0.08,
        "ws_rerank": 0.04,
    }
    return LearnablePathWeights(base_weights=base_weights)
