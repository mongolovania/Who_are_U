#!/usr/bin/env python3
# ============================================================
# Real Community Algorithm Implementations — 真实社区算法适配器
# v9: Wraps pip-installable memory/retrieval libraries behind
#     the unified answer() interface used by all benchmarks.
#
# Each adapter accepts list[BenchmarkMemory] at construction,
# builds its internal index, and exposes:
#     answer(query, top_k) -> (text, list[indices], score)
#
# 适配器将 pip 可安装的真实记忆/检索库包装在统一的
# answer() 接口后面，与11个模拟器保持合同一致。
# ============================================================

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.benchmarks.benchmark_dataset import BenchmarkMemory

# ── Library availability probes ──────────────────────────────

_SENTENCE_TRANSFORMERS_AVAILABLE = False
_FAISS_AVAILABLE = False
_BM25S_AVAILABLE = False
_MEM0_AVAILABLE = False
_GRAPHRAG_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    pass

try:
    import bm25s  # noqa: F401
    _BM25S_AVAILABLE = True
except ImportError:
    pass

try:
    import mem0  # noqa: F401
    _MEM0_AVAILABLE = True
except ImportError:
    pass

try:
    import graphrag  # noqa: F401
    _GRAPHRAG_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# Adapter 1: Sentence-Transformers + FAISS — Dense Semantic Search
# ═══════════════════════════════════════════════════════════════

class SentenceTransformersFAISSAdapter:
    """
    Real dense embedding retrieval using sentence-transformers + FAISS.

    Uses ``paraphrase-multilingual-MiniLM-L12-v2`` (384 dims) which handles
    Chinese text well. Embeddings are L2-normalized and searched via FAISS
    IndexFlatIP (inner product = cosine similarity on normalized vectors).

    This is a genuine community implementation — not a simulator.
    No API key required. Runs entirely locally.

    References:
        Reimers & Gurevych, "Sentence-BERT", EMNLP 2019
        Johnson et al., "Billion-scale similarity search with GPUs",
            IEEE Trans. Big Data 2019
    """

    # Default model: multilingual, 384-dim, ~120MB, good CJK support
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        memories: list[BenchmarkMemory],
        model_name: str | None = None,
    ):
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu not installed. "
                "Run: pip install faiss-cpu"
            )

        self.memories = memories
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: "SentenceTransformer"
        self._index: "faiss.Index"
        self._embeddings: np.ndarray
        self._build_time_ms: float = 0.0

        self._build()

    def _build(self) -> None:
        """Embed all memories and build FAISS index."""
        from sentence_transformers import SentenceTransformer
        import faiss

        t0 = time.perf_counter()

        self._model = SentenceTransformer(self._model_name)
        contents = [m.content for m in self.memories]

        # Normalize embeddings for cosine similarity via inner product
        self._embeddings = self._model.encode(
            contents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product = cosine
        self._index.add(self._embeddings.astype(np.float32))

        self._build_time_ms = (time.perf_counter() - t0) * 1000

    def answer(
        self, query: str, top_k: int = 10,
    ) -> tuple[str, list[int], float]:
        """Semantic search via dense embeddings + FAISS."""
        import faiss

        q_emb = self._model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self._index.search(q_emb, min(top_k, len(self.memories)))

        top_indices = [int(i) for i in indices[0] if i >= 0 and i < len(self.memories)]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        # Cosine similarity is in [-1, 1]; normalize to [0, 1]
        raw_score = float(scores[0][0])
        normalized_score = (raw_score + 1.0) / 2.0
        return " | ".join(contexts), top_indices, max(0.0, min(1.0, normalized_score))


# ═══════════════════════════════════════════════════════════════
# Adapter 2: BM25S — Faster BM25 (Rust backend)
# ═══════════════════════════════════════════════════════════════

class BM25sAdapter:
    """
    BM25S — faster BM25 implementation with a Rust backend.

    Serves as an independent verification of the rank-bm25 BM25Baseline.
    If both produce near-identical rankings, confidence in BM25 scoring
    is high. If they diverge, one implementation has a bug.

    Reference:
        Xing Han Lu, "bm25s: Fast BM25 in Python with scoring enhancements",
        GitHub 2024
    """

    def __init__(self, memories: list[BenchmarkMemory]):
        if not _BM25S_AVAILABLE:
            raise ImportError(
                "bm25s not installed. Run: pip install bm25s"
            )

        self.memories = memories
        self._retriever: "bm25s.BM25"
        self._build_time_ms: float = 0.0

        self._build()

    def _build(self) -> None:
        """Tokenize corpus and build BM25S index."""
        import bm25s

        t0 = time.perf_counter()

        corpus = [m.content for m in self.memories]
        # bm25s has built-in Chinese-friendly tokenization
        corpus_tokenized = bm25s.tokenize(corpus, stopwords=[])
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokenized)

        self._build_time_ms = (time.perf_counter() - t0) * 1000

    def answer(
        self, query: str, top_k: int = 10,
    ) -> tuple[str, list[int], float]:
        """BM25 keyword search via bm25s."""
        import bm25s

        q_tok = bm25s.tokenize([query], stopwords=[])
        results, scores = self._retriever.retrieve(
            q_tok, k=min(top_k, len(self.memories)),
        )

        indices = [int(i) for i in results[0]]
        if not indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in indices]

        # Normalize score
        raw_scores = scores[0]
        max_s = float(max(raw_scores)) if len(raw_scores) > 0 else 1.0
        norm_score = float(raw_scores[0]) / max(max_s, 1e-8)
        return " | ".join(contexts), indices, norm_score


# ═══════════════════════════════════════════════════════════════
# Adapter 3: Real Mem0 — Cloud-Hosted Memory System
# ═══════════════════════════════════════════════════════════════

class RealMem0Adapter:
    """
    Real Mem0 (mem0ai) — the most popular open-source AI memory layer.

    Uses MemoryClient.add() for ingestion and MemoryClient.search()
    for retrieval. Requires MEM0_API_KEY environment variable.

    Maintains a bidirectional mapping between Mem0 memory UIDs and
    corpus indices for benchmark scoring.

    Reference:
        Taranjeet Singh et al., "Mem0: A Memory Layer for AI Agents",
        GitHub 2024 (25k+ stars)
    """

    def __init__(
        self,
        memories: list[BenchmarkMemory],
        api_key: str | None = None,
        user_id: str = "benchmark_user",
    ):
        self.memories = memories
        self._client = None
        self._uid_to_idx: dict[str, int] = {}
        self._idx_to_uid: dict[int, str] = {}
        self._configured = False
        self._build_time_ms: float = 0.0

        if not _MEM0_AVAILABLE:
            return  # Will answer with "not configured" message

        try:
            from mem0 import MemoryClient

            key = api_key or os.environ.get("MEM0_API_KEY", "")
            if not key:
                return

            t0 = time.perf_counter()
            self._client = MemoryClient(api_key=key, org_id=None)
            self._configured = True
            self._ingest()
            self._build_time_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            self._configured = False

    def _ingest(self) -> None:
        """Add all benchmark memories to Mem0 cloud."""
        if not self._configured or not self._client:
            return

        for i, mem in enumerate(self.memories):
            try:
                # Mem0 API: add messages, get back memory objects
                result = self._client.add(
                    [
                        {"role": "user", "content": mem.content},
                    ],
                    user_id="benchmark_user",
                )
                # Mem0 returns list of dicts with "id" field
                items = result if isinstance(result, list) else [result]
                for r in items:
                    if isinstance(r, dict):
                        uid = r.get("id", f"mem_{i}")
                        self._uid_to_idx[uid] = i
                        self._idx_to_uid[i] = uid
            except Exception:
                continue

    def answer(
        self, query: str, top_k: int = 10,
    ) -> tuple[str, list[int], float]:
        """Search Mem0 cloud for relevant memories."""
        if not self._configured:
            return "Mem0 未配置（需要 MEM0_API_KEY 环境变量）。", [], 0.0

        try:
            results = self._client.search(
                query,
                user_id="benchmark_user",
                limit=min(top_k, len(self.memories)),
            )

            indices: list[int] = []
            # Mem0 search returns dict with "results" key
            items = (
                results.get("results", [])
                if isinstance(results, dict)
                else results
            )
            for r in items:
                if isinstance(r, dict):
                    uid = r.get("id", "")
                    if uid in self._uid_to_idx:
                        indices.append(self._uid_to_idx[uid])

            if not indices:
                return "未找到相关信息。", [], 0.0

            contexts = [
                self.memories[i].content
                for i in indices[:top_k]
            ]

            top_score = 0.5
            if items and isinstance(items[0], dict):
                top_score = float(items[0].get("score", 0.5))

            return " | ".join(contexts), indices[:top_k], min(1.0, top_score)
        except Exception as e:
            return f"Mem0 错误: {e}", [], 0.0


# ═══════════════════════════════════════════════════════════════
# Adapter 4: Real GraphRAG — Microsoft GraphRAG (Stretch)
# ═══════════════════════════════════════════════════════════════

class RealGraphRAGAdapter:
    """
    Microsoft GraphRAG — community detection + hierarchical summarization.

    Uses the graphrag Python SDK: builds an entity knowledge graph from
    memory texts, detects communities via Leiden algorithm, generates
    community summaries, and retrieves via local/global search.

    **Heavyweight**: requires an LLM API key (OpenAI-compatible) for
    entity extraction + summarization. Mark as stretch goal.

    Reference:
        Edge et al., "From Local to Global: A Graph RAG Approach to
        Query-Focused Summarization", arXiv:2404.16130, 2024
    """

    def __init__(
        self,
        memories: list[BenchmarkMemory],
        api_key: str | None = None,
        llm_model: str = "gpt-4o-mini",
    ):
        self.memories = memories
        self._configured = False
        self._build_time_ms: float = 0.0

        if not _GRAPHRAG_AVAILABLE:
            return

        # GraphRAG requires significant setup: write input files,
        # run indexing pipeline, configure LLM access.
        # For now, this adapter is a placeholder.
        # The existing GraphRAGSimulator (Louvain on Jaccard graph)
        # already captures the core community-detection idea.
        pass

    def answer(
        self, query: str, top_k: int = 10,
    ) -> tuple[str, list[int], float]:
        """Placeholder — GraphRAG not configured."""
        return (
            "GraphRAG 未配置（需要 graphrag pip 包 + LLM API key + 索引构建）。",
            [], 0.0,
        )


# ═══════════════════════════════════════════════════════════════
# Registry — all real implementations
# ═══════════════════════════════════════════════════════════════

REAL_IMPLEMENTATIONS: dict[str, type] = {
    "sentence-transformers+FAISS": SentenceTransformersFAISSAdapter,
    "Real BM25S": BM25sAdapter,
    "Real Mem0 (mem0ai)": RealMem0Adapter,
    "Real GraphRAG": RealGraphRAGAdapter,
}

# Availability flags for test skipping
REAL_AVAILABILITY: dict[str, bool] = {
    "sentence-transformers+FAISS": (
        _SENTENCE_TRANSFORMERS_AVAILABLE and _FAISS_AVAILABLE
    ),
    "Real BM25S": _BM25S_AVAILABLE,
    "Real Mem0 (mem0ai)": _MEM0_AVAILABLE,
    "Real GraphRAG": _GRAPHRAG_AVAILABLE,
}
