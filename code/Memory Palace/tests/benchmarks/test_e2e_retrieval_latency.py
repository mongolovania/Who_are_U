# ============================================================
# Track B: End-to-End Retrieval Latency Benchmarks
# Track B：全链路检索延迟基准测试
#
# Measures the full zero-LLM retrieval pipeline:
#   query → tokenize → BM25 → vector search → graph → emotion → fusion
# 测量全链路零LLM检索管道延迟。
#
# Target: P95 < 300ms (Track B requirement)
# 目标：P95 < 300ms
# ============================================================

from __future__ import annotations

import os
import sys
import time
import math
import json
import random
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from memory_node import DDILevel, DDAStrategy, HOT_STRATEGY, WARM_STRATEGY, RICH_STRATEGY, COLD_STRATEGY
from retrieval_engine import RetrievalEngine, BM25Retriever, _tokenize


# ── Synthetic test data ──────────────────────────────────

def _generate_synthetic_memories(count: int = 100, seed: int = 42) -> list[dict]:
    """
    Generate realistic synthetic memories for benchmarking.
    生成逼真的合成记忆数据用于基准测试。
    """
    rng = random.Random(seed)

    topics = [
        "工作", "情感", "家庭", "健康", "学习", "旅行", "财务", "社交",
        "职业规划", "生活感悟", "技术学习", "人际关系",
    ]
    templates = [
        "今天在{topic}方面有了一些新的想法，感觉{mood}。具体来说，我{action}。",
        "关于{topic}的事情，我一直在思考。最近{context}，所以心情{mood}。",
        "和{person}聊了聊{topic}，TA觉得我应该{todo}。我觉得{reaction}。",
        "{topic}方面遇到了一个问题：{problem}。不知道怎么处理，有些{mood}。",
        "今天在{topic}上取得了一些进展！{achievement}。这让我感到{mood}。",
    ]
    moods = ["很开心", "有点焦虑", "挺平静的", "特别兴奋", "有些迷茫",
             "非常满足", "有点失落", "充满希望", "有点不安", "很放松"]
    persons = ["朋友", "同事", "家人", "领导", "伴侣", "导师"]
    actions = ["制定了一个新计划", "重新思考了之前的决定", "做了一个重要的选择",
               "学到了新的东西", "完成了一个困扰已久的任务", "发现了一个新的方向"]
    problems = ["时间不够用", "和别人的意见不一致", "不知道该怎么选",
                "感觉有点力不从心", "资源有限需要权衡"]
    achievements = ["完成了阶段性的目标", "得到了别人的认可", "突破了之前的瓶颈",
                    "想到了一个好的解决方案", "收到了一个好消息"]
    reactions = ["TA说得有道理", "还是要按自己的想法来", "可以试试看",
                 "不完全同意", "值得考虑一下"]
    todos = ["再想想", "坚持自己的选择", "放松一下", "行动起来", "多沟通"]

    memories = []
    for i in range(count):
        template = rng.choice(templates)
        topic = rng.choice(topics)
        mood = rng.choice(moods)

        content = template.format(
            topic=topic,
            mood=mood,
            action=rng.choice(actions),
            context=f"最近在{topic}方面投入了很多时间",
            person=rng.choice(persons),
            todo=rng.choice(todos),
            reaction=rng.choice(reactions),
            problem=rng.choice(problems),
            achievement=rng.choice(achievements),
        )

        memories.append({
            "id": f"mem_{i:04d}",
            "content": content,
            "metadata": {
                "name": f"记忆_{i:04d}",
                "type": "dynamic",
                "memory_type": rng.choice(["chat", "emotion", "decision", "milestone"]),
                "domain": [topic],
                "tags": [topic, mood],
                "valence": round(rng.uniform(0.1, 0.9), 2),
                "arousal": round(rng.uniform(0.1, 0.9), 2),
                "importance": rng.randint(1, 10),
                "created": f"2026-0{rng.randint(1,6)}-{rng.randint(1,28):02d}T{rng.randint(8,23):02d}:00:00",
                "last_active": f"2026-06-{rng.randint(1,10):02d}T{rng.randint(8,23):02d}:00:00",
                "activation_count": rng.randint(0, 20),
                "resolved": rng.random() < 0.2,
                "pinned": rng.random() < 0.05,
            },
        })

    return memories


# ── Benchmark queries ────────────────────────────────────

BENCHMARK_QUERIES = [
    "最近工作方面有什么进展吗",
    "我最近感觉有些焦虑",
    "之前关于职业规划的想法",
    "和家人聊了什么",
    "哪些决定让我感到满意",
    "最近的社交活动",
    "技术学习方面遇到了什么问题",
    "旅行计划有变化吗",
    "财务方面需要考虑的事情",
    "人际关系最近怎么样",
    "我之前最开心的是什么时候",
    "有什么需要解决的工作问题",
    "最近的学习进度如何",
    "健康方面需要注意什么",
    "关于未来的规划",
]


# ═══════════════════════════════════════════════════════════
# BM25Retriever Latency
# ═══════════════════════════════════════════════════════════

class TestBM25Latency:
    """BM25 indexing and search performance."""

    def test_bm25_index_build_under_50ms(self):
        """Building BM25 index for 200 documents should be fast."""
        memories = _generate_synthetic_memories(200)
        documents = [(m["id"], m["content"]) for m in memories]

        retriever = BM25Retriever()
        start = time.perf_counter()
        retriever.build_index(documents)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert retriever.corpus_size == 200
        assert elapsed_ms < 50, f"BM25 index build too slow: {elapsed_ms:.2f}ms (target <50ms)"

    def test_bm25_search_under_10ms(self):
        """Single BM25 query should complete quickly."""
        memories = _generate_synthetic_memories(200)
        documents = [(m["id"], m["content"]) for m in memories]
        retriever = BM25Retriever()
        retriever.build_index(documents)

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            results = retriever.search(query, top_k=20)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(results, list)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        assert p50 < 5, f"BM25 P50 too slow: {p50:.2f}ms (target <5ms)"
        assert p95 < 10, f"BM25 P95 too slow: {p95:.2f}ms (target <10ms)"
        print(f"\nBM25 search: P50={p50:.2f}ms P95={p95:.2f}ms P99={p99:.2f}ms max={max(latencies):.2f}ms")


# ═══════════════════════════════════════════════════════════
# Tokenizer Latency
# ═══════════════════════════════════════════════════════════

class TestTokenizerLatency:
    """Tokenizer performance."""

    def test_tokenize_under_1ms(self):
        """Tokenization should be sub-millisecond."""
        queries = BENCHMARK_QUERIES + [
            "我" * 100,  # long Chinese text
            "hello world " * 20,  # long English text
            "今天写了个 function 来处理 async/await 的 error handling",  # mixed
        ]

        latencies = []
        for q in queries:
            start = time.perf_counter()
            tokens = _tokenize(q)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(tokens, list)
            assert len(tokens) > 0

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 1.0, f"Tokenizer P95 too slow: {p95:.3f}ms (target <1ms)"


# ═══════════════════════════════════════════════════════════
# RetrievalEngine Path Latency
# ═══════════════════════════════════════════════════════════

class TestRetrievalPathLatency:
    """Individual retrieval path performance."""

    @pytest.mark.asyncio
    async def test_emotion_resonance_batch_under_5ms(self):
        """Computing 100 emotion resonance scores should be fast."""
        engine = RetrievalEngine()

        start = time.perf_counter()
        for _ in range(100):
            engine.emotion_resonance(
                query_valence=random.uniform(0.0, 1.0),
                query_arousal=random.uniform(0.0, 1.0),
                memory_valence=random.uniform(0.0, 1.0),
                memory_arousal=random.uniform(0.0, 1.0),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100
        assert elapsed_ms < 0.05, f"Emotion resonance too slow: {elapsed_ms:.4f}ms per call"

    def test_query_emotion_extraction_under_1ms(self):
        """Query emotion extraction should be near-instant."""
        engine = RetrievalEngine()

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            v, a = engine._extract_query_emotion(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert 0.0 <= v <= 1.0
            assert 0.0 <= a <= 1.0

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 1.0, f"Emotion extraction P95 too slow: {p95:.3f}ms"

    def test_query_category_inference_under_1ms(self):
        """Query category inference should be near-instant."""
        engine = RetrievalEngine()

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            cat = engine._infer_query_category(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert cat in ("emotional", "causal", "temporal", "cross_reference", "factual")

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 1.0, f"Category inference P95 too slow: {p95:.3f}ms"


# ═══════════════════════════════════════════════════════════
# Full Pipeline Latency (COLD/WARM/HOT/RICH)
# ═══════════════════════════════════════════════════════════

class TestFullPipelineLatency:
    """End-to-end retrieval pipeline latency benchmarks."""

    @pytest.fixture
    def mock_bucket_mgr(self):
        """Create a mock bucket manager with synthetic memories."""
        memories = _generate_synthetic_memories(200)

        mgr = MagicMock()
        mgr.list_all = AsyncMock(return_value=memories)
        mgr.search = AsyncMock(return_value=[
            {**m, "score": random.uniform(50, 95)}
            for m in random.sample(memories, min(10, len(memories)))
        ])

        return mgr

    @pytest.fixture
    def mock_decay_engine(self):
        """Create a mock decay engine."""
        eng = MagicMock()
        eng.calculate_score = MagicMock(return_value=0.75)
        return eng

    @pytest.mark.asyncio
    async def test_cold_retrieval_p95_under_50ms(self, mock_bucket_mgr, mock_decay_engine):
        """
        COLD mode: return ALL — should be very fast (just list + sort).
        Target: P95 < 50ms
        """
        engine = RetrievalEngine()
        strategy = COLD_STRATEGY

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            results = await engine.search(
                query=query,
                strategy=strategy,
                ddi_level=DDILevel.COLD,
                bucket_mgr=mock_bucket_mgr,
                decay_engine=mock_decay_engine,
                top_k=20,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(results, list)

        _report_latencies("COLD (all)", latencies)
        p95 = _p95(latencies)
        assert p95 < 50, f"COLD P95 too slow: {p95:.2f}ms (target <50ms)"

    @pytest.mark.asyncio
    async def test_warm_retrieval_p95_under_100ms(self, mock_bucket_mgr, mock_decay_engine):
        """
        WARM mode: semantic + time — 2-path ranking.
        Target: P95 < 100ms
        """
        engine = RetrievalEngine()
        strategy = WARM_STRATEGY

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            results = await engine.search(
                query=query,
                strategy=strategy,
                ddi_level=DDILevel.WARM,
                bucket_mgr=mock_bucket_mgr,
                decay_engine=mock_decay_engine,
                top_k=20,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(results, list)

        _report_latencies("WARM (semantic+time)", latencies)
        p95 = _p95(latencies)
        assert p95 < 100, f"WARM P95 too slow: {p95:.2f}ms (target <100ms)"

    @pytest.mark.asyncio
    async def test_hot_retrieval_p95_under_300ms(self, mock_bucket_mgr, mock_decay_engine):
        """
        HOT mode: vector + BM25 + graph + emotion — 4-path fusion.
        This is the Track B target: P95 < 300ms.

        v8: Now includes temporal + cross_ref paths (6-path total).
        """
        engine = RetrievalEngine()
        strategy = HOT_STRATEGY

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            results = await engine.search(
                query=query,
                strategy=strategy,
                ddi_level=DDILevel.HOT,
                bucket_mgr=mock_bucket_mgr,
                decay_engine=mock_decay_engine,
                top_k=20,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(results, list)

        _report_latencies("HOT (4-path fusion)", latencies)
        p50 = _p50(latencies)
        p95 = _p95(latencies)
        p99 = _p99(latencies)
        max_lat = max(latencies)

        # Track B requirement
        assert p95 < 300, (
            f"HOT P95 exceeds Track B target: {p95:.2f}ms > 300ms\n"
            f"P50={p50:.2f}ms P95={p95:.2f}ms P99={p99:.2f}ms max={max_lat:.2f}ms"
        )
        print(f"\n✅ Track B P95 target met: {p95:.2f}ms < 300ms")

    @pytest.mark.asyncio
    async def test_rich_retrieval_p95_under_500ms(self, mock_bucket_mgr, mock_decay_engine):
        """
        RICH mode: HOT + Working Self re-rank — 5-path fusion.
        Target: P95 < 500ms
        """
        engine = RetrievalEngine()
        strategy = RICH_STRATEGY

        latencies = []
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            results = await engine.search(
                query=query,
                strategy=strategy,
                ddi_level=DDILevel.RICH,
                bucket_mgr=mock_bucket_mgr,
                decay_engine=mock_decay_engine,
                top_k=20,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(results, list)

        _report_latencies("RICH (5-path+WS)", latencies)
        p95 = _p95(latencies)
        assert p95 < 500, f"RICH P95 too slow: {p95:.2f}ms (target <500ms)"


# ═══════════════════════════════════════════════════════════
# Full zero-LLM pipeline (with BM25 indexing)
# ═══════════════════════════════════════════════════════════

class TestZeroLLMPipelineLatency:
    """
    Full zero-LLM retrieval pipeline including BM25 index building.

    This is the most realistic benchmark — it includes the BM25 corpus
    build step that happens on first retrieval.
    """

    @pytest.mark.asyncio
    async def test_full_hot_pipeline_with_bm25_p95_under_300ms(self):
        """
        Full HOT pipeline including BM25 index building from 200 memories.
        This is the Track B primary benchmark.
        """
        memories = _generate_synthetic_memories(200)

        mock_bm = MagicMock()
        mock_bm.list_all = AsyncMock(return_value=memories)

        # Fuzzy search fallback for when BM25 corpus < 10
        mock_bm.search = AsyncMock(return_value=[
            {**m, "score": random.uniform(50, 95)}
            for m in random.sample(memories, min(5, len(memories)))
        ])

        mock_decay = MagicMock()
        mock_decay.calculate_score = MagicMock(return_value=0.75)

        engine = RetrievalEngine()
        strategy = HOT_STRATEGY

        latencies = []
        # First query builds BM25 index (corpus=200)
        for i, query in enumerate(BENCHMARK_QUERIES):
            start = time.perf_counter()
            results = await engine.search(
                query=query,
                strategy=strategy,
                ddi_level=DDILevel.HOT,
                bucket_mgr=mock_bm,
                decay_engine=mock_decay,
                top_k=20,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert isinstance(results, list)

        _report_latencies("HOT full (BM25 index + 4-path)", latencies)
        p50 = _p50(latencies)
        p95 = _p95(latencies)
        p99 = _p99(latencies)
        max_lat = max(latencies)

        print(f"\n📊 Full Pipeline Breakdown:")
        print(f"   P50 = {p50:.2f}ms")
        print(f"   P95 = {p95:.2f}ms")
        print(f"   P99 = {p99:.2f}ms")
        print(f"   Max = {max_lat:.2f}ms")

        assert p95 < 300, (
            f"Full pipeline P95 exceeds Track B target: {p95:.2f}ms > 300ms"
        )


# ═══════════════════════════════════════════════════════════
# Scale tests
# ═══════════════════════════════════════════════════════════

class TestScaleLatency:
    """Latency scaling with corpus size."""

    @pytest.mark.parametrize("corpus_size", [50, 200, 500])
    def test_bm25_scales_reasonably(self, corpus_size):
        """BM25 should scale sub-linearly with corpus size."""
        memories = _generate_synthetic_memories(corpus_size)
        documents = [(m["id"], m["content"]) for m in memories]
        retriever = BM25Retriever()

        # Measure index build
        start = time.perf_counter()
        retriever.build_index(documents)
        build_ms = (time.perf_counter() - start) * 1000

        # Measure search
        search_latencies = []
        for query in BENCHMARK_QUERIES[:5]:
            start = time.perf_counter()
            retriever.search(query, top_k=20)
            search_latencies.append((time.perf_counter() - start) * 1000)

        avg_search = sum(search_latencies) / len(search_latencies)

        print(f"\nBM25@{corpus_size} docs: build={build_ms:.2f}ms avg_search={avg_search:.2f}ms")

        # Allow reasonable scaling: 500 docs should still build in < 200ms
        assert build_ms < 200, f"BM25 build @{corpus_size} too slow: {build_ms:.2f}ms"


# ── Helpers ──────────────────────────────────────────────

def _p50(latencies: list[float]) -> float:
    return sorted(latencies)[len(latencies) // 2]

def _p95(latencies: list[float]) -> float:
    return sorted(latencies)[int(len(latencies) * 0.95)]

def _p99(latencies: list[float]) -> float:
    return sorted(latencies)[int(len(latencies) * 0.99)]

def _report_latencies(label: str, latencies: list[float]):
    """Print latency distribution for debugging."""
    p50 = _p50(latencies)
    p95 = _p95(latencies)
    p99 = _p99(latencies)
    print(f"\n{label}: P50={p50:.2f}ms P95={p95:.2f}ms P99={p99:.2f}ms max={max(latencies):.2f}ms")
    return p50, p95, p99
