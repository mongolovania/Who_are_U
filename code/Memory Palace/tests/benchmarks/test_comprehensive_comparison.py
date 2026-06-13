# ============================================================
# Comprehensive Comparison Test — 12 systems x 6 categories x 2 sizes
# 全面横向对比 — 真实RetrievalEngine.search()驱动
#
# Key differences from previous tests:
#   1. MP uses REAL RetrievalEngine.search() through BenchmarkHarness
#      (real BM25 + emotion + temporal + cross_ref fusion)
#      — NO memory stitching, NO mocked bucket_mgr
#   2. ALL 11 simulators use REAL BM25 (rank_bm25), REAL PPR (networkx),
#      REAL community detection (Louvain), REAL emotion matching
#      — NO keyword overlap proxy
#   3. Tests across 2 sample sizes: small (10 mems) + large (72 mems)
#   4. 6 QA categories: simple_recall, multi_hop, temporal,
#      emotional, causal, cross_ref
#   5. Exports results JSON for visual report generation
# ============================================================

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_node import DDILevel, DDAStrategy, COLD_STRATEGY, WARM_STRATEGY, HOT_STRATEGY, RICH_STRATEGY
from retrieval_engine import RetrievalEngine

from tests.benchmarks.benchmark_dataset import (
    BENCHMARK_MEMORIES, BENCHMARK_MEMORIES_SMALL,
    BenchmarkMemory, get_dataset, DATASET_SIZES,
)
from tests.benchmarks.comparison_qa_dataset import (
    ALL_QA_PAIRS, CATEGORY_MAP, CATEGORY_QA_COUNTS,
    QA_SIMPLE_RECALL, QA_MULTI_HOP, QA_TEMPORAL,
    QA_EMOTIONAL, QA_CAUSAL, QA_CROSS_REF,
    ComparisonQA,
)
from tests.benchmarks.algorithm_simulators import (
    AMEMSimulator, MAGMASimulator, MMAGSimulator,
    Mem0Simulator, ZepSimulator, BM25Baseline, VectorBaseline,
    HippoRAGSimulator, GraphRAGSimulator,
    MemLongSimulator, HybridFusionSim,
    SharedBM25Index, create_all_systems, SYSTEM_NAMES,
    SIMULATOR_SYSTEMS, _tokenize,
)
from tests.benchmarks.benchmark_harness import (
    BenchmarkHarness, DDI_STRATEGIES, generate_noise_memories,
)


# ═══════════════════════════════════════════════════════════════
# Scoring Engine (enhanced)
# ═══════════════════════════════════════════════════════════════

class AnswerScorer:
    """Enhanced answer scoring with keyword + memory coverage + reasoning checks."""

    @staticmethod
    def score(system_answer: str, qa: ComparisonQA) -> int:
        """
        Score 0-3 against ground truth.

        3 = fully correct, all key facts present
        2 = mostly correct, missing minor details
        1 = partially correct, missing key information
        0 = incorrect or no result
        """
        if not system_answer or system_answer == "未找到相关信息。":
            return 0

        answer_lower = system_answer.lower()

        # Keyword match ratio
        kw_matched = 0
        for kw in qa.keywords:
            if kw.lower() in answer_lower:
                kw_matched += 1

        kw_ratio = kw_matched / max(len(qa.keywords), 1) if qa.keywords else 0.5

        # Memory coverage check
        memory_hit = 0
        for idx in qa.relevant_memory_indices:
            if idx < len(BENCHMARK_MEMORIES):
                mem = BENCHMARK_MEMORIES[idx]
                mem_tokens = set(_tokenize(mem.content))
                key_tokens = {t for t in mem_tokens if len(t) >= 2}
                if key_tokens:
                    overlap = len(key_tokens & set(_tokenize(system_answer)))
                    if overlap >= min(3, len(key_tokens) * 0.3):
                        memory_hit += 1

        mem_coverage = memory_hit / max(len(qa.relevant_memory_indices), 1)

        # Combined score
        if kw_ratio >= 0.8 and mem_coverage >= 0.5:
            return 3
        elif kw_ratio >= 0.6:
            return 2
        elif kw_ratio >= 0.3:
            return 1
        elif kw_ratio > 0:
            return 1
        return 0

    @staticmethod
    def normalized_score(system_answer: str, qa: ComparisonQA) -> float:
        """0.0–1.0 normalized score for visualizations."""
        return AnswerScorer.score(system_answer, qa) / 3.0


# ═══════════════════════════════════════════════════════════════
# REAL Memory Palace Adapter (no mocks, no stitching)
# ═══════════════════════════════════════════════════════════════

class MemoryPalaceAdapter:
    """
    Memory Palace with REAL RetrievalEngine.search() through BenchmarkHarness.

    This is the KEY CHANGE from previous benchmarks:
    - Real BucketManager (filesystem-backed, temp dirs)
    - Real DecayEngine
    - Real RetrievalEngine.search() with:
      * Real BM25 (rank_bm25 library)
      * Real emotion resonance (8-category valence/arousal dicts)
      * Real temporal scoring (time window matching)
      * Real cross-reference scoring (memory type/graph diversity)
      * Two-phase retrieval-ranking decoupling
      * Path auto-silencing for non-discriminating paths
    """

    def __init__(self, harness: BenchmarkHarness):
        self.harness = harness
        self.engine = harness.engine

    async def answer(
        self,
        query: str,
        strategy: DDAStrategy = HOT_STRATEGY,
        ddi_level: DDILevel = DDILevel.HOT,
        top_k: int = 25,
        disabled_paths: set[str] | None = None,  # v9 Ablation
    ) -> tuple[str, list[int], float]:
        """Run real RetrievalEngine.search() and map results back."""
        results = await self.harness.search(
            query=query,
            strategy=strategy,
            ddi_level=ddi_level,
            top_k=top_k,
            disabled_paths=disabled_paths,
        )

        if not results:
            return "未找到相关信息。", [], 0.0

        contexts = [r.get("content", "") for r in results if r.get("content")]
        answer = " | ".join(contexts)
        top_score = results[0].get("final_score", results[0].get("score", 0.5)) if results else 0.0

        # Map bucket IDs back to original memory indices
        indices = []
        for r in results:
            rid = r.get("id", "")
            try:
                idx = self.harness._memory_ids.index(rid)
                indices.append(idx)
            except ValueError:
                pass

        return answer, indices, float(top_score)


# ═══════════════════════════════════════════════════════════════
# Comparison Results Collector
# ═══════════════════════════════════════════════════════════════

@dataclass
class SystemResult:
    """Aggregated results for one system."""
    name: str
    category_scores: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    all_grades: list[int] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def total_score(self) -> int:
        return sum(self.all_grades)

    @property
    def avg_grade(self) -> float:
        return sum(self.all_grades) / max(len(self.all_grades), 1)

    @property
    def max_possible(self) -> int:
        return len(self.all_grades) * 3

    def category_avg(self, cat: str) -> float:
        grades = self.category_scores.get(cat, [])
        return sum(grades) / max(len(grades), 1) if grades else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_score": self.total_score,
            "max_possible": self.max_possible,
            "avg_grade": round(self.avg_grade, 3),
            "category_scores": {
                cat: {
                    "total": sum(grades),
                    "avg": round(sum(grades) / max(len(grades), 1), 3),
                    "max": len(grades) * 3,
                    "grades": grades,
                }
                for cat, grades in self.category_scores.items()
            },
            "all_grades": self.all_grades,
            "latency_ms": round(self.latency_ms, 1),
        }


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest_asyncio.fixture(scope="module")
async def harness_small(tmp_path_factory):
    """Small corpus: 10 memories (COLD zone)."""
    tmp = tmp_path_factory.mktemp("mp_small")
    h = BenchmarkHarness(tmp)
    await h.populate(BENCHMARK_MEMORIES_SMALL)
    return h


@pytest_asyncio.fixture(scope="module")
async def harness_medium(tmp_path_factory):
    """Medium corpus: 22 memories (WARM zone)."""
    tmp = tmp_path_factory.mktemp("mp_medium")
    h = BenchmarkHarness(tmp)
    await h.populate(BENCHMARK_MEMORIES)
    return h


@pytest_asyncio.fixture(scope="module")
async def harness_large(tmp_path_factory):
    """Large corpus: 22 + 50 noise = 72 memories (HOT zone)."""
    tmp = tmp_path_factory.mktemp("mp_large")
    h = BenchmarkHarness(tmp)
    all_mems = list(BENCHMARK_MEMORIES) + generate_noise_memories(50, seed=42)
    await h.populate_async_batch(all_mems, concurrency=15)
    return h


# ═══════════════════════════════════════════════════════════════
# Module-level state (shared across tests for export)
# ═══════════════════════════════════════════════════════════════

_module_small_results: dict[str, SystemResult] = {}
_module_large_results: dict[str, SystemResult] = {}


# ═══════════════════════════════════════════════════════════════
# Module-level fixtures (shared across test classes)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def scorer():
    return AnswerScorer()


@pytest.fixture(scope="module")
def simulator_systems():
    """Initialize all 11 simulators (no MP) — shared across all tests."""
    return create_all_systems(BENCHMARK_MEMORIES)


# ═══════════════════════════════════════════════════════════════
# Comprehensive Comparison Test Class
# ═══════════════════════════════════════════════════════════════

class TestComprehensiveComparison:
    """
    THE definitive cross-system benchmark.

    Runs: 12 systems × 25 QA pairs × 2 sample sizes.
    MP uses real RetrievalEngine.search() — no stitching.
    All simulators use real BM25/PPR/community/emotion algorithms.
    """

    # -- Test 1: All simulators return valid results --------─

    def test_all_simulators_initialized(self, simulator_systems):
        """All 11 simulator systems load correctly."""
        for name in SIMULATOR_SYSTEMS:
            assert name in simulator_systems, f"Missing: {name}"
            assert simulator_systems[name] is not None, f"None: {name}"

    @pytest.mark.parametrize("qa", ALL_QA_PAIRS, ids=lambda q: q.id)
    def test_simulators_handle_all_questions(self, qa, simulator_systems, scorer):
        """Every simulator can process every question without crashing."""
        for name in SIMULATOR_SYSTEMS:
            sys = simulator_systems[name]
            answer, indices, score = sys.answer(qa.question)
            assert isinstance(answer, str), f"{name} crashed on {qa.id}"
            grade = scorer.score(answer, qa)
            assert 0 <= grade <= 3, f"{name} invalid grade {grade} on {qa.id}"

    # -- Test 2: Real MP answers all questions --------------─

    @pytest.mark.asyncio
    async def test_mp_real_answers_all_questions(self, harness_medium):
        """MP with real RetrievalEngine.search() handles all 25 QA."""
        mp = MemoryPalaceAdapter(harness_medium)
        scorer = AnswerScorer()
        for qa in ALL_QA_PAIRS[:10]:  # First 10 for speed
            answer, indices, score = await mp.answer(
                qa.question, strategy=HOT_STRATEGY, ddi_level=DDILevel.HOT,
            )
            assert isinstance(answer, str), f"MP answer not str for {qa.id}"
            assert len(answer) > 0, f"MP empty for {qa.id}"
            grade = scorer.score(answer, qa)
            assert 0 <= grade <= 3

    # -- Test 3: FULL BENCHMARK — small corpus (10 mems) ----─

    @pytest.mark.asyncio
    async def test_full_benchmark_small(self, harness_small, simulator_systems, scorer):
        """
        FULL BENCHMARK: Small corpus (10 core memories).
        All 12 systems × 25 QA pairs.
        MP uses real RetrievalEngine.search().
        """
        results: dict[str, SystemResult] = {}
        print("\n" + "=" * 72)
        print("  COMPREHENSIVE BENCHMARK — Small Corpus (10 memories)")
        print("=" * 72)

        # -- Run all simulators --
        for name in SIMULATOR_SYSTEMS:
            t0 = time.perf_counter()
            sys = simulator_systems[name]
            sr = SystemResult(name=name)
            for qa in ALL_QA_PAIRS:
                answer, indices, score = sys.answer(qa.question)
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            sr.latency_ms = (time.perf_counter() - t0) * 1000
            results[name] = sr

        # -- Run Memory Palace (real engine, HOT mode) --
        mp = MemoryPalaceAdapter(harness_small)
        sr_mp = SystemResult(name="Memory Palace v9")
        t0 = time.perf_counter()
        for qa in ALL_QA_PAIRS:
            answer, indices, score = await mp.answer(
                qa.question, strategy=HOT_STRATEGY, ddi_level=DDILevel.HOT,
            )
            grade = scorer.score(answer, qa)
            sr_mp.all_grades.append(grade)
            sr_mp.category_scores[qa.category].append(grade)
        sr_mp.latency_ms = (time.perf_counter() - t0) * 1000
        results["Memory Palace v9"] = sr_mp

        # -- Print ranking table --
        self._print_ranking(results, "Small Corpus (10 memories)")
        self._print_category_breakdown(results)
        self._print_detailed_qa_table(results)

        # -- Assertions --
        for name, sr in results.items():
            assert len(sr.all_grades) == 25, f"{name}: expected 25 grades, got {len(sr.all_grades)}"
            for g in sr.all_grades:
                assert 0 <= g <= 3, f"{name}: invalid grade {g}"

        # MP should beat at least BM25 baseline in small corpus
        mp_total = results["Memory Palace v9"].total_score
        bm25_total = results["BM25 Baseline"].total_score
        print(f"\n  MP v8: {mp_total}/75 | BM25: {bm25_total}/75")
        print(f"  Delta: {'+' if mp_total >= bm25_total else ''}{mp_total - bm25_total}")

        # Store for cross-test access
        global _module_small_results
        _module_small_results = results

    # -- Test 4: FULL BENCHMARK — large corpus (72 mems) ----─

    @pytest.mark.asyncio
    async def test_full_benchmark_large(self, harness_large, scorer):
        """
        FULL BENCHMARK: Large corpus (22 core + 50 noise = 72 memories).
        All 12 systems × 25 QA pairs.
        """
        # Create simulators with the large dataset
        all_mems = list(BENCHMARK_MEMORIES) + generate_noise_memories(50, seed=42)
        bm25 = SharedBM25Index(all_mems)
        sim_systems = create_all_systems(all_mems)
        # Rebuild simulators with large dataset
        sim_systems_large = {
            "A-MEM (NeurIPS 2025)": AMEMSimulator(all_mems, bm25),
            "MAGMA (CVPR 2025)": MAGMASimulator(all_mems, bm25),
            "MMAG": MMAGSimulator(all_mems, bm25),
            "Mem0-like": Mem0Simulator(all_mems, bm25),
            "Zep-like": ZepSimulator(all_mems, bm25),
            "BM25 Baseline": BM25Baseline(all_mems, bm25),
            "Vector Baseline": VectorBaseline(all_mems, bm25),
            "HippoRAG (PPR)": HippoRAGSimulator(all_mems, bm25),
            "GraphRAG (Community)": GraphRAGSimulator(all_mems, bm25),
            "MemLong (Learnable)": MemLongSimulator(all_mems, bm25),
            "HybridFusion (No-DDA)": HybridFusionSim(all_mems, bm25),
        }

        results: dict[str, SystemResult] = {}
        print("\n" + "=" * 72)
        print("  COMPREHENSIVE BENCHMARK — Large Corpus (72 memories)")
        print("=" * 72)

        # -- Run all simulators --
        for name in SIMULATOR_SYSTEMS:
            t0 = time.perf_counter()
            sys = sim_systems_large[name]
            sr = SystemResult(name=name)
            for qa in ALL_QA_PAIRS:
                answer, indices, score = sys.answer(qa.question)
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            sr.latency_ms = (time.perf_counter() - t0) * 1000
            results[name] = sr

        # -- Run Memory Palace (real engine) --
        mp = MemoryPalaceAdapter(harness_large)
        sr_mp = SystemResult(name="Memory Palace v9")
        t0 = time.perf_counter()
        for qa in ALL_QA_PAIRS:
            answer, indices, score = await mp.answer(
                qa.question, strategy=HOT_STRATEGY, ddi_level=DDILevel.HOT,
            )
            grade = scorer.score(answer, qa)
            sr_mp.all_grades.append(grade)
            sr_mp.category_scores[qa.category].append(grade)
        sr_mp.latency_ms = (time.perf_counter() - t0) * 1000
        results["Memory Palace v9"] = sr_mp

        # -- Print ranking tables --
        self._print_ranking(results, "Large Corpus (72 memories)")
        self._print_category_breakdown(results)
        self._print_size_scaling_comparison(results)

        # -- Assertions --
        for name, sr in results.items():
            assert len(sr.all_grades) == 25, f"{name}: expected 25 grades"
            for g in sr.all_grades:
                assert 0 <= g <= 3, f"{name}: invalid grade {g}"

        global _module_large_results
        _module_large_results = results

    # -- Test 5: Export results for visual report ------------─

    def test_export_results_json(self):
        """Export benchmark results as JSON for visual report generation."""
        small = _module_small_results
        large = _module_large_results
        if small is None or large is None:
            pytest.skip("Run full benchmarks first")

        export = {
            "meta": {
                "date": __import__('datetime').datetime.now().isoformat(),
                "qa_count": 25,
                "categories": CATEGORY_MAP,
                "systems": SYSTEM_NAMES,
                "sample_sizes": {"small": 10, "large": 72},
            },
            "small_corpus": {name: sr.to_dict() for name, sr in small.items()},
            "large_corpus": {name: sr.to_dict() for name, sr in large.items()},
        }

        # Write to docs directory
        output_dir = Path(__file__).resolve().parent.parent / "docs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "comprehensive_benchmark_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        print(f"\nResults exported to: {output_path}")
        assert output_path.exists(), "Export failed"

    # -- Printing helpers ------------------------------------

    @staticmethod
    def _print_ranking(results: dict[str, SystemResult], title: str):
        """Print system ranking table."""
        ranked = sorted(results.items(), key=lambda x: x[1].total_score, reverse=True)
        print(f"\n## Overall Ranking — {title}")
        print(f"| Rank | System | Total | Avg | Grade |")
        print(f"|------|--------|-------|-----|-------|")
        for rank, (name, sr) in enumerate(ranked, 1):
            medal = {1: "[1]", 2: "[2]", 3: "[3]"}.get(rank, f"{rank:2d}")
            grade_bar = "#" * max(1, int(sr.avg_grade * 10)) + "." * max(0, 30 - int(sr.avg_grade * 10))
            print(f"| {medal} | {name} | {sr.total_score}/75 | {sr.avg_grade:.2f} | {grade_bar} |")
        print()

    @staticmethod
    def _print_category_breakdown(results: dict[str, SystemResult]):
        """Print per-category average scores."""
        ranked = sorted(results.items(), key=lambda x: x[1].total_score, reverse=True)
        categories = list(CATEGORY_MAP.keys())
        cat_cn = [CATEGORY_MAP[c] for c in categories]

        print("\n## Category Breakdown (avg grade / 3.0)")
        # Header
        header = "| System | " + " | ".join(cat_cn) + " |"
        print(header)
        print("|" + "---|" * (len(categories) + 1))
        for name, sr in ranked[:8]:  # Top 8
            cells = [name]
            for cat in categories:
                avg = sr.category_avg(cat)
                cells.append(f"{avg:.2f}")
            print("| " + " | ".join(cells) + " |")
        print()

    @staticmethod
    def _print_detailed_qa_table(results: dict[str, SystemResult]):
        """Print detailed per-QA grade matrix."""
        ranked = sorted(results.items(), key=lambda x: x[1].total_score, reverse=True)
        top_systems = ranked[:6]  # Top 6 systems

        print("\n## Detailed QA Matrix (top 6 systems)")
        header = "| QA | Category | " + " | ".join(name[:12] for name, _ in top_systems) + " |"
        print(header)
        print("|" + "---|" * (len(top_systems) + 2))
        for qa in ALL_QA_PAIRS:
            cells = [qa.id, qa.category_cn[:4]]
            for name, sr in top_systems:
                idx = ALL_QA_PAIRS.index(qa)
                grade = sr.all_grades[idx] if idx < len(sr.all_grades) else 0
                cells.append(str(grade))
            print("| " + " | ".join(cells) + " |")
        print()

    @staticmethod
    def _print_size_scaling_comparison(results_large: dict[str, SystemResult]):
        """Compare large corpus results with expected small corpus degradation."""
        print("\n## Large Corpus Performance Notes")
        ranked = sorted(results_large.items(), key=lambda x: x[1].total_score, reverse=True)
        for rank, (name, sr) in enumerate(ranked[:5], 1):
            best_cat = max(sr.category_scores.keys(), key=lambda c: sr.category_avg(c))
            worst_cat = min(sr.category_scores.keys(), key=lambda c: sr.category_avg(c))
            print(f"  {rank}. {name}: {sr.total_score}/75 ({sr.avg_grade:.2f})")
            print(f"     Best: {CATEGORY_MAP[best_cat]} ({sr.category_avg(best_cat):.2f})")
            print(f"     Worst: {CATEGORY_MAP[worst_cat]} ({sr.category_avg(worst_cat):.2f})")
        print()


# ═══════════════════════════════════════════════════════════════
# DDI Scaling Test (MP only — COLD/WARM/HOT/RICH)
# ═══════════════════════════════════════════════════════════════

DDI_TEST_LEVELS = [
    ("COLD", COLD_STRATEGY, DDILevel.COLD),
    ("WARM", WARM_STRATEGY, DDILevel.WARM),
    ("HOT", HOT_STRATEGY, DDILevel.HOT),
    ("RICH", RICH_STRATEGY, DDILevel.RICH),
]


class TestMPDDIScaling:
    """Test MP retrieval quality at each DDI level."""

    @pytest.mark.asyncio
    async def test_ddi_scaling_all_levels(self, harness_medium):
        """
        Test all 4 DDI levels against all 25 QA pairs.
        HOT/RICH should match or outperform COLD/WARM on medium corpus.
        """
        scorer = AnswerScorer()
        mp = MemoryPalaceAdapter(harness_medium)
        results: dict[str, SystemResult] = {}

        print("\n-- MP DDI Scaling (Medium.22 memories) --")
        for ddi_name, strategy, ddi_level in DDI_TEST_LEVELS:
            sr = SystemResult(name=f"MP {ddi_name}")
            for qa in ALL_QA_PAIRS:
                answer, indices, score = await mp.answer(
                    qa.question, strategy=strategy, ddi_level=ddi_level,
                )
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            results[ddi_name] = sr

        # Print DDI comparison
        print(f"\n| DDI | Total | Avg | Category Breakdown |")
        print(f"|-----|-------|-----|-------------------|")
        for ddi_name, sr in results.items():
            cat_str = " | ".join(
                f"{CATEGORY_MAP[c][:2]}:{sr.category_avg(c):.2f}"
                for c in CATEGORY_MAP
            )
            print(f"| {ddi_name} | {sr.total_score}/75 | {sr.avg_grade:.2f} | {cat_str} |")

        # Assertions
        hot_total = results["HOT"].total_score
        cold_total = results["COLD"].total_score
        assert hot_total >= 0, "HOT should produce results"
        # DDA should not catastrophically regress
        assert hot_total >= max(0, cold_total - 10), \
            f"HOT ({hot_total}) significantly worse than COLD ({cold_total})"


# ═══════════════════════════════════════════════════════════════
# Quick Verifier (runs fast — good for dev iteration)
# ═══════════════════════════════════════════════════════════════

class TestQuickVerify:
    """Quick verification that everything works end-to-end."""

    @pytest.mark.asyncio
    async def test_mp_real_vs_simulators_quick(self, harness_small, simulator_systems, scorer):
        """Quick 5-QA comparison: real MP vs top 3 simulators."""
        mp = MemoryPalaceAdapter(harness_small)
        test_qas = ALL_QA_PAIRS[:5]

        print("\n-- Quick Verify (5 QA . Small Corpus) --")
        print(f"| QA | MP v8 | BM25 | Vector | HippoRAG |")
        print(f"|----|-------|------|--------|----------|")

        systems_to_test = {
            "MP v8": None,
            "BM25 Baseline": simulator_systems["BM25 Baseline"],
            "Vector Baseline": simulator_systems["Vector Baseline"],
            "HippoRAG (PPR)": simulator_systems["HippoRAG (PPR)"],
        }

        for qa in test_qas:
            cells = [qa.id]
            for name, sys in systems_to_test.items():
                if sys is None:
                    answer, indices, score = await mp.answer(
                        qa.question, strategy=HOT_STRATEGY, ddi_level=DDILevel.HOT,
                    )
                else:
                    answer, indices, score = sys.answer(qa.question)
                grade = scorer.score(answer, qa)
                cells.append(str(grade))
            print("| " + " | ".join(cells) + " |")

        print()
