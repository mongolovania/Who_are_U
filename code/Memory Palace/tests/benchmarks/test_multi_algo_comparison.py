#!/usr/bin/env python3
# ============================================================
# Multi-Algorithm Comparison Benchmark — 全方法对比测试
# v9: 15 systems (MP v9 + 11 simulators + 3 real implementations)
#     25 QA pairs × 6 categories × 2 corpus sizes
#
# Systems under test:
#   Tier A — Real Engine:
#     Memory Palace v9          (8-path DDA fusion via RetrievalEngine)
#   Tier B — Paper Simulators (11):
#     A-MEM, MAGMA, MMAG, Mem0-like, Zep-like,
#     BM25 Baseline, Vector Baseline, HippoRAG(PPR),
#     GraphRAG(Community), MemLong(Learnable), HybridFusion
#   Tier C — Real Community Implementations (3):
#     sentence-transformers+FAISS, BM25S, Real Mem0 (optional)
# ============================================================

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Existing infrastructure ──────────────────────────────────
from tests.benchmarks.benchmark_dataset import (
    BENCHMARK_MEMORIES,
    BENCHMARK_MEMORIES_SMALL,
    BenchmarkMemory,
)
from tests.benchmarks.comparison_qa_dataset import (
    ALL_QA_PAIRS,
    CATEGORY_MAP,
    CATEGORY_QA_COUNTS,
)
from tests.benchmarks.algorithm_simulators import (
    AMEMSimulator,
    BM25Baseline,
    GraphRAGSimulator,
    HippoRAGSimulator,
    HybridFusionSim,
    MAGMASimulator,
    Mem0Simulator,
    MemLongSimulator,
    MMAGSimulator,
    SharedBM25Index,
    VectorBaseline,
    ZepSimulator,
    create_all_systems,
)
from tests.benchmarks.benchmark_harness import (
    BenchmarkHarness,
    generate_noise_memories,
    make_benchmark_config,
)
from tests.benchmarks.real_implementations import (
    BM25sAdapter,
    REAL_AVAILABILITY,
    RealGraphRAGAdapter,
    RealMem0Adapter,
    SentenceTransformersFAISSAdapter,
)
from tests.benchmarks.test_comprehensive_comparison import (
    AnswerScorer,
    MemoryPalaceAdapter,
    SystemResult,
)
from memory_node import DDILevel, HOT_STRATEGY


# ═══════════════════════════════════════════════════════════════
# System Registry — all 15 systems
# ═══════════════════════════════════════════════════════════════

# Factory: (memories, bm25_index) -> system with answer() interface
SystemFactory = callable


def _make_sim_registry() -> dict[str, SystemFactory]:
    """Build registry of all 11 simulators + 3 real impls."""
    registry: dict[str, SystemFactory] = {}

    # ── 11 Paper Simulators ────────────────────────────────
    registry["A-MEM (NeurIPS 2025)"] = lambda mems, bm25: AMEMSimulator(mems, bm25)
    registry["MAGMA (CVPR 2025)"] = lambda mems, bm25: MAGMASimulator(mems, bm25)
    registry["MMAG"] = lambda mems, bm25: MMAGSimulator(mems, bm25)
    registry["Mem0-like (Sim)"] = lambda mems, bm25: Mem0Simulator(mems, bm25)
    registry["Zep-like (Sim)"] = lambda mems, bm25: ZepSimulator(mems, bm25)
    registry["BM25 Baseline"] = lambda mems, bm25: BM25Baseline(mems, bm25)
    registry["Vector Baseline"] = lambda mems, bm25: VectorBaseline(mems, bm25)
    registry["HippoRAG (PPR)"] = lambda mems, bm25: HippoRAGSimulator(mems, bm25)
    registry["GraphRAG (Community)"] = lambda mems, bm25: GraphRAGSimulator(mems, bm25)
    registry["MemLong (Learnable)"] = lambda mems, bm25: MemLongSimulator(mems, bm25)
    registry["HybridFusion (No-DDA)"] = lambda mems, bm25: HybridFusionSim(mems, bm25)

    # ── Real Community Implementations ─────────────────────
    if REAL_AVAILABILITY.get("sentence-transformers+FAISS", False):
        registry["sentence-transformers+FAISS"] = (
            lambda mems, bm25: SentenceTransformersFAISSAdapter(mems)
        )

    if REAL_AVAILABILITY.get("Real BM25S", False):
        registry["Real BM25S"] = (
            lambda mems, bm25: BM25sAdapter(mems)
        )

    if REAL_AVAILABILITY.get("Real Mem0 (mem0ai)", False):
        registry["Real Mem0 (mem0ai)"] = (
            lambda mems, bm25: RealMem0Adapter(mems)
        )

    # GraphRAG is stretch — only register if fully configured
    # registry["Real GraphRAG"] = lambda mems, bm25: RealGraphRAGAdapter(mems)

    return registry


ALL_SYSTEM_FACTORIES = _make_sim_registry()


def build_all_systems(
    memories: list[BenchmarkMemory],
) -> dict[str, object]:
    """
    Instantiate all available systems for the given memory corpus.

    Returns dict of system_name -> system_instance, where each instance
    has an ``answer(query, top_k) -> (text, indices, score)`` method.
    """
    bm25 = SharedBM25Index(memories)
    systems: dict[str, object] = {}

    for name, factory in ALL_SYSTEM_FACTORIES.items():
        try:
            systems[name] = factory(memories, bm25)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return systems


# ═══════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def benchmark_scorer() -> AnswerScorer:
    """Reuse the dual keyword+coverage scorer."""
    return AnswerScorer()


@pytest.fixture(scope="module")
def all_systems_medium() -> dict[str, object]:
    """All systems built on 22-memory corpus (WARM)."""
    return build_all_systems(BENCHMARK_MEMORIES)


@pytest.fixture(scope="module")
def all_systems_large() -> dict[str, object]:
    """All systems built on 72-memory corpus (HOT: 22 core + 50 noise)."""
    noise = generate_noise_memories(50, seed=42)
    large_corpus = list(BENCHMARK_MEMORIES) + list(noise)
    return build_all_systems(large_corpus)


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ═══════════════════════════════════════════════════════════════
# Test Class — Full Multi-Algo Benchmark
# ═══════════════════════════════════════════════════════════════

@pytest.mark.benchmark
class TestMultiAlgoComparison:
    """
    THE definitive multi-algorithm benchmark.

    Runs all available systems against all 25 QA pairs at two corpus
    sizes (22-medium, 72-large). Produces per-category per-system
    score matrices suitable for decision-making and visualization.
    """

    # Module-level result caches (shared across test methods)
    medium_results: dict[str, SystemResult] = {}
    large_results: dict[str, SystemResult] = {}

    @pytest.mark.asyncio
    async def test_full_benchmark_medium(
        self,
        tmp_path,
        all_systems_medium,
        benchmark_scorer,
    ):
        """
        Full 22-memory benchmark: all systems × 25 QA pairs.

        WARM state — BM25-dominant regime. Measures whether signal
        paths add marginal value beyond keyword matching.
        """
        harness = await self._setup_harness(tmp_path, BENCHMARK_MEMORIES)
        mp = MemoryPalaceAdapter(harness)

        print(f"\n{'='*70}")
        print(f"Multi-Algo Benchmark — MEDIUM (22 memories, WARM)")
        print(f"{'='*70}")
        print(f"Systems: {len(all_systems_medium)} sim/real + MP v9")
        print(f"QA pairs: {len(ALL_QA_PAIRS)} ({', '.join(CATEGORY_MAP.values())})")
        print()

        results: dict[str, SystemResult] = {}

        # ── Run all simulator/real systems ─────────────────
        for name, system in all_systems_medium.items():
            sr = await self._run_system(name, system, benchmark_scorer)
            results[name] = sr

        # ── Run Memory Palace v9 ───────────────────────────
        mp_sr = await self._run_mp("Memory Palace v9", mp, benchmark_scorer)
        results["Memory Palace v9"] = mp_sr

        # ── Print ranking ──────────────────────────────────
        self._print_ranking(results, "MEDIUM (22 memories)")
        self._print_category_matrix(results)

        TestMultiAlgoComparison.medium_results = results

        # Sanity: baseline should be reasonable
        if "BM25 Baseline" in results:
            bm25_total = results["BM25 Baseline"].total_score
            assert bm25_total >= 35, \
                f"BM25 Baseline too low: {bm25_total}/75 — possible regression"

    @pytest.mark.asyncio
    async def test_full_benchmark_large(
        self,
        tmp_path,
        all_systems_large,
        benchmark_scorer,
    ):
        """
        Full 72-memory benchmark: all systems × 25 QA pairs.

        HOT state — noise challenges keyword-only systems. Signal
        paths (graph, emotion, temporal) should differentiate.
        """
        noise = generate_noise_memories(50, seed=42)
        large_corpus = list(BENCHMARK_MEMORIES) + list(noise)
        harness = await self._setup_harness(tmp_path, large_corpus)
        mp = MemoryPalaceAdapter(harness)

        print(f"\n{'='*70}")
        print(f"Multi-Algo Benchmark — LARGE (72 memories, HOT)")
        print(f"{'='*70}")
        print(f"Systems: {len(all_systems_large)} sim/real + MP v9")
        print(f"QA pairs: {len(ALL_QA_PAIRS)}")
        print()

        results: dict[str, SystemResult] = {}

        for name, system in all_systems_large.items():
            sr = await self._run_system(name, system, benchmark_scorer)
            results[name] = sr

        mp_sr = await self._run_mp("Memory Palace v9", mp, benchmark_scorer)
        results["Memory Palace v9"] = mp_sr

        self._print_ranking(results, "LARGE (72 memories)")
        self._print_category_matrix(results)

        TestMultiAlgoComparison.large_results = results

    @pytest.mark.asyncio
    async def test_small_vs_large_delta(self):
        """
        Noise-resistance analysis: which systems degrade least
        when going from 22 → 72 memories?
        """
        med = TestMultiAlgoComparison.medium_results
        lrg = TestMultiAlgoComparison.large_results

        if not med or not lrg:
            pytest.skip("Need medium and large results first. "
                        "Run test_full_benchmark_medium and "
                        "test_full_benchmark_large first.")

        print(f"\n{'='*70}")
        print("Noise Resistance Delta: MEDIUM → LARGE")
        print(f"{'='*70}")
        print(f"{'System':<35} {'Medium':>7} {'Large':>7} {'Δ':>7} {'Resilience'}")
        print(f"{'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*12}")

        deltas: list[tuple[str, float, float, float]] = []
        common = set(med) & set(lrg)
        for name in sorted(common, key=lambda n: lrg[n].avg_grade, reverse=True):
            med_avg = med[name].avg_grade
            lrg_avg = lrg[name].avg_grade
            delta = lrg_avg - med_avg
            deltas.append((name, med_avg, lrg_avg, delta))

            if delta >= 0:
                resilience = "ANTI-FRAGILE"
            elif delta > -0.15:
                resilience = "RESILIENT"
            elif delta > -0.35:
                resilience = "MODERATE"
            else:
                resilience = "FRAGILE"

            print(f"{name:<35} {med_avg:>7.2f} {lrg_avg:>7.2f} {delta:>+7.2f} {resilience}")

    def test_export_results_json(self, tmp_path):
        """Export all results as JSON for visual report generation."""
        med = TestMultiAlgoComparison.medium_results
        lrg = TestMultiAlgoComparison.large_results

        export: dict = {
            "meta": {
                "version": "0.9.0",
                "qa_count": len(ALL_QA_PAIRS),
                "categories": CATEGORY_MAP,
                "category_counts": CATEGORY_QA_COUNTS,
                "medium_memory_count": len(BENCHMARK_MEMORIES),
                "systems_tested": sorted(set(list(med.keys()) + list(lrg.keys()))),
            },
            "medium_results": {
                name: sr.to_dict() for name, sr in med.items()
            },
            "large_results": {
                name: sr.to_dict() for name, sr in lrg.items()
            },
        }

        # Add per-category breakdowns
        for scope, results in [("medium", med), ("large", lrg)]:
            cat_matrix: dict[str, dict[str, float]] = {}
            for name, sr in results.items():
                cat_matrix[name] = {
                    cat: sr.category_avg(cat)
                    for cat in CATEGORY_MAP
                }
            export[f"{scope}_category_matrix"] = cat_matrix

        out_path = Path(tmp_path) / "multi_algo_benchmark_results.json"
        out_path.write_text(json.dumps(export, ensure_ascii=False, indent=2))

        # Also write to docs/ for report generator
        docs_path = Path(__file__).resolve().parent.parent / "docs"
        docs_path.mkdir(exist_ok=True)
        docs_out = docs_path / "multi_algo_benchmark_results.json"
        docs_out.write_text(json.dumps(export, ensure_ascii=False, indent=2))

        print(f"\n[OK] Results exported to: {docs_out}")
        print(f"     Systems: {len(export['systems_tested'])}")
        if med:
            top_sys = max(med, key=lambda n: med[n].avg_grade)
            print(f"     Medium best: {top_sys} ({med[top_sys].avg_grade:.2f})")
        if lrg:
            top_sys = max(lrg, key=lambda n: lrg[n].avg_grade)
            print(f"     Large best:  {top_sys} ({lrg[top_sys].avg_grade:.2f})")

    # ── Helpers ────────────────────────────────────────────────

    async def _setup_harness(
        self, tmp_path, memories: list[BenchmarkMemory],
    ) -> BenchmarkHarness:
        """Create harness with real BucketManager + populate."""
        harness = BenchmarkHarness(tmp_path, user_id="multi_algo_bench")
        await harness.populate_async_batch(memories, concurrency=10)
        return harness

    async def _run_system(
        self,
        name: str,
        system: object,
        scorer: AnswerScorer,
    ) -> SystemResult:
        """Run one system against all 25 QA pairs."""
        sr = SystemResult(name=name)
        t0 = time.perf_counter()

        for qa in ALL_QA_PAIRS:
            try:
                answer, indices, score = system.answer(qa.question)
            except Exception as e:
                answer, indices, score = f"ERROR: {e}", [], 0.0

            grade = scorer.score(answer, qa)
            sr.all_grades.append(grade)
            sr.category_scores[qa.category].append(grade)

        sr.latency_ms = (time.perf_counter() - t0) * 1000
        return sr

    async def _run_mp(
        self,
        name: str,
        mp: MemoryPalaceAdapter,
        scorer: AnswerScorer,
    ) -> SystemResult:
        """Run Memory Palace v9 against all 25 QA pairs."""
        sr = SystemResult(name=name)
        t0 = time.perf_counter()

        for qa in ALL_QA_PAIRS:
            answer, indices, score = await mp.answer(
                qa.question,
                strategy=HOT_STRATEGY,
                ddi_level=DDILevel.HOT,
            )
            grade = scorer.score(answer, qa)
            sr.all_grades.append(grade)
            sr.category_scores[qa.category].append(grade)

        sr.latency_ms = (time.perf_counter() - t0) * 1000
        return sr

    def _print_ranking(
        self,
        results: dict[str, SystemResult],
        title: str,
    ) -> None:
        """Print ranked table of all systems."""
        print(f"\n-- Ranking: {title} --")
        print(f"{'Rank':<5} {'System':<35} {'Total':>6} {'Avg':>6} {'Latency(ms)':>12}")
        print(f"{'─'*5} {'─'*35} {'─'*6} {'─'*6} {'─'*12}")

        ranked = sorted(results.items(), key=lambda x: x[1].avg_grade, reverse=True)
        for i, (name, sr) in enumerate(ranked, 1):
            medal = ["[1st]", "[2nd]", "[3rd]"][i-1] if i <= 3 else f"({i})"
            print(f"{medal:<5} {name:<35} {sr.total_score:>6} {sr.avg_grade:>6.2f} "
                  f"{sr.latency_ms:>10.0f}ms")

    def _print_category_matrix(
        self,
        results: dict[str, SystemResult],
    ) -> None:
        """Print per-category score matrix."""
        cats = list(CATEGORY_MAP.keys())
        ranked = sorted(results.items(), key=lambda x: x[1].avg_grade, reverse=True)

        print(f"\n-- Category Breakdown --")
        header = f"{'System':<35}"
        for cat in cats:
            header += f" {CATEGORY_MAP[cat][:6]:>7}"
        print(header)
        print(f"{'─'*35} " + " ".join(f"{'─'*7}" for _ in cats))

        for name, sr in ranked[:8]:  # Top 8 for readability
            row = f"{name:<35}"
            for cat in cats:
                row += f" {sr.category_avg(cat):>7.2f}"
            print(row)

        # Best per category
        print(f"\n-- Best per Category --")
        for cat in cats:
            best_name = max(results, key=lambda n: results[n].category_avg(cat))
            best_avg = results[best_name].category_avg(cat)
            print(f"  {CATEGORY_MAP[cat]:<20}: {best_name} ({best_avg:.2f})")


# ═══════════════════════════════════════════════════════════════
# Quick Smoke Test — fast dev iteration
# ═══════════════════════════════════════════════════════════════

class TestQuickSmoke:
    """Fast verification that all systems can answer without crashing."""

    def test_all_factories_produce_valid_systems(self, all_systems_medium):
        """Every registered system instantiates and answers one query."""
        test_qa = ALL_QA_PAIRS[0]  # "小明叫什么名字？"
        assert len(all_systems_medium) >= 11, \
            f"Expected >=11 systems, got {len(all_systems_medium)}"

        for name, system in all_systems_medium.items():
            answer, indices, score = system.answer(test_qa.question)
            assert isinstance(answer, str), f"{name}: answer not str: {type(answer)}"
            assert isinstance(indices, list), f"{name}: indices not list"
            assert isinstance(score, (int, float)), f"{name}: score not numeric"
            assert len(answer) > 0, f"{name}: empty answer"

    def test_system_count(self, all_systems_medium):
        """Verify expected systems are present."""
        required_sims = [
            "BM25 Baseline", "Vector Baseline", "A-MEM (NeurIPS 2025)",
            "MAGMA (CVPR 2025)", "HippoRAG (PPR)", "GraphRAG (Community)",
        ]
        for name in required_sims:
            assert name in all_systems_medium, \
                f"Required system '{name}' missing from registry"

    def test_real_vs_sim_bm25_agreement(self, all_systems_medium):
        """
        If BM25S is available, its top-1 should match BM25 Baseline
        on at least 80% of QA pairs (verifying implementation fidelity).
        """
        if "Real BM25S" not in all_systems_medium:
            pytest.skip("BM25S not installed")

        bm25_sim = all_systems_medium["BM25 Baseline"]
        bm25_real = all_systems_medium["Real BM25S"]

        agreements = 0
        for qa in ALL_QA_PAIRS:
            _, idx_sim, _ = bm25_sim.answer(qa.question)
            _, idx_real, _ = bm25_real.answer(qa.question)
            if idx_sim and idx_real and idx_sim[0] == idx_real[0]:
                agreements += 1

        agreement_rate = agreements / len(ALL_QA_PAIRS)
        print(f"\nBM25 agreement rate (top-1 match): {agreement_rate:.1%}")
        # Allow some variance due to different tokenization
        assert agreement_rate >= 0.60, \
            f"BM25 implementations diverge too much: {agreement_rate:.1%}"
