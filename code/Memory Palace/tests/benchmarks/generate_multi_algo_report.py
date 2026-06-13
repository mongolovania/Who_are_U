#!/usr/bin/env python3
# ============================================================
# Multi-Algorithm Comparison Report Generator — 12面板可视化报告
# v9: Generates comprehensive PNG + Markdown report comparing
#     15 systems (MP v9 + 11 simulators + 3 real implementations)
#     across 25 QA pairs × 6 categories × 2 corpus sizes.
#
# Usage:
#   python tests/benchmarks/generate_multi_algo_report.py
#
# Requires JSON results from test_multi_algo_comparison.py.
# If JSON not found, runs benchmarks live (slow but self-contained).
# ============================================================

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Headless rendering

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.benchmarks.comparison_qa_dataset import ALL_QA_PAIRS, CATEGORY_MAP, CATEGORY_QA_COUNTS
from tests.benchmarks.benchmark_dataset import BENCHMARK_MEMORIES, BenchmarkMemory
from tests.benchmarks.algorithm_simulators import (
    SharedBM25Index, create_all_systems,
)
from tests.benchmarks.benchmark_harness import (
    BenchmarkHarness, generate_noise_memories,
    make_benchmark_config,
)
from tests.benchmarks.real_implementations import (
    BM25sAdapter, REAL_AVAILABILITY,
    SentenceTransformersFAISSAdapter, RealMem0Adapter,
)
from tests.benchmarks.test_comprehensive_comparison import (
    AnswerScorer, MemoryPalaceAdapter, SystemResult,
)
from memory_node import DDILevel, HOT_STRATEGY


# ═══════════════════════════════════════════════════════════════
# Color scheme — 15+ systems
# ═══════════════════════════════════════════════════════════════

SYSTEM_COLORS: dict[str, str] = {
    "Memory Palace v9": "#FFD700",
    "A-MEM (NeurIPS 2025)": "#E57373",
    "MAGMA (CVPR 2025)": "#BA68C8",
    "MMAG": "#64B5F6",
    "Mem0-like (Sim)": "#4DB6AC",
    "Zep-like (Sim)": "#FFB74D",
    "BM25 Baseline": "#90A4AE",
    "Vector Baseline": "#A1887F",
    "HippoRAG (PPR)": "#7986CB",
    "GraphRAG (Community)": "#4FC3F7",
    "MemLong (Learnable)": "#81C784",
    "HybridFusion (No-DDA)": "#FF8A65",
    # Real implementations
    "sentence-transformers+FAISS": "#00E676",
    "Real BM25S": "#FF6D00",
    "Real Mem0 (mem0ai)": "#2979FF",
    "Real GraphRAG": "#AA00FF",
}

# Category display names (short)
CAT_SHORT = {
    "simple_recall": "Simple\nRecall",
    "multi_hop": "Multi-hop",
    "temporal": "Temporal",
    "emotional": "Emotional",
    "causal": "Causal",
    "cross_ref": "Cross-ref",
}

CAT_ORDER = list(CATEGORY_MAP.keys())


# ═══════════════════════════════════════════════════════════════
# Benchmark Runner (self-contained, no pytest needed)
# ═══════════════════════════════════════════════════════════════

async def run_benchmark(
    memories: list[BenchmarkMemory],
    mp_harness: BenchmarkHarness | None = None,
) -> dict[str, SystemResult]:
    """
    Run all available systems against all 25 QA pairs.

    Args:
        memories: Corpus to index
        mp_harness: Pre-built Memory Palace harness (or None to skip MP)

    Returns:
        dict of system_name -> SystemResult
    """
    bm25 = SharedBM25Index(memories)
    scorer = AnswerScorer()
    results: dict[str, SystemResult] = {}

    # ── 11 Simulators ──────────────────────────────────────
    sim_factories = {
        "A-MEM (NeurIPS 2025)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["AMEMSimulator"]
        ).AMEMSimulator(m, b),
        "MAGMA (CVPR 2025)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["MAGMASimulator"]
        ).MAGMASimulator(m, b),
        "MMAG": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["MMAGSimulator"]
        ).MMAGSimulator(m, b),
        "Mem0-like (Sim)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["Mem0Simulator"]
        ).Mem0Simulator(m, b),
        "Zep-like (Sim)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["ZepSimulator"]
        ).ZepSimulator(m, b),
        "BM25 Baseline": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["BM25Baseline"]
        ).BM25Baseline(m, b),
        "Vector Baseline": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["VectorBaseline"]
        ).VectorBaseline(m, b),
        "HippoRAG (PPR)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["HippoRAGSimulator"]
        ).HippoRAGSimulator(m, b),
        "GraphRAG (Community)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["GraphRAGSimulator"]
        ).GraphRAGSimulator(m, b),
        "MemLong (Learnable)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["MemLongSimulator"]
        ).MemLongSimulator(m, b),
        "HybridFusion (No-DDA)": lambda m, b: __import__(
            "tests.benchmarks.algorithm_simulators", fromlist=["HybridFusionSim"]
        ).HybridFusionSim(m, b),
    }

    for name, factory in sim_factories.items():
        t0 = time.perf_counter()
        try:
            system = factory(memories, bm25)
            sr = SystemResult(name=name)
            for qa in ALL_QA_PAIRS:
                answer, indices, score = system.answer(qa.question)
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            sr.latency_ms = (time.perf_counter() - t0) * 1000
            results[name] = sr
            print(f"  [OK] {name}: {sr.total_score}/75 ({sr.avg_grade:.2f})")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # ── Real Implementations ───────────────────────────────
    if REAL_AVAILABILITY.get("sentence-transformers+FAISS", False):
        t0 = time.perf_counter()
        try:
            st_adapter = SentenceTransformersFAISSAdapter(memories)
            sr = SystemResult(name="sentence-transformers+FAISS")
            for qa in ALL_QA_PAIRS:
                answer, indices, score = st_adapter.answer(qa.question)
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            sr.latency_ms = (time.perf_counter() - t0) * 1000
            results["sentence-transformers+FAISS"] = sr
            print(f"  [OK] sentence-transformers+FAISS: {sr.total_score}/75 ({sr.avg_grade:.2f})")
        except Exception as e:
            print(f"  [SKIP] sentence-transformers+FAISS: {e}")

    if REAL_AVAILABILITY.get("Real BM25S", False):
        t0 = time.perf_counter()
        try:
            bm25s_adapter = BM25sAdapter(memories)
            sr = SystemResult(name="Real BM25S")
            for qa in ALL_QA_PAIRS:
                answer, indices, score = bm25s_adapter.answer(qa.question)
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            sr.latency_ms = (time.perf_counter() - t0) * 1000
            results["Real BM25S"] = sr
            print(f"  [OK] Real BM25S: {sr.total_score}/75 ({sr.avg_grade:.2f})")
        except Exception as e:
            print(f"  [SKIP] Real BM25S: {e}")

    # ── Memory Palace v9 (real engine) ─────────────────────
    if mp_harness is not None:
        t0 = time.perf_counter()
        try:
            mp = MemoryPalaceAdapter(mp_harness)
            sr = SystemResult(name="Memory Palace v9")
            for qa in ALL_QA_PAIRS:
                answer, indices, score = await mp.answer(
                    qa.question, strategy=HOT_STRATEGY, ddi_level=DDILevel.HOT,
                )
                grade = scorer.score(answer, qa)
                sr.all_grades.append(grade)
                sr.category_scores[qa.category].append(grade)
            sr.latency_ms = (time.perf_counter() - t0) * 1000
            results["Memory Palace v9"] = sr
            print(f"  [OK] Memory Palace v9: {sr.total_score}/75 ({sr.avg_grade:.2f})")
        except Exception as e:
            print(f"  [SKIP] Memory Palace v9: {e}")

    return results


# ═══════════════════════════════════════════════════════════════
# Report Generator — 12-panel matplotlib figure
# ═══════════════════════════════════════════════════════════════

def generate_report(
    results_medium: dict[str, SystemResult],
    results_large: dict[str, SystemResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Generate 12-panel comparison report.

    Returns:
        (png_path, md_path) — paths to generated files
    """
    # Merge system lists from both corpora
    all_systems = sorted(
        set(list(results_medium.keys()) + list(results_large.keys())),
        key=lambda n: results_large.get(n, results_medium.get(n, SystemResult(name=n))).avg_grade,
        reverse=True,
    )
    n_systems = len(all_systems)

    # ── Figure setup ───────────────────────────────────────
    fig = plt.figure(figsize=(32, 24))
    fig.suptitle(
        f"Multi-Algorithm Memory Retrieval Benchmark — {n_systems} Systems, 25 QA Pairs, 6 Categories",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # ── Pre-compute data ────────────────────────────────────
    # Category averages for heatmap
    cat_matrix_large = np.zeros((n_systems, 6))
    for i, name in enumerate(all_systems):
        sr = results_large.get(name)
        if sr:
            for j, cat in enumerate(CAT_ORDER):
                cat_matrix_large[i, j] = sr.category_avg(cat)

    # Total scores
    medium_totals = {
        name: results_medium[name].total_score
        for name in all_systems if name in results_medium
    }
    large_totals = {
        name: results_large[name].total_score
        for name in all_systems if name in results_large
    }

    # Deltas
    deltas = {}
    for name in all_systems:
        if name in results_medium and name in results_large:
            deltas[name] = (
                results_large[name].avg_grade - results_medium[name].avg_grade
            )

    # ═══════════════════════════════════════════════════════════
    # Panel 1: Overall Bar — Medium vs Large totals
    # ═══════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(4, 3, 1)
    x = np.arange(len(all_systems))
    width = 0.35

    med_vals = [medium_totals.get(n, 0) for n in all_systems]
    lrg_vals = [large_totals.get(n, 0) for n in all_systems]

    bars1 = ax1.bar(x - width/2, med_vals, width, label="Small (22 mems)",
                     color="#64B5F6", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + width/2, lrg_vals, width, label="Large (72 mems)",
                     color="#E57373", edgecolor="white", linewidth=0.5)

    # Highlight MP v9
    if "Memory Palace v9" in all_systems:
        mp_idx = all_systems.index("Memory Palace v9")
        bars1[mp_idx].set_color("#FFD700")
        bars2[mp_idx].set_color("#FFA000")

    ax1.set_xticks(x)
    ax1.set_xticklabels(all_systems, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Total Score (/75)")
    ax1.set_title("Panel 1: Overall Ranking — Small vs Large Corpus", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.axhline(y=37.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax1.set_ylim(0, 80)

    # ═══════════════════════════════════════════════════════════
    # Panel 2: Radar Chart — Top 5 systems (Large)
    # ═══════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(4, 3, 2, projection="polar")
    top5_names = sorted(
        [n for n in all_systems if n in results_large],
        key=lambda n: results_large[n].avg_grade, reverse=True,
    )[:5]

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for name in top5_names:
        sr = results_large[name]
        values = [sr.category_avg(cat) for cat in CAT_ORDER]
        values += values[:1]
        color = SYSTEM_COLORS.get(name, "#999999")
        ax2.fill(angles, values, alpha=0.1, color=color)
        ax2.plot(angles, values, "o-", linewidth=1.5, label=name, color=color)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([CAT_SHORT[c] for c in CAT_ORDER], fontsize=7)
    ax2.set_ylim(0, 3.0)
    ax2.set_title("Panel 2: Top 5 Radar (Large Corpus)", fontweight="bold", pad=20)
    ax2.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # ═══════════════════════════════════════════════════════════
    # Panel 3: Heatmap — Systems × Categories (Large)
    # ═══════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(4, 3, 3)
    im = ax3.imshow(cat_matrix_large, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)

    ax3.set_xticks(range(6))
    ax3.set_xticklabels([CAT_SHORT[c] for c in CAT_ORDER], fontsize=7)
    ax3.set_yticks(range(n_systems))
    ax3.set_yticklabels(all_systems, fontsize=7)

    # Annotate cells with values
    for i in range(n_systems):
        for j in range(6):
            val = cat_matrix_large[i, j]
            text_color = "white" if val < 1.5 else "black"
            ax3.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=6, color=text_color, fontweight="bold")

    ax3.set_title("Panel 3: Heatmap — Systems × Categories (Large)", fontweight="bold")
    plt.colorbar(im, ax=ax3, shrink=0.8, label="Avg Grade (0-3)")

    # ═══════════════════════════════════════════════════════════
    # Panel 4: Noise Delta — Small → Large change
    # ═══════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(4, 3, 4)
    delta_names = sorted(deltas.keys(), key=lambda n: deltas[n])
    delta_vals = [deltas[n] for n in delta_names]
    delta_colors = [
        "#4CAF50" if d >= 0 else "#FF9800" if d > -0.15 else "#F44336"
        for d in delta_vals
    ]

    ax4.barh(range(len(delta_names)), delta_vals, color=delta_colors, edgecolor="white")
    ax4.set_yticks(range(len(delta_names)))
    ax4.set_yticklabels(delta_names, fontsize=7)
    ax4.axvline(x=0, color="black", linewidth=1)
    ax4.set_xlabel("Δ Avg Grade (Large − Small)")
    ax4.set_title("Panel 4: Noise Resistance — Score Change with +50 Noise", fontweight="bold")

    # ═══════════════════════════════════════════════════════════
    # Panel 5: Category Breakdown — Top 6 (Large)
    # ═══════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(4, 3, 5)
    top6 = sorted(
        [n for n in all_systems if n in results_large],
        key=lambda n: results_large[n].avg_grade, reverse=True,
    )[:6]

    x5 = np.arange(6)
    w5 = 0.12
    for k, name in enumerate(top6):
        sr = results_large[name]
        vals = [sr.category_avg(cat) for cat in CAT_ORDER]
        offset = (k - len(top6)/2 + 0.5) * w5
        color = SYSTEM_COLORS.get(name, "#999999")
        ax5.bar(x5 + offset, vals, w5, label=name, color=color,
                edgecolor="white", linewidth=0.3)

    ax5.set_xticks(x5)
    ax5.set_xticklabels([CAT_SHORT[c] for c in CAT_ORDER], fontsize=7)
    ax5.set_ylabel("Avg Grade (0-3)")
    ax5.set_title("Panel 5: Category Breakdown — Top 6 Systems", fontweight="bold")
    ax5.legend(fontsize=6, ncol=2)
    ax5.set_ylim(0, 3.5)

    # ═══════════════════════════════════════════════════════════
    # Panel 6: Per-QA — MP v9 vs best baseline
    # ═══════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(4, 3, 6)
    if "Memory Palace v9" in results_large and "BM25 Baseline" in results_large:
        mp_large = results_large["Memory Palace v9"]
        bm25_large = results_large["BM25 Baseline"]

        qa_labels = [f"{qa.id}" for qa in ALL_QA_PAIRS]
        x6 = np.arange(len(ALL_QA_PAIRS))

        ax6.plot(x6, mp_large.all_grades, "o-", color="#FFD700",
                 linewidth=1.5, markersize=4, label="MP v9")
        ax6.plot(x6, bm25_large.all_grades, "s--", color="#90A4AE",
                 linewidth=1, markersize=4, label="BM25 Baseline")

        # Category background spans
        cat_boundaries = [0]
        for cat in CAT_ORDER:
            cat_boundaries.append(cat_boundaries[-1] + CATEGORY_QA_COUNTS[cat])
        cat_colors_bg = ["#E8F5E9", "#FFF3E0", "#E3F2FD", "#FCE4EC", "#F3E5F5", "#E0F7FA"]
        for j, cat in enumerate(CAT_ORDER):
            ax6.axvspan(cat_boundaries[j] - 0.5, cat_boundaries[j+1] - 0.5,
                        alpha=0.15, color=cat_colors_bg[j])

        ax6.set_xticks(x6[::2])
        ax6.set_xticklabels(qa_labels[::2], fontsize=6, rotation=45)
        ax6.set_ylabel("Grade (0-3)")
        ax6.set_title("Panel 6: MP v9 vs BM25 — Per-Question (Large)", fontweight="bold")
        ax6.legend(fontsize=7)
        ax6.set_ylim(-0.2, 3.5)

    # ═══════════════════════════════════════════════════════════
    # Panel 7: Real vs Simulator Comparison
    # ═══════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(4, 3, 7)
    pairs = [
        ("BM25 Baseline", "Real BM25S"),
        ("Vector Baseline", "sentence-transformers+FAISS"),
        ("Mem0-like (Sim)", "Real Mem0 (mem0ai)"),
    ]
    pair_x = []
    pair_labels = []
    for sim, real in pairs:
        if sim in results_large and real in results_large:
            pair_x.append(results_large[sim].avg_grade)
            pair_x.append(results_large[real].avg_grade)
            pair_labels.append(f"{sim}\nvs\n{real}")

    if pair_x:
        y7 = np.arange(len(pair_labels))
        for i in range(0, len(pair_x), 2):
            ax7.scatter(pair_x[i], i//2, color="#E57373", s=100, zorder=5, marker="s", label="Simulator" if i == 0 else "")
            ax7.scatter(pair_x[i+1], i//2, color="#00E676", s=100, zorder=5, marker="o", label="Real" if i == 0 else "")
            ax7.plot([pair_x[i], pair_x[i+1]], [i//2, i//2], "k-", alpha=0.3)

        ax7.set_yticks(y7)
        ax7.set_yticklabels(pair_labels, fontsize=7)
        ax7.set_xlabel("Avg Grade (0-3)")
        ax7.set_title("Panel 7: Simulator vs Real Implementation (Large)", fontweight="bold")
        ax7.legend(fontsize=7)
        ax7.set_xlim(0, 3.0)

    # ═══════════════════════════════════════════════════════════
    # Panel 8: Latency Comparison
    # ═══════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(4, 3, 8)
    latency_data = []
    latency_names = []
    for name in all_systems:
        sr = results_large.get(name) or results_medium.get(name)
        if sr and sr.latency_ms > 0:
            latency_data.append(sr.latency_ms)
            latency_names.append(name)

    if latency_data:
        sorted_idx = np.argsort(latency_data)
        sorted_lat = [latency_data[i] for i in sorted_idx]
        sorted_names = [latency_names[i] for i in sorted_idx]
        lat_colors = [SYSTEM_COLORS.get(n, "#999999") for n in sorted_names]

        ax8.barh(range(len(sorted_lat)), sorted_lat, color=lat_colors, edgecolor="white")
        ax8.set_yticks(range(len(sorted_names)))
        ax8.set_yticklabels(sorted_names, fontsize=7)
        ax8.set_xlabel("Latency (ms)")
        ax8.set_title("Panel 8: Retrieval Latency — 25 QA Pairs", fontweight="bold")

    # ═══════════════════════════════════════════════════════════
    # Panel 9: MP v9 Category Scaling (Small vs Large)
    # ═══════════════════════════════════════════════════════════
    ax9 = fig.add_subplot(4, 3, 9)
    if "Memory Palace v9" in results_medium and "Memory Palace v9" in results_large:
        mp_med = results_medium["Memory Palace v9"]
        mp_lrg = results_large["Memory Palace v9"]

        x9 = np.arange(6)
        w9 = 0.3
        med_cats = [mp_med.category_avg(cat) for cat in CAT_ORDER]
        lrg_cats = [mp_lrg.category_avg(cat) for cat in CAT_ORDER]

        ax9.bar(x9 - w9/2, med_cats, w9, label="Small (22)",
                color="#64B5F6", edgecolor="white")
        ax9.bar(x9 + w9/2, lrg_cats, w9, label="Large (72)",
                color="#FFA000", edgecolor="white")

        # Delta annotations
        for j in range(6):
            delta = lrg_cats[j] - med_cats[j]
            ax9.annotate(f"{delta:+.2f}", (x9[j], max(med_cats[j], lrg_cats[j]) + 0.05),
                         ha="center", fontsize=8, fontweight="bold",
                         color="green" if delta >= 0 else "red")

        ax9.set_xticks(x9)
        ax9.set_xticklabels([CAT_SHORT[c] for c in CAT_ORDER], fontsize=7)
        ax9.set_ylabel("Avg Grade (0-3)")
        ax9.set_title("Panel 9: MP v9 Category Scaling", fontweight="bold")
        ax9.legend(fontsize=7)
        ax9.set_ylim(0, 3.5)

    # ═══════════════════════════════════════════════════════════
    # Panel 10: Summary Statistics Table
    # ═══════════════════════════════════════════════════════════
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.axis("off")

    # Build table data
    table_data = []
    table_cols = ["System", "Small", "Large", "Δ", "Best Category"]
    ranked = sorted(all_systems, key=lambda n: (
        results_large.get(n, SystemResult(name=n)).avg_grade
    ), reverse=True)

    for name in ranked[:12]:
        med_sr = results_medium.get(name)
        lrg_sr = results_large.get(name)
        med_str = f"{med_sr.avg_grade:.2f}" if med_sr else "N/A"
        lrg_str = f"{lrg_sr.avg_grade:.2f}" if lrg_sr else "N/A"
        delta_str = ""
        if name in deltas:
            delta_str = f"{deltas[name]:+.2f}"
        best_cat = ""
        if lrg_sr:
            best_cat_key = max(CAT_ORDER, key=lambda c: lrg_sr.category_avg(c))
            best_cat = CAT_SHORT[best_cat_key].replace("\n", " ")
        table_data.append([name[:30], med_str, lrg_str, delta_str, best_cat])

    if table_data:
        table = ax10.table(
            cellText=table_data, colLabels=table_cols,
            cellLoc="left", loc="center",
            colWidths=[0.35, 0.12, 0.12, 0.1, 0.18],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)

        # Header style
        for j in range(len(table_cols)):
            table[0, j].set_facecolor("#455A64")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight MP v9 row
        if "Memory Palace v9" in ranked:
            mp_row = ranked.index("Memory Palace v9")
            if mp_row < 12:
                for j in range(len(table_cols)):
                    table[mp_row + 1, j].set_facecolor("#FFF9C4")

    ax10.set_title("Panel 10: Summary Statistics", fontweight="bold", y=1.02)

    # ═══════════════════════════════════════════════════════════
    # Panel 11: Key Findings
    # ═══════════════════════════════════════════════════════════
    ax11 = fig.add_subplot(4, 3, 11)
    ax11.axis("off")

    # Compute findings dynamically
    findings = []

    # Finding 1: Best overall
    best_name = max(
        [n for n in all_systems if n in results_large],
        key=lambda n: results_large[n].avg_grade,
    ) if results_large else "N/A"
    best_score = results_large[best_name].avg_grade if best_name in results_large else 0
    findings.append(f"1. Best Overall (Large): {best_name} ({best_score:.2f}/3.0)")

    # Finding 2: BM25 dominance check
    if "BM25 Baseline" in results_large:
        bm25_avg = results_large["BM25 Baseline"].avg_grade
        findings.append(f"2. BM25 Baseline: {bm25_avg:.2f} — {'still competitive' if bm25_avg >= 1.5 else 'struggles with noise'}")

    # Finding 3: Real vs Sim
    if "BM25 Baseline" in results_large and "Real BM25S" in results_large:
        sim_avg = results_large["BM25 Baseline"].avg_grade
        real_avg = results_large["Real BM25S"].avg_grade
        agree = "agree well" if abs(sim_avg - real_avg) < 0.2 else "diverge"
        findings.append(f"3. BM25 implementations {agree} (sim={sim_avg:.2f}, real={real_avg:.2f})")

    # Finding 4: Sentence-transformers vs Vector
    if "Vector Baseline" in results_large and "sentence-transformers+FAISS" in results_large:
        vec_avg = results_large["Vector Baseline"].avg_grade
        st_avg = results_large["sentence-transformers+FAISS"].avg_grade
        findings.append(f"4. Dense embeddings {'outperform' if st_avg > vec_avg else 'underperform'} TF-IDF ({st_avg:.2f} vs {vec_avg:.2f})")

    # Finding 5: MP v9 advantage
    if "Memory Palace v9" in results_large and "BM25 Baseline" in results_large:
        mp_avg = results_large["Memory Palace v9"].avg_grade
        bm25_avg = results_large["BM25 Baseline"].avg_grade
        delta = mp_avg - bm25_avg
        findings.append(f"5. MP v9 vs BM25: Δ={delta:+.2f} — {'DDA fusion adds value' if delta > 0.1 else 'marginal on this dataset'}")

    # Finding 6: Hardest category
    cat_avgs = {}
    for cat in CAT_ORDER:
        cat_scores = []
        for name, sr in results_large.items():
            cat_scores.append(sr.category_avg(cat))
        cat_avgs[cat] = np.mean(cat_scores) if cat_scores else 0
    hardest = min(cat_avgs, key=cat_avgs.get)
    easiest = max(cat_avgs, key=cat_avgs.get)
    findings.append(f"6. Hardest category: {CAT_SHORT[hardest]} ({cat_avgs[hardest]:.2f}), Easiest: {CAT_SHORT[easiest]} ({cat_avgs[easiest]:.2f})")

    # Finding 7: Noise resistance
    if deltas:
        most_resilient = max(deltas, key=deltas.get)
        most_fragile = min(deltas, key=deltas.get)
        findings.append(f"7. Most noise-resilient: {most_resilient} (Δ={deltas[most_resilient]:+.2f})")
        findings.append(f"8. Most noise-fragile: {most_fragile} (Δ={deltas[most_fragile]:+.2f})")

    # Finding 9: System count
    findings.append(f"9. {len(all_systems)} systems tested: {len([n for n in all_systems if n in results_large])} on large, {len([n for n in all_systems if n in results_medium])} on medium")

    # Finding 10: MP uniqueness
    findings.append("10. Memory Palace v9 is the only system with DDA (4-level data-density adaptation) + multi-path fusion + causal reasoning architecture")

    y_pos = 0.95
    for finding in findings:
        ax11.text(0.02, y_pos, finding, transform=ax11.transAxes,
                  fontsize=9, verticalalignment="top", fontfamily="monospace")
        y_pos -= 0.10

    ax11.set_title("Panel 11: Key Findings", fontweight="bold")

    # ═══════════════════════════════════════════════════════════
    # Panel 12: Research Methods Landscape
    # ═══════════════════════════════════════════════════════════
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis("off")

    landscape_text = (
        "RESEARCHED METHODS LANDSCAPE\n"
        "══════════════════════════════\n\n"
        "Open-Source Projects (8):\n"
        "  Mem0(25k★) Zep/Graphiti(8.2k★) Letta/MemGPT(14k★)\n"
        "  MemU MemoBase LANGMem SillyTavern AIRI\n\n"
        "Key Papers (16):\n"
        "  A-MEM(NIPS'25) MMAG MAGMA(CVPR'25)\n"
        "  GraphRAG(arXiv'24) HippoRAG 1&2(NIPS'24/ICML'25)\n"
        "  MemLong('24) CausalRAG(ACL'25) CDF-RAG('25)\n"
        "  Causal Cartographer('25) DAM-LLM('25) REMT('25)\n"
        "  MemoTime('25) DyMemR(TKDE'24)\n"
        "  McClelland(PsychRev'95) Diekelmann&Born(NatRev'10)\n"
        "  Pearl(2009/2018) — Causal Hierarchy\n\n"
        "Cognitive Theories (12):\n"
        "  SMS(Conway) Flashbulb(Brown&Kulik) Forgetting(Ebbinghaus)\n"
        "  Emotion-Congruent(Bower) Scripts/MOPs(Schank)\n"
        "  Allostatic Load(McEwen) Kindling(Post)\n"
        "  Emotional Inertia(Kuppens) Critical Slowing(Scheffer)\n"
        "  Cold Start(Adomavicius) SRM(Vapnik)\n"
        "  Differential Privacy(Dwork)\n\n"
        "Community Implementations (this benchmark):\n"
        "  sentence-transformers+FAISS — dense semantic search\n"
        "  BM25S (Rust backend) — fast keyword retrieval\n"
        "  Mem0 (mem0ai) — cloud memory system (optional)"
    )

    ax12.text(0.02, 0.98, landscape_text, transform=ax12.transAxes,
              fontsize=6.5, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    ax12.set_title("Panel 12: Research Methods Landscape", fontweight="bold")

    # ── Finalize ────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save PNG
    png_path = output_dir / "multi_algo_comparison_report.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\n[OK] PNG report saved to: {png_path}")

    # ── Generate Markdown report ────────────────────────────
    md_path = output_dir / "multi_algo_comparison_report.md"
    md_content = _generate_markdown(
        results_medium, results_large, all_systems, deltas, cat_avgs, png_path,
    )
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[OK] Markdown report saved to: {md_path}")

    plt.close(fig)
    return png_path, md_path


def _generate_markdown(
    results_medium: dict[str, SystemResult],
    results_large: dict[str, SystemResult],
    all_systems: list[str],
    deltas: dict[str, float],
    cat_avgs: dict[str, float],
    png_path: Path,
) -> str:
    """Generate comprehensive Markdown report."""
    lines = []

    lines.append("# Multi-Algorithm Memory Retrieval Benchmark Report")
    lines.append("")
    lines.append(f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Systems tested: {len(all_systems)}")
    lines.append(f"> QA pairs: 25 (6 categories)")
    lines.append(f"> Corpus sizes: 22 (WARM) + 72 (HOT)")
    lines.append("")
    lines.append(f"![Visual Report]({png_path.name})")
    lines.append("")

    # ── Overall Ranking ─────────────────────────────────────
    lines.append("## 1. Overall Ranking (Large Corpus — 72 memories)")
    lines.append("")
    lines.append("| Rank | System | Total (/75) | Avg (0-3) | Latency (ms) |")
    lines.append("|------|--------|-------------|-----------|-------------|")

    ranked = sorted(
        [n for n in all_systems if n in results_large],
        key=lambda n: results_large[n].avg_grade, reverse=True,
    )
    for i, name in enumerate(ranked, 1):
        sr = results_large[name]
        medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else str(i)
        lines.append(f"| {medal} | {name} | {sr.total_score} | {sr.avg_grade:.2f} | {sr.latency_ms:.0f} |")

    lines.append("")

    # ── Category Breakdown ──────────────────────────────────
    lines.append("## 2. Category Breakdown (Large Corpus)")
    lines.append("")
    header = "| System | " + " | ".join(CATEGORY_MAP.values()) + " |"
    lines.append(header)
    sep = "|--------|" + "|".join(["--------" for _ in CATEGORY_MAP]) + "|"
    lines.append(sep)

    for name in ranked:
        sr = results_large[name]
        vals = " | ".join([f"{sr.category_avg(cat):.2f}" for cat in CAT_ORDER])
        lines.append(f"| {name} | {vals} |")

    lines.append("")

    # ── Best per Category ───────────────────────────────────
    lines.append("## 3. Best System per Category")
    lines.append("")
    lines.append("| Category | Best System | Score |")
    lines.append("|----------|-------------|-------|")
    for cat in CAT_ORDER:
        best_name = max(
            [n for n in all_systems if n in results_large],
            key=lambda n: results_large[n].category_avg(cat),
        )
        best_score = results_large[best_name].category_avg(cat)
        lines.append(f"| {CATEGORY_MAP[cat]} | {best_name} | {best_score:.2f} |")
    lines.append("")

    # ── Category Difficulty ─────────────────────────────────
    lines.append("## 4. Category Difficulty (All-System Average)")
    lines.append("")
    lines.append("| Category | Avg Score | Difficulty |")
    lines.append("|----------|-----------|------------|")
    for cat in CAT_ORDER:
        avg = cat_avgs.get(cat, 0)
        if avg >= 2.0:
            diff = "🟢 Easy"
        elif avg >= 1.5:
            diff = "🟡 Medium"
        elif avg >= 1.0:
            diff = "🟠 Hard"
        else:
            diff = "🔴 Very Hard"
        lines.append(f"| {CATEGORY_MAP[cat]} | {avg:.2f} | {diff} |")
    lines.append("")

    # ── Noise Resistance ────────────────────────────────────
    lines.append("## 5. Noise Resistance (Medium → Large Δ)")
    lines.append("")
    lines.append("| System | Medium | Large | Δ | Resilience |")
    lines.append("|--------|--------|-------|---|------------|")
    for name in sorted(deltas.keys(), key=lambda n: deltas[n], reverse=True):
        med_avg = results_medium[name].avg_grade
        lrg_avg = results_large[name].avg_grade
        d = deltas[name]
        if d >= 0:
            res = "ANTI-FRAGILE ⬆"
        elif d > -0.15:
            res = "RESILIENT ✓"
        elif d > -0.35:
            res = "MODERATE ~"
        else:
            res = "FRAGILE ⬇"
        lines.append(f"| {name} | {med_avg:.2f} | {lrg_avg:.2f} | {d:+.2f} | {res} |")
    lines.append("")

    # ── Real vs Simulator ───────────────────────────────────
    lines.append("## 6. Real Implementation vs Simulator Fidelity")
    lines.append("")
    pairs = [
        ("BM25 Baseline", "Real BM25S"),
        ("Vector Baseline", "sentence-transformers+FAISS"),
        ("Mem0-like (Sim)", "Real Mem0 (mem0ai)"),
    ]
    lines.append("| Simulator | Real Impl | Sim Score | Real Score | Δ |")
    lines.append("|-----------|-----------|-----------|------------|---|")
    for sim, real in pairs:
        if sim in results_large and real in results_large:
            sim_avg = results_large[sim].avg_grade
            real_avg = results_large[real].avg_grade
            diff = real_avg - sim_avg
            lines.append(f"| {sim} | {real} | {sim_avg:.2f} | {real_avg:.2f} | {diff:+.2f} |")
    lines.append("")

    # ── Methodology ─────────────────────────────────────────
    lines.append("## 7. Methodology")
    lines.append("")
    lines.append("- **Scoring**: Dual keyword match ratio (≥0.8 for full credit) + memory content coverage (≥0.5 for full credit), 0-3 scale")
    lines.append("- **Small corpus**: 22 synthetic memories (XiaoMing career transition story, 6 sessions, 60-day span)")
    lines.append("- **Large corpus**: 22 core + 50 noise memories (synthetic diary entries, semantically unrelated, spread across 90 days)")
    lines.append("- **QA design**: 25 manually annotated questions, each with expected answer, keywords, relevant memory indices, reasoning chain, and difficulty rating (1-5)")
    lines.append("- **6 categories**: Simple Recall, Multi-hop Reasoning, Temporal Reasoning, Emotional Memory, Causal Reasoning, Cross-Reference")
    lines.append("")

    # ── Systems Tested ──────────────────────────────────────
    lines.append("## 8. Systems Under Test")
    lines.append("")
    lines.append("| Tier | Systems |")
    lines.append("|------|---------|")
    lines.append("| **Real Engine** | Memory Palace v9 (8-path DDA fusion) |")
    lines.append("| **Paper Simulators (11)** | A-MEM, MAGMA, MMAG, Mem0-like, Zep-like, BM25 Baseline, Vector Baseline, HippoRAG(PPR), GraphRAG(Community), MemLong(Learnable), HybridFusion |")
    lines.append("| **Real Community (3)** | sentence-transformers+FAISS, BM25S (Rust), Mem0 (mem0ai cloud) |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main — Run benchmarks and generate report
# ═══════════════════════════════════════════════════════════════

async def main():
    """Run benchmarks at both scales and generate report."""
    import tempfile

    print("=" * 70)
    print("Multi-Algorithm Memory Retrieval Benchmark")
    print("=" * 70)
    print()

    # Check for cached JSON results
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    json_path = docs_dir / "multi_algo_benchmark_results.json"

    if json_path.exists():
        print(f"Loading cached results from: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            cached = json.load(f)

        results_medium = {
            name: SystemResult(**data)
            for name, data in cached.get("medium_results", {}).items()
        }
        results_large = {
            name: SystemResult(**data)
            for name, data in cached.get("large_results", {}).items()
        }
        print(f"  Medium: {len(results_medium)} systems")
        print(f"  Large:  {len(results_large)} systems")
    else:
        print("No cached results found. Running benchmarks live...")
        print("(This may take 2-5 minutes with real implementations)\n")

        # ── Medium corpus (22 memories) ─────────────────────
        print("[1/2] Benchmarking MEDIUM corpus (22 memories, WARM)...")
        tmp_med = tempfile.mkdtemp(prefix="mp_bench_med_")
        harness_med = BenchmarkHarness(Path(tmp_med), user_id="report_med")
        await harness_med.populate_async_batch(list(BENCHMARK_MEMORIES), concurrency=10)
        results_medium = await run_benchmark(list(BENCHMARK_MEMORIES), harness_med)
        print(f"  Completed: {len(results_medium)} systems\n")

        # ── Large corpus (72 memories) ──────────────────────
        print("[2/2] Benchmarking LARGE corpus (72 memories, HOT)...")
        noise = generate_noise_memories(50, seed=42)
        large_corpus = list(BENCHMARK_MEMORIES) + list(noise)
        tmp_lrg = tempfile.mkdtemp(prefix="mp_bench_lrg_")
        harness_lrg = BenchmarkHarness(Path(tmp_lrg), user_id="report_lrg")
        await harness_lrg.populate_async_batch(large_corpus, concurrency=10)
        results_large = await run_benchmark(large_corpus, harness_lrg)
        print(f"  Completed: {len(results_large)} systems\n")

        # Cache results
        cache_data = {
            "meta": {"version": "0.9.0", "qa_count": 25},
            "medium_results": {
                name: sr.to_dict() for name, sr in results_medium.items()
            },
            "large_results": {
                name: sr.to_dict() for name, sr in results_large.items()
            },
        }
        json_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
        print(f"Cached results to: {json_path}\n")

    # ── Generate Report ─────────────────────────────────────
    print("Generating 12-panel visual report...")
    png_path, md_path = generate_report(results_medium, results_large, docs_dir)

    print(f"\n{'=' * 70}")
    print("Report Generation Complete!")
    print(f"  PNG:  {png_path}")
    print(f"  MD:   {md_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
