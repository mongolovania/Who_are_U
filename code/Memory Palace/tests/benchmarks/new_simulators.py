# ============================================================
# New Algorithm Simulators — 新增论文算法模拟器
# v9 Enhancement (2026-06-10)
#
# 5 new paper-based simulators filling coverage gaps:
#   12. CausalRAGSimulator   — CausalRAG (ACL 2025): causal graph BFS
#   13. DAMLLMSimulator      — DAM-LLM (2025): dynamic emotional state EMA
#   14. MemoTimeSimulator    — MemoTime (2025): explicit time-indexed retrieval
#   15. DyMemRSimulator      — DyMemR (TKDE 2024): co-retrieval consolidation
#   16. REMTSimulator        — REMT (2025): emotion-weighted graph + edge RL
#
# All follow the unified contract:
#   answer(query, top_k=10) -> tuple[str, list[int], float]
# ============================================================

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import numpy as np

from tests.benchmarks.benchmark_dataset import BENCHMARK_MEMORIES, BenchmarkMemory
from tests.benchmarks.simulator_utils import (
    SharedBM25Index, _tokenize, _jaccard, _days_ago,
    _emotion_resonance, _extract_query_emotion, _infer_query_category,
)


# ── Causal pattern extraction for CausalRAG ──────────────────────

# Chinese causal patterns
_CAUSAL_PATTERNS = [
    (r"因为(.+?)所以", "cause"),        # 因为A所以B → A causes B
    (r"由于(.+?)[，,]", "cause"),
    (r"导致(.+?)[，。]", "effect"),
    (r"引起了(.+?)[，。]", "effect"),
    (r"原因是(.+?)[，。]", "cause"),
    (r"之所以.+?是因为(.+?)[，。]", "cause"),
    (r"如果(.+?)[，,]那么", "hypothetical"),
    (r"假如(.+?)[，,]", "hypothetical"),
    (r"要不是(.+?)[，,]", "counterfactual"),
    (r"让(.+?)更", "effect"),
    (r"使得(.+?)[，。]", "effect"),
    (r"造成了(.+?)[，。]", "effect"),
]

# English causal patterns (for mixed-language content)
_CAUSAL_PATTERNS_EN = [
    (r"because\s+(.+?)[,\.]", "cause"),
    (r"due to\s+(.+?)[,\.]", "cause"),
    (r"led to\s+(.+?)[,\.]", "effect"),
    (r"caused\s+(.+?)[,\.]", "effect"),
    (r"resulted in\s+(.+?)[,\.]", "effect"),
    (r"if\s+(.+?),", "hypothetical"),
]


def _extract_causal_phrases(text: str) -> list[tuple[str, str]]:
    """Extract (phrase, relation_type) from text using causal patterns."""
    results = []
    for pattern, rel_type in _CAUSAL_PATTERNS:
        for match in re.finditer(pattern, text):
            phrase = match.group(1).strip()
            if len(phrase) >= 2:
                results.append((phrase, rel_type))
    for pattern, rel_type in _CAUSAL_PATTERNS_EN:
        for match in re.finditer(pattern, text.lower()):
            phrase = match.group(1).strip()
            if len(phrase) >= 3:
                results.append((phrase, rel_type))
    return results


# ═══════════════════════════════════════════════════════════════
# 12. CausalRAG Simulator (ACL 2025)
#    Causal graph construction + BM25 + causal BFS traversal
# ═══════════════════════════════════════════════════════════════

class CausalRAGSimulator:
    """
    CausalRAG: Build a directed causal graph from memory texts,
    then use BM25 + causal edge traversal for retrieval.

    Core mechanism:
    1. Extract cause-effect patterns from all memory texts
    2. Build directed causal graph (A causes B)
    3. BM25 for content scoring
    4. Traverse outgoing causal edges from top candidates (depth=2)
    5. Boost causally-linked memories by depth_factor * 0.30
    """

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._causal_graph: nx.DiGraph = nx.DiGraph()
        self._build_causal_graph()

    def _build_causal_graph(self):
        """Build directed causal graph from text patterns."""
        n = len(self.memories)
        for i in range(n):
            self._causal_graph.add_node(i)

        # Extract causal phrases and link memories
        for i, mem in enumerate(self.memories):
            phrases = _extract_causal_phrases(mem.content)
            for phrase, rel_type in phrases:
                # Find which other memories contain this phrase
                for j, mem2 in enumerate(self.memories):
                    if i == j:
                        continue
                    if phrase in mem2.content:
                        if rel_type in ("cause", "effect"):
                            self._causal_graph.add_edge(i, j, weight=0.6, rel_type=rel_type)
                        elif rel_type in ("hypothetical", "counterfactual"):
                            self._causal_graph.add_edge(i, j, weight=0.4, rel_type=rel_type)

        # Add weak causal links based on temporal proximity + keyword overlap
        for i in range(n):
            ti = self.bm25.get_document_tokens(i)
            di = _days_ago(self.memories[i].created)
            for j in range(i + 1, n):
                tj = self.bm25.get_document_tokens(j)
                dj = _days_ago(self.memories[j].created)
                jac = _jaccard(ti, tj)
                # Memories close in time with some overlap → potential causal link
                if jac > 0.08 and abs(di - dj) < 10:
                    if not self._causal_graph.has_edge(i, j):
                        self._causal_graph.add_edge(i, j, weight=0.2, rel_type="temporal")
                    if not self._causal_graph.has_edge(j, i):
                        self._causal_graph.add_edge(j, i, weight=0.2, rel_type="temporal")

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = self.bm25.search(query)
        if not bm25_results:
            return "未找到相关信息。", [], 0.0

        scores: dict[int, float] = {}
        visited: set[int] = set()

        # Phase 1: BM25 scoring
        for idx, bm_score in bm25_results:
            scores[idx] = bm_score * 0.60
            visited.add(idx)

        # Phase 2: Causal BFS from top candidates (depth=2)
        top_candidates = [idx for idx, s in bm25_results[:5] if s > 0.1]
        for start_node in top_candidates:
            # BFS with depth limit
            queue = [(start_node, 0)]
            bfs_visited = {start_node}
            while queue:
                node, depth = queue.pop(0)
                if depth >= 2:
                    continue
                for succ in self._causal_graph.successors(node):
                    if succ in bfs_visited:
                        continue
                    bfs_visited.add(succ)
                    edge_weight = self._causal_graph[node][succ].get("weight", 0.2)
                    rel_type = self._causal_graph[node][succ].get("rel_type", "unknown")
                    depth_factor = 0.30 if depth == 0 else 0.15
                    rel_boost = 0.10 if rel_type == "cause" else 0.05
                    causal_score = depth_factor + rel_boost + edge_weight * 0.10
                    scores[succ] = max(scores.get(succ, 0), causal_score)
                    queue.append((succ, depth + 1))

        # Phase 3: Add predecessor context (what led to relevant memories?)
        for idx in list(scores.keys()):
            for pred in self._causal_graph.predecessors(idx):
                if pred not in scores:
                    edge_weight = self._causal_graph[pred][idx].get("weight", 0.2)
                    scores[pred] = max(scores.get(pred, 0), 0.15 + edge_weight * 0.10)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:top_k]]
        if not top_indices:
            top_indices = [i for i, _ in bm25_results[:top_k]]

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 13. DAM-LLM Simulator (2025)
#    Dynamic Affective Memory — EMA emotional state + congruence
# ═══════════════════════════════════════════════════════════════

class DAMLLMSimulator:
    """
    DAM-LLM: Dynamic emotional state tracking via EMA,
    weighted retrieval by emotional congruence.

    Core mechanism:
    1. Maintain a dynamic emotional state vector (valence_hat, arousal_hat)
    2. On each query, update state via EMA: state_new = 0.7*state_old + 0.3*query_emotion
    3. Score memories by emotional congruence (Russell circumplex distance)
    4. Blend: final = 0.50 * BM25 + 0.35 * emotion_congruence + 0.15 * importance
    """

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Dynamic state initialized to neutral
        self._valence_hat = 0.5
        self._arousal_hat = 0.3
        self._query_count = 0

    def _update_state(self, query: str):
        """EMA update of emotional state from query."""
        q_val, q_ar = _extract_query_emotion(query)
        alpha = 0.3  # learning rate for emotional state
        self._valence_hat = 0.7 * self._valence_hat + alpha * q_val
        self._arousal_hat = 0.7 * self._arousal_hat + alpha * q_ar
        # Clamp to valid range
        self._valence_hat = max(0.0, min(1.0, self._valence_hat))
        self._arousal_hat = max(0.0, min(1.0, self._arousal_hat))
        self._query_count += 1

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        self._update_state(query)
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        query_cat = _infer_query_category(query)

        scores = []
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            # Emotional congruence with current dynamic state
            emotion_congruence = _emotion_resonance(
                self._valence_hat, self._arousal_hat,
                mem.valence, mem.arousal,
            )

            # Importance normalization
            importance_norm = mem.importance / 10.0

            # DAM-LLM blend
            if query_cat == "emotional":
                # Boost emotion weight for emotional queries
                final = content_score * 0.35 + emotion_congruence * 0.45 + importance_norm * 0.20
            else:
                final = content_score * 0.50 + emotion_congruence * 0.35 + importance_norm * 0.15

            scores.append((i, final))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, scores[0][1] if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 14. MemoTime Simulator (2025)
#    Explicit time-indexed retrieval with temporal operator parsing
# ═══════════════════════════════════════════════════════════════

class MemoTimeSimulator:
    """
    MemoTime: Build explicit time-indexed structure,
    parse temporal operators from query, and filter/boost by time.

    Core mechanism:
    1. Sort all memories by created timestamp
    2. Extract temporal operators from query:
       "之前" (before), "之后" (after), "之间" (between),
       "多久" (how long), explicit day counts, ordering clues
    3. For temporal queries: boost memories within operator-constrained window
    4. For non-temporal queries: exponential recency decay blended with BM25
    """

    # Temporal operator patterns
    TEMPORAL_OPS = [
        (r"(\d+)天前", "exact_days_ago"),
        (r"(\d+)天[以之]?前", "exact_days_ago"),
        (r"(\d+)个?月前", "months_ago"),
        (r"(\d+)周前", "weeks_ago"),
        (r"(\d+)天[以之]?后", "days_after"),
        (r"之前", "before"),
        (r"以后", "after"),
        (r"之后", "after"),
        (r"之间", "between"),
        (r"多久", "duration"),
        (r"多长时间", "duration"),
        (r"持续了", "duration"),
        (r"间隔", "interval"),
        (r"什么时候", "when"),
        (r"什么时候开始", "when_start"),
        (r"什么时候好", "when_end"),
        (r"时间顺序", "ordering"),
        (r"按时间", "ordering"),
        (r"顺序排列", "ordering"),
    ]

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Time-indexed structure: sorted by creation time
        self._time_index: list[tuple[int, float]] = []
        self._build_time_index()

    def _build_time_index(self):
        """Build sorted time index (index, days_ago)."""
        for i, mem in enumerate(self.memories):
            days = _days_ago(mem.created)
            self._time_index.append((i, days))
        self._time_index.sort(key=lambda x: x[1])  # most recent first

    def _parse_temporal_operators(self, query: str) -> dict:
        """Parse temporal constraints from query text."""
        ops = {
            "has_temporal": False,
            "exact_days": None,
            "before": False,
            "after": False,
            "between": False,
            "duration": False,
            "ordering": False,
            "days_range": None,
        }
        for pattern, op_type in self.TEMPORAL_OPS:
            match = re.search(pattern, query)
            if match:
                ops["has_temporal"] = True
                if op_type == "exact_days_ago":
                    ops["exact_days"] = int(match.group(1))
                elif op_type == "months_ago":
                    ops["exact_days"] = int(match.group(1)) * 30
                elif op_type == "weeks_ago":
                    ops["exact_days"] = int(match.group(1)) * 7
                elif op_type in ("before",):
                    ops["before"] = True
                elif op_type in ("after", "days_after"):
                    ops["after"] = True
                elif op_type == "between":
                    ops["between"] = True
                elif op_type == "duration":
                    ops["duration"] = True
                elif op_type == "ordering":
                    ops["ordering"] = True
                elif op_type == "when":
                    ops["has_temporal"] = True
        return ops

    def _temporal_window_score(self, mem_idx: int, days_ago: float, ops: dict) -> float:
        """Score a memory based on temporal operator constraints."""
        score = 0.0

        if ops.get("exact_days"):
            target = ops["exact_days"]
            # Gaussian decay around target day
            diff = abs(days_ago - target)
            sigma = max(3, target * 0.2)  # 20% tolerance
            score = max(0.0, math.exp(-0.5 * (diff / sigma) ** 2))
        elif ops.get("before"):
            # Memories older than some implicit reference
            score = max(0.0, min(1.0, days_ago / 60.0))
        elif ops.get("after"):
            # Newer memories preferred
            score = max(0.0, 1.0 - days_ago / 60.0)
        elif ops.get("between"):
            # Middle-range memories
            score = max(0.0, 1.0 - abs(days_ago - 30) / 30.0)
        else:
            # Default recency scoring
            score = max(0.1, 1.0 - days_ago / 90.0)

        return score

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        ops = self._parse_temporal_operators(query)
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        query_cat = _infer_query_category(query)

        scores = []
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            days = _days_ago(mem.created)
            temporal_s = self._temporal_window_score(i, days, ops)

            if ops["has_temporal"] or query_cat == "temporal":
                # Temporal-heavy blend for temporal queries
                final = content_score * 0.45 + temporal_s * 0.40 + (mem.importance / 10.0) * 0.15
            else:
                # Standard blend with recency
                final = content_score * 0.60 + temporal_s * 0.30 + (mem.importance / 10.0) * 0.10

            scores.append((i, final))

        # Ordering bonus: re-rank by actual timestamp order if ordering query
        if ops.get("ordering"):
            scores.sort(key=lambda x: _days_ago(self.memories[x[0]].created))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, scores[0][1] if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 15. DyMemR Simulator (IEEE TKDE 2024)
#    Dynamic Memory-enhanced Retrieval with co-retrieval consolidation
# ═══════════════════════════════════════════════════════════════

class DyMemRSimulator:
    """
    DyMemR: Co-retrieval consolidation mechanism.
    Memory pairs frequently retrieved together are "consolidated"
    into virtual nodes that boost both members.

    Core mechanism:
    1. Maintain a co-retrieval counter matrix
    2. When co-retrieval count >= 3, create "consolidation boost"
    3. Retrieve: BM25 + recency decay + consolidation boost
    4. Consolidation is query-adaptive (accumulates over session)
    """

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Co-retrieval counter (i, j) -> count
        self._co_retrieval: dict[tuple[int, int], int] = defaultdict(int)
        # Consolidated pairs (set of frozen pairs)
        self._consolidated_pairs: set[tuple[int, int]] = set()
        self._query_history: list[set[int]] = []

    def _record_co_retrieval(self, retrieved_indices: list[int]):
        """Record co-retrieved pairs and update consolidation."""
        retrieved_set = set(retrieved_indices[:8])
        self._query_history.append(retrieved_set)

        # Update co-retrieval counters
        indices_list = list(retrieved_set)
        for i in range(len(indices_list)):
            for j in range(i + 1, len(indices_list)):
                pair = (min(indices_list[i], indices_list[j]),
                        max(indices_list[i], indices_list[j]))
                self._co_retrieval[pair] += 1
                # Consolidate if threshold reached
                if self._co_retrieval[pair] >= 3:
                    self._consolidated_pairs.add(pair)

    def _consolidation_boost(self, idx: int, other_retrieved: set[int]) -> float:
        """Boost score based on consolidated pairs."""
        boost = 0.0
        for other in other_retrieved:
            pair = (min(idx, other), max(idx, other))
            if pair in self._consolidated_pairs:
                boost += 0.15
            elif self._co_retrieval.get(pair, 0) >= 2:
                boost += 0.08
        return min(0.5, boost)  # Cap at 0.5

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        if not bm25_results:
            return "未找到相关信息。", [], 0.0

        scores = []
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            # Recency decay
            days = _days_ago(mem.created)
            recency = max(0.1, 1.0 - days / 90.0)

            # Importance
            importance_norm = mem.importance / 10.0

            # DyMemR blend (without consolidation initially)
            final = content_score * 0.55 + recency * 0.25 + importance_norm * 0.20
            scores.append((i, final))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]

        # Apply consolidation boost retroactively
        retrieved_set = set(top_indices)
        consolidated_scores = []
        for i, base_score in scores:
            boost = self._consolidation_boost(i, retrieved_set)
            consolidated_scores.append((i, base_score + boost))

        consolidated_scores.sort(key=lambda x: x[1], reverse=True)
        final_indices = [i for i, _ in consolidated_scores[:top_k]]

        # Record this retrieval for future consolidation
        self._record_co_retrieval(final_indices)

        if not final_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in final_indices]
        return " | ".join(contexts), final_indices, consolidated_scores[0][1] if consolidated_scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 16. REMT Simulator (2025)
#    Real-time Editable Memory Topology — emotion-weighted graph
#    with edge reinforcement learning on each retrieval
# ═══════════════════════════════════════════════════════════════

class REMTSimulator:
    """
    REMT: Emotion-weighted memory graph with edge reinforcement.
    Edge weights update on each retrieval (RL proxy), and
    2-hop neighbor expansion boosts emotionally similar memories.

    Core mechanism:
    1. Build emotion-weighted graph: edge_w = 1 - |valence_diff|*0.7 + tag_overlap*0.3
    2. BM25 content scoring
    3. 2-hop neighbor expansion from top candidates
    4. Boost neighbor by edge_weight * 0.25
    5. Edge reinforcement: edge weight += 0.05 on co-retrieval
    """

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._graph: nx.Graph = nx.Graph()
        self._build_emotion_graph()

    def _build_emotion_graph(self):
        """Build emotion-weighted graph with initial edge weights."""
        n = len(self.memories)
        for i in range(n):
            self._graph.add_node(i)

        for i in range(n):
            mi = self.memories[i]
            for j in range(i + 1, n):
                mj = self.memories[j]
                # Emotion-based weight
                valence_diff = abs(mi.valence - mj.valence)
                arousal_diff = abs(mi.arousal - mj.arousal)
                emotion_sim = 1.0 - (valence_diff * 0.5 + arousal_diff * 0.3)
                emotion_sim = max(0.1, emotion_sim)

                # Tag overlap
                tag_overlap = len(set(mi.tags) & set(mj.tags))
                tag_sim = min(1.0, tag_overlap * 0.3)

                # Content similarity via BM25 token Jaccard
                ti = self.bm25.get_document_tokens(i)
                tj = self.bm25.get_document_tokens(j)
                content_sim = _jaccard(ti, tj)

                # Combined weight
                weight = emotion_sim * 0.35 + tag_sim + content_sim * 0.35
                if weight > 0.12:
                    self._graph.add_edge(i, j, weight=weight)

        # Add typed emotion edges
        for i in range(n):
            mi = self.memories[i]
            for j in range(i + 1, n):
                mj = self.memories[j]
                if mi.memory_type == "emotion" and mj.memory_type == "emotion":
                    if self._graph.has_edge(i, j):
                        self._graph[i][j]["weight"] = min(1.0, self._graph[i][j]["weight"] + 0.15)
                    else:
                        self._graph.add_edge(i, j, weight=0.25)

    def _reinforce_edge(self, i: int, j: int):
        """Reinforcement learning proxy: strengthen co-retrieved edge."""
        if self._graph.has_edge(i, j):
            current = self._graph[i][j]["weight"]
            self._graph[i][j]["weight"] = min(1.0, current + 0.05)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        if not bm25_results:
            return "未找到相关信息。", [], 0.0

        q_val, q_ar = _extract_query_emotion(query)

        # Phase 1: Base scoring (BM25 + emotion match)
        base_scores: dict[int, float] = {}
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue
            emotion_s = _emotion_resonance(q_val, q_ar, mem.valence, mem.arousal)
            base_scores[i] = content_score * 0.60 + emotion_s * 0.25 + (mem.importance / 10.0) * 0.15

        # Phase 2: 2-hop neighbor expansion
        boosted_scores = dict(base_scores)
        top_candidates = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        for seed_idx, seed_score in top_candidates:
            # 1-hop neighbors
            if seed_idx not in self._graph:
                continue
            for neighbor in self._graph.neighbors(seed_idx):
                edge_w = self._graph[seed_idx][neighbor].get("weight", 0.2)
                boost = seed_score * edge_w * 0.25
                boosted_scores[neighbor] = max(boosted_scores.get(neighbor, 0), boost)

                # 2-hop neighbors
                if neighbor in self._graph:
                    for n2 in self._graph.neighbors(neighbor):
                        if n2 == seed_idx:
                            continue
                        edge_w2 = self._graph[neighbor][n2].get("weight", 0.15)
                        boost2 = boost * edge_w2 * 0.15
                        boosted_scores[n2] = max(boosted_scores.get(n2, 0), boost2)

        ranked = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:top_k]]
        if not top_indices:
            top_indices = [i for i, _ in bm25_results[:top_k]]

        # Edge reinforcement: strengthen edges between co-retrieved memories
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                self._reinforce_edge(top_indices[i], top_indices[j])

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0
