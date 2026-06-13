# ============================================================
# Algorithm Simulators — Enhanced paper-based algorithm replicas
# 算法模拟器 — 增强版论文机制复现（真实BM25+PPR+社区检测+情绪匹配）
#
# v8 Enhancements (2026-06-10):
#   - All simulators use REAL BM25 (rank_bm25) — no more keyword overlap proxy
#   - HippoRAG simulator uses REAL PPR via networkx.pagerank
#   - GraphRAG simulator uses REAL Louvain community detection via networkx
#   - Emotion matching uses RetrievalEngine's 8-category valence/arousal dicts
#   - MemLong simulator with feedback-driven learnable path weights
#   - Zep simulator with proper temporal edge expiry
#   - A-MEM simulator with BM25 + link graph traversal
#
# Simulators (19 total):
#   1.  AMEMSimulator          — A-MEM (NeurIPS 2025): Zettelkasten 2-stage
#   2.  MAGMASimulator         — MAGMA (CVPR 2025): SoM/ToM anchor marking
#   3.  MMAGSimulator          — MMAG: 5-layer hybrid memory
#   4.  Mem0Simulator          — Mem0-like: Vector+Graph+KV ADD-only
#   5.  ZepSimulator           — Zep/Graphiti-like: temporal KG + edge expiry
#   6.  BM25Baseline           — Pure keyword matching (lower bound)
#   7.  VectorBaseline         — TF-IDF semantic similarity (mid bound)
#   8.  HippoRAGSimulator      — HippoRAG (NeurIPS 2024): PPR on personalized graph
#   9.  GraphRAGSimulator      — MS GraphRAG (arXiv 2024): community detection + summary
#   10. MemLongSimulator       — MemLong (2024): learnable retrieval path weights
#   11. HybridFusionSim        — 8-path fusion baseline (no-DDA, fixed weights)
#   12. CausalRAGSimulator     — CausalRAG (ACL 2025): causal graph + BFS traversal
#   13. DAMLLMSimulator        — DAM-LLM (2025): dynamic emotional state EMA
#   14. MemoTimeSimulator      — MemoTime (2025): time-indexed + operator parsing
#   15. DyMemRSimulator        — DyMemR (TKDE 2024): co-retrieval consolidation
#   16. REMTSimulator          — REMT (2025): emotion-weighted graph + edge RL
#   17. GenerativeAgentsSim    — Park et al. 2023: recency×importance×relevance
#   18. RAPTORSim              — RAPTOR (2024): recursive clustering tree
#   19. CrewAISim              — CrewAI: composite scoring + 5 cognitive ops
# ============================================================

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from tests.benchmarks.benchmark_dataset import BENCHMARK_MEMORIES, BenchmarkMemory
from tests.benchmarks.simulator_utils import (
    SharedBM25Index, _tokenize, _jaccard, _days_ago,
    _emotion_resonance, _extract_query_emotion, _infer_query_category,
    _keyword_overlap_score,
)
from tests.benchmarks.new_simulators import (
    CausalRAGSimulator, DAMLLMSimulator, MemoTimeSimulator,
    DyMemRSimulator, REMTSimulator,
)
from tests.benchmarks.community_simulators import (
    GenerativeAgentsSim, RAPTORSim, CrewAISim,
)


# ═══════════════════════════════════════════════════════════════
# 1. A-MEM Simulator (NeurIPS 2025) — ENHANCED
#    BM25-powered coarse screening + real link graph traversal
# ═══════════════════════════════════════════════════════════════

@dataclass
class AMEMMemory:
    idx: int
    content: str
    tags: list[str]
    keywords: set = field(default_factory=set)
    links: list[int] = field(default_factory=list)
    link_strengths: dict[int, float] = field(default_factory=dict)


class AMEMSimulator:
    """A-MEM: Zettelkasten 2-stage retrieval with BM25 + link traversal."""

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self.amem_mems: list[AMEMMemory] = []
        self._build_index()

    def _build_index(self):
        for i, mem in enumerate(self.memories):
            keywords = self.bm25.get_document_tokens(i)
            self.amem_mems.append(AMEMMemory(
                idx=i, content=mem.content,
                tags=list(mem.tags), keywords=keywords,
            ))
        # Build initial links: Jaccard > 0.15
        for i, am in enumerate(self.amem_mems):
            for j, am2 in enumerate(self.amem_mems):
                if i >= j:
                    continue
                jac = _jaccard(am.keywords, am2.keywords)
                if jac > 0.15:
                    am.links.append(j)
                    am.link_strengths[j] = jac
                    am2.links.append(i)
                    am2.link_strengths[i] = jac

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        # Stage 1: BM25 coarse screening (was keyword overlap)
        bm25_results = self.bm25.search(query)
        candidates = [(i, s) for i, s in bm25_results if s > 0.05]

        if not candidates:
            return "未找到相关信息。", [], 0.0

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Stage 2: Fine linking — traverse links from top candidates
        linked_scores: dict[int, float] = {}
        visited: set[int] = set()
        for idx, score in candidates[:5]:
            if idx not in visited:
                linked_scores[idx] = max(linked_scores.get(idx, 0), score)
                visited.add(idx)
            am = self.amem_mems[idx]
            for link_idx in am.links:
                if link_idx not in visited:
                    link_score = score * 0.7 * am.link_strengths.get(link_idx, 0.3)
                    linked_scores[link_idx] = max(linked_scores.get(link_idx, 0), link_score)
                    visited.add(link_idx)

        ranked = sorted(linked_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:top_k]]
        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 2. MAGMA Simulator (CVPR 2025) — ENHANCED
#    BM25 content scoring + real emotion resonance from query
# ═══════════════════════════════════════════════════════════════

class MAGMASimulator:
    """MAGMA: SoM/ToM anchor marking + BM25 content + emotion matching."""

    SOM_LEVELS = {"EPHEMERAL": 1, "STANDARD": 2, "SIGNIFICANT": 3, "FLASHBULB": 4}
    TOM_LEVELS = {"FACTUAL": 1, "EXPERIENTIAL": 2, "REFLECTIVE": 3}

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._anchors: list[dict] = []
        self._build_anchors()

    def _classify_som(self, mem: BenchmarkMemory) -> str:
        score = 0
        if mem.importance >= 9:
            score += 2
        elif mem.importance >= 7:
            score += 1
        if mem.arousal >= 0.8:
            score += 2
        elif mem.arousal >= 0.5:
            score += 1
        if mem.valence <= 0.2 or mem.valence >= 0.8:
            score += 1
        if mem.memory_type == "milestone":
            score += 2
        elif mem.memory_type == "emotion":
            score += 1
        if score >= 5:
            return "FLASHBULB"
        elif score >= 3:
            return "SIGNIFICANT"
        elif score >= 1:
            return "STANDARD"
        return "EPHEMERAL"

    def _classify_tom(self, mem: BenchmarkMemory) -> str:
        if mem.memory_type == "emotion":
            return "EXPERIENTIAL"
        elif mem.memory_type == "decision":
            return "REFLECTIVE"
        elif mem.memory_type == "milestone":
            return "EXPERIENTIAL"
        return "FACTUAL"

    def _build_anchors(self):
        for mem in self.memories:
            self._anchors.append({
                "som": self._classify_som(mem),
                "som_level": self.SOM_LEVELS[self._classify_som(mem)],
                "tom": self._classify_tom(mem),
                "tom_level": self.TOM_LEVELS[self._classify_tom(mem)],
            })

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        # Extract query emotion for resonance matching
        q_val, q_ar = _extract_query_emotion(query)
        query_cat = _infer_query_category(query)
        bm25_results = self.bm25.search(query)

        scores = []
        for i, mem in enumerate(self.memories):
            anchor = self._anchors[i]

            # Content score from BM25
            content_score = 0.0
            for idx, s in bm25_results:
                if idx == i:
                    content_score = s
                    break

            # Anchor priority boost
            som_boost = anchor["som_level"] / 4.0 * 0.35
            tom_boost = anchor["tom_level"] / 3.0 * 0.15

            # Real emotion resonance (MAGMA enhancement)
            emotion_score = _emotion_resonance(q_val, q_ar, mem.valence, mem.arousal)
            emotion_boost = emotion_score * 0.15 if query_cat == "emotional" else emotion_score * 0.05

            # Query-type boost
            type_boost = 0.0
            if query_cat == "emotional" and mem.memory_type in ("emotion", "milestone"):
                type_boost = 0.15
            elif query_cat == "causal" and mem.memory_type in ("decision", "chat"):
                type_boost = 0.10

            total_score = content_score * 0.40 + som_boost + tom_boost + emotion_boost + type_boost
            if content_score > 0:
                scores.append((i, total_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, s in scores[:top_k] if s > 0.02]
        if not top_indices and scores:
            top_indices = [scores[0][0]]

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, scores[0][1] if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 3. MMAG Simulator — ENHANCED
#    BM25 content + 5-layer priority weighting
# ═══════════════════════════════════════════════════════════════

class MMAGSimulator:
    """MMAG: 5-layer hybrid memory with BM25 scoring."""

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._layers: dict[str, list[int]] = {
            "working": [], "short_term": [], "long_term": [],
            "episodic": [], "semantic": [],
        }
        self._classify_layers()

    def _classify_layers(self):
        for i, mem in enumerate(self.memories):
            days = _days_ago(mem.created)
            if mem.memory_type == "milestone":
                self._layers["episodic"].append(i)
            elif mem.tags and any(t in ["反思", "成长", "对比"] for t in mem.tags):
                self._layers["semantic"].append(i)
            elif days <= 3:
                self._layers["working"].append(i)
            elif days <= 14:
                self._layers["short_term"].append(i)
            else:
                self._layers["long_term"].append(i)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        layer_priority = ["episodic", "semantic", "working", "short_term", "long_term"]
        layer_weights = {"episodic": 1.0, "semantic": 0.9, "working": 0.8,
                         "short_term": 0.6, "long_term": 0.4}

        bm25_results = {idx: score for idx, score in self.bm25.search(query)}

        all_scores: dict[int, float] = {}
        for layer in layer_priority:
            for idx in self._layers[layer]:
                if idx in all_scores:
                    continue
                content_score = bm25_results.get(idx, 0.0)
                if content_score > 0:
                    all_scores[idx] = content_score * layer_weights[layer]

        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 4. Mem0-like Simulator — ENHANCED
#    BM25 + entity graph + recency + importance multi-signal fusion
# ═══════════════════════════════════════════════════════════════

class Mem0Simulator:
    """Mem0-like: Vector-first hybrid retrieval with BM25 content scoring."""

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._entity_graph: dict[str, set[int]] = defaultdict(set)
        self._build_entity_graph()

    def _build_entity_graph(self):
        for i, mem in enumerate(self.memories):
            for tag in mem.tags:
                self._entity_graph[tag.lower()].add(i)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}

        # Extract query entities from entity graph
        query_entities = set()
        for entity in self._entity_graph:
            if entity in query.lower():
                query_entities.add(entity)

        query_cat = _infer_query_category(query)
        q_val, q_ar = _extract_query_emotion(query)

        scores = []
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            # Graph boost: co-entity matching
            graph_boost = 0.0
            if query_entities:
                mem_tags_lower = {t.lower() for t in mem.tags}
                overlap = query_entities & mem_tags_lower
                graph_boost = len(overlap) / max(len(query_entities), 1) * 0.25

            # Recency signal
            days = _days_ago(mem.created)
            recency = max(0.0, 1.0 - days / 90.0)

            # Importance signal
            importance = mem.importance / 10.0

            # Emotion resonance (Mem0 enhancement)
            emotion_score = _emotion_resonance(q_val, q_ar, mem.valence, mem.arousal)

            # Multi-signal fusion (Mem0-style)
            final_score = (
                content_score * 0.35 +
                graph_boost * 0.18 +
                recency * 0.22 +
                importance * 0.12 +
                emotion_score * 0.13
            )
            scores.append((i, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, scores[0][1] if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 5. Zep/Graphiti-like Simulator — ENHANCED
#    Real temporal KG with proper edge expiry + BM25 content
# ═══════════════════════════════════════════════════════════════

@dataclass
class ZepEdge:
    from_entity: str
    to_entity: str
    relation: str
    created_days_ago: float
    expired: bool = False


class ZepSimulator:
    """Zep/Graphiti: Temporal KG with proper edge expiry logic."""

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self.edges: list[ZepEdge] = []
        self._entity_to_memories: dict[str, set[int]] = defaultdict(set)
        self._build_temporal_graph()

    def _extract_entities(self, content: str) -> list[str]:
        entities = []
        for pat in ["大厂", "创业公司", "AI创业公司", "A公司", "B公司", "新公司"]:
            if pat in content:
                entities.append(pat)
        for pat in ["小明", "leader", "HR", "同事", "妈妈", "我妈"]:
            if pat in content:
                entities.append(pat)
        for pat in ["offer", "面试", "离职", "裁员", "Python", "Go", "LLM", "失眠",
                     "焦虑", "薪资", "技术分享"]:
            if pat in content:
                entities.append(pat)
        return list(set(entities))

    def _build_temporal_graph(self):
        for i, mem in enumerate(self.memories):
            entities = self._extract_entities(mem.content)
            days = _days_ago(mem.created)
            for e in entities:
                self._entity_to_memories[e].add(i)

            # Create co-occurrence edges
            for ei, e1 in enumerate(entities):
                for e2 in entities[ei + 1:]:
                    self.edges.append(ZepEdge(
                        from_entity=e1, to_entity=e2,
                        relation="CO_OCCURS", created_days_ago=days,
                    ))

            # Create typed decision edges
            if mem.memory_type == "decision":
                for e in entities:
                    if e not in ("小明",):
                        self.edges.append(ZepEdge(
                            from_entity="小明", to_entity=e,
                            relation="DECIDES_ABOUT", created_days_ago=days,
                        ))

        # Detect edge expiry (contradictory facts about same entities)
        self._detect_edge_expiry()

    def _detect_edge_expiry(self):
        entity_facts: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, edge in enumerate(self.edges):
            if edge.relation == "DECIDES_ABOUT":
                key = f"{edge.from_entity}_{edge.to_entity}"
                entity_facts[key].append((i, edge.created_days_ago))

        for key, facts in entity_facts.items():
            if len(facts) > 1:
                facts.sort(key=lambda x: x[1])
                for edge_idx, _ in facts[:-1]:
                    self.edges[edge_idx].expired = True

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        query_entities = self._extract_entities(query)
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}

        memory_scores: dict[int, float] = {}
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            mem_entities = set(self._extract_entities(mem.content))
            entity_overlap = len(set(query_entities) & mem_entities)

            # Edge freshness bonus
            edge_freshness = 0.0
            for edge in self.edges:
                if edge.from_entity in query_entities or edge.to_entity in query_entities:
                    if edge.from_entity in mem_entities or edge.to_entity in mem_entities:
                        if not edge.expired:
                            edge_freshness = max(edge_freshness, 0.15)
                        else:
                            edge_freshness = max(edge_freshness, 0.03)

            total_score = content_score * 0.50 + entity_overlap * 0.05 + edge_freshness
            memory_scores[i] = total_score

        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 6. BM25 Baseline — REAL rank_bm25 (not keyword overlap)
# ═══════════════════════════════════════════════════════════════

class BM25Baseline:
    """Pure BM25 keyword matching — lower bound baseline."""

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        results = self.bm25.search(query)
        top_indices = [i for i, _ in results[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0
        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, results[0][1] if results else 0.0


# ═══════════════════════════════════════════════════════════════
# 7. Vector Baseline — TF-IDF enhanced (real IDF weighting)
# ═══════════════════════════════════════════════════════════════

class VectorBaseline:
    """TF-IDF semantic similarity baseline with IDF-weighted term matching."""

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._term_index: dict[str, set[int]] = defaultdict(set)
        for i, mem in enumerate(memories):
            for token in _tokenize(mem.content):
                self._term_index[token].add(i)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        q_tokens = _tokenize(query)
        scores: dict[int, float] = {}

        for token in q_tokens:
            matching_mems = self._term_index.get(token, set())
            idf = math.log(1 + len(self.memories) / max(1, len(matching_mems)))
            for mem_idx in matching_mems:
                scores[mem_idx] = scores.get(mem_idx, 0) + idf

        # Normalize by memory length
        for mem_idx in list(scores.keys()):
            mem_len = len(_tokenize(self.memories[mem_idx].content))
            scores[mem_idx] /= math.log(1 + mem_len)

        # Tag match bonus
        for i, mem in enumerate(self.memories):
            tag_overlap = len(set(q_tokens) & {t.lower() for t in mem.tags})
            if tag_overlap > 0:
                scores[i] = scores.get(i, 0) + tag_overlap * 0.2

        # Also blend in BM25 scores for better ranking
        bm25_results = {idx: s * 0.3 for idx, s in self.bm25.search(query)}
        for idx, s in bm25_results.items():
            scores[idx] = scores.get(idx, 0.0) + s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 8. HippoRAG Simulator (NeurIPS 2024 / ICML 2025) — NEW
#    Personalized PageRank on memory graph (REAL networkx.pagerank)
# ═══════════════════════════════════════════════════════════════

class HippoRAGSimulator:
    """
    HippoRAG: Personalized PageRank on a memory graph.

    Core mechanism (per paper):
    1. Build a weighted graph where nodes=memories, edges=similarity
    2. For each query, compute BM25 scores as seed vector
    3. Run PPR (personalized PageRank) from seeds
    4. Return top-k nodes by PPR score

    Uses REAL networkx.pagerank with personalization vector.
    """

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._graph: nx.Graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """Build memory similarity graph for PPR computation."""
        n = len(self.memories)
        for i in range(n):
            self._graph.add_node(i)

        # Add edges based on BM25 token similarity
        for i in range(n):
            ti = self.bm25.get_document_tokens(i)
            for j in range(i + 1, n):
                tj = self.bm25.get_document_tokens(j)
                jac = _jaccard(ti, tj)
                if jac > 0.08:
                    self._graph.add_edge(i, j, weight=jac)

        # Add typed edges based on shared tags
        for i in range(n):
            for j in range(i + 1, n):
                shared_tags = set(self.memories[i].tags) & set(self.memories[j].tags)
                if shared_tags:
                    if self._graph.has_edge(i, j):
                        self._graph[i][j]["weight"] = min(1.0, self._graph[i][j]["weight"] + 0.1 * len(shared_tags))
                    else:
                        self._graph.add_edge(i, j, weight=0.1 * len(shared_tags))

        # Add temporal edges (memories close in time)
        for i in range(n):
            di = _days_ago(self.memories[i].created)
            for j in range(i + 1, n):
                dj = _days_ago(self.memories[j].created)
                if abs(di - dj) < 3:  # within 3 days
                    if self._graph.has_edge(i, j):
                        self._graph[i][j]["weight"] = min(1.0, self._graph[i][j]["weight"] + 0.05)
                    else:
                        self._graph.add_edge(i, j, weight=0.08)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        # Build personalization vector from BM25 scores
        bm25_results = self.bm25.search(query)
        if not bm25_results:
            return "未找到相关信息。", [], 0.0

        personalization: dict[int, float] = {}
        total_bm25 = sum(s for _, s in bm25_results) or 1.0
        for idx, score in bm25_results:
            personalization[idx] = score / total_bm25

        # Ensure all nodes have at least a tiny weight
        for node in self._graph.nodes():
            if node not in personalization:
                personalization[node] = 1e-6

        # Run PPR (real networkx.pagerank)
        try:
            ppr = nx.pagerank(self._graph, personalization=personalization, alpha=0.85, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            # Fallback: use BM25 directly
            ranked = bm25_results[:top_k]
            top_indices = [i for i, _ in ranked]
            contexts = [self.memories[i].content for i in top_indices]
            return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0

        # Sort by PPR score
        ranked = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:top_k] if i in personalization and personalization[i] > 1e-6]
        if not top_indices and ranked:
            top_indices = [ranked[0][0]]

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 9. GraphRAG Simulator (MS GraphRAG, arXiv 2024) — NEW
#    Leiden/Louvain community detection + community-level retrieval
# ═══════════════════════════════════════════════════════════════

class GraphRAGSimulator:
    """
    Microsoft GraphRAG: Community detection + hierarchical summarization.

    Core mechanism:
    1. Build a memory similarity graph
    2. Detect communities using Louvain (networkx)
    3. For each query, find the most relevant community
    4. Return memories from that community + cross-community bridges
    """

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._graph: nx.Graph = nx.Graph()
        self._communities: dict[int, int] = {}  # node -> community_id
        self._community_members: dict[int, list[int]] = defaultdict(list)
        self._build_graph()
        self._detect_communities()

    def _build_graph(self):
        """Build memory graph for community detection."""
        n = len(self.memories)
        for i in range(n):
            self._graph.add_node(i)

        for i in range(n):
            ti = self.bm25.get_document_tokens(i)
            for j in range(i + 1, n):
                tj = self.bm25.get_document_tokens(j)
                jac = _jaccard(ti, tj)
                if jac > 0.1:
                    self._graph.add_edge(i, j, weight=jac)

        # Add tag-based edges
        for i in range(n):
            for j in range(i + 1, n):
                shared = set(self.memories[i].tags) & set(self.memories[j].tags)
                if shared:
                    if self._graph.has_edge(i, j):
                        self._graph[i][j]["weight"] = min(1.0, self._graph[i][j]["weight"] + 0.15 * len(shared))
                    else:
                        self._graph.add_edge(i, j, weight=0.15 * len(shared))

    def _detect_communities(self):
        """Run Louvain community detection (real networkx)."""
        if self._graph.number_of_edges() < 2:
            for node in self._graph.nodes():
                self._communities[node] = 0
                self._community_members[0].append(node)
            return

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(self._graph, weight="weight", seed=42)
            for cid, community in enumerate(communities):
                for node in community:
                    self._communities[node] = cid
                    self._community_members[cid].append(node)
        except ImportError:
            # Fallback: connected components
            for cid, component in enumerate(nx.connected_components(self._graph)):
                for node in component:
                    self._communities[node] = cid
                    self._community_members[cid].append(node)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = self.bm25.search(query)
        if not bm25_results:
            return "未找到相关信息。", [], 0.0

        # Compute community relevance scores
        community_scores: dict[int, float] = defaultdict(float)
        community_hits: dict[int, int] = defaultdict(int)
        for idx, score in bm25_results:
            cid = self._communities.get(idx, -1)
            if cid >= 0:
                community_scores[cid] += score
                community_hits[cid] += 1

        # Normalize by community size to avoid large-community bias
        for cid in list(community_scores.keys()):
            size = len(self._community_members.get(cid, [1]))
            community_scores[cid] /= max(1, math.log(1 + size))

        # Rank communities
        ranked_communities = sorted(community_scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked_communities:
            top_indices = [i for i, _ in bm25_results[:top_k]]
            contexts = [self.memories[i].content for i in top_indices]
            return " | ".join(contexts), top_indices, bm25_results[0][1]

        # Collect from top communities
        collected: set[int] = set()
        all_scored: dict[int, float] = {}
        for cid, comm_score in ranked_communities[:3]:
            members = self._community_members.get(cid, [])
            # Score each member: community relevance × BM25 score
            for node in members:
                bm25_s = dict(bm25_results).get(node, 0.0)
                all_scored[node] = bm25_s * 0.6 + comm_score * 0.4

        # Add cross-community bridges (nodes with high degree)
        for node in self._graph.nodes():
            if node not in all_scored:
                degree = self._graph.degree(node, weight="weight")
                bm25_s = dict(bm25_results).get(node, 0.0)
                if degree > 2 and bm25_s > 0:
                    all_scored[node] = bm25_s * 0.5 + 0.05 * min(degree, 10) / 10

        ranked = sorted(all_scored.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:top_k]]
        if not top_indices:
            top_indices = [i for i, _ in bm25_results[:top_k]]

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# 10. MemLong Simulator (2024) — NEW
#     Learnable retrieval path weights with feedback adjustment
# ═══════════════════════════════════════════════════════════════

class MemLongSimulator:
    """
    MemLong: Feedback-driven learnable path weights for retrieval.

    Core mechanism:
    1. Multiple retrieval paths with learnable weights
    2. Weights updated based on implicit feedback (score × engagement)
    3. Query-category-aware weight selection

    Simulates the MemLong weight learning loop with pre-trained defaults.
    """

    # Pre-trained path weights per query category (simulated training)
    CATEGORY_WEIGHTS = {
        "emotional": {"content": 0.30, "emotion": 0.35, "temporal": 0.10, "graph": 0.15, "cross": 0.10},
        "causal": {"content": 0.25, "emotion": 0.05, "temporal": 0.15, "graph": 0.35, "cross": 0.20},
        "temporal": {"content": 0.25, "emotion": 0.05, "temporal": 0.40, "graph": 0.20, "cross": 0.10},
        "cross_reference": {"content": 0.20, "emotion": 0.10, "temporal": 0.15, "graph": 0.30, "cross": 0.25},
        "factual": {"content": 0.50, "emotion": 0.05, "temporal": 0.05, "graph": 0.25, "cross": 0.15},
    }

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Build a simple link graph for graph-path scoring
        self._link_graph: dict[int, list[int]] = defaultdict(list)
        self._build_link_graph()

    def _build_link_graph(self):
        n = len(self.memories)
        for i in range(n):
            ti = self.bm25.get_document_tokens(i)
            for j in range(i + 1, n):
                tj = self.bm25.get_document_tokens(j)
                if _jaccard(ti, tj) > 0.12:
                    self._link_graph[i].append(j)
                    self._link_graph[j].append(i)

    def _temporal_score(self, mem_idx: int) -> float:
        """Recency-based temporal score."""
        days = _days_ago(self.memories[mem_idx].created)
        return max(0.1, 1.0 - days / 90.0)

    def _cross_ref_score(self, mem_idx: int) -> float:
        """Cross-reference score based on graph degree and type diversity."""
        degree = len(self._link_graph.get(mem_idx, []))
        mem = self.memories[mem_idx]
        score = 0.2
        if degree >= 3:
            score += 0.3
        elif degree >= 1:
            score += 0.15
        if mem.memory_type in ("emotion", "milestone", "decision"):
            score += 0.15
        if mem.importance >= 8:
            score += 0.15
        return min(1.0, score)

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        query_cat = _infer_query_category(query)
        weights = self.CATEGORY_WEIGHTS.get(query_cat, self.CATEGORY_WEIGHTS["factual"])
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        q_val, q_ar = _extract_query_emotion(query)

        scores = []
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            emotion_score = _emotion_resonance(q_val, q_ar, mem.valence, mem.arousal)
            temporal_s = self._temporal_score(i)
            # Graph score: neighbor overlap with other BM25 hits
            graph_score = 0.0
            neighbors = self._link_graph.get(i, [])
            if neighbors:
                hit_neighbors = sum(1 for n in neighbors if n in bm25_results)
                graph_score = hit_neighbors / max(len(neighbors), 1)
            cross_s = self._cross_ref_score(i)

            final = (
                content_score * weights["content"] +
                emotion_score * weights["emotion"] +
                temporal_s * weights["temporal"] +
                graph_score * weights["graph"] +
                cross_s * weights["cross"]
            )
            scores.append((i, final))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, scores[0][1] if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 11. HybridFusion Simulator — NEW
#     8-path fusion baseline (fixed weights, no DDA adaptation)
#     This is what earlier MP versions aimed for — serves as
#     an ablation baseline to measure DDA's contribution.
# ═══════════════════════════════════════════════════════════════

class HybridFusionSim:
    """
    8-path fixed-weight fusion baseline.
    No DDA adaptation — always uses full fusion regardless of data density.
    This isolates the value of DDA vs. the value of multi-path fusion.
    """

    FUSION_WEIGHTS = {
        "bm25": 0.28, "vector": 0.22, "graph": 0.18,
        "emotion": 0.10, "temporal": 0.08,
        "cross_ref": 0.06, "ppr": 0.05, "narrative": 0.03,
    }

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Build graph for PPR-like scoring
        self._graph: nx.Graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        n = len(self.memories)
        for i in range(n):
            self._graph.add_node(i)
            ti = self.bm25.get_document_tokens(i)
            for j in range(i + 1, n):
                tj = self.bm25.get_document_tokens(j)
                jac = _jaccard(ti, tj)
                if jac > 0.08:
                    self._graph.add_edge(i, j, weight=jac)
        for i in range(n):
            for j in range(i + 1, n):
                shared = set(self.memories[i].tags) & set(self.memories[j].tags)
                if shared and self._graph.has_edge(i, j):
                    self._graph[i][j]["weight"] = min(1.0, self._graph[i][j]["weight"] + 0.1 * len(shared))

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        q_val, q_ar = _extract_query_emotion(query)
        query_cat = _infer_query_category(query)

        # Compute PPR scores
        ppr_scores: dict[int, float] = {}
        if bm25_results and self._graph.number_of_edges() > 0:
            personalization = {n: 1e-6 for n in self._graph.nodes()}
            total_bm = sum(bm25_results.values()) or 1.0
            for idx, s in bm25_results.items():
                personalization[idx] = s / total_bm
            try:
                ppr = nx.pagerank(self._graph, personalization=personalization, alpha=0.85, max_iter=100)
                max_p = max(ppr.values()) if ppr else 1.0
                ppr_scores = {n: v / max_p for n, v in ppr.items()}
            except nx.PowerIterationFailedConvergence:
                pass

        # Compute all path scores and fuse
        scores: dict[int, float] = {}
        for i, mem in enumerate(self.memories):
            bm25_s = bm25_results.get(i, 0.0)
            if bm25_s == 0:
                continue

            # Vector proxy: TF-IDF-like (blend with BM25)
            vector_s = bm25_s * 0.8 + (mem.importance / 10.0) * 0.2

            # Graph score: degree-weighted
            graph_s = min(1.0, self._graph.degree(i, weight="weight") / 10.0) if self._graph.has_node(i) else 0.0

            # Emotion score
            emotion_s = _emotion_resonance(q_val, q_ar, mem.valence, mem.arousal)

            # Temporal score
            days = _days_ago(mem.created)
            temporal_s = max(0.1, 1.0 - days / 90.0)

            # Cross-ref score
            neighbor_types = set()
            if self._graph.has_node(i):
                for n in self._graph.neighbors(i):
                    neighbor_types.add(self.memories[n].memory_type)
            cross_s = min(1.0, 0.2 + 0.15 * len(neighbor_types) + 0.1 * (mem.importance >= 8))

            # PPR score
            ppr_s = ppr_scores.get(i, 0.0)

            # Narrative proxy: milestone/emotion content
            narrative_s = 0.3
            if mem.memory_type == "milestone":
                narrative_s = 0.9
            elif mem.memory_type == "emotion":
                narrative_s = 0.7
            elif mem.memory_type == "decision":
                narrative_s = 0.6

            w = self.FUSION_WEIGHTS
            final = (
                bm25_s * w["bm25"] + vector_s * w["vector"] + graph_s * w["graph"] +
                emotion_s * w["emotion"] + temporal_s * w["temporal"] +
                cross_s * w["cross_ref"] + ppr_s * w["ppr"] + narrative_s * w["narrative"]
            )
            scores[i] = final

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, ranked[0][1] if ranked else 0.0


# ═══════════════════════════════════════════════════════════════
# ALL_SYSTEMS registry — 11 comparison systems + MP slot
# ═══════════════════════════════════════════════════════════════

# All system names used in comparisons
SYSTEM_NAMES = [
    "Memory Palace v9",
    # ── Paper simulators (11) ──
    "A-MEM (NeurIPS 2025)",
    "MAGMA (CVPR 2025)",
    "MMAG",
    "Mem0-like",
    "Zep-like",
    "BM25 Baseline",
    "Vector Baseline",
    "HippoRAG (PPR)",
    "GraphRAG (Community)",
    "MemLong (Learnable)",
    "HybridFusion (No-DDA)",
    # ── New paper simulators (5) ──
    "CausalRAG (ACL 2025)",
    "DAM-LLM (2025)",
    "MemoTime (2025)",
    "DyMemR (TKDE 2024)",
    "REMT (2025)",
    # ── Community classic simulators (3) ──
    "Generative Agents (2023)",
    "RAPTOR (2024)",
    "CrewAI Cognitive",
]

SIMULATOR_SYSTEMS = [n for n in SYSTEM_NAMES if n != "Memory Palace v9"]


def create_shared_bm25(memories: list[BenchmarkMemory]) -> SharedBM25Index:
    """Create a shared BM25 index for all simulators."""
    return SharedBM25Index(memories)


def create_all_systems(memories: list[BenchmarkMemory] = None):
    """Create all 11 comparison systems with shared BM25 index."""
    if memories is None:
        memories = BENCHMARK_MEMORIES

    bm25 = create_shared_bm25(memories)

    return {
        "Memory Palace v8": None,  # handled via real RetrievalEngine adapter
        "A-MEM (NeurIPS 2025)": AMEMSimulator(memories, bm25),
        "MAGMA (CVPR 2025)": MAGMASimulator(memories, bm25),
        "MMAG": MMAGSimulator(memories, bm25),
        "Mem0-like": Mem0Simulator(memories, bm25),
        "Zep-like": ZepSimulator(memories, bm25),
        "BM25 Baseline": BM25Baseline(memories, bm25),
        "Vector Baseline": VectorBaseline(memories, bm25),
        "HippoRAG (PPR)": HippoRAGSimulator(memories, bm25),
        "GraphRAG (Community)": GraphRAGSimulator(memories, bm25),
        "MemLong (Learnable)": MemLongSimulator(memories, bm25),
        "HybridFusion (No-DDA)": HybridFusionSim(memories, bm25),
        # ── New paper simulators (5) ──
        "CausalRAG (ACL 2025)": CausalRAGSimulator(memories, bm25),
        "DAM-LLM (2025)": DAMLLMSimulator(memories, bm25),
        "MemoTime (2025)": MemoTimeSimulator(memories, bm25),
        "DyMemR (TKDE 2024)": DyMemRSimulator(memories, bm25),
        "REMT (2025)": REMTSimulator(memories, bm25),
        # ── Community classic simulators (3) ──
        "Generative Agents (2023)": GenerativeAgentsSim(memories, bm25),
        "RAPTOR (2024)": RAPTORSim(memories, bm25),
        "CrewAI Cognitive": CrewAISim(memories, bm25),
    }
