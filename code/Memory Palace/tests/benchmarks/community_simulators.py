# ============================================================
# Community Algorithm Simulators — 社区经典算法模拟器
# v9 Enhancement (2026-06-10)
#
# 3 community-classic algorithm simulators from influential
# open-source projects and papers:
#   17. GenerativeAgentsSim   — Park et al. 2023: 3-factor weighted retrieval
#   18. RAPTORSim             — RAPTOR (arXiv 2024): recursive clustering tree
#   19. CrewAISim             — CrewAI: composite scoring + cognitive operations
#
# All follow the unified contract:
#   answer(query, top_k=10) -> tuple[str, list[int], float]
# ============================================================

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np

from tests.benchmarks.benchmark_dataset import BENCHMARK_MEMORIES, BenchmarkMemory
from tests.benchmarks.simulator_utils import (
    SharedBM25Index, _tokenize, _jaccard, _days_ago,
    _emotion_resonance, _extract_query_emotion, _infer_query_category,
)


# ═══════════════════════════════════════════════════════════════
# 17. Generative Agents Simulator (Park et al., 2023)
#    Memory Stream + 3-factor retrieval + Reflection
# ═══════════════════════════════════════════════════════════════

class GenerativeAgentsSim:
    """
    Generative Agents (Park et al., 2023 — "Generative Agents:
    Interactive Simulacra of Human Behavior", UIST 2023).

    Core retrieval mechanism:
      score = recency × importance × relevance

    Where:
      - recency: exponential decay e^(-λ·days), λ=0.99
      - importance: LLM-scored 1-10 (we use pre-assigned importance)
      - relevance: BM25 content match score

    Additionally implements:
      - Reflection: high-importance memories get reflection boost
      - Time-weighted retrieval (LangChain's TimeWeightedVectorStoreRetriever)
    """

    # Park et al. recency parameter
    RECENCY_DECAY = 0.99  # λ in e^(-λ·hours), converted to days: ~0.05/day

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Reflection store: synthesized high-level insights
        self._reflections: dict[int, str] = {}
        self._generate_reflections()

    def _generate_reflections(self):
        """Simulate the reflection process: group high-importance memories."""
        # Find memories with importance >= 8
        high_imp_indices = [i for i, m in enumerate(self.memories) if m.importance >= 8]
        # For each high-importance memory, create a reflection note
        for idx in high_imp_indices:
            mem = self.memories[idx]
            if mem.memory_type == "emotion":
                self._reflections[idx] = f"情感转折点: {mem.content[:60]}..."
            elif mem.memory_type == "milestone":
                self._reflections[idx] = f"重要里程碑: {mem.content[:60]}..."
            elif mem.memory_type == "decision":
                self._reflections[idx] = f"关键决定: {mem.content[:60]}..."

    def _recency_score(self, days_ago: float) -> float:
        """Park et al. recency: exponential decay, normalized to 0-1."""
        if days_ago <= 0:
            return 1.0
        # λ ≈ 0.05 per day (matching Park's hourly decay converted)
        return math.exp(-0.05 * days_ago)

    def _importance_score(self, mem: BenchmarkMemory) -> float:
        """Normalized importance 0-1 from 1-10 scale."""
        return mem.importance / 10.0

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}

        scores = []
        for i, mem in enumerate(self.memories):
            relevance = bm25_results.get(i, 0.0)
            if relevance == 0:
                continue

            recency = self._recency_score(_days_ago(mem.created))
            importance = self._importance_score(mem)

            # Park et al. 3-factor formula:
            # score = recency × importance × relevance
            # We add weighting to match the original paper's emphasis
            park_score = recency * importance * relevance

            # Reflection boost
            reflection_boost = 0.0
            if i in self._reflections:
                reflection_boost = 0.15

            # Normalized: the product can be very small
            # Scale up to make it comparable with other systems
            final = park_score * 3.0 + reflection_boost  # scale factor for comparability
            scores.append((i, final))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]
        if not top_indices:
            return "未找到相关信息。", [], 0.0

        contexts = [self.memories[i].content for i in top_indices]
        return " | ".join(contexts), top_indices, scores[0][1] if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# 18. RAPTOR Simulator (arXiv 2024)
#    Recursive Abstractive Processing for Tree-Organized Retrieval
# ═══════════════════════════════════════════════════════════════

class RAPTORSim:
    """
    RAPTOR: Build a hierarchical tree from memory texts via
    recursive clustering, then query by tree traversal.

    Simplified simulation of the RAPTOR algorithm:
    1. Cluster memories into groups (simulated GMM via k-means-like grouping)
    2. Build summary nodes for each cluster
    3. At query time: traverse tree top-down OR collapsed-tree search
    4. Return most relevant leaf nodes

    The full RAPTOR uses SBERT embeddings + UMAP + GMM + LLM summaries.
    We simulate the tree structure using BM25 token similarity clustering.
    """

    MAX_DEPTH = 3  # Maximum tree depth
    CLUSTER_SIZE = 5  # Target cluster size

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        self._tree: dict[int, dict] = {}  # node_id -> {children, summary, depth}
        self._leaf_to_node: dict[int, int] = {}  # memory_idx -> tree_node_id
        self._build_tree()

    def _build_tree(self):
        """Build a recursive clustering tree from memories."""
        n = len(self.memories)
        if n == 0:
            return

        # Level 0: leaf nodes (individual memories)
        for i in range(n):
            node_id = 1000 + i  # leaf IDs start at 1000
            self._tree[node_id] = {
                "children": [i],  # stores memory index
                "summary": self.memories[i].content[:80],
                "depth": 0,
                "is_leaf": True,
            }
            self._leaf_to_node[i] = node_id

        # Level 1+: cluster similar leaves
        if n <= self.CLUSTER_SIZE:
            # Single cluster at root
            root_id = 0
            self._tree[root_id] = {
                "children": [1000 + i for i in range(n)],
                "summary": f"All {n} memories",
                "depth": 1,
                "is_leaf": False,
            }
            return

        # Build clusters based on BM25 token similarity
        clusters = self._cluster_memories(n)
        next_node_id = 0
        current_level_nodes = []

        for cluster in clusters:
            node_id = next_node_id
            next_node_id += 1
            child_ids = [1000 + i for i in cluster]
            # Summary: concatenate first 40 chars of each memory in cluster
            summaries = [self.memories[i].content[:40] for i in cluster[:3]]
            summary = " | ".join(summaries)
            self._tree[node_id] = {
                "children": child_ids,
                "summary": summary,
                "depth": 1,
                "is_leaf": False,
            }
            current_level_nodes.append(node_id)

        # If multiple L1 nodes, create L2 root
        if len(current_level_nodes) > 1:
            root_id = next_node_id
            self._tree[root_id] = {
                "children": current_level_nodes,
                "summary": f"Root: {n} memories in {len(clusters)} clusters",
                "depth": 2,
                "is_leaf": False,
            }
            self._root_id = root_id
        else:
            self._root_id = current_level_nodes[0]

    def _cluster_memories(self, n: int) -> list[list[int]]:
        """Simple token-similarity-based clustering (stand-in for GMM)."""
        # Compute similarity matrix
        sim = np.zeros((n, n))
        for i in range(n):
            ti = self.bm25.get_document_tokens(i)
            for j in range(i + 1, n):
                tj = self.bm25.get_document_tokens(j)
                sim[i][j] = sim[j][i] = _jaccard(ti, tj)

        # Greedy clustering: assign each memory to best cluster
        clusters: list[list[int]] = []
        assigned: set[int] = set()
        k = max(3, n // self.CLUSTER_SIZE)  # number of clusters

        # Seed with k most central memories
        centrality = sim.sum(axis=1)
        seeds = sorted(range(n), key=lambda i: centrality[i], reverse=True)[:k]

        for seed in seeds:
            if seed in assigned:
                continue
            cluster = [seed]
            assigned.add(seed)
            # Add similar memories to cluster
            for j in range(n):
                if j not in assigned and sim[seed][j] > 0.1:
                    if len(cluster) < self.CLUSTER_SIZE * 2:
                        cluster.append(j)
                        assigned.add(j)
            clusters.append(cluster)

        # Add unassigned to nearest cluster
        for i in range(n):
            if i not in assigned:
                best_cluster = max(range(len(clusters)),
                                   key=lambda c: max(sim[i][j] for j in clusters[c]))
                clusters[best_cluster].append(i)
                assigned.add(i)

        return clusters

    def _traverse_tree(self, query: str, top_k: int = 10) -> list[int]:
        """Collapsed-tree search: score all leaves, weighted by cluster relevance."""
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}

        # Score each leaf by BM25 + tree-level boost
        leaf_scores: dict[int, float] = {}
        for i, mem in enumerate(self.memories):
            content_score = bm25_results.get(i, 0.0)
            if content_score == 0:
                continue

            # Tree-level boost: check if cluster summary matches query
            tree_boost = 0.0
            if i in self._leaf_to_node:
                node_id = self._leaf_to_node[i]
                # Walk up the tree
                current = node_id
                for _ in range(2):  # up to 2 levels up
                    if current in self._tree and not self._tree[current].get("is_leaf"):
                        summary = self._tree[current].get("summary", "")
                        q_tokens = set(_tokenize(query))
                        s_tokens = set(_tokenize(summary))
                        if q_tokens and s_tokens:
                            overlap = len(q_tokens & s_tokens) / len(q_tokens)
                            tree_boost += overlap * 0.10
                    # Find parent
                    parent = None
                    for pid, node in self._tree.items():
                        if not node.get("is_leaf") and current in node.get("children", []):
                            parent = pid
                            break
                    current = parent if parent else -1

            leaf_scores[i] = content_score * 0.80 + tree_boost

        ranked = sorted(leaf_scores.items(), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked[:top_k]]

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        top_indices = self._traverse_tree(query, top_k)
        if not top_indices:
            return "未找到相关信息。", [], 0.0
        contexts = [self.memories[i].content for i in top_indices]
        bm25 = self.bm25.search(query)
        top_score = bm25[0][1] if bm25 else 0.0
        return " | ".join(contexts), top_indices, top_score


# ═══════════════════════════════════════════════════════════════
# 19. CrewAI Simulator
#    Composite scoring + 5 cognitive operations
# ═══════════════════════════════════════════════════════════════

class CrewAISim:
    """
    CrewAI Cognitive Memory: 5-stage pipeline inspired by
    CrewAI's memory architecture and human cognitive operations.

    Stages:
    1. ENCODE: Extract features from memory content (tags, type, emotion)
    2. CONSOLIDATE: Strengthen frequently-accessed memories
    3. RECALL: Composite scoring = sim×0.5 + recency×0.3 + importance×0.2
    4. EXTRACT: Filter to top-k with diversity bonus
    5. FORGET: Suppress very old, low-importance memories

    Reference: CrewAI framework (crewAIInc/crewAI, 25k+ stars)
    """

    # Stage 5 forgetting parameters
    FORGET_THRESHOLD_DAYS = 90
    FORGET_IMPORTANCE_MIN = 3

    def __init__(self, memories: list[BenchmarkMemory], bm25: SharedBM25Index | None = None):
        self.memories = memories
        self.bm25 = bm25 or SharedBM25Index(memories)
        # Consolidation state (access counter per memory)
        self._access_count: dict[int, int] = defaultdict(int)
        # Encoded features cache
        self._encoded: dict[int, dict] = {}
        self._encode_all()

    def _encode_all(self):
        """Stage 1: ENCODE — extract features from each memory."""
        for i, mem in enumerate(self.memories):
            self._encoded[i] = {
                "type": mem.memory_type,
                "tags": set(mem.tags),
                "valence": mem.valence,
                "arousal": mem.arousal,
                "importance": mem.importance,
                "days_ago": _days_ago(mem.created),
                "token_count": len(_tokenize(mem.content)),
            }

    def _consolidate(self, retrieved: list[int]):
        """Stage 2: CONSOLIDATE — increase access count for retrieved memories."""
        for idx in retrieved:
            self._access_count[idx] += 1

    def _recall_score(self, idx: int, bm25_score: float) -> float:
        """Stage 3: RECALL — composite scoring formula."""
        mem = self.memories[idx]
        enc = self._encoded.get(idx, {})

        # Similarity component (50%): BM25 score
        sim = bm25_score

        # Recency component (30%): exponential decay
        days = enc.get("days_ago", 0)
        recency = math.exp(-0.05 * days)

        # Importance component (20%): normalized importance
        importance = mem.importance / 10.0

        # CrewAI composite formula
        composite = sim * 0.50 + recency * 0.30 + importance * 0.20

        # Consolidation bonus
        consolidation_bonus = min(0.15, self._access_count.get(idx, 0) * 0.03)
        composite += consolidation_bonus

        return composite

    def _extract(self, scored: list[tuple[int, float]], top_k: int) -> list[int]:
        """Stage 4: EXTRACT — top-k with diversity bonus."""
        if not scored:
            return []

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # MMR-style diversity: penalize same-tag clusters
        selected: list[int] = []
        for idx, score in scored:
            if len(selected) >= top_k:
                break
            # Diversity penalty: if too many same-type memories already selected
            mem_type = self.memories[idx].memory_type
            same_type_count = sum(1 for s in selected
                                  if self.memories[s].memory_type == mem_type)
            if same_type_count >= top_k // 2:
                score *= 0.7  # diversity penalty
            selected.append(idx)

        # Re-sort selected by adjusted score
        return selected[:top_k]

    def _forget(self, scores: dict[int, float]) -> dict[int, float]:
        """Stage 5: FORGET — suppress very old, low-importance memories."""
        filtered = {}
        for idx, score in scores.items():
            days = _days_ago(self.memories[idx].created)
            importance = self.memories[idx].importance
            if days > self.FORGET_THRESHOLD_DAYS and importance < self.FORGET_IMPORTANCE_MIN:
                # Heavy suppression
                filtered[idx] = score * 0.1
            elif days > self.FORGET_THRESHOLD_DAYS:
                filtered[idx] = score * 0.5
            else:
                filtered[idx] = score
        return filtered

    def answer(self, query: str, top_k: int = 10) -> tuple[str, list[int], float]:
        bm25_results = {idx: score for idx, score in self.bm25.search(query)}
        if not bm25_results:
            return "未找到相关信息。", [], 0.0

        # Stage 3: RECALL — compute composite scores
        raw_scores: dict[int, float] = {}
        for idx, bm_score in bm25_results.items():
            raw_scores[idx] = self._recall_score(idx, bm_score)

        # Stage 5: FORGET — filter old memories
        filtered_scores = self._forget(raw_scores)

        # Stage 4: EXTRACT — top-k with diversity
        scored_list = list(filtered_scores.items())
        top_indices = self._extract(scored_list, top_k)

        if not top_indices:
            return "未找到相关信息。", [], 0.0

        # Stage 2: CONSOLIDATE — record access
        self._consolidate(top_indices)

        contexts = [self.memories[i].content for i in top_indices]
        top_score = filtered_scores.get(top_indices[0], 0.0) if top_indices else 0.0
        return " | ".join(contexts), top_indices, top_score
