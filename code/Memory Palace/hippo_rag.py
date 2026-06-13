# ============================================================
# Module: HippoRAG Personalized PageRank Retrieval (hippo_rag.py)
# Track C Task 2: Personalized PageRank on the Memory Graph
# for personalized retrieval.
#
# Theoretical foundation:
#   1. Gutiérrez-Basulto et al. (2024), OSU/Stanford —
#      "HippoRAG: Neurobiologically Inspired Long-Term Memory
#      for Large Language Models." Uses Personalized PageRank
#      + Ontology-guided retrieval for hippocampus-like memory.
#   2. Page, Brin, Motwani & Winograd (1999) — The PageRank
#      Citation Ranking. PPR = (1-α) * P * v + α * e_s
#      where v is the rank vector, e_s is the seed vector,
#      and α is the teleport probability.
#   3. Haveliwala (2003), IEEE TKDE — "Topic-Sensitive PageRank."
#      Topic-biased PPR using per-topic teleport vectors.
#
# Implementation:
#   - Power iteration PPR on the memory graph adjacency matrix
#   - Personalized seed selection from user profile signals
#   - Query-biased PPR for retrieval (teleport to query-relevant nodes)
#   - Zero external dependencies (pure numpy)
# ============================================================

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("memory_palace.hippo_rag")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class PPRResult:
    """Personalized PageRank result for a single node."""
    node_id: str
    ppr_score: float
    rank: int = 0
    seed_contribution: dict[str, float] = field(default_factory=dict)
    # Which seeds contributed most to this node's PPR score

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "ppr_score": round(self.ppr_score, 6),
            "rank": self.rank,
            "seed_contribution": {
                k: round(v, 6) for k, v in self.seed_contribution.items()
            },
        }


@dataclass
class PPRSeed:
    """A seed node for personalized PageRank."""
    node_id: str
    weight: float = 1.0      # Teleport probability weight
    source: str = "default"  # importance | flashbulb | goal | recent | query


# ═══════════════════════════════════════════════════════════════
# Personalized PageRank Engine
# ═══════════════════════════════════════════════════════════════


class PersonalizedPageRank:
    """
    Personalized PageRank on the Memory Graph.

    Computes topic/person-sensitive PageRank scores that reflect
    the user's unique memory landscape rather than global graph
    centrality. This enables "what's important to THIS user"
    retrieval — the core HippoRAG insight.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        """
        Args:
            alpha: teleport probability (0.85 = standard PageRank).
                   Higher α = more personalization, less graph exploration.
            max_iterations: max power iteration steps
            tolerance: convergence tolerance for L1 norm
        """
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Cached state
        self._last_ppr_vector: dict[str, float] = {}
        self._last_seeds: list[str] = []
        self._node_index: dict[str, int] = {}
        self._index_node: dict[int, str] = {}
        self._transition_matrix: np.ndarray | None = None

    # ── Main PPR computation ──────────────────────────────────

    def compute_ppr(
        self,
        graph_edges: list[dict],
        seeds: list[PPRSeed],
        top_k: int = 50,
    ) -> list[PPRResult]:
        """
        Compute Personalized PageRank over the memory graph.

        Args:
            graph_edges: list of {from_id, to_id, weight, ...} from MemoryGraph
            seeds: personalized seed nodes with weights
            top_k: number of top PPR results to return

        Returns:
            Top-k PPR results sorted by score desc
        """
        if not graph_edges or not seeds:
            return []

        # Step 1: Build node index and adjacency
        self._build_graph_structure(graph_edges)

        n = len(self._node_index)
        if n == 0:
            return []

        # Step 2: Build transition matrix
        self._build_transition_matrix(graph_edges)

        # Step 3: Build personalization vector from seeds
        personalization = np.zeros(n)
        total_seed_weight = 0.0
        for seed in seeds:
            idx = self._node_index.get(seed.node_id)
            if idx is not None:
                personalization[idx] += seed.weight
                total_seed_weight += seed.weight

        if total_seed_weight > 0:
            personalization /= total_seed_weight
        else:
            # Uniform if no valid seeds
            personalization = np.ones(n) / n

        # Step 4: Power iteration
        ppr = self._power_iteration(personalization)

        # Step 5: Rank results
        results = []
        for i in range(n):
            if ppr[i] > 0:
                node_id = self._index_node[i]
                results.append(PPRResult(
                    node_id=node_id,
                    ppr_score=float(ppr[i]),
                ))

        results.sort(key=lambda r: r.ppr_score, reverse=True)

        # Assign ranks
        for rank, r in enumerate(results):
            r.rank = rank + 1

        # Cache
        self._last_ppr_vector = {
            self._index_node[i]: float(ppr[i])
            for i in range(n)
        }

        return results[:top_k]

    def _power_iteration(self, personalization: np.ndarray) -> np.ndarray:
        """
        Power iteration for PPR.

        v_{t+1} = (1 - α) * M^T * v_t + α * p
        where M is the column-stochastic transition matrix,
        p is the personalization vector, α is the teleport prob.
        """
        n = len(personalization)
        v = np.ones(n) / n  # uniform initial

        for iteration in range(self.max_iterations):
            if self._transition_matrix is not None:
                v_new = (1 - self.alpha) * self._transition_matrix.dot(v)
            else:
                v_new = np.zeros(n)
            v_new += self.alpha * personalization

            # Check convergence (L1 norm)
            delta = np.sum(np.abs(v_new - v))
            v = v_new

            if delta < self.tolerance:
                logger.debug(f"PPR converged in {iteration + 1} iterations (δ={delta:.8f})")
                break

        return v

    # ── Graph structure building ──────────────────────────────

    def _build_graph_structure(self, graph_edges: list[dict]):
        """Extract node index from edges."""
        node_set: set[str] = set()
        for edge in graph_edges:
            node_set.add(edge.get("from_id", ""))
            node_set.add(edge.get("to_id", ""))

        node_set.discard("")
        nodes = sorted(node_set)

        self._node_index = {node_id: i for i, node_id in enumerate(nodes)}
        self._index_node = {i: node_id for node_id, i in self._node_index.items()}

    def _build_transition_matrix(self, graph_edges: list[dict]):
        """Build column-stochastic transition matrix from edges."""
        n = len(self._node_index)
        if n == 0:
            self._transition_matrix = None
            return

        # Build adjacency matrix
        adj = np.zeros((n, n))
        out_degree = np.zeros(n)

        for edge in graph_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            weight = edge.get("weight", 1.0)

            from_idx = self._node_index.get(from_id)
            to_idx = self._node_index.get(to_id)

            if from_idx is not None and to_idx is not None:
                # Column-stochastic: M[j,i] = probability of going from i to j
                adj[to_idx, from_idx] += weight
                out_degree[from_idx] += weight

        # Normalize columns (handle dangling nodes)
        for i in range(n):
            if out_degree[i] > 0:
                adj[:, i] /= out_degree[i]
            else:
                # Dangling node: uniform teleport
                adj[:, i] = 1.0 / n

        self._transition_matrix = adj

    # ── Seed selection strategies ─────────────────────────────

    @staticmethod
    def personalize_seeds(
        memories: list[dict],
        working_self=None,
        flashbulb_ids: set[str] | None = None,
        recent_window_days: int = 7,
    ) -> list[PPRSeed]:
        """
        Select personalized seed nodes (HippoRAG-style).

        Seeds based on:
          1. High importance (importance >= 7)
          2. Flashbulb memories
          3. Working Self active goal domains
          4. Recently activated memories
          5. Pinned/protected memories

        Args:
            memories: list of {id, importance, valence, arousal, type, domain, created, ...}
            working_self: optional WorkingSelf for goal matching
            flashbulb_ids: set of flashbulb memory IDs
            recent_window_days: recency window in days

        Returns:
            List of PPRSeed objects
        """
        seeds: list[PPRSeed] = []
        seen: set[str] = set()

        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)

        for mem in memories:
            mem_id = mem.get("id", "")
            if not mem_id or mem_id in seen:
                continue

            importance = mem.get("importance", 5)
            is_flashbulb = mem.get("is_flashbulb", False)
            is_pinned = mem.get("pinned", False)
            is_protected = mem.get("protected", False)
            valence = mem.get("valence", 0.5)
            arousal = mem.get("arousal", 0.3)
            created = mem.get("created", "")
            memory_type = mem.get("memory_type", mem.get("type", "chat"))

            # Priority 1: Flashbulb memories (highest weight)
            if is_flashbulb or (flashbulb_ids and mem_id in flashbulb_ids):
                seeds.append(PPRSeed(node_id=mem_id, weight=3.0, source="flashbulb"))
                seen.add(mem_id)
                continue

            # Priority 2: Pinned/protected
            if is_pinned or is_protected:
                seeds.append(PPRSeed(node_id=mem_id, weight=2.5, source="pinned"))
                seen.add(mem_id)
                continue

            # Priority 3: High importance
            if importance >= 8:
                seeds.append(PPRSeed(node_id=mem_id, weight=2.0, source="importance"))
                seen.add(mem_id)
                continue
            elif importance >= 7:
                seeds.append(PPRSeed(node_id=mem_id, weight=1.5, source="importance"))
                seen.add(mem_id)
                continue

            # Priority 4: High emotional intensity (extreme valence + arousal)
            if (valence <= 0.2 or valence >= 0.8) and arousal >= 0.7:
                seeds.append(PPRSeed(node_id=mem_id, weight=1.5, source="emotional_peak"))
                seen.add(mem_id)
                continue

            # Priority 5: Recency
            if created:
                try:
                    created_dt = datetime.datetime.fromisoformat(created)
                    days_ago = (now - created_dt).total_seconds() / 86400.0
                    if days_ago <= recent_window_days:
                        seeds.append(PPRSeed(
                            node_id=mem_id,
                            weight=1.0 * max(0.3, 1.0 - days_ago / recent_window_days),
                            source="recent",
                        ))
                        seen.add(mem_id)
                        continue
                except (ValueError, TypeError):
                    pass

        # Priority 6: Milestone memories
        for mem in memories:
            mem_id = mem.get("id", "")
            if mem_id in seen:
                continue
            if memory_type in ("milestone",):
                seeds.append(PPRSeed(node_id=mem_id, weight=1.0, source="milestone"))
                seen.add(mem_id)

        # Working Self goal-domain seeds
        if working_self and hasattr(working_self, 'get_active_goal_domains'):
            try:
                goal_domains = working_self.get_active_goal_domains()
                for mem in memories:
                    mem_id = mem.get("id", "")
                    if mem_id in seen:
                        continue
                    mem_domain = mem.get("domain", [])
                    if isinstance(mem_domain, str):
                        mem_domain = [mem_domain]
                    if any(d in goal_domains for d in mem_domain):
                        seeds.append(PPRSeed(
                            node_id=mem_id,
                            weight=1.2,
                            source="working_self",
                        ))
                        seen.add(mem_id)
            except Exception:
                pass

        return seeds

    @staticmethod
    def query_biased_seeds(
        query: str,
        memories: list[dict],
        max_seeds: int = 10,
    ) -> list[PPRSeed]:
        """
        Select seeds biased toward query relevance.

        For query-time retrieval: teleport probability is biased
        toward nodes that are directly relevant to the query.
        """
        seeds: list[PPRSeed] = []
        query_lower = query.lower()

        for mem in memories:
            mem_id = mem.get("id", "")
            content = mem.get("content", "")
            name = mem.get("name", "")

            # Simple keyword match scoring
            score = 0.0
            if content:
                content_lower = content.lower()
                # Count keyword overlap
                query_terms = [t for t in query_lower.split() if len(t) >= 1]
                matched = sum(1 for t in query_terms if t in content_lower)
                score = matched / max(len(query_terms), 1)

            if name and name.lower() in query_lower:
                score += 0.3

            if score > 0.1:
                seeds.append(PPRSeed(
                    node_id=mem_id,
                    weight=score,
                    source="query",
                ))

        # Sort and limit
        seeds.sort(key=lambda s: s.weight, reverse=True)
        return seeds[:max_seeds]

    # ── Retrieval integration ─────────────────────────────────

    def retrieve(
        self,
        graph_edges: list[dict],
        seeds: list[PPRSeed],
        top_k: int = 20,
    ) -> list[PPRResult]:
        """
        Full PPR-based retrieval pipeline.

        1. Compute PPR with personalization vector from seeds
        2. Return top-k results

        This is the primary integration point for retrieval_engine.
        """
        return self.compute_ppr(
            graph_edges=graph_edges,
            seeds=seeds,
            top_k=top_k,
        )

    def combine_with_other_scores(
        self,
        ppr_results: list[PPRResult],
        other_results: dict[str, dict],
        ppr_weight: float = 0.10,
    ) -> dict[str, dict]:
        """
        Fuse PPR scores into existing retrieval results.

        For integration in retrieval_engine._retrieve_three_way():
        combine PPR scores with vector, BM25, graph, emotion scores.
        """
        combined = dict(other_results)

        # Build PPR lookup
        ppr_map = {r.node_id: r.ppr_score for r in ppr_results}

        max_ppr = max(ppr_map.values()) if ppr_map else 1.0
        if max_ppr == 0:
            max_ppr = 1.0

        for node_id, ppr_score in ppr_map.items():
            normalized = ppr_score / max_ppr
            if node_id in combined:
                # Blend PPR into existing result
                existing_final = combined[node_id].get("final_score", 0)
                combined[node_id]["ppr_score"] = normalized
                combined[node_id]["final_score"] = (
                    existing_final * (1 - ppr_weight)
                    + normalized * ppr_weight
                )
                combined[node_id]["source"] += "+ppr"
            else:
                # New result from PPR
                combined[node_id] = {
                    "id": node_id,
                    "ppr_score": normalized,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "graph_score": 0.0,
                    "emotion_score": 0.0,
                    "temporal_score": 0.0,
                    "cross_ref_score": 0.0,
                    "final_score": normalized * ppr_weight,
                    "source": "ppr",
                }

        return combined

    # ── Graph extraction from MemoryGraph ─────────────────────

    @staticmethod
    def extract_edges(memory_graph) -> list[dict]:
        """
        Extract all active edges from a MemoryGraph instance.

        Returns list of {from_id, to_id, weight, ...} dicts
        suitable for PPR computation.
        """
        edges = []
        try:
            for etype in ["causal", "thematic", "temporal", "emotional"]:
                typed_edges = memory_graph.get_edges_by_type(etype, limit=2000)
                for edge in typed_edges:
                    # Only include active edges
                    if edge.get("valid_until"):
                        continue
                    edges.append({
                        "from_id": edge["from_id"],
                        "to_id": edge["to_id"],
                        "weight": edge.get("weight", 1.0),
                        "relation_type": edge.get("relation_type", etype),
                    })
        except Exception as e:
            logger.warning(f"Edge extraction failed: {e}")

        return edges


# ═══════════════════════════════════════════════════════════════
# HippoRAG integration class
# ═══════════════════════════════════════════════════════════════


class HippoRAGRetriever:
    """
    HippoRAG-style retriever combining PPR with memory graph.

    Wraps PersonalizedPageRank with caching and integration
    conveniences for the Memory Palace retrieval pipeline.
    """

    def __init__(self, alpha: float = 0.85):
        self.ppr = PersonalizedPageRank(alpha=alpha)
        self._cached_seeds: list[PPRSeed] = []
        self._cached_edges: list[dict] = []
        self._cache_valid = False

    def update_graph(
        self,
        memory_graph,
        bucket_mgr=None,
        working_self=None,
    ):
        """
        Update the cached graph structure and seeds.

        Called periodically (e.g., in dream()) to refresh the
        PPR cache with new memories and edges.
        """
        # Extract edges
        self._cached_edges = self.ppr.extract_edges(memory_graph)

        # Recompute seeds
        if bucket_mgr:
            try:
                # We need memories for seed selection
                # In practice, this runs in dream() which is async,
                # but seed selection is synchronous
                import asyncio
                pass
            except Exception:
                pass

        self._cache_valid = True

    async def update_graph_async(
        self,
        memory_graph,
        bucket_mgr=None,
        working_self=None,
    ):
        """Async version with bucket_mgr content access."""
        self._cached_edges = self.ppr.extract_edges(memory_graph)

        if bucket_mgr:
            try:
                all_buckets = await bucket_mgr.list_all(include_archive=False)
                memories = []
                for b in all_buckets:
                    meta = b.get("metadata", {})
                    memories.append({
                        "id": b["id"],
                        "content": b.get("content", ""),
                        "name": meta.get("name", ""),
                        "importance": meta.get("importance", 5),
                        "valence": meta.get("valence", 0.5),
                        "arousal": meta.get("arousal", 0.3),
                        "is_flashbulb": meta.get("is_flashbulb", False),
                        "pinned": meta.get("pinned", False),
                        "protected": meta.get("protected", False),
                        "domain": meta.get("domain", []),
                        "memory_type": meta.get("memory_type", "chat"),
                        "type": meta.get("type", "dynamic"),
                        "created": meta.get("created", ""),
                    })

                self._cached_seeds = self.ppr.personalize_seeds(
                    memories=memories,
                    working_self=working_self,
                )
            except Exception as e:
                logger.warning(f"Graph update async failed: {e}")

        self._cache_valid = True

    def retrieve(
        self,
        query: str = "",
        memories: list[dict] | None = None,
        top_k: int = 20,
    ) -> list[PPRResult]:
        """
        Retrieve using cached PPR + optional query-biased seeds.

        Args:
            query: optional query for query-biased seeds
            memories: optional memory list for query seed selection
            top_k: number of results
        """
        seeds = list(self._cached_seeds)

        # Add query-biased seeds
        if query and memories:
            query_seeds = self.ppr.query_biased_seeds(query, memories)
            # Merge: query seeds get higher weight
            existing_ids = {s.node_id for s in seeds}
            for qs in query_seeds:
                if qs.node_id not in existing_ids:
                    seeds.append(PPRSeed(
                        node_id=qs.node_id,
                        weight=qs.weight * 2.0,  # Query bias boost
                        source="query",
                    ))

        if not seeds:
            return []

        return self.ppr.retrieve(
            graph_edges=self._cached_edges,
            seeds=seeds,
            top_k=top_k,
        )
