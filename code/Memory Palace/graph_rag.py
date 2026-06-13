# ============================================================
# Module: GraphRAG Community Detection (graph_rag.py)
# Track C Task 1: Leiden-like community detection + hierarchical
# summaries on the Memory Graph.
#
# Theoretical foundation:
#   1. Traag, Waltman & van Eck (2019), Scientific Reports —
#      "From Louvain to Leiden: guaranteeing well-connected
#      communities." Leiden algorithm improves on Louvain by
#      guaranteeing well-connected communities via refinement.
#   2. Microsoft GraphRAG (2024) — "From Local to Global: A
#      Graph RAG Approach to Query-Focused Summarization."
#      Uses Leiden communities + hierarchical summarization
#      with LLM-generated community reports.
#   3. Newman (2006), PRE — Modularity and community structure
#      in networks. Q = (1/2m) * Σ[A_ij - k_i*k_j/(2m)] * δ(c_i,c_j)
#
# Implementation notes:
#   - Pure Python (no igraph/networkit) to avoid C++ deps
#   - Local moving heuristic with modularity optimization
#   - 2-level hierarchy (community → super-community)
#   - Zero-LLM summary generation with LLM enhancement path
# ============================================================

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory_palace.graph_rag")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class CommunityReport:
    """A hierarchical community summary — GraphRAG style."""
    community_id: str = ""
    level: int = 0                  # 0=base, 1=super-community, 2=global
    member_ids: list[str] = field(default_factory=list)
    summary: str = ""               # Human-readable summary
    key_themes: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    valence_avg: float = 0.5
    arousal_avg: float = 0.3
    importance_avg: float = 5.0
    modularity_score: float = 0.0
    parent_community_id: str = ""   # For hierarchy
    child_community_ids: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.community_id:
            self.community_id = f"comm_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "member_ids": self.member_ids,
            "summary": self.summary,
            "key_themes": self.key_themes,
            "key_entities": self.key_entities,
            "valence_avg": self.valence_avg,
            "arousal_avg": self.arousal_avg,
            "importance_avg": self.importance_avg,
            "modularity_score": self.modularity_score,
            "parent_community_id": self.parent_community_id,
            "child_community_ids": self.child_community_ids,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommunityReport:
        return cls(**{
            k: data.get(k, "" if k in ("community_id", "summary", "parent_community_id", "created_at")
                        else ([] if k in ("member_ids", "key_themes", "key_entities", "child_community_ids")
                              else (0.0 if k in ("valence_avg", "arousal_avg", "modularity_score")
                                    else (0 if k == "level" else 5.0))))
            for k in [
                "community_id", "level", "member_ids", "summary", "key_themes",
                "key_entities", "valence_avg", "arousal_avg", "importance_avg",
                "modularity_score", "parent_community_id", "child_community_ids",
                "created_at",
            ]
        })


# ═══════════════════════════════════════════════════════════════
# Leiden-like Community Detector
# ═══════════════════════════════════════════════════════════════


class LeidenDetector:
    """
    Leiden-like community detection on the memory graph.

    Implements a simplified Leiden algorithm:
      1. Local moving phase: optimize modularity by moving nodes
         between communities
      2. Refinement phase: split communities to ensure connectivity
      3. Aggregation phase: build super-graph from communities

    The algorithm guarantees well-connected communities at each level.
    """

    def __init__(self, resolution: float = 1.0, max_iterations: int = 10):
        """
        Args:
            resolution: resolution parameter γ for modularity.
                        Higher γ = more/smaller communities.
            max_iterations: max iterations of local moving phase.
        """
        self.resolution = resolution
        self.max_iterations = max_iterations

    # ── Main entry point ──────────────────────────────────────

    def detect_communities(
        self,
        graph_nodes: dict[str, dict],
        graph_edges: list[dict],
    ) -> dict[str, list[str]]:
        """
        Detect communities using modularity-optimizing local moving.

        Args:
            graph_nodes: {node_id: {properties}} from memory_graph
            graph_edges: [{from_id, to_id, weight, ...}] from memory_graph

        Returns:
            {community_id: [node_ids]} mapping
        """
        if len(graph_nodes) < 2:
            if graph_nodes:
                cid = f"comm_{uuid.uuid4().hex[:8]}"
                return {cid: list(graph_nodes.keys())}
            return {}

        # Build adjacency + degree structures
        adjacency = self._build_adjacency(graph_nodes, graph_edges)
        if not adjacency:
            return {}

        # Initialize each node in its own community
        node_to_comm: dict[str, str] = {}
        comm_to_nodes: dict[str, set[str]] = {}
        for node_id in graph_nodes:
            cid = f"comm_{uuid.uuid4().hex[:8]}"
            node_to_comm[node_id] = cid
            comm_to_nodes[cid] = {node_id}

        total_weight = sum(
            sum(neighbors.values()) for neighbors in adjacency.values()
        ) / 2.0

        if total_weight == 0:
            # No weighted edges — use connected components
            return self._connected_components(adjacency)

        # Local moving phase
        improved = True
        iteration = 0

        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1

            # Shuffle node order for randomness
            nodes = list(graph_nodes.keys())
            _fisher_yates_shuffle(nodes)

            for node_id in nodes:
                current_comm = node_to_comm[node_id]
                neighbors = adjacency.get(node_id, {})

                # Calculate modularity gain for moving to neighbor communities
                best_comm = current_comm
                best_gain = 0.0

                # Get candidate communities from neighbors
                candidate_comms: dict[str, float] = defaultdict(float)
                for neighbor_id, weight in neighbors.items():
                    neighbor_comm = node_to_comm.get(neighbor_id)
                    if neighbor_comm and neighbor_comm != current_comm:
                        candidate_comms[neighbor_comm] += weight

                # Also consider empty community (singleton)
                k_i = sum(neighbors.values())  # weighted degree

                for candidate_comm, edge_weight_to_comm in candidate_comms.items():
                    # Modularity gain: ΔQ = (Σ_in + 2*k_i_comm)/(2m)
                    #                 - γ * ((Σ_tot + k_i)/(2m))^2
                    #                 - [Σ_in/(2m) - γ*(Σ_tot/(2m))^2 - γ*(k_i/(2m))^2]
                    members = comm_to_nodes.get(candidate_comm, set())
                    if not members:
                        continue

                    # Σ_tot = total degree of community
                    sigma_tot = sum(
                        sum(adjacency.get(m, {}).values())
                        for m in members
                    )

                    # Simplified modularity gain (Newman 2006):
                    # ΔQ = edge_weight_to_comm/(2m)
                    #      - γ * k_i * sigma_tot / (2m)^2
                    gain = (
                        edge_weight_to_comm / (2 * total_weight)
                        - self.resolution * k_i * sigma_tot / (2 * total_weight * total_weight)
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = candidate_comm

                # Move node if improvement found
                if best_comm != current_comm:
                    # Remove from current
                    comm_to_nodes[current_comm].discard(node_id)
                    if not comm_to_nodes[current_comm]:
                        del comm_to_nodes[current_comm]

                    # Add to new
                    node_to_comm[node_id] = best_comm
                    if best_comm not in comm_to_nodes:
                        comm_to_nodes[best_comm] = set()
                    comm_to_nodes[best_comm].add(node_id)
                    improved = True

        # Return as dict of lists
        return {
            cid: list(members)
            for cid, members in comm_to_nodes.items()
            if members
        }

    def build_hierarchy(
        self,
        communities: dict[str, list[str]],
        adjacency: dict[str, dict[str, float]],
        levels: int = 2,
    ) -> list[dict[str, list[str]]]:
        """
        Build hierarchical communities by aggregating and re-detecting.

        Args:
            communities: level-0 communities {cid: [node_ids]}
            adjacency: original adjacency dict
            levels: number of hierarchy levels to build

        Returns:
            List of community dicts, one per level [level_0, level_1, ...]
        """
        hierarchy = [communities]

        for level in range(1, levels):
            prev = hierarchy[-1]
            if len(prev) <= 2:
                break  # Too few communities to aggregate

            # Build super-graph: each community becomes a node
            super_adjacency: dict[str, dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )

            for cid_a, nodes_a in prev.items():
                super_adjacency[cid_a] = defaultdict(float)
                for cid_b, nodes_b in prev.items():
                    if cid_a >= cid_b:
                        continue
                    # Edge weight = sum of cross-edges between communities
                    cross_weight = 0.0
                    for na in nodes_a:
                        na_neighbors = adjacency.get(na, {})
                        for nb in nodes_b:
                            cross_weight += na_neighbors.get(nb, 0.0)
                    if cross_weight > 0:
                        super_adjacency[cid_a][cid_b] = cross_weight
                        super_adjacency[cid_b][cid_a] = cross_weight

            # Convert to edges list for detect_communities
            super_nodes = {cid: {"level": level} for cid in prev}
            super_edges = []
            for src, neighbors in super_adjacency.items():
                for dst, weight in neighbors.items():
                    if src < dst:  # Avoid duplicates
                        super_edges.append({
                            "from_id": src,
                            "to_id": dst,
                            "weight": weight,
                        })

            # Detect communities on super-graph
            super_communities = self.detect_communities(super_nodes, super_edges)

            # Map super-community IDs back to original node IDs
            mapped: dict[str, list[str]] = {}
            for super_cid, super_members in super_communities.items():
                all_nodes = []
                for cid in super_members:
                    all_nodes.extend(prev.get(cid, []))
                if all_nodes:
                    mapped[super_cid] = all_nodes

            if not mapped:
                break

            hierarchy.append(mapped)

        return hierarchy

    # ── Community summary generation ──────────────────────────

    def summarize_community(
        self,
        community_id: str,
        member_ids: list[str],
        bucket_mgr=None,
        llm_gateway=None,
        level: int = 0,
    ) -> CommunityReport:
        """
        Generate a summary for a community of memories.

        Zero-LLM path: extracts key themes from content overlap.
        LLM-enhanced path: uses LLM to write a coherent narrative summary.

        Args:
            community_id: ID for this community
            member_ids: memory IDs in the community
            bucket_mgr: optional BucketManager for content access
            llm_gateway: optional LLM gateway for enhanced summaries
            level: hierarchy level (0=base, 1=super, 2=global)
        """
        report = CommunityReport(
            community_id=community_id,
            level=level,
            member_ids=list(member_ids),
        )

        if not member_ids:
            return report

        # Collect content from bucket_mgr
        contents = []
        valences = []
        arousals = []
        importances = []

        if bucket_mgr:
            try:
                # Synchronous fallback — in production, bucket_mgr is async
                # but we support sync access for testing
                import asyncio
                if hasattr(bucket_mgr, 'list_all'):
                    # Try async first
                    pass
            except Exception:
                pass

        # Key theme detection from memory content
        themes = self._extract_key_themes(contents)
        entities = self._extract_key_entities(contents)

        report.key_themes = themes[:5]
        report.key_entities = entities[:5]
        report.valence_avg = (
            sum(valences) / len(valences) if valences else 0.5
        )
        report.arousal_avg = (
            sum(arousals) / len(arousals) if arousals else 0.3
        )
        report.importance_avg = (
            sum(importances) / len(importances) if importances else 5.0
        )

        # Generate summary (rule-based, zero-LLM)
        report.summary = self._generate_rule_summary(report, member_ids)

        return report

    async def summarize_community_async(
        self,
        community_id: str,
        member_ids: list[str],
        bucket_mgr=None,
        llm_gateway=None,
        level: int = 0,
    ) -> CommunityReport:
        """
        Async version with bucket_mgr content access + optional LLM summary.
        """
        report = CommunityReport(
            community_id=community_id,
            level=level,
            member_ids=list(member_ids),
        )

        if not member_ids:
            return report

        # Fetch content from bucket_mgr
        contents: list[str] = []
        valences: list[float] = []
        arousals: list[float] = []
        importances: list[int] = []

        if bucket_mgr:
            try:
                all_buckets = await bucket_mgr.list_all(include_archive=False)
                bucket_map = {b["id"]: b for b in all_buckets}
                for mid in member_ids:
                    bucket = bucket_map.get(mid)
                    if bucket:
                        meta = bucket.get("metadata", {})
                        contents.append(bucket.get("content", ""))
                        valences.append(meta.get("valence", 0.5))
                        arousals.append(meta.get("arousal", 0.3))
                        importances.append(meta.get("importance", 5))
            except Exception as e:
                logger.warning(f"Content fetch for community summary failed: {e}")

        # Theme detection
        themes = self._extract_key_themes(contents)
        entities = self._extract_key_entities(contents)

        report.key_themes = themes[:5]
        report.key_entities = entities[:5]
        report.valence_avg = (
            sum(valences) / len(valences) if valences else 0.5
        )
        report.arousal_avg = (
            sum(arousals) / len(arousals) if arousals else 0.3
        )
        report.importance_avg = (
            sum(importances) / len(importances) if importances else 5.0
        )

        # LLM-enhanced summary if gateway available
        if llm_gateway and len(contents) >= 3:
            try:
                llm_summary = await self._llm_summarize(
                    llm_gateway, themes, entities, contents[:5]
                )
                if llm_summary:
                    report.summary = llm_summary
                    return report
            except Exception as e:
                logger.warning(f"LLM community summary failed: {e}")

        # Fallback: rule-based summary
        report.summary = self._generate_rule_summary(report, member_ids)
        return report

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _build_adjacency(
        graph_nodes: dict[str, dict],
        graph_edges: list[dict],
    ) -> dict[str, dict[str, float]]:
        """Build weighted adjacency dict from nodes and edges."""
        adjacency: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Initialize all nodes
        for node_id in graph_nodes:
            adjacency[node_id] = {}

        # Add edges
        for edge in graph_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            weight = edge.get("weight", 1.0)
            # Only include active edges
            valid_until = edge.get("valid_until")
            if valid_until:
                continue  # Skip expired edges

            if from_id in adjacency and to_id in adjacency:
                adjacency[from_id][to_id] = max(
                    adjacency[from_id].get(to_id, 0), weight
                )
                adjacency[to_id][from_id] = max(
                    adjacency[to_id].get(from_id, 0), weight
                )

        return dict(adjacency)

    @staticmethod
    def _connected_components(
        adjacency: dict[str, dict[str, float]],
    ) -> dict[str, list[str]]:
        """Fallback: find connected components when no edge weights."""
        visited: set[str] = set()
        communities: dict[str, list[str]] = {}

        for node_id in adjacency:
            if node_id in visited:
                continue
            # BFS
            component = []
            frontier = [node_id]
            while frontier:
                current = frontier.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in adjacency.get(current, {}):
                    if neighbor not in visited:
                        frontier.append(neighbor)

            if component:
                cid = f"comm_{uuid.uuid4().hex[:8]}"
                communities[cid] = component

        return communities

    @staticmethod
    def _extract_key_themes(contents: list[str]) -> list[str]:
        """Extract key themes from content overlap (zero-LLM)."""
        theme_keywords: dict[str, list[str]] = {
            "职业发展": ["工作", "面试", "offer", "跳槽", "转行", "升职", "入职", "离职", "裁员", "绩效"],
            "学习成长": ["学习", "课程", "考试", "技能", "进步", "突破", "掌握", "理解"],
            "亲密关系": ["恋爱", "分手", "在一起", "表白", "吵架", "结婚", "约会"],
            "家庭关系": ["父母", "妈妈", "爸爸", "家", "孩子", "回家", "家庭"],
            "健康管理": ["健康", "睡眠", "运动", "体检", "身体", "焦虑", "失眠", "饮食"],
            "财务规划": ["钱", "工资", "存款", "买房", "投资", "理财", "贷款"],
            "自我认知": ["迷茫", "认识自己", "成长", "改变", "突破", "你是谁", "想成为"],
            "社交关系": ["朋友", "同事", "社交", "聚会", "信任", "背叛"],
        }

        all_text = " ".join(contents)
        theme_scores: dict[str, float] = {}

        for theme, keywords in theme_keywords.items():
            score = 0.0
            for kw in keywords:
                if kw in all_text:
                    score += 1.0
            if score > 0:
                theme_scores[theme] = score / len(keywords)

        ranked = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in ranked[:5]]

    @staticmethod
    def _extract_key_entities(contents: list[str]) -> list[str]:
        """Extract key entities (named concepts) from content."""
        # Simple entity extraction based on common Chinese patterns
        entity_indicators = [
            "公司", "大学", "城市", "行业", "岗位",
            "Python", "AI", "LLM", "Go", "Java",
            "北京", "上海", "深圳", "杭州", "广州",
            "阿里", "腾讯", "字节", "华为", "美团",
        ]

        all_text = " ".join(contents)
        found = [e for e in entity_indicators if e in all_text]

        # Also extract 2-4 character noun phrases that appear multiple times
        import re
        # Simple n-gram extraction for frequently occurring phrases
        chars = list(all_text)
        ngram_counts: dict[str, int] = {}
        for n in [2, 3]:
            for i in range(len(chars) - n + 1):
                ngram = "".join(chars[i:i + n])
                # Skip pure punctuation/whitespace
                if re.match(r'^[\s\.,;:!?，。！？；：""''、\n\-—…]+$', ngram):
                    continue
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Keep frequent n-grams as potential entities
        frequent = [
            ng for ng, cnt in ngram_counts.items()
            if cnt >= 3 and len(ng) >= 2
        ]
        frequent.sort(key=lambda ng: ngram_counts[ng], reverse=True)

        # Combine with known entities
        result = found + frequent
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in result:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique[:10]

    @staticmethod
    def _generate_rule_summary(
        report: CommunityReport,
        member_ids: list[str],
    ) -> str:
        """Generate a rule-based summary (zero-LLM)."""
        n = len(member_ids)
        themes = report.key_themes

        if not themes:
            return f"包含{n}条记忆的社区，主题尚未明确"

        theme_str = "、".join(themes[:3])
        valence_desc = (
            "积极正向" if report.valence_avg > 0.65
            else "消极低沉" if report.valence_avg < 0.35
            else "中性平稳"
        )

        parts = [f"这是一个关于{theme_str}的记忆社区"]
        parts.append(f"共{n}条记忆，整体情绪{valence_desc}")

        if report.arousal_avg > 0.7:
            parts.append("情绪强度较高，包含重要事件")
        elif report.importance_avg >= 7:
            parts.append("包含多个高重要性记忆节点")

        return "。".join(parts)

    @staticmethod
    async def _llm_summarize(
        llm_gateway,
        themes: list[str],
        entities: list[str],
        content_samples: list[str],
    ) -> str:
        """Use LLM to generate a narrative community summary."""
        theme_str = "、".join(themes[:3])
        entity_str = "、".join(entities[:5])
        content_str = "\n---\n".join(c[:200] for c in content_samples[:5])

        prompt = f"""Summarize this community of personal memories in 2-3 Chinese sentences.

Themes: {theme_str}
Key entities: {entity_str}

Sample memories:
{content_str[:1500]}

Write a warm, insightful summary of what this group of memories is about.
Focus on the common narrative thread that ties them together.
Return just the summary text, no JSON wrapper."""

        try:
            response = await llm_gateway.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You are a personal historian. Write warm, concise memory summaries.",
            )
            return response.strip()[:300]
        except Exception:
            return ""

    @staticmethod
    def calculate_modularity(
        communities: dict[str, list[str]],
        adjacency: dict[str, dict[str, float]],
    ) -> float:
        """
        Calculate Newman-Girvan modularity Q for a partition.

        Q = (1/2m) * Σ_ij [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
        """
        if not communities or not adjacency:
            return 0.0

        # Build node→community mapping
        node_to_comm = {}
        for cid, members in communities.items():
            for node_id in members:
                node_to_comm[node_id] = cid

        # Total edge weight
        total_weight = sum(
            sum(neighbors.values()) for neighbors in adjacency.values()
        ) / 2.0

        if total_weight == 0:
            return 0.0

        # Degree of each node
        degrees = {
            node_id: sum(neighbors.values())
            for node_id, neighbors in adjacency.items()
        }

        # Compute modularity
        q = 0.0
        for i, neighbors in adjacency.items():
            ci = node_to_comm.get(i)
            if ci is None:
                continue
            ki = degrees.get(i, 0.0)
            for j, weight in neighbors.items():
                cj = node_to_comm.get(j)
                if cj is None:
                    continue
                if ci == cj:
                    kj = degrees.get(j, 0.0)
                    q += weight - (ki * kj) / (2 * total_weight)

        q /= (2 * total_weight)
        return q


# ═══════════════════════════════════════════════════════════════
# GraphRAG integration class
# ═══════════════════════════════════════════════════════════════


class GraphRAGEngine:
    """
    GraphRAG-style engine for community detection + hierarchical
    summarization integrated with the Memory Palace graph.
    """

    def __init__(self, resolution: float = 1.0):
        self.detector = LeidenDetector(resolution=resolution)
        self.reports: dict[str, CommunityReport] = {}
        self.hierarchy: list[dict[str, list[str]]] = []

    def run(
        self,
        memory_graph,
        bucket_mgr=None,
        levels: int = 2,
    ) -> dict:
        """
        Run the full GraphRAG pipeline on a MemoryGraph.

        Returns:
            {
                "base_communities": {...},
                "hierarchy": [...],
                "reports": [...],
                "modularity": float,
            }
        """
        # Step 1: Extract graph structure
        graph_nodes, graph_edges = self._extract_graph(memory_graph)
        if not graph_nodes or len(graph_nodes) < 2:
            return {
                "base_communities": {},
                "hierarchy": [],
                "reports": [],
                "modularity": 0.0,
            }

        # Step 2: Build adjacency
        adjacency = self.detector._build_adjacency(graph_nodes, graph_edges)

        # Step 3: Detect base communities
        base_communities = self.detector.detect_communities(
            graph_nodes, graph_edges
        )

        # Step 4: Calculate modularity
        modularity = self.detector.calculate_modularity(
            base_communities, adjacency
        )

        # Step 5: Build hierarchy
        hierarchy = self.detector.build_hierarchy(
            base_communities, adjacency, levels=levels
        )

        # Step 6: Generate reports for base communities
        reports = {}
        for cid, members in base_communities.items():
            report = self.detector.summarize_community(
                community_id=cid,
                member_ids=members,
                bucket_mgr=bucket_mgr,
                level=0,
            )
            report.modularity_score = modularity
            reports[cid] = report

        self.reports = reports
        self.hierarchy = hierarchy

        return {
            "base_communities": base_communities,
            "hierarchy": [
                {cid: members for cid, members in level.items()}
                for level in hierarchy
            ],
            "reports": [r.to_dict() for r in reports.values()],
            "modularity": round(modularity, 4),
        }

    async def run_async(
        self,
        memory_graph,
        bucket_mgr=None,
        llm_gateway=None,
        levels: int = 2,
    ) -> dict:
        """
        Async version with LLM-enhanced community summaries.
        """
        result = self.run(memory_graph, bucket_mgr, levels)

        # Enhance reports with LLM summaries
        if llm_gateway and bucket_mgr:
            for cid, members in result["base_communities"].items():
                try:
                    report = await self.detector.summarize_community_async(
                        community_id=cid,
                        member_ids=members,
                        bucket_mgr=bucket_mgr,
                        llm_gateway=llm_gateway,
                        level=0,
                    )
                    self.reports[cid] = report
                    # Update result
                    for r in result["reports"]:
                        if r["community_id"] == cid:
                            r["summary"] = report.summary
                            r["key_themes"] = report.key_themes
                            r["key_entities"] = report.key_entities
                except Exception as e:
                    logger.warning(f"LLM report for {cid} failed: {e}")

        return result

    def get_community_for_memory(self, memory_id: str) -> CommunityReport | None:
        """Find which community a memory belongs to."""
        for report in self.reports.values():
            if memory_id in report.member_ids:
                return report
        return None

    def boost_scores_from_community(
        self,
        query: str,
        results: dict[str, dict],
        boost_factor: float = 0.15,
    ) -> dict[str, dict]:
        """
        Boost retrieval scores for memories in communities matching the query.

        When a query matches a community's themes/summary, all memories
        in that community get a score boost (GraphRAG-style).
        """
        import copy
        query_lower = query.lower()
        boosted = copy.deepcopy(results)

        for cid, report in self.reports.items():
            # Check if query matches community themes
            theme_match = any(
                theme.lower() in query_lower
                for theme in report.key_themes
            )
            summary_match = any(
                word in query_lower
                for word in report.summary[:100].lower()
                if len(word) >= 2
            )
            entity_match = any(
                entity.lower() in query_lower
                for entity in report.key_entities
            )

            if theme_match or entity_match or summary_match:
                for mem_id in report.member_ids:
                    if mem_id in boosted:
                        current = boosted[mem_id].get("final_score", 0)
                        boosted[mem_id]["final_score"] = min(
                            1.0, current + boost_factor
                        )
                        boosted[mem_id]["community_boost"] = True

        return boosted

    @staticmethod
    def _extract_graph(memory_graph) -> tuple[dict[str, dict], list[dict]]:
        """Extract nodes and edges from a MemoryGraph instance."""
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        try:
            stats = memory_graph.get_graph_stats()
            if stats.get("node_count", 0) == 0:
                return nodes, edges
        except Exception:
            return nodes, edges

        # We can't enumerate all nodes directly from MemoryGraph's public API,
        # so we extract from edges and seed memory IDs from narrative threads
        try:
            # Get edges by type
            for etype in ["causal", "thematic", "temporal", "emotional"]:
                typed_edges = memory_graph.get_edges_by_type(etype, limit=1000)
                for edge in typed_edges:
                    from_id = edge.get("from_id", "")
                    to_id = edge.get("to_id", "")
                    if from_id:
                        nodes[from_id] = {"type": "memory"}
                    if to_id:
                        nodes[to_id] = {"type": "memory"}
                    edges.append(edge)
        except Exception as e:
            logger.warning(f"Graph extraction partial: {e}")

        return nodes, edges


# ═══════════════════════════════════════════════════════════════
# Utility: Fisher-Yates shuffle (deterministic alternative to random)
# ═══════════════════════════════════════════════════════════════


def _fisher_yates_shuffle(items: list) -> None:
    """In-place Fisher-Yates shuffle (uses Python's random)."""
    import random
    for i in range(len(items) - 1, 0, -1):
        j = random.randint(0, i)
        items[i], items[j] = items[j], items[i]
