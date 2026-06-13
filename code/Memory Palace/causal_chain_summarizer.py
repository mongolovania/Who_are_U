# ============================================================
# Module: Causal Chain Summarizer (causal_chain_summarizer.py)
# L2: Transforms causal graph paths into human-readable summaries.
# L2：因果链摘要器 — 因果路径 → 可读叙事摘要
#
# Theoretical foundation:
#   1. CausalRAG (ACL 2025). — Causal graph constraints for RAG;
#      summarization of causal chains improves decision understanding
#      by providing context about WHY things happened.
#   2. CDF-RAG (2025). — Causal Discovery Framework for RAG:
#      causal path extraction + natural language summarization.
#   3. Causal Cartographer (2025). arXiv:2505.14396. — Graph RAG
#      agent with causal chain export for human interpretation.
#   4. Pearl (2009). Causality. — Causal diagrams (DAGs) as the
#      substrate for understanding; summarization = translating
#      DAG paths into natural language explanations.
#
# Design §12.5:
#   - Find all causal paths between two memories
#   - Summarize each path into natural language
#   - Export causal chains as markdown for user presentation
#   - Optional LLM polishing for narrative quality
#
# Integration points:
#   - memory_orchestrator.dream(): summarize_all_chains in EVOLVE stage
#   - memory_graph: BFS path finding + edge type filtering
#   - llm_gateway: optional lightweight LLM call for narrative polishing
# ============================================================

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("memory_palace.causal_summarizer")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class CausalLink:
    """A single link in a causal chain."""
    from_id: str
    to_id: str
    from_summary: str = ""      # Short summary of cause
    to_summary: str = ""        # Short summary of effect
    confidence: float = 0.5     # Edge weight / verification confidence
    relation_type: str = "causal"
    edge_properties: dict = field(default_factory=dict)


@dataclass
class CausalChain:
    """A complete causal chain: A → B → C → D."""
    id: str = ""
    chain: list[CausalLink] = field(default_factory=list)
    summary: str = ""               # Human-readable summary
    total_confidence: float = 0.0   # Product of individual confidences
    depth: int = 0
    domain: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        if self.chain and not self.depth:
            self.depth = len(self.chain)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "chain": [
                {
                    "from_id": link.from_id,
                    "to_id": link.to_id,
                    "from_summary": link.from_summary,
                    "to_summary": link.to_summary,
                    "confidence": link.confidence,
                    "relation_type": link.relation_type,
                }
                for link in self.chain
            ],
            "summary": self.summary,
            "total_confidence": self.total_confidence,
            "depth": self.depth,
            "domain": self.domain,
            "generated_at": self.generated_at,
        }


# ═══════════════════════════════════════════════════════════════
# Causal Chain Summarizer
# ═══════════════════════════════════════════════════════════════


class CausalChainSummarizer:
    """
    Transform causal graph paths into human-readable chain summaries.

    Supports:
      - Point-to-point chain summarization
      - Domain-wide chain discovery
      - Markdown export for user presentation
      - Optional LLM narrative polishing
    """

    def __init__(self, user_id: str = "", llm_gateway=None):
        self.user_id = user_id
        self.llm = llm_gateway
        self._chains: dict[str, CausalChain] = {}  # chain_id → chain
        self._summarization_count: int = 0

        # Config
        self.min_chain_length: int = 2       # Minimum causal links
        self.max_chain_depth: int = 5        # Max BFS depth for path finding
        self.min_confidence: float = 0.2     # Minimum edge confidence to include

    # ── Point-to-point chain summarization ─────────────────────

    def summarize_chain(
        self,
        from_id: str,
        to_id: str,
        graph,
        bucket_mgr=None,
        max_depth: int = 5,
    ) -> list[CausalChain]:
        """
        Find and summarize all causal paths between two memories.

        Args:
            from_id: Starting memory node
            to_id: Target memory node
            graph: MemoryGraph instance
            bucket_mgr: BucketManager for content lookup
            max_depth: Max BFS depth

        Returns:
            List of CausalChain objects, sorted by total_confidence desc
        """
        if graph is None:
            return []

        chains: list[CausalChain] = []

        # Find all causal paths using BFS (via graph.get_path)
        all_paths = self._find_causal_paths(from_id, to_id, graph, max_depth)

        for path_edges in all_paths:
            if len(path_edges) < self.min_chain_length:
                continue

            chain = self._build_chain(path_edges, graph, bucket_mgr)
            if chain and chain.total_confidence >= self.min_confidence:
                chain.summary = self._generate_summary(chain)
                chains.append(chain)
                self._chains[chain.id] = chain

        chains.sort(key=lambda c: c.total_confidence, reverse=True)
        self._summarization_count += 1

        logger.debug(
            f"Summarized {len(chains)} causal chains from {from_id[:8]} "
            f"to {to_id[:8]} (depth={max_depth})"
        )

        return chains

    # ── Domain-wide chain discovery ────────────────────────────

    def summarize_all_chains(
        self,
        graph,
        bucket_mgr=None,
        domain_filter: list[str] | None = None,
    ) -> list[CausalChain]:
        """
        Find all non-trivial causal chains in the graph.

        Called from sleeptime EVOLVE stage.

        Scans for chains where:
          - Chain length ≥ 2 causal edges
          - Each edge confidence ≥ min_confidence
          - Domain filter applied if specified

        Returns:
            All discovered causal chains
        """
        if graph is None:
            return []

        # Get all causal edges
        try:
            all_edges = graph.get_edges_by_type("causal", limit=1000)
        except Exception as e:
            logger.warning(f"Failed to get causal edges: {e}")
            return []

        if len(all_edges) < 2:
            return []

        # Build adjacency list for causal edges
        adjacency: dict[str, list[dict]] = {}
        for edge in all_edges:
            from_id = edge.get("from_id", "")
            if from_id not in adjacency:
                adjacency[from_id] = []
            adjacency[from_id].append(edge)

        # Find chains: DFS from each node with outgoing edges
        all_chains: list[CausalChain] = []
        seen_chain_sigs: set[str] = set()

        for start_id in list(adjacency.keys())[:50]:  # Cap to avoid explosion
            chains_from_node = self._dfs_find_chains(
                start_id, adjacency, graph, bucket_mgr, seen_chain_sigs, depth=0
            )
            all_chains.extend(chains_from_node)

        # Apply domain filter
        if domain_filter:
            all_chains = [
                c for c in all_chains
                if any(d in c.domain for d in domain_filter)
            ]

        # Sort and store
        all_chains.sort(key=lambda c: c.total_confidence, reverse=True)
        for chain in all_chains:
            self._chains[chain.id] = chain

        logger.info(
            f"Discovered {len(all_chains)} causal chains "
            f"(from {len(all_edges)} edges, {len(adjacency)} source nodes)"
        )

        return all_chains

    # ── Markdown export ────────────────────────────────────────

    def export_chain_to_markdown(self, chain: CausalChain) -> str:
        """Export a single causal chain as readable Markdown."""
        if not chain.chain:
            return "（空的因果链）"

        lines = [
            f"### 🔗 因果链：{chain.summary}",
            "",
            f"*置信度：{chain.total_confidence:.0%} · 深度：{chain.depth}层*",
            "",
            "```",
        ]

        for i, link in enumerate(chain.chain):
            arrow = "  ↓" if i < len(chain.chain) - 1 else ""
            lines.append(
                f"{link.from_summary[:60]}"
            )
            if i < len(chain.chain) - 1:
                lines.append(f"  │ 因为 → 导致 ({link.confidence:.0%})")
            lines.append(f"  ↓")
            if i == len(chain.chain) - 1:
                lines.append(f"{link.to_summary[:60]}")

        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def export_all_to_markdown(
        self,
        max_chains: int = 10,
    ) -> str:
        """Export all stored causal chains as a Markdown document."""
        chains = sorted(
            self._chains.values(),
            key=lambda c: c.total_confidence,
            reverse=True,
        )[:max_chains]

        if not chains:
            return "# 🔗 因果链报告\n\n*暂无因果链。随着你与系统的交互增多，因果关系会自动浮现。*"

        lines = [
            "# 🔗 你的因果链图",
            "",
            f"*共 {len(self._chains)} 条因果链，以下展示前 {len(chains)} 条*",
            "",
            "---",
            "",
        ]

        for chain in chains:
            lines.append(self.export_chain_to_markdown(chain))
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # ── Private: Path finding ──────────────────────────────────

    def _find_causal_paths(
        self,
        from_id: str,
        to_id: str,
        graph,
        max_depth: int,
    ) -> list[list[dict]]:
        """Find all causal paths between two nodes using BFS."""
        if from_id == to_id:
            return []

        all_paths: list[list[dict]] = []

        # Use DFS with path tracking to find all paths
        def dfs(current: str, path: list[dict], visited: set[str]):
            if len(path) >= max_depth:
                return

            neighbors = graph.get_neighbors(
                current, depth=1, relation_types=["causal"], active_only=True
            )

            for edge in neighbors:
                next_node = (
                    edge.get("to_id") if edge.get("from_id") == current
                    else edge.get("from_id")
                )

                if next_node in visited:
                    continue

                new_path = path + [edge]

                if next_node == to_id:
                    all_paths.append(new_path)
                else:
                    visited.add(next_node)
                    dfs(next_node, new_path, visited)
                    visited.discard(next_node)

        dfs(from_id, [], {from_id})
        return all_paths

    def _dfs_find_chains(
        self,
        node_id: str,
        adjacency: dict[str, list[dict]],
        graph,
        bucket_mgr,
        seen_sigs: set[str],
        depth: int,
    ) -> list[CausalChain]:
        """DFS to find all causal chains starting from a node."""
        if depth >= self.max_chain_depth or node_id not in adjacency:
            return []

        chains: list[CausalChain] = []
        outgoing = adjacency[node_id]

        for edge in outgoing:
            to_id = edge.get("to_id", "")
            if not to_id or to_id == node_id:
                continue

            weight = edge.get("weight", 0.5)
            if weight < self.min_confidence:
                continue

            # Check if this edge continues a chain
            if to_id in adjacency:
                sub_chains = self._dfs_find_chains(
                    to_id, adjacency, graph, bucket_mgr, seen_sigs, depth + 1
                )

                for sub in sub_chains:
                    # Prepend this edge to the sub-chain
                    from_summary = self._get_node_summary(node_id, bucket_mgr)
                    to_summary = self._get_node_summary(to_id, bucket_mgr)

                    link = CausalLink(
                        from_id=node_id,
                        to_id=to_id,
                        from_summary=from_summary,
                        to_summary=to_summary,
                        confidence=weight,
                        edge_properties=edge.get("properties", {}),
                    )

                    new_chain = CausalChain(
                        chain=[link] + sub.chain,
                        total_confidence=round(
                            weight * sub.total_confidence, 3
                        ) if sub.total_confidence > 0 else weight,
                        domain=sub.domain,
                    )
                    new_chain.summary = self._generate_summary(new_chain)

                    sig = self._chain_signature(new_chain)
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        chains.append(new_chain)

            # Also create a terminal chain if this edge has at least min length
            # (handled by building 2-link chains directly)
            if depth == 0:  # Only from root level
                from_summary = self._get_node_summary(node_id, bucket_mgr)
                to_summary = self._get_node_summary(to_id, bucket_mgr)

                link = CausalLink(
                    from_id=node_id,
                    to_id=to_id,
                    from_summary=from_summary,
                    to_summary=to_summary,
                    confidence=weight,
                    edge_properties=edge.get("properties", {}),
                )

                chain = CausalChain(
                    chain=[link],
                    total_confidence=weight,
                )
                chain.summary = self._generate_summary(chain)

                sig = self._chain_signature(chain)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    chains.append(chain)

        return chains

    # ── Private: Chain building ────────────────────────────────

    def _build_chain(
        self,
        path_edges: list[dict],
        graph,
        bucket_mgr,
    ) -> CausalChain | None:
        """Build a CausalChain from a list of edges forming a path."""
        if not path_edges:
            return None

        links: list[CausalLink] = []
        total_conf = 1.0
        domain = ""

        for edge in path_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            weight = edge.get("weight", 0.5)
            props = edge.get("properties", {})

            from_summary = self._get_node_summary(from_id, bucket_mgr)
            to_summary = self._get_node_summary(to_id, bucket_mgr)

            links.append(CausalLink(
                from_id=from_id,
                to_id=to_id,
                from_summary=from_summary,
                to_summary=to_summary,
                confidence=weight,
                edge_properties=props,
            ))

            total_conf *= weight

            # Infer domain from edge properties
            if not domain:
                domain = props.get("thread_title", props.get("domain", ""))

        chain = CausalChain(
            chain=links,
            total_confidence=round(total_conf, 3),
            domain=domain,
        )
        chain.summary = self._generate_summary(chain)
        return chain

    # ── Private: Summary generation ────────────────────────────

    def _generate_summary(self, chain: CausalChain) -> str:
        """
        Generate a human-readable summary of a causal chain.

        Uses template-based summarization. Falls back to rule-based.
        LLM polishing can be added via llm_gateway in future.
        """
        if not chain.chain:
            return ""

        links = chain.chain
        parts: list[str] = []

        for i, link in enumerate(links):
            cause = link.from_summary[:40] or f"事件{link.from_id[:6]}"
            effect = link.to_summary[:40] or f"事件{link.to_id[:6]}"

            if i == 0:
                parts.append(f"「{cause}」")
            parts.append(f"→ 导致「{effect}」")

        summary = " ".join(parts)

        # Add confidence indicator
        if chain.total_confidence >= 0.7:
            summary += " （高置信度因果链）"
        elif chain.total_confidence >= 0.4:
            summary += " （中等置信度因果链）"
        else:
            summary += " （低置信度因果链·需验证）"

        return summary

    # ── Private: Helpers ───────────────────────────────────────

    @staticmethod
    def _get_node_summary(memory_id: str, bucket_mgr) -> str:
        """Get a short content summary for a node."""
        if bucket_mgr is None:
            return ""
        try:
            import asyncio
            if hasattr(bucket_mgr, 'read'):
                result = bucket_mgr.read(memory_id)
                if asyncio.iscoroutine(result):
                    return ""
                if isinstance(result, dict):
                    content = result.get("content", "")
                    # Extract first meaningful sentence
                    for sep in ["。", "！", "？", "\n", "；"]:
                        if sep in content:
                            content = content.split(sep)[0]
                            break
                    return content[:80]
        except Exception:
            pass
        return ""

    @staticmethod
    def _chain_signature(chain: CausalChain) -> str:
        """Generate a unique signature for deduplication."""
        ids = [link.from_id[:8] for link in chain.chain]
        ids.append(chain.chain[-1].to_id[:8] if chain.chain else "")
        return "→".join(ids)

    # ── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get summarizer statistics."""
        chains = list(self._chains.values())
        avg_depth = (
            sum(c.depth for c in chains) / max(len(chains), 1)
            if chains else 0
        )
        avg_conf = (
            sum(c.total_confidence for c in chains) / max(len(chains), 1)
            if chains else 0
        )

        return {
            "total_chains": len(self._chains),
            "summarizations": self._summarization_count,
            "avg_depth": round(avg_depth, 2),
            "avg_confidence": round(avg_conf, 3),
            "high_confidence_chains": sum(
                1 for c in chains if c.total_confidence >= 0.7
            ),
        }
