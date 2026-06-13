# ============================================================
# Module: Memory Graph (memory_graph.py)
# L1: SQLite temporal knowledge graph for memory relationships.
# L1：SQLite 时序知识图谱 — 记忆间的多对多关系网络
#
# Design §3.2:
#   - Nodes: memory_id + properties (JSON)
#   - Edges: from_id → to_id with relation_type + valid_from/valid_until
#   - Edge expiry, NOT deletion — preserves full history
#   - Relation types: causal, thematic, temporal, emotional
#
# Key innovation from Zep/Graphiti:
#   - Dual temporal model: tracks both event time (t_valid/t_invalid)
#     AND ingestion time → supports precise time-point queries
#   - Edge expiration: when new fact contradicts old, mark old edge
#     as expired rather than deleting → "当时认为X，后来发现Y"
#
# Privacy: Per-user SQLite database (namespace isolation).
# ============================================================

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory_node import RelationType

logger = logging.getLogger("memory_palace.graph")


class MemoryGraph:
    """
    Per-user temporal knowledge graph for memory relationships.

    Stores:
      - Nodes: (memory_id → properties)
      - Edges: (from_id → to_id, relation_type, time range)

    All operations are synchronous (no async needed for SQLite).
    """

    def __init__(self, user_id: str = "", db_dir: str = "./buckets"):
        self.user_id = user_id
        db_path = Path(db_dir)
        if user_id:
            db_path = db_path / user_id
        os.makedirs(db_path, exist_ok=True)
        self.db_path = str(db_path / "memory_edges.db")
        self._init_db()

    def _init_db(self):
        """Create tables if not exists."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                memory_id TEXT PRIMARY KEY,
                properties_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation_type TEXT NOT NULL CHECK(
                    relation_type IN ('causal','thematic','temporal','emotional')
                ),
                valid_from TEXT NOT NULL,
                valid_until TEXT,           -- NULL = permanent (still valid)
                weight REAL NOT NULL DEFAULT 1.0,
                properties_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (from_id) REFERENCES graph_nodes(memory_id),
                FOREIGN KEY (to_id) REFERENCES graph_nodes(memory_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges(from_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges(to_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(relation_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_valid ON graph_edges(valid_until)
        """)
        conn.commit()
        conn.close()

    # ── Node CRUD ──────────────────────────────────────────

    def add_node(self, memory_id: str, properties: dict | None = None) -> str:
        """
        Add or update a memory node.
        添加或更新记忆节点。返回 memory_id。
        """
        now = datetime.now(timezone.utc).isoformat()
        props_json = json.dumps(properties or {}, ensure_ascii=False)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO graph_nodes (memory_id, properties_json, created_at, updated_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM graph_nodes WHERE memory_id=?), ?), ?)
        """, (memory_id, props_json, memory_id, now, now))
        conn.commit()
        conn.close()
        logger.debug(f"Graph node added: {memory_id}")
        return memory_id

    def get_node(self, memory_id: str) -> dict | None:
        """Get a node by memory_id."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT memory_id, properties_json, created_at, updated_at FROM graph_nodes WHERE memory_id=?",
            (memory_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "memory_id": row[0],
            "properties": json.loads(row[1]),
            "created_at": row[2],
            "updated_at": row[3],
        }

    def remove_node(self, memory_id: str):
        """Remove a node and all its edges."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM graph_edges WHERE from_id=? OR to_id=?", (memory_id, memory_id))
        conn.execute("DELETE FROM graph_nodes WHERE memory_id=?", (memory_id,))
        conn.commit()
        conn.close()
        logger.debug(f"Graph node removed: {memory_id}")

    # ── Edge CRUD ──────────────────────────────────────────

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: RelationType | str,
        valid_from: str | None = None,
        valid_until: str | None = None,
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> str:
        """
        Add a directed edge between two memory nodes.
        添加一条有向边。

        Args:
            from_id: source memory
            to_id: target memory
            relation_type: causal | thematic | temporal | emotional
            valid_from: when the relationship started (ISO timestamp)
            valid_until: when it ended (None = still valid)
            weight: edge weight (0-1, higher = stronger)
            properties: additional edge metadata
        """
        now = datetime.now(timezone.utc).isoformat()
        edge_id = uuid.uuid4().hex[:12]
        rt = relation_type.value if isinstance(relation_type, RelationType) else relation_type

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO graph_edges
                (edge_id, from_id, to_id, relation_type, valid_from, valid_until,
                 weight, properties_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge_id, from_id, to_id, rt,
            valid_from or now, valid_until,
            weight, json.dumps(properties or {}, ensure_ascii=False), now,
        ))
        conn.commit()
        conn.close()
        logger.debug(f"Graph edge added: {from_id} --[{rt}]--> {to_id}")
        return edge_id

    def expire_edge(self, edge_id: str, at: str | None = None):
        """
        Expire an edge — mark valid_until, NOT delete.
        边失效而非删除 —— 保留完整历史追溯。

        Design §3.2: When new fact contradicts old, mark old edge
        as expired, preserving the full history.
        """
        now = at or datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE graph_edges SET valid_until=? WHERE edge_id=? AND valid_until IS NULL",
            (now, edge_id),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Graph edge expired: {edge_id} at {now}")

    def get_edge(self, edge_id: str) -> dict | None:
        """Get a single edge by ID."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT edge_id, from_id, to_id, relation_type, valid_from, valid_until, "
            "weight, properties_json, created_at FROM graph_edges WHERE edge_id=?",
            (edge_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_edge(row)

    # ── Graph traversal ────────────────────────────────────

    def get_neighbors(
        self,
        memory_id: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
        active_only: bool = True,
    ) -> list[dict]:
        """
        Get neighboring nodes up to [depth] hops.
        获取邻居节点（深度可配）。

        Args:
            memory_id: center node
            depth: how many hops (1 = direct neighbors only)
            relation_types: filter by relation type(s)
            active_only: only return currently-valid edges
        """
        if depth < 1:
            return []

        seen: set[str] = {memory_id}
        frontier = [memory_id]
        results: list[dict] = []

        for _ in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                neighbors = self._get_direct_neighbors(
                    node_id, relation_types, active_only
                )
                for n in neighbors:
                    target = n["to_id"] if n["from_id"] == node_id else n["from_id"]
                    if target not in seen:
                        seen.add(target)
                        results.append(n)
                        next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break

        return results

    def _get_direct_neighbors(
        self,
        memory_id: str,
        relation_types: list[str] | None = None,
        active_only: bool = True,
    ) -> list[dict]:
        """Get direct (1-hop) neighbors."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT edge_id, from_id, to_id, relation_type, valid_from, valid_until,
                   weight, properties_json, created_at
            FROM graph_edges
            WHERE (from_id=? OR to_id=?)
        """
        params: list = [memory_id, memory_id]

        if active_only:
            query += " AND (valid_until IS NULL OR valid_until > datetime('now'))"
        if relation_types:
            placeholders = ",".join("?" * len(relation_types))
            query += f" AND relation_type IN ({placeholders})"
            params.extend(relation_types)

        query += " ORDER BY weight DESC, created_at DESC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [self._row_to_edge(r) for r in rows]

    def get_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 4,
    ) -> list[dict] | None:
        """
        Find a path between two memories (BFS).
        两个记忆之间的关联路径。
        Returns list of edges forming the path, or None if no path.
        """
        if from_id == to_id:
            return []

        # BFS with path tracking
        from collections import deque
        queue = deque([(from_id, [])])
        visited = {from_id}

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            neighbors = self._get_direct_neighbors(current, active_only=True)
            for edge in neighbors:
                next_node = edge["to_id"] if edge["from_id"] == current else edge["from_id"]
                if next_node in visited:
                    continue
                new_path = path + [edge]
                if next_node == to_id:
                    return new_path
                visited.add(next_node)
                queue.append((next_node, new_path))

        return None

    # ── Bulk operations ────────────────────────────────────

    def get_all_edges_for_node(self, memory_id: str) -> list[dict]:
        """Get all edges (active and expired) for a node."""
        return self._get_direct_neighbors(memory_id, active_only=False)

    def get_edges_by_type(
        self,
        relation_type: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get all edges of a specific type."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT edge_id, from_id, to_id, relation_type, valid_from, valid_until,
                      weight, properties_json, created_at
               FROM graph_edges WHERE relation_type=?
               ORDER BY created_at DESC LIMIT ?""",
            (relation_type, limit),
        ).fetchall()
        conn.close()
        return [self._row_to_edge(r) for r in rows]

    def get_graph_stats(self) -> dict:
        """Get graph statistics for pulse endpoint."""
        conn = sqlite3.connect(self.db_path)
        node_count = conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        active_edge_count = conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE valid_until IS NULL"
        ).fetchone()[0]
        conn.close()
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "active_edge_count": active_edge_count,
            "expired_edge_count": edge_count - active_edge_count,
        }

    # ── Similarity-based edge creation ─────────────────────

    def create_similarity_edges(
        self,
        memory_id: str,
        similar_ids: list[tuple[str, float]],  # [(id, similarity_score)]
        threshold: float = 0.5,
    ) -> int:
        """
        Create EMOTIONAL edges for embedding-similar memories.
        Called after embedding generation to build the similarity network.
        """
        count = 0
        for similar_id, score in similar_ids:
            if similar_id == memory_id:
                continue
            if score >= threshold:
                self.add_edge(
                    from_id=memory_id,
                    to_id=similar_id,
                    relation_type=RelationType.EMOTIONAL,
                    weight=score,
                    properties={"source": "embedding_similarity", "score": score},
                )
                count += 1
        return count

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _row_to_edge(row: tuple) -> dict:
        return {
            "edge_id": row[0],
            "from_id": row[1],
            "to_id": row[2],
            "relation_type": row[3],
            "valid_from": row[4],
            "valid_until": row[5],
            "weight": row[6],
            "properties": json.loads(row[7]) if row[7] else {},
            "created_at": row[8],
        }
