# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via Gemini API (OpenAI-compatible),
# stores them in ChromaDB with HNSW indexing for ANN search.
# 通过 Gemini API（OpenAI 兼容）生成 embedding，
# 存储在 ChromaDB 中，使用 HNSW 索引进行近似最近邻搜索。
#
# v2 (Track B): Replaced SQLite+JSON linear scan with ChromaDB.
# v2 (Track B)：用 ChromaDB 替换 SQLite+JSON 线性扫描。
#
# Depended on by: server.py, bucket_manager.py, app.py
# 被谁依赖：server.py, bucket_manager.py, app.py
# ============================================================

from __future__ import annotations

import os
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.embedding")


class ChromaBackend:
    """
    Thin wrapper around ChromaDB PersistentClient for embedding storage and ANN search.
    ChromaDB 持久化客户端的薄封装，用于 embedding 存储和近似最近邻搜索。
    """

    def __init__(self, persist_dir: str, collection_name: str = "memory_embeddings"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, bucket_id: str, embedding: list[float], metadata: dict | None = None) -> None:
        """Insert or update an embedding vector."""
        meta = metadata or {}
        self._collection.upsert(
            ids=[bucket_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def get(self, bucket_id: str) -> list[float] | None:
        """Retrieve a stored embedding by bucket ID. Returns None if not found."""
        result = self._collection.get(ids=[bucket_id], include=["embeddings"])
        if result and result["embeddings"] and len(result["embeddings"]) > 0:
            emb = result["embeddings"][0]
            return list(emb) if emb else None
        return None

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        ANN search returning (bucket_id, similarity_score) sorted by score desc.
        近似最近邻搜索，返回 (bucket_id, 相似度分数) 列表，按分数降序排列。
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["distances"],
        )
        if not result or not result["ids"] or not result["ids"][0]:
            return []
        ids = result["ids"][0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else []
        if not distances:
            return [(bid, 0.0) for bid in ids]
        # ChromaDB cosine distance → similarity: sim = 1 - distance
        scored = [(bid, round(1.0 - min(dist, 2.0), 6)) for bid, dist in zip(ids, distances)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def delete(self, bucket_id: str) -> None:
        """Remove an embedding from the collection."""
        self._collection.delete(ids=[bucket_id])

    def count(self) -> int:
        """Return the number of stored embeddings."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it (for testing)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


class EmbeddingEngine:
    """
    Embedding generation + ChromaDB vector storage + ANN search.
    向量生成 + ChromaDB 向量存储 + 近似最近邻搜索。
    """

    def __init__(self, config: dict, user_id: str = ""):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        self.api_key = (embed_cfg.get("api_key") or dehy_cfg.get("api_key") or "").strip()
        self.base_url = (
            (embed_cfg.get("base_url") or "").strip()
            or (dehy_cfg.get("base_url") or "").strip()
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)

        # --- Per-user ChromaDB directory (v2: namespace isolation) ---
        buckets_dir = config["buckets_dir"]
        if user_id:
            buckets_dir = os.path.join(buckets_dir, user_id)
        persist_dir = os.path.join(buckets_dir, "chroma_db")
        self.persist_dir = persist_dir
        self.user_id = user_id

        # --- Initialize client ---
        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,
            )
        else:
            self.client = None

        # --- Initialize ChromaDB backend ---
        self._backend: Optional[ChromaBackend] = None
        if self.enabled:
            try:
                self._backend = ChromaBackend(persist_dir=persist_dir)
                logger.info(
                    f"ChromaDB initialized at {persist_dir} "
                    f"({self._backend.count()} existing embeddings)"
                )
            except Exception as e:
                logger.warning(f"ChromaDB init failed, will try on first use: {e}")
                self._backend = None

    def _ensure_backend(self) -> Optional[ChromaBackend]:
        """Lazy-init ChromaDB backend if it failed during __init__."""
        if self._backend is None and self.enabled:
            try:
                self._backend = ChromaBackend(persist_dir=self.persist_dir)
            except Exception as e:
                logger.warning(f"ChromaDB lazy init failed: {e}")
        return self._backend

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in ChromaDB.
        为内容生成 embedding 并存入 ChromaDB。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content)
            if not embedding:
                return False
            backend = self._ensure_backend()
            if backend is None:
                return False
            backend.upsert(bucket_id, embedding)
            return True
        except Exception as e:
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def generate_and_store_batch(self, items: list[tuple[str, str]]) -> int:
        """
        Generate embeddings for multiple (bucket_id, content) pairs and store them.
        批量为多个 (bucket_id, content) 对生成 embedding 并存储。
        Returns count of successfully stored embeddings.
        """
        if not self.enabled or not items:
            return 0

        success_count = 0
        for bucket_id, content in items:
            if await self.generate_and_store(bucket_id, content):
                success_count += 1
        return success_count

    async def _generate_embedding(self, text: str) -> list[float]:
        """Call API to generate embedding vector."""
        truncated = text[:2000]
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted."""
        backend = self._ensure_backend()
        if backend is None:
            return
        try:
            backend.delete(bucket_id)
        except Exception as e:
            logger.warning(f"Failed to delete embedding {bucket_id}: {e}")

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        backend = self._ensure_backend()
        if backend is None:
            return None
        return backend.get(bucket_id)

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text using ANN (HNSW).
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        使用 ANN (HNSW) 搜索与查询文本相似的桶。
        返回 (bucket_id, 相似度分数) 列表，按分数降序排列。
        """
        if not self.enabled:
            return []

        backend = self._ensure_backend()
        if backend is None:
            return []

        try:
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        try:
            return backend.query(query_embedding, top_k=top_k)
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return []

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Used for offline comparison (e.g., dream connection hints).
        计算两个向量之间的余弦相似度。用于离线比较（如 dream 关联提示）。
        """
        import math
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
