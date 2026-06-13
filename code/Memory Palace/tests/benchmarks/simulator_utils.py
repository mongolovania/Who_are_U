# ============================================================
# Simulator Utilities — Shared across all simulator modules
# Extracted from algorithm_simulators.py to break circular import
# ============================================================

import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from tests.benchmarks.benchmark_dataset import BenchmarkMemory


# ── Shared BM25 Engine ───────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Chinese+English tokenizer — character-level for CJK, word-level for EN."""
    tokens = []
    text_lower = text.lower()
    en_tokens = re.findall(r"[a-zA-Z]+|\d+", text_lower)
    tokens.extend(en_tokens)
    cjk_chars = re.findall(r"[一-鿿]", text_lower)
    tokens.extend(cjk_chars)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    return tokens


class SharedBM25Index:
    """
    Singleton BM25 index shared across all simulators.
    Uses rank_bm25.BM25Okapi for true probabilistic scoring (k1=1.5, b=0.75).
    """

    def __init__(self, memories: list[BenchmarkMemory]):
        self._memories = memories
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._build()

    def _build(self):
        self._corpus = [_tokenize(m.content) for m in self._memories]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str) -> list[tuple[int, float]]:
        """Return [(memory_index, bm25_score), ...] sorted desc."""
        if self._bm25 is None or not self._corpus:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        raw_scores = self._bm25.get_scores(q_tokens)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        scored = [(i, float(s / max_score)) for i, s in enumerate(raw_scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_document_tokens(self, idx: int) -> set[str]:
        if 0 <= idx < len(self._corpus):
            return set(self._corpus[idx])
        return set()

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)


# ── Shared Utilities ─────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _keyword_overlap_score(query: str, text: str) -> float:
    """Simple keyword overlap score for test backward compatibility."""
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / len(q_tokens)


def _days_ago(date_str: str) -> float:
    if not date_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(date_str)
        return (datetime.now() - dt).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return 0.0


# ── Shared Emotion Engine ────────────────────────────────────

_VALENCE_KEYWORDS: dict[str, float] = {}
_AROUSAL_KEYWORDS: dict[str, float] = {}
_EMOTION_QUERY_WORDS: set[str] = set()
_CAUSAL_QUERY_WORDS: set[str] = set()
_TEMPORAL_QUERY_WORDS: set[str] = set()
_CROSS_REFERENCE_WORDS: set[str] = set()
_NEGATION_WORDS = {"不", "没", "无", "非", "别", "莫", "未", "否", "休", "从不", "绝不"}

try:
    from retrieval_engine import (
        _VALENCE_KEYWORDS as _vk,
        _AROUSAL_KEYWORDS as _ak,
        _EMOTION_QUERY_WORDS as _eqw,
        _CAUSAL_QUERY_WORDS as _cqw,
        _TEMPORAL_QUERY_WORDS as _tqw,
        _CROSS_REFERENCE_WORDS as _crw,
    )
    _VALENCE_KEYWORDS = _vk
    _AROUSAL_KEYWORDS = _ak
    _EMOTION_QUERY_WORDS = _eqw
    _CAUSAL_QUERY_WORDS = _cqw
    _TEMPORAL_QUERY_WORDS = _tqw
    _CROSS_REFERENCE_WORDS = _crw
except ImportError:
    pass


def _extract_query_emotion(query: str) -> tuple[float, float]:
    """Extract (valence, arousal) from query using emotion keyword matching."""
    if not query or not query.strip():
        return (0.5, 0.3)
    valence_sum, valence_count = 0.0, 0
    arousal_sum, arousal_count = 0.0, 0

    for kw, v in _VALENCE_KEYWORDS.items():
        if kw in query:
            valence_sum += v
            valence_count += 1
    for kw, a in _AROUSAL_KEYWORDS.items():
        if kw in query:
            arousal_sum += a
            arousal_count += 1

    if valence_count == 0 and arousal_count == 0:
        return (0.5, 0.3)
    valence = valence_sum / valence_count if valence_count > 0 else 0.5
    arousal = arousal_sum / arousal_count if arousal_count > 0 else 0.3
    return (max(0.0, min(1.0, valence)), max(0.0, min(1.0, arousal)))


def _emotion_resonance(q_val: float, q_ar: float, m_val: float, m_ar: float) -> float:
    """Russell circumplex distance → resonance 0-1."""
    dv = q_val - m_val
    da = q_ar - m_ar
    distance = math.sqrt(dv * dv + da * da)
    return max(0.0, 1.0 - distance / math.sqrt(2))


def _infer_query_category(query: str) -> str:
    """Infer query type: emotional | causal | temporal | cross_reference | factual."""
    if not query:
        return "factual"
    scores: dict[str, int] = {}
    for w in _EMOTION_QUERY_WORDS:
        if w in query:
            scores["emotional"] = scores.get("emotional", 0) + 1
    for w in _CAUSAL_QUERY_WORDS:
        if w in query:
            scores["causal"] = scores.get("causal", 0) + 1
    for w in _TEMPORAL_QUERY_WORDS:
        if w in query:
            scores["temporal"] = scores.get("temporal", 0) + 1
    for w in _CROSS_REFERENCE_WORDS:
        if w in query:
            scores["cross_reference"] = scores.get("cross_reference", 0) + 1
    return max(scores, key=scores.get) if scores else "factual"
