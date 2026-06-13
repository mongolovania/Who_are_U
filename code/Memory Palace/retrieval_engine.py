# ============================================================
# Module: Retrieval Engine (retrieval_engine.py)
# L2: DDA-adaptive multi-path memory retrieval.
# L2：DDA 自适应多路检索引擎
#
# Design §5.1:
#   COLD: return ALL (no data to filter)
#   WARM: semantic similarity + time ranking (2 paths)
#   HOT:  vector(30%) + BM25(15%) + graph(25%) + emotion(15%) + random(15%)
#         (4-path fusion with content preservation)
#   RICH: HOT + Working Self re-rank (5-path)
#
# v7 (Phase 1) improvements:
#   P0-1: Content-preserving fusion — results carry content through all paths
#   P0-2: Emotion resonance wired into retrieval (MAGMA-style query inference)
#   P0-3: Typed graph traversal with depth=3 + relation_type filtering
#
# All retrieval is zero-LLM (<500ms target for sync path).
# LLM is only used in the async path for deep analysis.
# ============================================================

from __future__ import annotations

__version__ = "9.0.0"

import logging
import math
import random
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from memory_node import MemoryNode, DDILevel, DDAStrategy

logger = logging.getLogger("memory_palace.retrieval")


# ── Tokenizer (shared utility) ───────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Chinese+English tokenizer — character-level for CJK, word-level for EN."""
    import re
    tokens = []
    text_lower = text.lower()
    en_tokens = re.findall(r"[a-zA-Z]+|\d+", text_lower)
    tokens.extend(en_tokens)
    cjk_chars = re.findall(r"[一-鿿]", text_lower)
    tokens.extend(cjk_chars)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    return tokens


# ── BM25 Retriever (Track B: replaces token overlap) ─────────

class BM25Retriever:
    """
    True BM25 probabilistic retrieval using rank_bm25 library.
    基于 rank_bm25 库的真正 BM25 概率检索引擎。

    Replaces the old token overlap approach (len(A&B)/len(A))
    with proper BM25Okapi scoring (k1=1.5, b=0.75).
    用正确的 BM25Okapi 评分（k1=1.5, b=0.75）替代旧的 token 重叠方法。
    """

    def __init__(self):
        self._corpus_tokens: list[list[str]] = []
        self._corpus_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self, documents: list[tuple[str, str]]) -> None:
        """
        Build BM25 index from (doc_id, content) pairs.
        从 (doc_id, content) 对构建 BM25 索引。
        """
        self._corpus_tokens = []
        self._corpus_ids = []
        for doc_id, content in documents:
            tokens = _tokenize(content)
            if tokens:  # skip empty documents
                self._corpus_tokens.append(tokens)
                self._corpus_ids.append(doc_id)

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search for documents matching query using BM25.
        Returns list of (doc_id, score) sorted by score desc.
        使用 BM25 搜索与查询匹配的文档。
        返回 (doc_id, 分数) 列表，按分数降序排列。
        """
        if self._bm25 is None or not self._corpus_tokens:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # Normalize scores to 0~1 range using sigmoid-like normalization
        # BM25 scores are unbounded; typical range is 0~20 for well-matched docs
        normalized = 1.0 / (1.0 + np.exp(-scores / 3.0))

        # Sort by score desc and return top_k
        indexed = list(enumerate(normalized))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed[:top_k]:
            if score > 0.0:  # filter zero-score results
                results.append((self._corpus_ids[idx], float(score)))

        return results

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_ids)

# ═══════════════════════════════════════════════════════════════
# Query emotion keyword maps — expanded with multi-theory grounding.
#
# Theoretical sources (all peer-reviewed, high-impact):
#   1. Russell (1980), JPSP — Circumplex Model: 28 core words × (valence, arousal)
#   2. Plutchik (1980) — Psycho-evolutionary theory: 8 primary emotions × 3 intensities
#   3. Bradley & Lang (1999) — ANEW: 1,034 English words rated on valence/arousal/dominance
#   4. Warriner, Kuperman & Brysbaert (2013), Behav Res Methods — 13,915 English lemmas
#   5. Mohammad & Turney (2013), Computational Intelligence — NRC Emotion Lexicon: 14,182 words × 8 Plutchik emotions
#   6. 王一牛, 周立明, 罗跃嘉 (2008), 中国心理卫生杂志 — CAWS: 1,500 Chinese two-character words (valence/arousal/dominance)
#   7. 徐琳宏, 林鸿飞等 (2008), 情报学报 — DLUT Emotion Ontology: 27,466 Chinese words, 7大类×21小类
#   8. Chan & Tse (2024), Behav Res Methods — 25,281 Chinese two-character words (valence/arousal norms, open data)
#   9. Chan & Tse (2025), Behav Res Methods — 3,971 Chinese characters (valence/arousal norms, open data)
#
# Organization: DLUT 7-category system (乐/好/怒/哀/惧/恶/惊), cross-mapped to:
#   - Plutchik 8 primary emotions (Joy/Trust/Anger/Sadness/Fear/Disgust/Surprise/Anticipation)
#   - Russell circumplex quadrant (HA+P / HA+U / LA+U / LA+P)
#   - Intensity levels (mild / moderate / intense) per Plutchik
#
# Valence scores: empirically grounded in CAWS + Chan & Tse norms where available;
#   inferred from Russell circumplex angular position for remaining words.
#   Scale: 0.0 (extremely negative) → 0.5 (neutral) → 1.0 (extremely positive)
#
# Arousal scores: empirically grounded in CAWS + Chan & Tse norms where available.
#   Scale: 0.0 (completely calm/sleepy) → 1.0 (extremely activated/aroused)
# ═══════════════════════════════════════════════════════════════

# ── Category 1: 乐 (Joy / Happiness) — DLUT大类 PA+PE ──────
# Plutchik: Joy (serenity → joy → ecstasy). Russell: Q2 (HA+P) & Q4 (LA+P).
# CAWS norms: 开心=0.87v/0.65a, 快乐=0.85v/0.60a, 满足=0.78v/0.28a

_VALENCE_JOY: dict[str, float] = {
    # Intense (ecstasy-level, very high arousal + valence)
    "狂喜": 0.95, "欣喜若狂": 0.95, "欢欣鼓舞": 0.92, "雀跃": 0.90,
    # Moderate (basic joy, high-moderate arousal)
    "开心": 0.90, "高兴": 0.88, "快乐": 0.87, "兴奋": 0.90,
    "欢喜": 0.86, "愉悦": 0.85, "喜悦": 0.88, "兴高采烈": 0.90,
    "欢快": 0.85, "乐呵呵": 0.87, "乐陶陶": 0.88, "陶然": 0.82,
    # Mild (serenity-level, low arousal)
    "满足": 0.80, "满意": 0.78, "惬意": 0.82, "舒适": 0.75,
    "愉快": 0.84, "舒心": 0.80, "称心": 0.78, "遂心": 0.80,
    # Security/peace (安心 subcategory PE)
    "安心": 0.78, "踏实": 0.72, "宽心": 0.75, "放心": 0.74,
    "安稳": 0.73, "平和": 0.68, "安然": 0.70, "宁静": 0.67,
    # Compound joy expressions
    "开心得不得了": 0.93, "心满意足": 0.82, "心旷神怡": 0.88,
    "喜出望外": 0.92, "乐不可支": 0.92, "乐不思蜀": 0.85,
    "欢笑": 0.86, "欢声笑语": 0.90,
}

_AROUSAL_JOY: dict[str, float] = {
    "狂喜": 0.95, "欣喜若狂": 0.95, "欢欣鼓舞": 0.88, "雀跃": 0.85,
    "开心": 0.70, "高兴": 0.65, "快乐": 0.62, "兴奋": 0.90,
    "欢喜": 0.68, "愉悦": 0.60, "喜悦": 0.72, "兴高采烈": 0.88,
    "欢快": 0.65, "乐呵呵": 0.62, "乐陶陶": 0.60, "陶然": 0.45,
    "满足": 0.30, "满意": 0.28, "惬意": 0.32, "舒适": 0.25,
    "愉快": 0.55, "舒心": 0.35, "称心": 0.30, "遂心": 0.32,
    "安心": 0.25, "踏实": 0.28, "宽心": 0.22, "放心": 0.25,
    "安稳": 0.22, "平和": 0.20, "安然": 0.22, "宁静": 0.18,
    "开心得不得了": 0.92, "心满意足": 0.35, "心旷神怡": 0.52,
    "喜出望外": 0.88, "乐不可支": 0.88, "乐不思蜀": 0.70,
    "欢笑": 0.75, "欢声笑语": 0.78,
}

# ── Category 2: 好 (Positive/Good) — DLUT大类 PD/PH/PG/PB/PK ─
# Plutchik: Trust + subsets. Russell: Q4 & Q2 (pleasant, variable arousal).
# CAWS-like: 温暖=0.82v/0.32a, 感动=0.85v/0.70a

_VALENCE_GOOD: dict[str, float] = {
    # Trust/faith 相信 PG
    "信任": 0.78, "信赖": 0.80, "可靠": 0.76, "相信": 0.72,
    "信仰": 0.70, "信心": 0.75, "信念": 0.73, "笃信": 0.76,
    # Respect/admiration 尊敬 PD
    "尊敬": 0.75, "敬爱": 0.80, "敬佩": 0.82, "钦佩": 0.80,
    "仰慕": 0.78, "尊重": 0.74, "崇敬": 0.82, "敬仰": 0.80,
    # Praise/appreciation 赞扬 PH
    "赞赏": 0.78, "赞美": 0.80, "欣赏": 0.80, "称赞": 0.76,
    "叹服": 0.82, "好评": 0.74, "喝彩": 0.84,
    # Affection/love 喜爱 PB
    "喜欢": 0.80, "喜爱": 0.82, "爱": 0.88, "热爱": 0.90,
    "倾慕": 0.84, "爱慕": 0.86, "溺爱": 0.78, "宠爱": 0.82,
    "心动": 0.85, "依恋": 0.76, "眷恋": 0.78,
    # Well-wishing 祝愿 PK
    "期待": 0.75, "希望": 0.72, "盼望": 0.76, "渴望": 0.78,
    "憧憬": 0.80, "向往": 0.78, "祝愿": 0.82, "期盼": 0.76,
    # Warmth & gratitude (cross-category)
    "温暖": 0.80, "感动": 0.83, "感激": 0.85, "感恩": 0.84,
    "关怀": 0.80, "体贴": 0.78, "亲切": 0.76, "温馨": 0.82,
    # Pride & achievement (cross-category)
    "自豪": 0.88, "骄傲": 0.82, "得意": 0.78, "光荣": 0.84,
    "荣耀": 0.85, "扬眉吐气": 0.88,
    # Relief & release
    "如释重负": 0.90, "轻松": 0.78, "释然": 0.80, "解放": 0.82,
    "解脱": 0.84, "释怀": 0.78, "松了一口气": 0.85,
}

_AROUSAL_GOOD: dict[str, float] = {
    "信任": 0.30, "信赖": 0.32, "可靠": 0.28, "相信": 0.35,
    "信仰": 0.38, "信心": 0.40, "信念": 0.38, "笃信": 0.42,
    "尊敬": 0.32, "敬爱": 0.42, "敬佩": 0.45, "钦佩": 0.44,
    "仰慕": 0.52, "尊重": 0.28, "崇敬": 0.48, "敬仰": 0.46,
    "赞赏": 0.48, "赞美": 0.52, "欣赏": 0.45, "称赞": 0.42,
    "叹服": 0.55, "好评": 0.40, "喝彩": 0.68,
    "喜欢": 0.55, "喜爱": 0.58, "爱": 0.65, "热爱": 0.75,
    "倾慕": 0.62, "爱慕": 0.64, "溺爱": 0.55, "宠爱": 0.55,
    "心动": 0.72, "依恋": 0.48, "眷恋": 0.50,
    "期待": 0.72, "希望": 0.55, "盼望": 0.62, "渴望": 0.70,
    "憧憬": 0.65, "向往": 0.60, "祝愿": 0.52, "期盼": 0.60,
    "温暖": 0.35, "感动": 0.78, "感激": 0.65, "感恩": 0.58,
    "关怀": 0.42, "体贴": 0.38, "亲切": 0.35, "温馨": 0.38,
    "自豪": 0.75, "骄傲": 0.72, "得意": 0.65, "光荣": 0.62,
    "荣耀": 0.68, "扬眉吐气": 0.80,
    "如释重负": 0.28, "轻松": 0.22, "释然": 0.25, "解放": 0.55,
    "解脱": 0.45, "释怀": 0.30, "松了一口气": 0.35,
}

# ── Category 3: 怒 (Anger) — DLUT大类 NA ──────────────────
# Plutchik: Anger (annoyance → anger → rage). Russell: Q1 (HA+U).
# CAWS: 愤怒=0.12v/0.88a

_VALENCE_ANGER: dict[str, float] = {
    # Intense (rage)
    "暴怒": 0.04, "狂怒": 0.05, "激愤": 0.08, "震怒": 0.05,
    "勃然大怒": 0.04, "怒气冲天": 0.05,
    # Moderate (basic anger)
    "愤怒": 0.15, "气愤": 0.16, "恼火": 0.15, "恼怒": 0.14,
    "发怒": 0.12, "动怒": 0.14, "气恼": 0.16, "愤慨": 0.13,
    "愤懑": 0.12, "愤然": 0.13, "不满": 0.22, "不平": 0.20,
    "可气": 0.18, "可恼": 0.16,
    # Mild (annoyance/irritation)
    "不悦": 0.25, "烦躁": 0.22, "不耐烦": 0.24, "急躁": 0.22,
    "烦心": 0.24, "闹心": 0.22, "烦闷": 0.23,
    # Compound anger
    "咬牙切齿": 0.06, "怒火中烧": 0.05, "火冒三丈": 0.05,
    "怒不可遏": 0.04, "大发雷霆": 0.05, "愤愤不平": 0.14,
}

_AROUSAL_ANGER: dict[str, float] = {
    "暴怒": 0.95, "狂怒": 0.95, "激愤": 0.92, "震怒": 0.93,
    "勃然大怒": 0.95, "怒气冲天": 0.93,
    "愤怒": 0.90, "气愤": 0.85, "恼火": 0.82, "恼怒": 0.84,
    "发怒": 0.86, "动怒": 0.82, "气恼": 0.80, "愤慨": 0.84,
    "愤懑": 0.80, "愤然": 0.82, "不满": 0.62, "不平": 0.65,
    "可气": 0.68, "可恼": 0.65,
    "不悦": 0.50, "烦躁": 0.55, "不耐烦": 0.58, "急躁": 0.62,
    "烦心": 0.52, "闹心": 0.55, "烦闷": 0.50,
    "咬牙切齿": 0.92, "怒火中烧": 0.93, "火冒三丈": 0.92,
    "怒不可遏": 0.94, "大发雷霆": 0.92, "愤愤不平": 0.78,
}

# ── Category 4: 哀 (Sadness) — DLUT大类 NB/NJ/NH/PF ────────
# Plutchik: Sadness (pensiveness → sadness → grief). Russell: Q3 (LA+U).
# CAWS: 悲伤=0.10v/0.55a, 难过=0.12v/0.48a

_VALENCE_SADNESS: dict[str, float] = {
    # Intense (grief/despair)
    "悲痛": 0.06, "沉痛": 0.08, "悲恸": 0.05, "撕心裂肺": 0.04,
    "悲痛欲绝": 0.03, "痛不欲生": 0.03,
    # Moderate (basic sadness)
    "悲伤": 0.13, "难过": 0.15, "伤心": 0.12, "忧伤": 0.16,
    "悲哀": 0.12, "悲凉": 0.14, "凄楚": 0.10, "凄然": 0.13,
    "心酸": 0.10, "酸楚": 0.12,
    # Mild (pensiveness/melancholy)
    "低落": 0.20, "消沉": 0.18, "沮丧": 0.17, "失落": 0.20,
    "怅然": 0.22, "黯然": 0.18, "惆怅": 0.20, "惘然": 0.24,
    "忧郁": 0.18, "抑郁": 0.12, "沉闷": 0.24, "闷闷不乐": 0.20,
    # Disappointment/regret 失望 NJ + 疚 NH
    "失望": 0.20, "绝望": 0.05, "遗憾": 0.22, "惋惜": 0.20,
    "悔恨": 0.12, "后悔": 0.15, "懊悔": 0.14, "懊恼": 0.16,
    "内疚": 0.13, "愧疚": 0.12, "自责": 0.14, "惭愧": 0.20,
    "过意不去": 0.22, "问心有愧": 0.15,
    # Longing/rumination 思 PF
    "思念": 0.42, "想念": 0.44, "牵挂": 0.38, "惦念": 0.40,
    "怀念": 0.42, "追忆": 0.40, "缅怀": 0.38,
    # Loneliness
    "孤独": 0.18, "寂寞": 0.18, "孤单": 0.18, "无助": 0.12,
    "凄凉": 0.12, "落寞": 0.20,
}

_AROUSAL_SADNESS: dict[str, float] = {
    "悲痛": 0.80, "沉痛": 0.75, "悲恸": 0.85, "撕心裂肺": 0.92,
    "悲痛欲绝": 0.92, "痛不欲生": 0.90,
    "悲伤": 0.55, "难过": 0.45, "伤心": 0.55, "忧伤": 0.48,
    "悲哀": 0.52, "悲凉": 0.45, "凄楚": 0.55, "凄然": 0.50,
    "心酸": 0.55, "酸楚": 0.50,
    "低落": 0.35, "消沉": 0.32, "沮丧": 0.40, "失落": 0.38,
    "怅然": 0.32, "黯然": 0.30, "惆怅": 0.32, "惘然": 0.28,
    "忧郁": 0.38, "抑郁": 0.35, "沉闷": 0.28, "闷闷不乐": 0.35,
    "失望": 0.42, "绝望": 0.92, "遗憾": 0.38, "惋惜": 0.38,
    "悔恨": 0.55, "后悔": 0.42, "懊悔": 0.48, "懊恼": 0.52,
    "内疚": 0.48, "愧疚": 0.50, "自责": 0.52, "惭愧": 0.42,
    "过意不去": 0.38, "问心有愧": 0.42,
    "思念": 0.42, "想念": 0.42, "牵挂": 0.45, "惦念": 0.38,
    "怀念": 0.35, "追忆": 0.30, "缅怀": 0.28,
    "孤独": 0.38, "寂寞": 0.36, "孤单": 0.35, "无助": 0.55,
    "凄凉": 0.48, "落寞": 0.38,
}

# ── Category 5: 惧 (Fear) — DLUT大类 NI/NC/NG ─────────────
# Plutchik: Fear (apprehension → fear → terror). Russell: Q1 (HA+U).
# CAWS: 恐惧=0.08v/0.90a, 害怕=0.10v/0.82a

_VALENCE_FEAR: dict[str, float] = {
    # Intense (terror/panic)
    "恐惧": 0.08, "惊恐": 0.06, "恐慌": 0.07, "惧怕": 0.08,
    "胆战心惊": 0.05, "魂飞魄散": 0.04, "毛骨悚然": 0.08,
    "心惊肉跳": 0.06, "惶惶不安": 0.08,
    # Moderate (basic fear / anxiety)
    "害怕": 0.10, "畏惧": 0.12, "胆怯": 0.14, "惧怕": 0.10,
    "畏缩": 0.13, "忌惮": 0.15,
    # Mild (apprehension/worry)
    "担心": 0.25, "担忧": 0.22, "不安": 0.22, "忐忑": 0.20,
    "顾虑": 0.26, "忧心": 0.22, "烦忧": 0.24,
    # Anxiety spectrum (慌 NI)
    "焦虑": 0.20, "紧张": 0.25, "慌张": 0.20, "心慌": 0.18,
    "慌乱": 0.18, "惶恐": 0.15, "坐立不安": 0.16, "不知所措": 0.22,
    # Shame/shyness 羞 NG
    "害羞": 0.35, "羞怯": 0.32, "羞涩": 0.34, "害臊": 0.30,
    "羞愧": 0.15, "羞耻": 0.12, "丢脸": 0.14, "难堪": 0.18,
    "尴尬": 0.28, "窘迫": 0.22,
    # Panic/dread compound
    "心惊胆战": 0.06, "惶惶不可终日": 0.06, "六神无主": 0.12,
}

_AROUSAL_FEAR: dict[str, float] = {
    "恐惧": 0.90, "惊恐": 0.93, "恐慌": 0.90, "惧怕": 0.88,
    "胆战心惊": 0.94, "魂飞魄散": 0.95, "毛骨悚然": 0.88,
    "心惊肉跳": 0.88, "惶惶不安": 0.85,
    "害怕": 0.88, "畏惧": 0.80, "胆怯": 0.75, "惧怕": 0.85,
    "畏缩": 0.72, "忌惮": 0.68,
    "担心": 0.70, "担忧": 0.72, "不安": 0.72, "忐忑": 0.76,
    "顾虑": 0.58, "忧心": 0.68, "烦忧": 0.62,
    "焦虑": 0.85, "紧张": 0.80, "慌张": 0.78, "心慌": 0.80,
    "慌乱": 0.78, "惶恐": 0.82, "坐立不安": 0.82, "不知所措": 0.72,
    "害羞": 0.55, "羞怯": 0.52, "羞涩": 0.52, "害臊": 0.55,
    "羞愧": 0.62, "羞耻": 0.65, "丢脸": 0.65, "难堪": 0.60,
    "尴尬": 0.55, "窘迫": 0.58,
    "心惊胆战": 0.92, "惶惶不可终日": 0.88, "六神无主": 0.82,
}

# ── Category 6: 恶 (Disgust/Aversion) — DLUT大类 NE/ND/NN/NK/NL ─
# Plutchik: Disgust (boredom → disgust → loathing). Russell: Q3↔Q1 (unpleasant).
# Also covers: 烦闷NE, 憎恶ND, 贬责NN, 妒忌NK, 怀疑NL

_VALENCE_DISGUST: dict[str, float] = {
    # Intense (loathing)
    "憎恶": 0.06, "厌恶": 0.08, "憎恨": 0.05, "反感": 0.12,
    "恶心": 0.08, "厌烦": 0.14, "深恶痛绝": 0.04, "恨之入骨": 0.03,
    # Moderate (basic disgust)
    "讨厌": 0.16, "嫌弃": 0.14, "鄙视": 0.15, "轻视": 0.20,
    "鄙夷": 0.14, "蔑视": 0.16, "看轻": 0.20, "瞧不起": 0.15,
    # Mild (boredom/aversion) 烦闷 NE
    "无聊": 0.30, "厌倦": 0.22, "腻烦": 0.22, "生厌": 0.20,
    "憋闷": 0.24, "窝火": 0.18, "窝囊": 0.18,
    # Criticism/blame 贬责 NN
    "谴责": 0.18, "责备": 0.20, "责难": 0.18, "不满": 0.22,
    # Jealousy/envy 妒忌 NK
    "妒忌": 0.14, "嫉妒": 0.15, "羡慕": 0.58, "眼红": 0.18,
    "吃醋": 0.20, "嫉恨": 0.12,
    # Suspicion/doubt 怀疑 NL
    "怀疑": 0.28, "猜疑": 0.25, "疑虑": 0.26, "多心": 0.28,
    "将信将疑": 0.32, "疑神疑鬼": 0.20,
    # Compound
    "鄙视链": 0.16, "阴阳怪气": 0.18, "冷嘲热讽": 0.15,
}

_AROUSAL_DISGUST: dict[str, float] = {
    "憎恶": 0.80, "厌恶": 0.72, "憎恨": 0.82, "反感": 0.68,
    "恶心": 0.72, "厌烦": 0.55, "深恶痛绝": 0.88, "恨之入骨": 0.92,
    "讨厌": 0.55, "嫌弃": 0.55, "鄙视": 0.58, "轻视": 0.48,
    "鄙夷": 0.58, "蔑视": 0.60, "看轻": 0.48, "瞧不起": 0.58,
    "无聊": 0.30, "厌倦": 0.35, "腻烦": 0.38, "生厌": 0.36,
    "憋闷": 0.42, "窝火": 0.58, "窝囊": 0.52,
    "谴责": 0.62, "责备": 0.55, "责难": 0.58, "不满": 0.62,
    "妒忌": 0.62, "嫉妒": 0.60, "羡慕": 0.52, "眼红": 0.58,
    "吃醋": 0.58, "嫉恨": 0.65,
    "怀疑": 0.42, "猜疑": 0.45, "疑虑": 0.42, "多心": 0.40,
    "将信将疑": 0.45, "疑神疑鬼": 0.55,
    "鄙视链": 0.52, "阴阳怪气": 0.58, "冷嘲热讽": 0.62,
}

# ── Category 7: 惊 (Surprise) — DLUT大类 PC ─────────────────
# Plutchik: Surprise (distraction → surprise → amazement). Russell: Q2↔Q1.
# Surprise valence is context-dependent; base valence near-neutral.

_VALENCE_SURPRISE: dict[str, float] = {
    # Intense (amazement / shock)
    "震惊": 0.25, "惊愕": 0.28, "大吃一惊": 0.30, "目瞪口呆": 0.28,
    "瞠目结舌": 0.28, "震撼": 0.45, "轰动": 0.48,
    # Moderate (basic surprise)
    "惊讶": 0.38, "惊奇": 0.42, "诧异": 0.36, "惊异": 0.40,
    "吃惊": 0.35, "意外": 0.38, "愕然": 0.32,
    # Mild (curiosity/distraction)
    "好奇": 0.60, "新鲜": 0.62, "奇妙": 0.68, "意想不到": 0.42,
    # Positive-leaning surprise
    "惊喜": 0.85, "喜出望外": 0.92, "惊艳": 0.82, "赞叹": 0.78,
    "难以置信": 0.35, "不可思议": 0.40,
    # Negative-leaning surprise
    "惊恐": 0.06, "惊吓": 0.10, "吓一跳": 0.18, "骇然": 0.12,
}

_AROUSAL_SURPRISE: dict[str, float] = {
    "震惊": 0.90, "惊愕": 0.85, "大吃一惊": 0.88, "目瞪口呆": 0.86,
    "瞠目结舌": 0.85, "震撼": 0.78, "轰动": 0.72,
    "惊讶": 0.72, "惊奇": 0.68, "诧异": 0.68, "惊异": 0.72,
    "吃惊": 0.72, "意外": 0.65, "愕然": 0.72,
    "好奇": 0.55, "新鲜": 0.50, "奇妙": 0.52, "意想不到": 0.65,
    "惊喜": 0.85, "喜出望外": 0.88, "惊艳": 0.78, "赞叹": 0.65,
    "难以置信": 0.75, "不可思议": 0.72,
    "惊恐": 0.93, "惊吓": 0.88, "吓一跳": 0.85, "骇然": 0.88,
}

# ── Category 8: 综合/复杂情绪 (Complex/Compound Emotions) ──
# Mixed emotions not cleanly fitting one DLUT category.
# Includes Plutchik dyads: shame, guilt, envy already covered; adding more.

_VALENCE_COMPLEX: dict[str, float] = {
    # Confusion/ambivalence
    "迷茫": 0.25, "困惑": 0.30, "矛盾": 0.28, "纠结": 0.24,
    "犹豫": 0.30, "彷徨": 0.22, "迟疑": 0.32,
    # Pressure/overwhelm (非稳态负荷 markers)
    "压抑": 0.15, "崩溃": 0.05, "疲惫": 0.18, "倦怠": 0.20,
    "身心俱疲": 0.10, "筋疲力尽": 0.12, "累垮": 0.10,
    "呼吸困难": 0.10, "喘不过气": 0.10,
    # Resignation/acceptance
    "释然": 0.60, "认命": 0.25, "看开": 0.58, "随缘": 0.55,
    "顺其自然": 0.55, "放下": 0.60, "看淡": 0.52,
    # Awe/wonder (mixed valence)
    "敬畏": 0.55, "惊叹": 0.65, "叹为观止": 0.72, "肃然起敬": 0.68,
    # Bittersweet/nostalgia
    "苦乐参半": 0.42, "感触": 0.48, "感慨": 0.45, "百感交集": 0.42,
    # Sensitive/fragile states
    "委屈": 0.18, "敏感": 0.28, "脆弱": 0.22, "不堪一击": 0.12,
    "玻璃心": 0.22,
}

_AROUSAL_COMPLEX: dict[str, float] = {
    "迷茫": 0.42, "困惑": 0.40, "矛盾": 0.55, "纠结": 0.58,
    "犹豫": 0.45, "彷徨": 0.50, "迟疑": 0.42,
    "压抑": 0.40, "崩溃": 0.95, "疲惫": 0.32, "倦怠": 0.28,
    "身心俱疲": 0.38, "筋疲力尽": 0.45, "累垮": 0.50,
    "呼吸困难": 0.72, "喘不过气": 0.72,
    "释然": 0.35, "认命": 0.30, "看开": 0.28, "随缘": 0.22,
    "顺其自然": 0.25, "放下": 0.30, "看淡": 0.28,
    "敬畏": 0.55, "惊叹": 0.68, "叹为观止": 0.72, "肃然起敬": 0.55,
    "苦乐参半": 0.42, "感触": 0.42, "感慨": 0.45, "百感交集": 0.62,
    "委屈": 0.75, "敏感": 0.58, "脆弱": 0.55, "不堪一击": 0.62,
    "玻璃心": 0.55,
}

# ═══════════════════════════════════════════════════════════════
# Merged lookup dictionaries (union of all 8 categories)
# ═══════════════════════════════════════════════════════════════

_VALENCE_KEYWORDS: dict[str, float] = {
    **_VALENCE_JOY, **_VALENCE_GOOD, **_VALENCE_ANGER, **_VALENCE_SADNESS,
    **_VALENCE_FEAR, **_VALENCE_DISGUST, **_VALENCE_SURPRISE, **_VALENCE_COMPLEX,
}

_AROUSAL_KEYWORDS: dict[str, float] = {
    **_AROUSAL_JOY, **_AROUSAL_GOOD, **_AROUSAL_ANGER, **_AROUSAL_SADNESS,
    **_AROUSAL_FEAR, **_AROUSAL_DISGUST, **_AROUSAL_SURPRISE, **_AROUSAL_COMPLEX,
}

# ── Emotion category labels (DLUT 7大类 + Plutchik 8) ──────
# Maps each keyword → set of emotion categories for richer profiling.
# DLUT categories: 乐/好/怒/哀/惧/恶/惊
# Plutchik categories: joy/trust/anger/sadness/fear/disgust/surprise/anticipation

_EMOTION_CATEGORIES: dict[str, set[str]] = {
    # ── 乐 (Joy) ──
    **{w: {"乐", "joy"} for w in _VALENCE_JOY},
    # ── 好 (Positive/Good) ──
    **{w: {"好", "trust"} for w in _VALENCE_GOOD},
    # ── 怒 (Anger) ──
    **{w: {"怒", "anger"} for w in _VALENCE_ANGER},
    # ── 哀 (Sadness) ──
    **{w: {"哀", "sadness"} for w in _VALENCE_SADNESS},
    # ── 惧 (Fear) ──
    **{w: {"惧", "fear"} for w in _VALENCE_FEAR},
    # ── 恶 (Disgust) ──
    **{w: {"恶", "disgust"} for w in _VALENCE_DISGUST},
    # ── 惊 (Surprise) ──
    **{w: {"惊", "surprise"} for w in _VALENCE_SURPRISE},
    # ── Complex/compound ──
    **{w: {"综合"} for w in _VALENCE_COMPLEX},
    # ── Override some compounds with more specific labels ──
    "敬畏": {"惊", "好", "surprise", "trust"},
    "惊喜": {"乐", "惊", "joy", "surprise"},
    "惊恐": {"惧", "惊", "fear", "surprise"},
    "羡慕": {"好", "恶", "trust", "disgust"},  # ambiguous
    "释然": {"乐", "好", "joy", "trust"},
    "感慨": {"哀", "惊", "sadness", "surprise"},
}

# Negation words that flip valence
_NEGATION_WORDS = {"不", "没", "无", "非", "别", "莫", "未", "否", "休", "从不", "绝不"}

# ── Query category keyword sets ─────────────────────────────

_EMOTION_QUERY_WORDS = {
    # Query type indicators
    "感觉", "情绪", "心情", "感受", "心态", "状态",
    "什么感受", "心情如何", "心里", "心理",
    # Emotion words (high-frequency query terms)
    "焦虑", "开心", "失眠", "难过", "害怕", "幸福", "低落",
    "压抑", "痛苦", "兴奋", "崩溃", "感动", "愤怒",
    "紧张", "担心", "失落", "后悔", "孤独", "迷茫",
    "绝望", "委屈", "烦躁", "悲伤", "恐惧", "厌恶",
    "惊喜", "满足", "自豪", "欣慰", "失望", "内疚",
    "嫉妒", "惭愧", "不安", "疲惫", "空虚", "愤怒",
    "沮丧", "平静", "放松", "期待", "害羞", "尴尬",
    # Emotion extremes
    "情绪最低点", "情绪最高点", "最开心", "最难过的时刻",
    "最失落", "最崩溃", "情绪波动", "情绪变化",
    # Emotion trajectory
    "情绪轨迹", "情绪趋势", "心情变好", "心情变差",
    "情绪好转", "情绪恶化", "从早到晚", "情绪历程",
    # Emotion triggers
    "为什么难过", "为什么开心", "什么导致", "让我高兴",
    "让我难过", "触发", "引爆",
}

_CAUSAL_QUERY_WORDS = {
    "为什么", "原因", "导致", "引起", "促成", "因素",
    "怎么造成的", "根本原因", "源头", "起因",
    "后果", "结果", "影响", "造成", "引发",
    "触发", "诱因", "根源", "来龙去脉", "前因后果",
    "关联", "相关", "有关系", "有联系",
    "促使", "推动", "驱使", "动机",
}

_TEMPORAL_QUERY_WORDS = {
    "什么时候", "多久", "多少天", "持续", "时间线", "顺序",
    "先后", "从早到晚", "间隔", "几天", "上周", "上个月",
    "半年前", "最近", "第一", "后来", "接着",
    "前", "后", "之前", "之后", "以前", "以后",
    "开始", "结束", "持续了", "历时", "经过",
    "一开始", "最终", "最后", "然后", "接下来",
    "期间", "这段时间", "那段时间",
}

_CROSS_REFERENCE_WORDS = {
    "怎么看", "经历", "变化", "成长", "影响", "关系",
    "联系", "对比", "和", "之间",
    "转变", "转折", "改变", "不同", "差异",
    "相提并论", "联系起来", "同时", "交叉",
    "比较", "相比之下", "相比", "比起",
    "发展", "演变", "进程", "过程",
}

# Mapping from query category → preferred graph relation types
_CATEGORY_RELATION_MAP: dict[str, list[str]] = {
    "emotional": ["emotional", "temporal"],
    "causal": ["causal", "thematic"],
    "temporal": ["temporal", "causal"],
    "cross_reference": None,  # all types
    "factual": ["thematic"],
}


class RetrievalEngine:
    """
    DDA-adaptive multi-path memory retrieval.

    The number and type of retrieval paths activate progressively
    as the user's data density (DDI) increases.

    v7: Content-preserving fusion + emotion resonance + typed graph traversal.
    """

    def __init__(
        self,
        narrative_engine=None,
        hippo_rag=None,
        graph_rag=None,
        learnable_weights=None,
    ):
        # Track B: BM25 retriever with proper probabilistic scoring
        self._bm25_retriever = BM25Retriever()

        # Track C: Optional enhanced retrieval modules
        self.narrative_engine = narrative_engine
        self.hippo_rag = hippo_rag
        self.graph_rag = graph_rag

        # Path weights for multi-path retrieval (HOT+)
        # v7: emotion path now active (was dead code in v6)
        # v8: temporal + cross_ref paths added (were missing — P1/P2)
        # v9 Track C: narrative path added (wired in mcp_server.py)
        # v9: ppr path set to 0.0 — hippo_rag never wired in any entry point;
        #     weight is reserved for future wiring. When hippo_rag is None,
        #     Fix 1 zeroes it at runtime; this explicit 0.0 avoids dead-weight
        #     allocation in the normalize step.
        self.path_weights = {
            "vector": 0.22,      # semantic embedding similarity
            "bm25": 0.12,        # keyword match (boosted from 0.10 → fills ppr gap)
            "graph": 0.18,       # typed graph traversal with depth=3
            "emotion": 0.12,     # emotional resonance (boosted from 0.10)
            "temporal": 0.14,    # temporal relevance + ordering (boosted from 0.12)
            "cross_ref": 0.10,   # cross-memory-type linking (boosted from 0.08)
            "narrative": 0.08,   # story-indexed retrieval (P3 Track C, wired in mcp_server)
            "ppr": 0.00,         # v9: reserved — requires hippo_rag wiring (not yet done)
            "ws_rerank": 0.04,   # Working Self relevance
        }

        # Normalize to 1.0
        _total = sum(self.path_weights.values())
        if _total != 1.0:
            for k in self.path_weights:
                self.path_weights[k] /= _total

        # Track C: Learnable weights (MemLong-style)
        if learnable_weights:
            self.learnable_weights = learnable_weights
        else:
            from learnable_weights import LearnablePathWeights
            self.learnable_weights = LearnablePathWeights(
                base_weights=self.path_weights,
            )

        # Random surface probability for diversity
        # Fix 3: Reduced from 0.15 to 0.03 — 15% was injecting noise
        # into ~1/7 queries in small-corpus scenarios. 3% preserves the
        # serendipity benefit without polluting small result sets.
        self.random_surface_probability: float = 0.03

        # Track C: Feedback accumulation for weight learning
        self._pending_feedback: list = []

    # ── Main retrieval ─────────────────────────────────────

    async def search(
        self,
        query: str,
        context: dict | None = None,
        strategy: DDAStrategy | None = None,
        ddi_level: DDILevel = DDILevel.COLD,
        bucket_mgr=None,
        embedding_engine=None,
        memory_graph=None,
        working_self=None,
        decay_engine=None,
        user_id: str = "",
        top_k: int = 20,
        # Track C: New optional modules
        narrative_engine=None,
        hippo_rag=None,
        graph_rag=None,
        # v9 Ablation: path-level disable + debug
        disabled_paths: set[str] | None = None,
        return_debug: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        DDA-adaptive memory retrieval.

        The retrieval mode is determined by the strategy:
          COLD: return ALL memories
          WARM: semantic + time ranking
          HOT:  vector + BM25 + graph + emotion (4 paths, content-preserving)
          RICH: HOT + Working Self re-rank (5 paths)

        Track C v9: Added narrative (P3), PPR (HippoRAG), and
        community boost (GraphRAG) paths.

        v9 Ablation: disabled_paths zeroes specified paths for ablation
        studies. return_debug returns (results, debug_info) tuple with
        per-path weights and contributions.
        """
        # Update module references if provided
        if narrative_engine is not None:
            self.narrative_engine = narrative_engine
        if hippo_rag is not None:
            self.hippo_rag = hippo_rag
        if graph_rag is not None:
            self.graph_rag = graph_rag

        # v9 Ablation: store disabled paths for internal methods
        self._disabled_paths = disabled_paths or set()
        self._return_debug = return_debug

        try:
            if strategy is None:
                strategy = DDAStrategy()

            mode = strategy.retrieval_mode

            if mode == "all":
                result = await self._retrieve_all(bucket_mgr, decay_engine, top_k)

            elif mode == "cold_fusion":
                result = await self._retrieve_cold_fusion(
                    query, bucket_mgr, decay_engine, top_k,
                )

            elif mode == "semantic_time":
                result = await self._retrieve_semantic_time(
                    query, bucket_mgr, embedding_engine, decay_engine, top_k
                )

            elif mode == "three_way":
                result = await self._retrieve_three_way(
                    query, bucket_mgr, embedding_engine, memory_graph,
                    decay_engine, top_k
                )

            elif mode == "four_way_ws":
                result = await self._retrieve_four_way_ws(
                    query, bucket_mgr, embedding_engine, memory_graph,
                    working_self, decay_engine, top_k
                )

            else:
                logger.warning(f"Unknown retrieval mode: {mode}, falling back to 'all'")
                result = await self._retrieve_all(bucket_mgr, decay_engine, top_k)

            # v9 Ablation: unwrap debug tuple if present
            if return_debug and isinstance(result, tuple):
                return result
            elif return_debug:
                return (result, getattr(self, '_last_ablation_debug', {}))
            return result
        finally:
            self._disabled_paths = set()
            self._return_debug = False

    # ── Retrieval modes ────────────────────────────────────

    async def _retrieve_all(
        self,
        bucket_mgr,
        decay_engine,
        top_k: int,
    ) -> list[dict]:
        """COLD: return ALL memories (there aren't many)."""
        if bucket_mgr is None:
            return []
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Retrieval all failed: {e}")
            return []

        results = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            results.append({
                "id": b["id"],
                "name": meta.get("name", ""),
                "content": b.get("content", ""),
                "type": meta.get("type", "dynamic"),
                "memory_type": meta.get("memory_type", "chat"),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "importance": meta.get("importance", 5),
            })

        results.sort(key=lambda r: r.get("importance", 5), reverse=True)
        return results[:top_k]

    @staticmethod
    def _temporal_recency_score(created_str: str) -> float:
        """
        Score memory by recency. 1.0 = today, ~0.1 after 365 days.

        Exponential decay with half-life of 60 days.
        Zero-dependency: only needs timestamp string.
        """
        from datetime import datetime, timezone
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            return math.exp(-days_ago / 86.5)  # 86.5 = 60/ln(2)
        except Exception:
            return 0.5  # neutral default

    async def _retrieve_cold_fusion(
        self,
        query: str,
        bucket_mgr,
        decay_engine,
        top_k: int,
    ) -> list[dict]:
        """
        COLD: Light 3-path fusion for sparse data (DDI < 10).

        BM25 (50%) + Emotion Resonance (25%) + Temporal Recency (25%)

        All three paths are zero-dependency:
        - BM25: rank_bm25 library, rebuilds index per call
        - Emotion: Russell circumplex distance on existing valence/arousal
        - Temporal: exponential decay on existing timestamps

        Fallback chain: BM25 → content_search → _retrieve_all
        """
        candidates: list[dict] = []

        # ── Phase 1: BM25 recall ──
        try:
            bm25_results = await self._bm25_search(query, bucket_mgr, top_k=top_k * 3)
        except Exception:
            bm25_results = []

        # Fallback 1: bucket_mgr fuzzy search when corpus < 10
        if not bm25_results and bucket_mgr:
            try:
                fuzzy = await bucket_mgr.search(query, limit=top_k * 3)
                bm25_results = [(b["id"], b.get("score", 0) / 100.0) for b in fuzzy]
            except Exception:
                pass

        # Fallback 2: return all memories sorted by importance
        if not bm25_results:
            return await self._retrieve_all(bucket_mgr, decay_engine, top_k)

        # ── Phase 2: Fetch metadata ──
        bucket_map: dict[str, dict] = {}
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            bucket_map = {b["id"]: b for b in all_buckets}
        except Exception:
            pass

        # ── Phase 3: Query emotion extraction ──
        q_valence, q_arousal = self._extract_query_emotion(query)

        # ── Phase 4: Score fusion ──
        for mem_id, bm25_score in bm25_results:
            bucket = bucket_map.get(mem_id, {})
            meta = bucket.get("metadata", {})
            content = bucket.get("content", "")

            mem_valence = float(meta.get("valence", 0.5))
            mem_arousal = float(meta.get("arousal", 0.3))
            created = meta.get("created", "")

            # Emotion resonance
            emotion = self.emotion_resonance(q_valence, q_arousal, mem_valence, mem_arousal)

            # Temporal recency
            temporal = self._temporal_recency_score(created) if created else 0.5

            # Cold fusion: BM25-dominant with emotion + temporal as tiebreakers
            final_score = bm25_score * 0.50 + emotion * 0.25 + temporal * 0.25

            candidates.append({
                "id": mem_id,
                "name": meta.get("name", ""),
                "content": content,
                "type": meta.get("type", "dynamic"),
                "memory_type": meta.get("memory_type", "chat"),
                "valence": mem_valence,
                "arousal": mem_arousal,
                "importance": meta.get("importance", 5),
                "created": created,
                "bm25_score": bm25_score,
                "emotion_score": emotion,
                "temporal_score": temporal,
                "final_score": final_score,
                "score": final_score,
                "source": "cold_fusion",
            })

        candidates.sort(key=lambda r: r.get("final_score", 0), reverse=True)
        result = candidates[:top_k]

        # v9 Ablation: return debug info
        if getattr(self, '_return_debug', False):
            debug = {
                "path_weights": {"bm25": 0.50, "emotion": 0.25, "temporal": 0.25},
                "disabled_paths": list(getattr(self, '_disabled_paths', set())),
                "n_candidates": len(candidates),
                "n_returned": len(result),
                "mode": "cold_fusion",
            }
            self._last_ablation_debug = debug
            return (result, debug)
        return result

    async def _retrieve_semantic_time(
        self,
        query: str,
        bucket_mgr,
        embedding_engine,
        decay_engine,
        top_k: int,
    ) -> list[dict]:
        """WARM: semantic similarity + time ranking."""
        results = []

        if bucket_mgr and query.strip():
            try:
                matches = await bucket_mgr.search(query, limit=top_k)
                for bucket in matches:
                    meta = bucket.get("metadata", {})
                    results.append({
                        "id": bucket["id"],
                        "name": meta.get("name", ""),
                        "content": bucket.get("content", ""),
                        "score": bucket.get("score", 0),
                        "type": meta.get("type", "dynamic"),
                        "memory_type": meta.get("memory_type", "chat"),
                        "valence": meta.get("valence", 0.5),
                        "arousal": meta.get("arousal", 0.3),
                        "importance": meta.get("importance", 5),
                        "source": "semantic",
                    })
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Supplement with high-importance unresolved
        if bucket_mgr:
            try:
                all_b = await bucket_mgr.list_all(include_archive=False)
                unresolved = [
                    b for b in all_b
                    if not b["metadata"].get("resolved")
                    and b["metadata"].get("type") not in ("permanent", "feel")
                    and not b["metadata"].get("pinned")
                ]
                if decay_engine:
                    unresolved.sort(
                        key=lambda b: decay_engine.calculate_score(b["metadata"]),
                        reverse=True,
                    )
                seen = {r["id"] for r in results}
                for b in unresolved[:5]:
                    if b["id"] not in seen:
                        meta = b.get("metadata", {})
                        results.append({
                            "id": b["id"],
                            "name": meta.get("name", ""),
                            "content": b.get("content", ""),
                            "score": decay_engine.calculate_score(meta) if decay_engine else 0,
                            "type": meta.get("type", "dynamic"),
                            "memory_type": meta.get("memory_type", "chat"),
                            "valence": meta.get("valence", 0.5),
                            "arousal": meta.get("arousal", 0.3),
                            "importance": meta.get("importance", 5),
                            "source": "unresolved",
                        })
            except Exception as e:
                logger.warning(f"Unresolved supplement failed: {e}")

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[:top_k]

    async def _retrieve_three_way(
        self,
        query: str,
        bucket_mgr,
        embedding_engine,
        memory_graph,
        decay_engine,
        top_k: int,
    ) -> list[dict]:
        """
        HOT: vector + BM25 + graph + emotion (4-path fusion).

        v7 improvements:
          P0-1: Content is preserved through fusion via backfill.
          P0-2: Emotion resonance path activated with query emotion extraction.
          P0-3: Graph traversal uses depth=3 + type filtering (was depth=1 blind).
        """
        all_results: dict[str, dict] = {}

        # ── P0-2: Extract query emotion once (zero-LLM) ──
        q_valence, q_arousal = self._extract_query_emotion(query)

        # ── P0-3: Infer query category and relation types ──
        query_category = self._infer_query_category(query)
        relation_types = self._infer_query_relation_types(query)

        # ── P1: Parse time constraints for temporal path ──
        time_constraints = self._parse_time_constraints(query)

        # ── Track C: Get learnable weights for this query category ──
        weights = self.learnable_weights.get_weights(query_category)

        # ── P0-2: Dynamic path weight adjustment based on query type ──
        # v9 Track C: Use learnable weights as base, then adjust per category
        if query_category == "emotional":
            weights["emotion"] = max(weights.get("emotion", 0.08), 0.22)
            weights["vector"] = max(weights.get("vector", 0.18), 0.18)
            weights["bm25"] = min(weights.get("bm25", 0.05), 0.10)
            weights["graph"] = max(weights.get("graph", 0.18), 0.20)
            weights["temporal"] = min(weights.get("temporal", 0.03), 0.08)
            weights["cross_ref"] = min(weights.get("cross_ref", 0.03), 0.08)
            weights["narrative"] = max(weights.get("narrative", 0.05), 0.12)
            weights["ppr"] = min(weights.get("ppr", 0.03), 0.06)
        elif query_category == "causal":
            weights["graph"] = max(weights.get("graph", 0.20), 0.28)
            weights["vector"] = max(weights.get("vector", 0.18), 0.20)
            weights["bm25"] = max(weights.get("bm25", 0.08), 0.12)
            weights["emotion"] = min(weights.get("emotion", 0.04), 0.08)
            weights["cross_ref"] = max(weights.get("cross_ref", 0.08), 0.14)
            weights["temporal"] = min(weights.get("temporal", 0.04), 0.10)
            weights["narrative"] = max(weights.get("narrative", 0.05), 0.10)
            weights["ppr"] = min(weights.get("ppr", 0.03), 0.06)
        elif query_category == "temporal":
            weights["temporal"] = max(weights.get("temporal", 0.10), 0.30)
            weights["graph"] = max(weights.get("graph", 0.16), 0.20)
            weights["vector"] = max(weights.get("vector", 0.14), 0.16)
            weights["bm25"] = min(weights.get("bm25", 0.05), 0.10)
            weights["emotion"] = min(weights.get("emotion", 0.03), 0.06)
            weights["cross_ref"] = min(weights.get("cross_ref", 0.04), 0.08)
            weights["narrative"] = max(weights.get("narrative", 0.05), 0.12)
            weights["ppr"] = min(weights.get("ppr", 0.03), 0.06)
        elif query_category == "cross_reference":
            weights["cross_ref"] = max(weights.get("cross_ref", 0.08), 0.25)
            weights["graph"] = max(weights.get("graph", 0.20), 0.25)
            weights["vector"] = max(weights.get("vector", 0.14), 0.16)
            weights["bm25"] = min(weights.get("bm25", 0.05), 0.08)
            weights["temporal"] = max(weights.get("temporal", 0.08), 0.14)
            weights["emotion"] = min(weights.get("emotion", 0.03), 0.06)
            weights["narrative"] = max(weights.get("narrative", 0.08), 0.18)
            weights["ppr"] = max(weights.get("ppr", 0.05), 0.10)
        elif query_category == "narrative":
            # P3: Strong boost for narrative/story queries
            weights["narrative"] = max(weights.get("narrative", 0.10), 0.28)
            weights["vector"] = max(weights.get("vector", 0.14), 0.18)
            weights["graph"] = max(weights.get("graph", 0.16), 0.20)
            weights["bm25"] = min(weights.get("bm25", 0.05), 0.08)
            weights["temporal"] = max(weights.get("temporal", 0.08), 0.14)
            weights["emotion"] = min(weights.get("emotion", 0.03), 0.06)
            weights["cross_ref"] = max(weights.get("cross_ref", 0.05), 0.10)
            weights["ppr"] = min(weights.get("ppr", 0.03), 0.06)
        else:
            # ── v8 fix: For factual queries, content match dominates ──
            # v9 Track C: Add small narrative/ppr weights
            weights["vector"] = max(weights.get("vector", 0.20), 0.28)
            weights["bm25"] = max(weights.get("bm25", 0.15), 0.22)
            weights["graph"] = max(weights.get("graph", 0.16), 0.20)
            weights["emotion"] = min(weights.get("emotion", 0.03), 0.06)
            weights["temporal"] = min(weights.get("temporal", 0.04), 0.08)
            weights["cross_ref"] = min(weights.get("cross_ref", 0.03), 0.06)
            weights["narrative"] = min(weights.get("narrative", 0.03), 0.06)
            weights["ppr"] = min(weights.get("ppr", 0.03), 0.06)

        # ── Fix 1: Zero out unavailable paths, no redistribution ──
        # Unavailable paths (missing infra) get weight=0 instead of
        # redistributing their weight to remaining (potentially noisy) paths.
        # This prevents noise amplification when vector/graph/narrative/ppr
        # infrastructure is absent (e.g., in testing/benchmarking).
        #
        # Only content-matching paths (vector + bm25) are renormalized among
        # themselves. Noise paths (emotion/temporal/cross_ref) keep their
        # original absolute design weights — they participate but don't get
        # amplified by the absence of content-matching infrastructure.
        _available = {
            "vector": embedding_engine is not None,
            "bm25": bucket_mgr is not None,
            "graph": memory_graph is not None,
            "emotion": True,
            "temporal": True,
            "cross_ref": True,
            "narrative": self.narrative_engine is not None,
            "ppr": self.hippo_rag is not None,
        }
        # Zero out unavailable paths
        for k, available in _available.items():
            if not available and k in weights:
                weights[k] = 0.0

        # v9 Ablation: Zero out explicitly disabled paths (bypasses Fix 1 availability check)
        _disabled = getattr(self, '_disabled_paths', set())
        for dp in _disabled:
            if dp in weights:
                weights[dp] = 0.0

        # Only renormalize among content-matching paths (vector + bm25)
        # when one of them is missing — keep noise paths at original weights.
        # v9: Exclude disabled paths from renormalization as well.
        _content_paths = {"vector", "bm25"}
        _active_content = sum(
            weights[k] for k in _content_paths
            if _available.get(k, False) and k not in _disabled
        )
        if _active_content > 0 and abs(_active_content - sum(weights[k] for k in _content_paths)) > 0.001:
            for k in _content_paths:
                if _available.get(k, False) and weights[k] > 0 and k not in _disabled:
                    weights[k] /= _active_content

        # Path 1: Vector search (30%)
        if embedding_engine and query.strip() and "vector" not in getattr(self, '_disabled_paths', set()):
            try:
                similar = await embedding_engine.search_similar(query, top_k=top_k)
                for mem_id, sim_score in similar:
                    all_results[mem_id] = {
                        "id": mem_id,
                        "vector_score": sim_score,
                        "bm25_score": 0.0,
                        "graph_score": 0.0,
                        "emotion_score": 0.0,
                        "source": "vector",
                    }
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # Path 2: BM25 keyword search (12%) — Track B: true BM25, not fuzzy
        if bucket_mgr and query.strip() and "bm25" not in getattr(self, '_disabled_paths', set()):
            try:
                bm25_results = await self._bm25_search(query, bucket_mgr, top_k=top_k)
                for mem_id, bm25_score in bm25_results:
                    if mem_id in all_results:
                        all_results[mem_id]["bm25_score"] = bm25_score
                        all_results[mem_id]["source"] += "+bm25"
                    else:
                        all_results[mem_id] = {
                            "id": mem_id,
                            "vector_score": 0.0,
                            "bm25_score": bm25_score,
                            "graph_score": 0.0,
                            "emotion_score": 0.0,
                            "source": "bm25",
                        }
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # ── Fallback: content-based search when advanced paths unavailable ──
        # v9 Ablation: skip fallback if content paths are intentionally disabled
        _disabled = getattr(self, '_disabled_paths', set())
        _content_disabled = {"vector", "bm25"} & _disabled
        if (not embedding_engine and not memory_graph and not all_results
                and not _content_disabled):
            content_results = await self._content_search(query, bucket_mgr, top_k)
            if content_results:
                all_results = content_results

        # Path 3: Typed graph traversal (22%) — P0-3: depth=3 + type filtering
        if memory_graph and all_results and "graph" not in getattr(self, '_disabled_paths', set()):
            try:
                # P0-3: Expand from top-15 seeds (was top-10), depth=3 (was depth=1)
                seed_ids = list(all_results.keys())[:15]
                for mem_id in seed_ids:
                    neighbors = memory_graph.get_neighbors(
                        mem_id,
                        depth=3,                    # P0-3: was depth=1
                        relation_types=relation_types,  # P0-3: type filtering
                        active_only=True,
                    )
                    for n in neighbors:
                        nid = n["to_id"] if n["from_id"] == mem_id else n["from_id"]
                        # P0-3: Weight by edge type and depth
                        edge_weight = n.get("weight", 0.5)
                        # Causal edges get higher boost for multi-hop reasoning
                        if n.get("relation_type") == "causal":
                            edge_weight *= 1.2
                        elif n.get("relation_type") == "temporal":
                            edge_weight *= 1.0

                        if nid not in all_results:
                            all_results[nid] = {
                                "id": nid,
                                "vector_score": 0.0,
                                "bm25_score": 0.0,
                                "graph_score": edge_weight,
                                "emotion_score": 0.0,
                                "source": "graph",
                            }
                        else:
                            all_results[nid]["graph_score"] = max(
                                all_results[nid].get("graph_score", 0),
                                edge_weight,
                            )
            except Exception as e:
                logger.warning(f"Graph traversal failed: {e}")

        # ── P0-1: Backfill content BEFORE fusion so we have valence/arousal ──
        await self._backfill_content(all_results, bucket_mgr)

        # ── P0-2: Compute emotion resonance scores ──
        # v9 Ablation: skip if emotion path is disabled
        if "emotion" not in getattr(self, '_disabled_paths', set()):
            for mem_id, scores in all_results.items():
                mem_valence = scores.get("valence", 0.5)
                mem_arousal = scores.get("arousal", 0.3)
                scores["emotion_score"] = self.emotion_resonance(
                    q_valence, q_arousal, mem_valence, mem_arousal
                )

        # ── P1: Compute temporal scores ──
        # v9 Ablation: skip if temporal path is disabled
        if "temporal" not in getattr(self, '_disabled_paths', set()):
            for mem_id, scores in all_results.items():
                mem_created = scores.get("created", "")
                scores["temporal_score"] = self.temporal_score(
                    mem_created, time_constraints
                )

        # ── P2: Compute cross-reference scores (uses graph topology) ──
        # v9 Ablation: skip if cross_ref path is disabled
        if "cross_ref" not in getattr(self, '_disabled_paths', set()):
            for mem_id, scores in all_results.items():
                mem_type = scores.get("memory_type", "chat")
                mem_tags = scores.get("tags", [])
                mem_imp = scores.get("importance", 5)

                # Collect neighbor types from graph topology
                neighbor_types: set[str] = set()
                graph_degree: int = 0
                if memory_graph:
                    try:
                        neighbors = memory_graph.get_neighbors(
                            mem_id, depth=1, active_only=True
                        )
                        graph_degree = len(neighbors)
                        for n in neighbors:
                            ntype = n.get("memory_type", "chat")
                            neighbor_types.add(ntype)
                    except Exception:
                        pass

                scores["cross_ref_score"] = self.cross_reference_score(
                    memory_type=mem_type,
                    memory_valence=scores.get("valence", 0.5),
                    memory_arousal=scores.get("arousal", 0.3),
                    memory_importance=mem_imp,
                    memory_tags=mem_tags,
                    query_category=query_category,
                    neighbor_types=neighbor_types if neighbor_types else None,
                    graph_degree=graph_degree,
                )

        # ── Track C: PPR (HippoRAG) path ──
        # v9: Reserved for future wiring. hippo_rag is never instantiated in
        # any production entry point (app.py, server.py, mcp_server.py all
        # pass None). When wired, uncomment the import and this path activates.
        ppr_scores: dict[str, float] = {}
        if self.hippo_rag is not None and memory_graph:
            try:
                # Get graph edges for PPR computation
                from hippo_rag import PPRSeed
                edges = self.hippo_rag.ppr.extract_edges(memory_graph)
                if edges:
                    # Build seeds from current results + importance
                    seeds: list = []
                    existing_ids = set(all_results.keys())
                    for mem_id, scores in all_results.items():
                        imp = scores.get("importance", 5)
                        if imp >= 7:
                            seeds.append(PPRSeed(
                                node_id=mem_id,
                                weight=imp / 10.0,
                                source="retrieval",
                            ))

                    if seeds:
                        ppr_results = self.hippo_rag.ppr.compute_ppr(
                            graph_edges=edges,
                            seeds=seeds,
                            top_k=top_k * 2,
                        )
                        max_ppr = max((r.ppr_score for r in ppr_results), default=1.0)
                        for r in ppr_results:
                            ppr_scores[r.node_id] = r.ppr_score / max(max_ppr, 0.001)

                        # Add PPR-discovered nodes to results
                        for r in ppr_results:
                            if r.node_id not in all_results:
                                all_results[r.node_id] = {
                                    "id": r.node_id,
                                    "vector_score": 0.0,
                                    "bm25_score": 0.0,
                                    "graph_score": 0.0,
                                    "emotion_score": 0.0,
                                    "temporal_score": 0.0,
                                    "cross_ref_score": 0.0,
                                    "source": "ppr",
                                }
            except Exception as e:
                logger.warning(f"PPR path failed: {e}")

        # ── Track C: Narrative retrieval path (P3) ──
        narrative_scores: dict[str, float] = {}
        if self.narrative_engine and query.strip() and "narrative" not in getattr(self, '_disabled_paths', set()):
            try:
                stories = self.narrative_engine.find_story_for_query(
                    query=query,
                    top_k=3,
                )
                for story in stories:
                    story_score = story.get("score", 0.0)
                    # Boost all key moments in this story
                    for moment in story.get("key_moments", []):
                        mid = moment.get("memory_id", "")
                        if mid:
                            # Score based on story relevance × moment importance
                            is_tp = moment.get("is_turning_point", False)
                            moment_score = story_score * (1.5 if is_tp else 1.0)
                            narrative_scores[mid] = max(
                                narrative_scores.get(mid, 0),
                                moment_score,
                            )
                    # Also boost seed memories in the thread
                    for mid in story.get("seed_memory_ids", []):
                        if mid not in narrative_scores:
                            narrative_scores[mid] = story_score * 0.8

                # Add narrative-discovered nodes to results
                for mid, nscore in narrative_scores.items():
                    if mid not in all_results:
                        all_results[mid] = {
                            "id": mid,
                            "vector_score": 0.0,
                            "bm25_score": 0.0,
                            "graph_score": 0.0,
                            "emotion_score": 0.0,
                            "temporal_score": 0.0,
                            "cross_ref_score": 0.0,
                            "source": "narrative",
                        }
            except Exception as e:
                logger.warning(f"Narrative path (P3) failed: {e}")

        # Fuse scores (v9 Track C: narrative + ppr paths now active)
        # Track C: Store path contributions for feedback learning

        # ── Fix 2: Auto-silence non-discriminating noise paths ──
        # Paths whose scores have near-zero variance across candidates
        # (e.g., temporal_score all 0.5 for queries without time constraints)
        # are zeroed out to prevent dead-weight dilution of the fusion.
        # v9 Ablation: Exclude disabled paths from discrimination detection —
        # they are already zeroed by intent, don't let Fix 2 re-activate them
        # through re-normalization.
        _score_to_weight = {
            "vector_score": "vector", "bm25_score": "bm25",
            "graph_score": "graph", "emotion_score": "emotion",
            "temporal_score": "temporal", "cross_ref_score": "cross_ref",
            "narrative_score": "narrative", "ppr_score": "ppr",
        }
        # Filter out disabled paths' score keys before discrimination detection
        _disabled_weight_keys = getattr(self, '_disabled_paths', set())
        _active_score_keys = [
            sk for sk, wk in _score_to_weight.items()
            if wk not in _disabled_weight_keys
        ]
        _discriminating = self._detect_discriminating_paths(
            all_results,
            score_keys=_active_score_keys,
        )
        for score_key, weight_key in _score_to_weight.items():
            if score_key not in _discriminating and weight_key in weights:
                weights[weight_key] = 0.0

        # Re-normalize among remaining discriminating paths (exclude disabled)
        _remaining = sum(
            v for k, v in weights.items() if k not in _disabled_weight_keys
        )
        if _remaining > 0.001 and abs(_remaining - 1.0) > 0.01:
            for k in weights:
                if k not in _disabled_weight_keys:
                    weights[k] /= _remaining

        # ── Fix 5: Two-phase retrieval-ranking decoupling ──
        # Phase 1 (Recall): Rank by content-matching paths only (bm25 + vector).
        #   This ensures the candidate pool is relevance-gated before noise paths
        #   (emotion, temporal, cross_ref, etc.) can influence ordering.
        # Phase 2 (Rerank): Apply full multi-path fusion on a narrowed top-k×3 set.
        #   This lets emotion/temporal/cross_ref/graph paths add value by
        #   reordering within the already-relevant set, without letting them
        #   pull irrelevant memories into the final top-k.
        _phase1_candidates = []
        for mem_id, scores in all_results.items():
            relevance_score = (
                scores.get("vector_score", 0) * weights.get("vector", 0)
                + scores.get("bm25_score", 0) * weights.get("bm25", 0)
            )
            _phase1_candidates.append((mem_id, scores, relevance_score))

        _phase1_candidates.sort(key=lambda x: x[2], reverse=True)
        _phase1_limit = max(top_k * 3, 15)  # at least 15 for small top_k
        _phase1_candidates = _phase1_candidates[:_phase1_limit]

        fused = []
        for mem_id, scores, _rel_score in _phase1_candidates:
            # Compute path contributions
            path_contribs = {
                "vector": scores.get("vector_score", 0),
                "bm25": scores.get("bm25_score", 0),
                "graph": scores.get("graph_score", 0),
                "emotion": scores.get("emotion_score", 0),
                "temporal": scores.get("temporal_score", 0),
                "cross_ref": scores.get("cross_ref_score", 0),
                "narrative": narrative_scores.get(mem_id, 0),
                "ppr": ppr_scores.get(mem_id, 0),
            }

            final_score = (
                scores.get("vector_score", 0) * weights["vector"]
                + scores.get("bm25_score", 0) * weights["bm25"]
                + scores.get("graph_score", 0) * weights["graph"]
                + scores.get("emotion_score", 0) * weights["emotion"]
                + scores.get("temporal_score", 0) * weights["temporal"]
                + scores.get("cross_ref_score", 0) * weights["cross_ref"]
                + narrative_scores.get(mem_id, 0) * weights["narrative"]
                + ppr_scores.get(mem_id, 0) * weights["ppr"]
            )

            # Determine dominant path
            dominant_path = "vector"
            dominant_val = 0.0
            for path_name in path_contribs:
                adj_val = path_contribs[path_name] * weights.get(path_name, 0)
                if adj_val > dominant_val:
                    dominant_val = adj_val
                    dominant_path = path_name

            # P0-1: content is already backfilled — carry it through
            fused.append({
                **scores,
                "final_score": final_score,
                "dominant_path": dominant_path,
                "path_contributions": path_contribs,
            })

        # Track C: Community boost from GraphRAG
        # v9: Reserved for future wiring. graph_rag is never instantiated in
        # any production entry point. When wired, this provides Louvain community
        # detection boost on the memory graph.
        if self.graph_rag is not None and query.strip():
            try:
                fused = self.graph_rag.boost_scores_from_community(
                    query=query,
                    results={r["id"]: r for r in fused},
                    boost_factor=0.08,
                )
                # Convert back to list
                fused = list(fused.values())
            except Exception as e:
                logger.warning(f"Community boost failed: {e}")

        # Random surface for diversity (15% chance)
        if random.random() < self.random_surface_probability and bucket_mgr:
            try:
                all_b = await bucket_mgr.list_all(include_archive=False)
                existing_ids = {r["id"] for r in fused}
                candidates = [b for b in all_b if b["id"] not in existing_ids]
                if candidates:
                    rand_b = random.choice(candidates)
                    meta = rand_b.get("metadata", {})
                    fused.append({
                        "id": rand_b["id"],
                        "name": meta.get("name", ""),
                        "content": rand_b.get("content", ""),
                        "type": meta.get("type", "dynamic"),
                        "memory_type": meta.get("memory_type", "chat"),
                        "valence": meta.get("valence", 0.5),
                        "arousal": meta.get("arousal", 0.3),
                        "importance": meta.get("importance", 5),
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "graph_score": 0.0,
                        "emotion_score": 0.0,
                        "temporal_score": 0.0,
                        "cross_ref_score": 0.0,
                        "narrative_score": 0.0,
                        "ppr_score": 0.0,
                        "final_score": 0.01,
                        "source": "random_surface",
                        "dominant_path": "random",
                        "path_contributions": {},
                    })
            except Exception:
                pass

        fused.sort(key=lambda r: r.get("final_score", 0), reverse=True)
        result = fused[:top_k]

        # v9 Ablation: return debug info with per-path weights and avg contributions
        if getattr(self, '_return_debug', False):
            debug = {
                "path_weights": dict(weights),
                "disabled_paths": list(getattr(self, '_disabled_paths', set())),
                "n_candidates": len(fused),
                "n_returned": len(result),
                "path_avg_contrib": {},
            }
            for path_name in weights:
                if result:
                    avg = sum(
                        r.get("path_contributions", {}).get(path_name, 0)
                        for r in result
                    ) / len(result)
                else:
                    avg = 0.0
                debug["path_avg_contrib"][path_name] = round(avg, 4)
            self._last_ablation_debug = debug
            return (result, debug)
        return result

    async def _retrieve_four_way_ws(
        self,
        query: str,
        bucket_mgr,
        embedding_engine,
        memory_graph,
        working_self,
        decay_engine,
        top_k: int,
    ) -> list[dict]:
        """
        RICH: 4-way + Working Self re-rank.

        v7: _retrieve_three_way now returns content, so WS matching works correctly.
        v9 Ablation: handles debug tuple from _retrieve_three_way when return_debug is set.
        """
        # Get 4-way results first (now with content — P0-1)
        # v9: _retrieve_three_way may return (results, debug) when return_debug is set
        three_way_out = await self._retrieve_three_way(
            query, bucket_mgr, embedding_engine, memory_graph, decay_engine, top_k * 2
        )
        _three_debug = {}
        if isinstance(three_way_out, tuple):
            results, _three_debug = three_way_out
        else:
            results = three_way_out

        # Apply Working Self re-rank (P0-1: content is now available)
        if working_self and working_self.has_goals:
            for r in results:
                content = r.get("content", "")
                if content:
                    ws_match = working_self.match(content)
                else:
                    ws_match = 0.0
                r["ws_match"] = ws_match
                r["final_score"] = (
                    r.get("final_score", 0) * 0.85
                    + ws_match * self.path_weights["ws_rerank"]
                )

        # Apply decay engine scores if available
        if decay_engine:
            for r in results:
                decay_score = r.get("decay_score", 0.5)
                r["final_score"] = r.get("final_score", 0) * 0.9 + decay_score * 0.1

        results.sort(key=lambda r: r.get("final_score", 0), reverse=True)
        result = results[:top_k]

        if getattr(self, '_return_debug', False):
            debug = dict(_three_debug)
            debug["ws_rerank_applied"] = bool(working_self and working_self.has_goals)
            self._last_ablation_debug = debug
            return (result, debug)
        return result

    # ── P0-1: Content backfill ──────────────────────────────

    async def _backfill_content(
        self,
        results: dict[str, dict],
        bucket_mgr,
    ) -> None:
        """
        Backfill content/metadata into fused results from bucket_mgr.

        P0-1: In v6, _retrieve_three_way() only carried scores, not content.
        This method fills in content, valence, arousal, importance, etc.
        from the storage layer so downstream consumers (WS re-rank,
        prompt injection) have access to full memory data.
        """
        if bucket_mgr is None:
            return

        # Collect IDs that need backfill
        ids_to_fetch = [
            mem_id for mem_id, scores in results.items()
            if not scores.get("content")
        ]

        if not ids_to_fetch:
            return

        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            bucket_map = {b["id"]: b for b in all_buckets}

            for mem_id in ids_to_fetch:
                bucket = bucket_map.get(mem_id)
                if bucket:
                    meta = bucket.get("metadata", {})
                    results[mem_id]["content"] = bucket.get("content", "")
                    results[mem_id]["name"] = meta.get("name", "")
                    results[mem_id]["type"] = meta.get("type", "dynamic")
                    results[mem_id]["memory_type"] = meta.get("memory_type", "chat")
                    results[mem_id]["valence"] = meta.get("valence", 0.5)
                    results[mem_id]["arousal"] = meta.get("arousal", 0.3)
                    results[mem_id]["importance"] = meta.get("importance", 5)
                    results[mem_id]["created"] = meta.get("created", "")
                    results[mem_id]["tags"] = meta.get("tags", [])
                else:
                    # Memory not found in storage — set defaults
                    results[mem_id]["content"] = ""
                    results[mem_id]["name"] = ""
                    results[mem_id]["type"] = "dynamic"
                    results[mem_id]["memory_type"] = "chat"
                    results[mem_id]["valence"] = 0.5
                    results[mem_id]["arousal"] = 0.3
                    results[mem_id]["importance"] = 5
                    results[mem_id]["created"] = ""
                    results[mem_id]["tags"] = []
        except Exception as e:
            logger.warning(f"Content backfill failed: {e}")

    # ── P0-2: Query emotion extraction (zero-LLM) ───────────

    @staticmethod
    def _extract_query_emotion(query: str) -> tuple[float, float]:
        """
        Extract (valence, arousal) from query text using keyword matching.
        Zero-LLM, <1ms. Handles negation ("不开心" → low valence).

        P0-2: This feeds into emotion_resonance() for Bower-style
        mood-congruent retrieval.
        """
        if not query or not query.strip():
            return (0.5, 0.3)  # neutral default

        valence_sum = 0.0
        valence_count = 0
        arousal_sum = 0.0
        arousal_count = 0

        # Check for negation in context (±3 chars before keyword)
        # Simple heuristic: if negation word appears in same clause
        has_negation_nearby = lambda pos, text: any(
            text[max(0, pos - 3):pos].strip().endswith(n) for n in _NEGATION_WORDS
        )

        for kw, v in _VALENCE_KEYWORDS.items():
            idx = query.find(kw)
            if idx >= 0:
                if has_negation_nearby(idx, query):
                    valence_sum += (1.0 - v)  # flip
                else:
                    valence_sum += v
                valence_count += 1

        for kw, a in _AROUSAL_KEYWORDS.items():
            if kw in query:
                arousal_sum += a
                arousal_count += 1

        if valence_count == 0 and arousal_count == 0:
            return (0.5, 0.3)  # neutral

        valence = valence_sum / valence_count if valence_count > 0 else 0.5
        arousal = arousal_sum / arousal_count if arousal_count > 0 else 0.3

        return (
            max(0.0, min(1.0, valence)),
            max(0.0, min(1.0, arousal)),
        )

    @staticmethod
    def get_emotion_categories(keyword: str) -> set[str]:
        """
        Return DLUT + Plutchik emotion category labels for a keyword.

        References:
          - 徐琳宏,林鸿飞等 (2008), 情报学报 — DLUT 7大类: 乐/好/怒/哀/惧/恶/惊
          - Mohammad & Turney (2013), Computational Intelligence — NRC Emotion Lexicon,
            Plutchik 8: joy/trust/anger/sadness/fear/disgust/surprise/anticipation

        Returns empty set if the keyword is not in the emotion lexicon.
        """
        return _EMOTION_CATEGORIES.get(keyword, set())

    @staticmethod
    def _infer_query_category(query: str) -> str:
        """
        Infer query type for dynamic path weight adjustment (MAGMA-style).

        P0-2: Determines whether the query is emotional, causal, temporal,
        cross-reference, or factual. Used to adjust path_weights dynamically.
        """
        if not query:
            return "factual"

        query_lower = query.lower()
        scores: dict[str, int] = {}

        for w in _EMOTION_QUERY_WORDS:
            if w in query_lower:
                scores["emotional"] = scores.get("emotional", 0) + 1

        for w in _CAUSAL_QUERY_WORDS:
            if w in query_lower:
                scores["causal"] = scores.get("causal", 0) + 1

        for w in _TEMPORAL_QUERY_WORDS:
            if w in query_lower:
                scores["temporal"] = scores.get("temporal", 0) + 1

        for w in _CROSS_REFERENCE_WORDS:
            if w in query_lower:
                scores["cross_reference"] = scores.get("cross_reference", 0) + 1

        if not scores:
            return "factual"

        return max(scores, key=scores.get)

    @staticmethod
    def _infer_query_relation_types(query: str) -> list[str] | None:
        """
        Map query category to preferred graph relation types.

        P0-3: Enables type-filtered graph traversal so that:
          - Emotional queries traverse emotional + temporal edges
          - Causal queries traverse causal + thematic edges
          - Temporal queries traverse temporal + causal edges
          - Cross-reference queries traverse ALL types
          - Factual queries traverse thematic edges
        """
        category = RetrievalEngine._infer_query_category(query)
        return _CATEGORY_RELATION_MAP.get(category)

    # ── Fix 2: Path discrimination detection ──────────────────

    @staticmethod
    def _detect_discriminating_paths(
        all_results: dict[str, dict],
        score_keys: list[str],
        variance_threshold: float = 0.005,
        range_threshold: float = 0.05,
    ) -> set[str]:
        """
        Detect which paths produce scores with meaningful discrimination.

        A path is considered "discriminating" if its scores across all
        candidate memories show sufficient variance AND range — meaning
        the path actually differentiates between memories rather than
        contributing a near-constant value.

        Fix 2: Non-discriminating paths (e.g., temporal_score all 0.5
        for queries without time constraints) have their weights zeroed
        out to prevent them from diluting the fusion with dead weight.

        Args:
            all_results: dict of {mem_id: {score_key: value, ...}}
            score_keys: list of score key names to check (e.g. ["bm25_score", ...])
            variance_threshold: minimum variance to consider discriminating
            range_threshold: minimum (max-min) range to consider discriminating

        Returns:
            Set of score_key names that have meaningful discrimination.
        """
        discriminating: set[str] = set()
        for key in score_keys:
            values = [
                scores.get(key, 0.0)
                for scores in all_results.values()
            ]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            value_range = max(values) - min(values)
            if variance > variance_threshold and value_range > range_threshold:
                discriminating.add(key)
        return discriminating

    # ── Emotion resonance scoring ──────────────────────────

    @staticmethod
    def emotion_resonance(
        query_valence: float,
        query_arousal: float,
        memory_valence: float,
        memory_arousal: float,
    ) -> float:
        """
        Calculate emotional resonance between query and memory.
        Russell circumplex Euclidean distance, normalized to 0-1.

        P0-2: Now actively called from _retrieve_three_way().
        In v6 this was orphan code — implemented but never wired in.
        """
        dv = query_valence - memory_valence
        da = query_arousal - memory_arousal
        distance = math.sqrt(dv * dv + da * da)
        resonance = 1.0 - (distance / math.sqrt(2))
        return max(0.0, min(1.0, resonance))

    # ── P1: Temporal retrieval path ─────────────────────────

    @staticmethod
    def _parse_time_constraints(query: str) -> dict:
        """
        Extract time constraints from a query (zero-LLM).

        Returns dict with optional keys:
          - "before_days": memories older than N days
          - "after_days": memories newer than N days
          - "between": (min_days, max_days) range
          - "ordering": "chronological" | "reverse_chronological" | None
          - "duration": whether query asks about duration/spans
          - "sequence": whether query asks about event ordering

        P1: Enables time-window filtering for temporal reasoning queries.
        """
        import re

        result: dict = {
            "before_days": None,
            "after_days": None,
            "between": None,
            "ordering": None,
            "duration": False,
            "sequence": False,
        }

        # Detect temporal query type
        if any(w in query for w in ["顺序", "先后", "第一", "接着", "然后", "接下来", "最后"]):
            result["sequence"] = True
            result["ordering"] = "chronological"

        if any(w in query for w in ["持续", "历时", "多久", "多长时间", "间隔", "几天"]):
            result["duration"] = True

        if any(w in query for w in ["最近", "近来", "最近几天", "近期"]):
            result["ordering"] = "reverse_chronological"

        # Extract numeric time references
        # Pattern: N天前, N天, N周前, N个月前, N天之后, N天后
        day_patterns = [
            (r"(\d+)\s*天前", "before_days"),
            (r"(\d+)\s*天之后", "after_days"),
            (r"(\d+)\s*天后", "after_days"),
            (r"(\d+)\s*周前", "before_days_weeks"),
            (r"(\d+)\s*个月前", "before_days_months"),
        ]

        for pattern, key in day_patterns:
            match = re.search(pattern, query)
            if match:
                num = int(match.group(1))
                if key == "before_days_weeks":
                    result["before_days"] = num * 7
                elif key == "before_days_months":
                    result["before_days"] = num * 30
                elif key == "before_days":
                    if result["before_days"] is None or num < result["before_days"]:
                        result["before_days"] = num
                elif key == "after_days":
                    if result["after_days"] is None or num > result["after_days"]:
                        result["after_days"] = num

        # Range: "在X到Y天之间", "X天到Y天前"
        range_match = re.search(r"(\d+)\s*[天到至]\s*(\d+)\s*天", query)
        if range_match:
            a, b = int(range_match.group(1)), int(range_match.group(2))
            result["between"] = (min(a, b), max(a, b))

        # Default: if query has temporal words but no specific numbers,
        # favor chronological ordering for "how things changed" queries
        if not any([result["before_days"], result["after_days"], result["between"]]):
            if any(w in query for w in _TEMPORAL_QUERY_WORDS):
                result["ordering"] = result["ordering"] or "chronological"

        return result

    @staticmethod
    def temporal_score(
        memory_created_iso: str,
        time_constraints: dict,
        query_reference_days: float | None = None,
    ) -> float:
        """
        Score a memory's temporal relevance to the query.

        P1: Feeds into the temporal path of multi-path fusion.

        Args:
            memory_created_iso: ISO datetime string of memory creation
            time_constraints: from _parse_time_constraints()
            query_reference_days: days-ago value for "now" reference point
        """
        if not memory_created_iso:
            return 0.5  # neutral for unknown times

        try:
            from datetime import datetime
            dt = datetime.fromisoformat(memory_created_iso)
            days_ago = (datetime.now() - dt).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 0.5

        score = 0.5  # base neutral

        # Time window matching
        before_days = time_constraints.get("before_days")
        after_days = time_constraints.get("after_days")
        between = time_constraints.get("between")

        if between:
            lo, hi = between
            if lo <= days_ago <= hi:
                score = 0.95  # exact match
            elif days_ago < lo:
                score = 0.7 - (lo - days_ago) / max(lo, 1) * 0.3
            else:
                score = 0.7 - (days_ago - hi) / max(hi, 1) * 0.3
        elif before_days is not None:
            if days_ago >= before_days:
                score = 0.9
            else:
                # Penalty proportional to distance
                dist = before_days - days_ago
                score = max(0.1, 0.9 - dist / max(before_days, 1) * 0.6)
        elif after_days is not None:
            if days_ago <= after_days:
                score = 0.9
            else:
                dist = days_ago - after_days
                score = max(0.1, 0.9 - dist / max(after_days, 1) * 0.6)

        # Duration/sequence boost: more recent → higher for "recent" queries
        if time_constraints.get("ordering") == "reverse_chronological":
            score = max(score, max(0.1, 1.0 - days_ago / 90.0))

        # Duration queries: prefer memories with clear timestamps
        if time_constraints.get("duration"):
            if days_ago > 0:
                score = max(score, 0.6)

        return max(0.0, min(1.0, score))

    # ── P2: Cross-reference retrieval path ───────────────────

    @staticmethod
    def cross_reference_score(
        memory_type: str,
        memory_valence: float,
        memory_arousal: float,
        memory_importance: int,
        memory_tags: list[str],
        query_category: str,
        neighbor_types: set[str] | None = None,
        graph_degree: int = 0,
    ) -> float:
        """
        Score a memory's cross-reference value — how well it bridges
        different memory types and domains.

        P2: Cross-reference requires linking across memory types
        (chat ↔ emotion ↔ decision ↔ milestone). This path rewards
        memories that:
          - Have diverse neighbor types in the graph (high "betweenness")
          - Link different domains/tags
          - Bridge emotional and factual content
        """
        score = 0.3  # base

        # Diversity of neighbor types in graph → cross-reference value
        if neighbor_types:
            n_types = len(neighbor_types)
            if n_types >= 3:
                score += 0.30
            elif n_types >= 2:
                score += 0.20
            elif n_types >= 1:
                score += 0.10

        # High graph degree → well-connected = valuable for cross-ref
        if graph_degree >= 5:
            score += 0.20
        elif graph_degree >= 3:
            score += 0.12
        elif graph_degree >= 1:
            score += 0.05

        # Bridge memories: emotional content + factual tags = good cross-ref
        is_emotional = memory_type in ("emotion", "milestone")
        has_diverse_tags = len(memory_tags) >= 3
        if is_emotional and has_diverse_tags:
            score += 0.10

        # Extreme valence/arousal → likely an anchor memory → cross-ref value
        if memory_valence <= 0.2 or memory_valence >= 0.8:
            score += 0.05
        if memory_arousal >= 0.8:
            score += 0.05

        # High importance → likely referenced across contexts
        if memory_importance >= 8:
            score += 0.10
        elif memory_importance >= 6:
            score += 0.05

        return max(0.0, min(1.0, score))

    # ── BM25 search (Track B: true probabilistic ranking) ─────────

    async def _bm25_search(
        self,
        query: str,
        bucket_mgr,
        top_k: int = 20,
        min_corpus_size: int = 10,
    ) -> list[tuple[str, float]]:
        """
        True BM25 search using rank_bm25 library.

        Builds a BM25Okapi index from all bucket contents, then scores
        the query against it. Falls back to BucketManager fuzzy search
        when corpus is too small for BM25 to be meaningful.

        Track B: Replaces the old token overlap approach.
        """
        if bucket_mgr is None or not query.strip():
            return []

        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            return []

        if not all_buckets:
            return []

        # Build BM25 corpus from all buckets
        documents = [
            (b["id"], b.get("content", ""))
            for b in all_buckets
        ]

        # Fall back to fuzzy search if corpus too small
        if len(documents) < min_corpus_size:
            try:
                fuzzy_results = await bucket_mgr.search(query, limit=top_k)
                return [
                    (b["id"], b.get("score", 0) / 100.0)
                    for b in fuzzy_results
                ]
            except Exception:
                return []

        # Build fresh BM25 index and search
        self._bm25_retriever.build_index(documents)
        return self._bm25_retriever.search(query, top_k=top_k)

    # ── Content-based search fallback (Track B: BM25-powered) ─────

    async def _content_search(
        self,
        query: str,
        bucket_mgr,
        top_k: int,
    ) -> dict[str, dict]:
        """
        BM25-powered content search against all memories.

        Used as fallback when embedding_engine and memory_graph are
        unavailable (e.g., in testing/benchmarking without full infra).

        Track B: Replaced token overlap with true BM25 scoring.

        Returns same format as the initial results dict in _retrieve_three_way.
        """
        results: dict[str, dict] = {}
        if bucket_mgr is None or not query.strip():
            return results

        # Use BM25 for scoring
        bm25_results = await self._bm25_search(query, bucket_mgr, top_k=top_k * 3)
        if not bm25_results:
            return results

        # Fetch content for each BM25 result
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            bucket_map = {b["id"]: b for b in all_buckets}
        except Exception:
            return results

        for mem_id, bm25_score in bm25_results:
            bucket = bucket_map.get(mem_id)
            if bucket is None:
                continue
            meta = bucket.get("metadata", {})
            results[mem_id] = {
                "id": mem_id,
                "vector_score": bm25_score * 0.6,
                "bm25_score": bm25_score,
                "graph_score": 0.0,
                "emotion_score": 0.0,
                "temporal_score": 0.0,
                "cross_ref_score": 0.0,
                "source": "content_bm25",
                # Pre-fill content/metadata (skip backfill)
                "content": bucket.get("content", ""),
                "name": meta.get("name", ""),
                "type": meta.get("type", "dynamic"),
                "memory_type": meta.get("memory_type", "chat"),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "importance": meta.get("importance", 5),
                "created": meta.get("created", ""),
                "tags": meta.get("tags", []),
            }

        return results

    # ── Track C: Feedback recording for learnable weights ───────

    def record_feedback(
        self,
        result_id: str,
        path_contributions: dict[str, float],
        engaged: bool = False,
        referenced: bool = False,
        query: str = "",
        query_category: str = "factual",
    ):
        """
        Record user feedback to train learnable path weights.

        Called externally (e.g., by memory_orchestrator) after
        user engages with retrieved results.

        Args:
            result_id: memory ID that was engaged with
            path_contributions: {path_name: contribution_score}
            engaged: did user engage with this result?
            referenced: did user reference this memory in their reply?
            query: original query text
            query_category: emotional | causal | temporal | factual | cross_reference
        """
        self.learnable_weights.record_feedback(
            result_id=result_id,
            path_contributions=path_contributions,
            engaged=engaged,
            referenced=referenced,
            query=query,
            query_category=query_category,
        )

    def infer_query_category(self, query: str) -> str:
        """Public alias for query category inference."""
        return self._infer_query_category(query)

    def extract_query_emotion(self, query: str) -> tuple[float, float]:
        """Public alias for query emotion extraction."""
        return self._extract_query_emotion(query)
