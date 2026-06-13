# ============================================================
# Module: Memory Node (memory_node.py)
# Structured data model for all memory operations.
# 结构化记忆数据模型 — 替代全代码中的裸 dict 传递。
#
# All 15 Memory Palace modules pass MemoryNode instances,
# not raw YAML frontmatter dicts. This ensures type safety
# across L0→L1→L2→L3 layers.
# 所有模块传递 MemoryNode 实例而非裸 dict，确保类型安全。
#
# Compatible with existing Ombre Brain YAML frontmatter format.
# 兼容现有 Ombre Brain 的 YAML frontmatter 格式。
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class BucketType(str, Enum):
    """记忆桶类型 — 对应存储子目录"""
    DYNAMIC = "dynamic"       # 动态记忆（会衰减）
    PERMANENT = "permanent"   # 固化记忆（不衰减）
    FEEL = "feel"             # 模型感受（不浮现·日记性质）
    ARCHIVE = "archive"       # 已遗忘归档
    DECISION = "decision"     # 决策记忆（决策过程+结果+回顾）
    MILESTONE = "milestone"   # 里程碑记忆（锚点事件）


class MemoryType(str, Enum):
    """内容类型 — 用于策略差异化（设计 §六）"""
    CHAT = "chat"             # 闲聊片段
    DECISION = "decision"     # 决策节点
    MILESTONE = "milestone"   # 里程碑事件
    EMOTION = "emotion"       # 情感坐标


class DDILevel(str, Enum):
    """数据密度级别 — L0 DDA 自适应"""
    COLD = "COLD"     # 0-10: 新用户，无个人数据
    WARM = "WARM"     # 10-50: 有一些数据
    HOT = "HOT"       # 50-200: 数据充足
    RICH = "RICH"     # 200+: 数据丰富


class RelationType(str, Enum):
    """记忆图边类型"""
    CAUSAL = "causal"         # 因果关系
    THEMATIC = "thematic"     # 主题关联
    TEMPORAL = "temporal"     # 时间顺序
    EMOTIONAL = "emotional"   # 情感共鸣


@dataclass
class ValenceArousal:
    """Russell 环形情感模型坐标"""
    valence: float = 0.5   # 0=negative → 1=positive
    arousal: float = 0.3   # 0=calm → 1=excited

    def __post_init__(self):
        self.valence = max(0.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

    def to_dict(self) -> dict:
        return {"valence": self.valence, "arousal": self.arousal}


@dataclass
class MemoryNode:
    """
    统一记忆节点 — Memory Palace 中所有数据传递的基本单元。

    兼容 Ombre Brain YAML frontmatter 格式。
    所有字段可通过 from_frontmatter() / to_frontmatter() 与 MD 文件互转。
    """

    # ── 标识 ──
    id: str                                     # 12位短UUID
    name: str = ""                              # 可读名（≤80字）

    # ── 内容 ──
    content: str = ""                           # 正文（Markdown）
    summary: str = ""                           # 脱水压缩摘要

    # ── 分类 ──
    bucket_type: BucketType = BucketType.DYNAMIC
    memory_type: MemoryType = MemoryType.CHAT   # 内容类型（策略差异化）
    domain: list[str] = field(default_factory=list)   # 主题域 ["成长","求职"]
    tags: list[str] = field(default_factory=list)     # 关键词标签

    # ── 情感坐标 ──
    valence: float = 0.5                        # 效价 0~1
    arousal: float = 0.3                        # 唤醒度 0~1
    model_valence: Optional[float] = None        # 模型独立感受

    # ── 重要性 ──
    importance: int = 5                         # 1~10 基础重要性
    importance_sync: float = 5.0                # 同步路径即时评分
    importance_async: float = 5.0               # 异步路径深度评分
    importance_emergent: float = 5.0            # 涌现重要性（随时间演化）

    # ── 状态标记 ──
    resolved: bool = False                      # 已解决/沉底
    digested: bool = False                      # 已消化/写过 feel
    pinned: bool = False                        # 钉选（永不衰减）
    protected: bool = False                     # 保护（永不衰减）

    # ── 访问统计 ──
    activation_count: int = 0                   # 被想起次数
    retrieval_count: int = 0                    # 被检索次数
    created: str = ""                           # ISO 时间戳
    last_active: str = ""                       # ISO 时间戳
    updated: str = ""                           # ISO 时间戳

    # ── 闪光灯记忆 ──
    is_flashbulb: bool = False                  # 是否为闪光灯记忆
    flashbulb_context: dict = field(default_factory=dict)  # 接收情境

    # ── 图关联 ──
    edge_ids: list[str] = field(default_factory=list)     # 关联边 ID 列表

    def __post_init__(self):
        """Validate and clamp fields."""
        self.importance = max(1, min(10, self.importance))
        self.valence = max(0.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if not self.last_active:
            self.last_active = self.created
        if not self.updated:
            self.updated = self.created

    # ── 与 YAML frontmatter 互转 ──

    @classmethod
    def from_frontmatter(cls, post) -> MemoryNode:
        """
        从 python-frontmatter Post 对象构建 MemoryNode。
        兼容现有 Ombre Brain MD 文件格式。

        Usage:
            import frontmatter
            post = frontmatter.load("buckets/dynamic/成长/some_memory.md")
            node = MemoryNode.from_frontmatter(post)
        """
        meta = post.metadata or {}
        content = post.content or ""

        return cls(
            id=meta.get("id", ""),
            name=meta.get("name", ""),
            content=content,
            bucket_type=BucketType(meta.get("type", "dynamic")),
            memory_type=MemoryType(meta.get("memory_type", "chat")),
            domain=cls._parse_list(meta.get("domain", [])),
            tags=cls._parse_list(meta.get("tags", [])),
            valence=float(meta.get("valence", 0.5)),
            arousal=float(meta.get("arousal", 0.3)),
            model_valence=meta.get("model_valence"),
            importance=int(meta.get("importance", 5)),
            resolved=bool(meta.get("resolved", False)),
            digested=bool(meta.get("digested", False)),
            pinned=bool(meta.get("pinned", False)),
            protected=bool(meta.get("protected", False)),
            activation_count=int(meta.get("activation_count", 0)),
            retrieval_count=int(meta.get("retrieval_count", 0)),
            created=str(meta.get("created", "")),
            last_active=str(meta.get("last_active", "")),
            is_flashbulb=bool(meta.get("is_flashbulb", False)),
            flashbulb_context=meta.get("flashbulb_context", {}),
        )

    def to_frontmatter(self) -> dict:
        """
        转换为 YAML frontmatter 字典（用于写入 MD 文件）。
        不包含 content — content 是 MD 正文。
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.bucket_type.value,
            "memory_type": self.memory_type.value,
            "domain": self.domain,
            "tags": self.tags,
            "valence": self.valence,
            "arousal": self.arousal,
            "model_valence": self.model_valence,
            "importance": self.importance,
            "resolved": self.resolved,
            "digested": self.digested,
            "pinned": self.pinned,
            "protected": self.protected,
            "activation_count": self.activation_count,
            "retrieval_count": self.retrieval_count,
            "created": self.created,
            "last_active": self.last_active,
            "updated": self.updated,
            "is_flashbulb": self.is_flashbulb,
            "flashbulb_context": self.flashbulb_context,
        }

    @classmethod
    def from_meta_dict(cls, meta: dict, content: str = "") -> MemoryNode:
        """
        From existing bucket_manager metadata dict (backward compat).
        从现有 bucket_manager 返回的 metadata dict 构建。
        """
        return cls(
            id=meta.get("id", ""),
            name=meta.get("name", ""),
            content=content or meta.get("content", ""),
            bucket_type=BucketType(meta.get("type", "dynamic")),
            memory_type=MemoryType(meta.get("memory_type", "chat")),
            domain=cls._parse_list(meta.get("domain", [])),
            tags=cls._parse_list(meta.get("tags", [])),
            valence=float(meta.get("valence", 0.5)),
            arousal=float(meta.get("arousal", 0.3)),
            model_valence=meta.get("model_valence"),
            importance=int(meta.get("importance", 5)),
            resolved=bool(meta.get("resolved", False)),
            digested=bool(meta.get("digested", False)),
            pinned=bool(meta.get("pinned", False)),
            protected=bool(meta.get("protected", False)),
            activation_count=int(meta.get("activation_count", 0)),
            retrieval_count=int(meta.get("retrieval_count", 0)),
            created=str(meta.get("created", "")),
            last_active=str(meta.get("last_active", "")),
            is_flashbulb=bool(meta.get("is_flashbulb", False)),
            flashbulb_context=meta.get("flashbulb_context", {}),
        )

    # ── 辅助方法 ──

    @staticmethod
    def _parse_list(value) -> list:
        """Parse list from string or list."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            if not value.strip():
                return []
            return [v.strip() for v in value.split(",") if v.strip()]
        return []

    @property
    def days_since_active(self) -> float:
        """距上次激活的天数"""
        try:
            last = datetime.fromisoformat(self.last_active)
            return max(0.0, (datetime.now() - last).total_seconds() / 86400)
        except (ValueError, TypeError):
            return 30.0

    @property
    def emotion_coord(self) -> ValenceArousal:
        """情感坐标"""
        return ValenceArousal(valence=self.valence, arousal=self.arousal)

    @property
    def is_immortal(self) -> bool:
        """永不衰减的记忆"""
        return (
            self.pinned
            or self.protected
            or self.bucket_type == BucketType.PERMANENT
            or self.bucket_type == BucketType.FEEL
        )

    def touch(self):
        """记录一次激活（被 breath 浮现或被检索命中的时候调用）"""
        self.activation_count += 1
        self.retrieval_count += 1
        self.last_active = datetime.now(timezone.utc).isoformat()
        self.updated = self.last_active


# ── DDA 策略配置 ──

@dataclass
class DDAStrategy:
    """
    DDA 四级策略配置 — 由 dda_controller 根据 DDI 选择。

    Design §2.2 策略矩阵:
        COLD: 全量存储·全量返回·LLM先验情绪·无衰减
        WARM: 语义+LLM知识判断·语义相似+时间排序·LLM知识先验
        HOT:  统计偏离·三路混合·个人情感模型
        RICH: 全v6模型·四路+Working Self·微调个人模型
    """

    # 存储门禁
    store_all: bool = True                # True = 不筛选·全存
    use_llm_gate: bool = False            # 用LLM判断是否值得存
    use_statistical_gate: bool = False    # 用统计偏离判断
    use_vulnerability_gate: bool = False  # 用脆弱性调整门禁

    # 检索策略
    retrieval_mode: str = "all"           # all | semantic_time | three_way | four_way_ws
    retrieval_top_k: int = 20
    use_vector_search: bool = True
    use_bm25_search: bool = False
    use_graph_search: bool = False
    use_ws_rerank: bool = False

    # 情绪策略
    emotion_mode: str = "warm_default"    # warm_default | llm_prior | personal_baseline | fine_tuned

    # 脆弱性
    vulnerability_enabled: bool = False

    # 衰减
    decay_enabled: bool = False
    decay_lambda: float = 0.05

    # 重要性
    importance_mode: str = "sync_only"    # sync_only | sync_async | full_fusion


# ── 预定义四级策略 ──

COLD_STRATEGY = DDAStrategy(
    store_all=True,
    retrieval_mode="cold_fusion",       # v9: light 3-path fusion (was "all" — ignored query)
    use_vector_search=False,            # BM25 sufficient for tiny corpora
    use_bm25_search=True,               # v9: enabled — BM25 is zero-dep, works on 1 doc
    use_graph_search=False,             # needs node topology, not available at COLD
    use_ws_rerank=False,                # no goal history at COLD
    emotion_mode="query_driven",        # v9: match query emotion (was "warm_default")
    vulnerability_enabled=False,
    decay_enabled=False,
    importance_mode="sync_only",
)

WARM_STRATEGY = DDAStrategy(
    store_all=False,
    use_llm_gate=True,
    retrieval_mode="cold_fusion",       # v9: same as COLD — BM25 dominates at <100 docs (BEIR/Thakur 2021)
    use_vector_search=False,            # v9: dense embeddings unreliable at 10-50 docs
    use_bm25_search=True,               # v9: BM25 strictly superior at WARM scale
    use_graph_search=False,
    use_ws_rerank=False,
    emotion_mode="query_driven",        # v9: match query emotion, no LLM call needed (was "llm_prior")
    vulnerability_enabled=False,
    decay_enabled=True,                 # WARM differentiator: decay is active
    decay_lambda=0.05,
    importance_mode="sync_only",
)

HOT_STRATEGY = DDAStrategy(
    store_all=False,
    use_llm_gate=True,
    use_statistical_gate=True,
    retrieval_mode="three_way",
    use_vector_search=True,
    use_bm25_search=True,
    use_graph_search=True,
    use_ws_rerank=False,
    emotion_mode="personal_baseline",
    vulnerability_enabled=True,
    decay_enabled=True,
    decay_lambda=0.05,
    importance_mode="sync_async",
)

RICH_STRATEGY = DDAStrategy(
    store_all=False,
    use_llm_gate=True,
    use_statistical_gate=True,
    use_vulnerability_gate=True,
    retrieval_mode="four_way_ws",
    use_vector_search=True,
    use_bm25_search=True,
    use_graph_search=True,
    use_ws_rerank=True,
    emotion_mode="fine_tuned",
    vulnerability_enabled=True,
    decay_enabled=True,
    decay_lambda=0.05,
    importance_mode="full_fusion",
)

# 策略矩阵
STRATEGY_MATRIX: dict[DDILevel, DDAStrategy] = {
    DDILevel.COLD: COLD_STRATEGY,
    DDILevel.WARM: WARM_STRATEGY,
    DDILevel.HOT: HOT_STRATEGY,
    DDILevel.RICH: RICH_STRATEGY,
}
