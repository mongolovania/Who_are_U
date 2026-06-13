# ============================================================
# Benchmark Harness — Real RetrievalEngine with real BucketManager
# 基准测试夹具 — 用真实 BucketManager + DecayEngine 驱动检索引擎
#
# Replaces memory stitching (mock bucket_mgr + fake search) with
# actual filesystem-backed BucketManager, real DecayEngine, and
# RetrievalEngine.search() using real BM25 + multi-path fusion.
# ============================================================

from __future__ import annotations

import asyncio
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucket_manager import BucketManager
from decay_engine import DecayEngine
from retrieval_engine import RetrievalEngine
from memory_node import DDILevel, DDAStrategy, COLD_STRATEGY, WARM_STRATEGY, HOT_STRATEGY, RICH_STRATEGY

from tests.benchmarks.benchmark_dataset import BENCHMARK_MEMORIES, BenchmarkMemory


# ── Config factory for benchmark temp directories ──────────────

def make_benchmark_config(tmp_path) -> dict:
    """Build a config dict pointing to temp directories."""
    buckets_dir = str(tmp_path / "buckets")
    for d in ["permanent", "dynamic", "archive", "feel"]:
        os.makedirs(os.path.join(buckets_dir, d), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "merge_threshold": 75,
        "matching": {"fuzzy_threshold": 50, "max_results": 50},
        "wikilink": {"enabled": False},
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 1.5,
            "importance": 1.0,
            "content_weight": 1.0,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": os.environ.get("OMBRE_API_KEY", "test-key"),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash-lite",
        },
        "embedding": {
            "api_key": os.environ.get("OMBRE_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-embedding-001",
            "enabled": False,
        },
    }


# ── Synthetic noise memory generator ──────────────────────────

# Template pool for generating unrelated but realistic-looking memories
_NOISE_TEMPLATES = [
    ("今天去{location}吃了{food}，味道{mood}。", ["日常", "饮食"], 0.55, 0.25),
    ("{location}的天气{mood}，{action}了一下午。", ["日常", "天气"], 0.50, 0.30),
    ("买了一本关于{topic}的书，打算{timeframe}看完。", ["学习", "阅读"], 0.60, 0.35),
    ("跟朋友在{location}{activity}，聊了很多关于{topic}的事。", ["社交"], 0.65, 0.45),
    ("{platform}上看到一个关于{topic}的视频，{opinion}。", ["娱乐", "视频"], 0.55, 0.40),
    ("最近在学习{topic}，感觉{mood}。", ["学习"], 0.55, 0.35),
    ("{location}的{transport}太{mood}了，{action}。", ["日常", "通勤"], 0.40, 0.30),
    ("今天{activity_type}的时候想到了一个关于{topic}的点子。", ["创意"], 0.65, 0.50),
    ("收拾房间发现了{timeframe}前的{item}，{emotion}。", ["日常", "回忆"], 0.55, 0.40),
    ("{platform}上{mood}的消息让人{emotion}。", ["社交媒体", "情绪"], 0.35, 0.55),
    ("最近在追一部关于{topic}的{media_type}，{opinion}。", ["娱乐"], 0.60, 0.40),
    ("跟同事讨论了{topic}，大家意见{mood}。", ["工作", "社交"], 0.55, 0.45),
    ("健身{action}了一小时，感觉{mood}。", ["健康", "运动"], 0.65, 0.55),
    ("周末去{location}{activity}，{emotion}。", ["休闲"], 0.70, 0.45),
    ("听说{location}新开了家{shop_type}，{opinion}。", ["日常"], 0.55, 0.35),
    ("花了{amount}买了{product}，{opinion}。", ["购物"], 0.55, 0.30),
    ("{platform}上的{topic}讨论{mood}。", ["社交媒体"], 0.50, 0.45),
    ("整理了一下{tool}的使用笔记，{opinion}。", ["学习", "工具"], 0.60, 0.30),
    ("喝了杯{drink}，{emotion}。", ["日常"], 0.55, 0.25),
    ("最近睡眠{mood}，{action}。", ["健康"], 0.45, 0.35),
]

_LOCATIONS = ["朝阳区", "海淀区", "浦东", "天河", "南山", "西湖区", "秦淮区", "武侯区", "园区", "滨江区"]
_FOODS = ["麻辣烫", "日料", "火锅", "拉面", "汉堡", "披萨", "寿司", "烤肉", "米线", "饺子"]
_MOODS = ["还不错", "一般般", "挺好的", "不太行", "很赞", "有点失望", "超出预期", "中规中矩"]
_ACTIONS = ["发呆", "加班", "看书", "写代码", "画图", "整理笔记", "刷手机", "做家务"]
_TOPICS = ["系统设计", "机器学习", "认知科学", "日本战国史", "摄影构图", "咖啡烘焙", "攀岩技巧",
           "古罗马建筑", "榫卯结构", "调香", "剧本写作", "陶艺", "观鸟", "城市设计", "字体排印"]
_TIMEFRAMES = ["一周", "两周", "一个月", "三个月", "周末", "年底前", "明年", "暑假"]
_PLATFORMS = ["B站", "知乎", "小红书", "豆瓣", "Twitter", "即刻", "V2EX", "抖音"]
_OPINIONS = ["很有意思", "颠覆认知", "可以试试", "不太同意", "值得收藏", "有点水", "干货满满"]
_TRANSPORTS = ["地铁", "公交", "打车", "骑行", "开车", "步行"]
_ACTIVITIES = ["逛了一圈", "拍照", "写生", "跑步", "野餐", "闲逛"]
_ACTIVITY_TYPES = ["洗澡", "散步", "做饭", "拖地", "洗衣服", "浇花"]
_EMOTIONS = ["有点怀念", "挺感慨的", "笑了半天", "心情复杂", "莫名感动", "哭笑不得"]
_ITEMS = ["旧照片", "笔记本", "明信片", "票根", "磁带", "老手机", "贺卡"]
_MEDIA_TYPES = ["纪录片", "动漫", "电影", "综艺", "播客", "有声书"]
_SHOP_TYPES = ["咖啡馆", "书店", "面包店", "花店", "杂货铺", "唱片店", "茶室"]
_AMOUNTS = ["三十块", "五十块", "一百多", "两百多", "几十块", "不到一百"]
_PRODUCTS = ["蓝牙耳机", "机械键盘", "台灯", "收纳盒", "水杯", "文具套装", "充电宝"]
_TOOLS = ["Obsidian", "Notion", "Vim", "Figma", "Blender", "Anki", "Logseq"]
_DRINKS = ["美式咖啡", "拿铁", "抹茶", "柠檬水", "气泡水", "冷萃", "热可可"]
_EMOTION_LIST = ["心情好多了", "觉得还不错", "挺舒服的", "感觉很平静", "有点小开心", "觉得挺值"]


def _pick(lst: list) -> str:
    return random.choice(lst)


def generate_noise_memories(count: int, seed: int = 42) -> list[BenchmarkMemory]:
    """
    Generate synthetic noise memories that are semantically unrelated
    to the XiaoMing benchmark story but look like realistic diary entries.

    生成与小明故事语义无关但看起来像真实日记的合成噪声记忆。
    """
    rng = random.Random(seed)
    memories = []
    now = datetime.now()

    for i in range(count):
        template, tags, base_valence, base_arousal = rng.choice(_NOISE_TEMPLATES)
        content = template.format(
            location=_pick(_LOCATIONS),
            food=_pick(_FOODS),
            mood=_pick(_MOODS),
            action=_pick(_ACTIONS),
            topic=_pick(_TOPICS),
            timeframe=_pick(_TIMEFRAMES),
            platform=_pick(_PLATFORMS),
            opinion=_pick(_OPINIONS),
            transport=_pick(_TRANSPORTS),
            activity=_pick(_ACTIVITIES),
            activity_type=_pick(_ACTIVITY_TYPES),
            emotion=_pick(_EMOTIONS),
            item=_pick(_ITEMS),
            media_type=_pick(_MEDIA_TYPES),
            shop_type=_pick(_SHOP_TYPES),
            amount=_pick(_AMOUNTS),
            product=_pick(_PRODUCTS),
            tool=_pick(_TOOLS),
            drink=_pick(_DRINKS),
        )

        # Add small random jitter to valence/arousal
        valence = max(0.1, min(0.9, base_valence + rng.uniform(-0.15, 0.15)))
        arousal = max(0.1, min(0.9, base_arousal + rng.uniform(-0.15, 0.15)))

        # Timestamps spread across last 90 days
        days_ago = rng.randint(1, 90)
        created = (now - timedelta(days=days_ago)).isoformat()

        memories.append(BenchmarkMemory(
            content=content,
            memory_type=rng.choice(["chat", "chat", "chat", "emotion"]),
            importance=rng.randint(1, 5),
            valence=round(valence, 2),
            arousal=round(arousal, 2),
            tags=list(tags),
            created=created,
            session_id=f"noise_{i // 5}",
        ))

    return memories


# ── Benchmark Harness ─────────────────────────────────────────

class BenchmarkHarness:
    """
    Real benchmark harness: creates a filesystem-backed BucketManager,
    populates it with memories, and provides RetrievalEngine.search()
    with real infrastructure.

    Usage:
        harness = BenchmarkHarness(tmp_path)
        await harness.populate(BENCHMARK_MEMORIES)
        results = await harness.search("小明在哪里工作？", strategy=HOT_STRATEGY)
    """

    def __init__(self, tmp_path, user_id: str = "benchmark_user"):
        self.tmp_path = tmp_path
        self.user_id = user_id
        self.config = make_benchmark_config(tmp_path)

        # Real BucketManager with temp directories
        self.bucket_mgr = BucketManager(self.config, user_id=user_id)

        # Real DecayEngine
        self.decay_engine = DecayEngine(self.config, self.bucket_mgr, user_id=user_id)

        # RetrievalEngine with default weights (no external modules)
        self.engine = RetrievalEngine(
            narrative_engine=None,
            hippo_rag=None,
            graph_rag=None,
            learnable_weights=None,
        )

        # Track populated memory ID → index mapping
        self._memory_ids: list[str] = []
        self._memories: list[BenchmarkMemory] = []

    async def populate(
        self,
        memories: list[BenchmarkMemory],
        batch_size: int = 10,
    ) -> list[str]:
        """
        Write benchmark memories to disk via BucketManager.create().
        Returns list of bucket IDs in order.

        通过 BucketManager.create() 将基准记忆写入磁盘。
        """
        ids = []
        self._memories = memories

        for i, mem in enumerate(memories):
            # Generate a deterministic name for traceability
            name = f"bm_mem_{i:04d}"
            if mem.tags:
                name = f"{mem.tags[0]}_{i:04d}"

            bid = await self.bucket_mgr.create(
                content=mem.content,
                tags=mem.tags,
                importance=mem.importance,
                domain=mem.tags[:1] if mem.tags else ["未分类"],
                valence=mem.valence,
                arousal=mem.arousal,
                bucket_type="dynamic",
                name=name,
            )
            ids.append(bid)

            # Small delay to avoid filesystem contention on large batches
            if i > 0 and i % batch_size == 0:
                await asyncio.sleep(0)

        self._memory_ids = ids
        return ids

    async def populate_async_batch(
        self,
        memories: list[BenchmarkMemory],
        concurrency: int = 10,
    ) -> list[str]:
        """
        Faster populate using asyncio.gather for large datasets.
        使用 asyncio.gather 加速大批量写入。
        """
        self._memories = memories
        semaphore = asyncio.Semaphore(concurrency)
        ids_by_index: dict[int, str] = {}

        async def _create_one(idx: int, mem: BenchmarkMemory):
            async with semaphore:
                name = f"bm_mem_{idx:04d}"
                if mem.tags:
                    name = f"{mem.tags[0]}_{idx:04d}"
                bid = await self.bucket_mgr.create(
                    content=mem.content,
                    tags=mem.tags,
                    importance=mem.importance,
                    domain=mem.tags[:1] if mem.tags else ["未分类"],
                    valence=mem.valence,
                    arousal=mem.arousal,
                    bucket_type="dynamic",
                    name=name,
                )
                ids_by_index[idx] = bid

        await asyncio.gather(*[
            _create_one(i, mem) for i, mem in enumerate(memories)
        ])

        ids = [ids_by_index[i] for i in sorted(ids_by_index)]
        self._memory_ids = ids
        return ids

    async def search(
        self,
        query: str,
        strategy: DDAStrategy = HOT_STRATEGY,
        ddi_level: DDILevel = DDILevel.HOT,
        top_k: int = 25,
        disabled_paths: set[str] | None = None,  # v9 Ablation
    ) -> list[dict]:
        """
        Run real RetrievalEngine.search() with real BucketManager + DecayEngine.

        用真实的 BucketManager + DecayEngine 运行 RetrieveEngine.search()。
        """
        return await self.engine.search(
            query=query,
            strategy=strategy,
            ddi_level=ddi_level,
            bucket_mgr=self.bucket_mgr,
            embedding_engine=None,
            memory_graph=None,
            working_self=None,
            decay_engine=self.decay_engine,
            user_id=self.user_id,
            top_k=top_k,
            disabled_paths=disabled_paths,
        )

    @property
    def memory_count(self) -> int:
        """Total number of populated memories."""
        return len(self._memory_ids)

    async def get_all_buckets(self) -> list[dict]:
        """Get all buckets from the real BucketManager."""
        return await self.bucket_mgr.list_all(include_archive=False)


# ── Convenience: DDI strategy lookup ──────────────────────────

DDI_STRATEGIES = {
    "COLD": (COLD_STRATEGY, DDILevel.COLD),
    "WARM": (WARM_STRATEGY, DDILevel.WARM),
    "HOT": (HOT_STRATEGY, DDILevel.HOT),
    "RICH": (RICH_STRATEGY, DDILevel.RICH),
}
