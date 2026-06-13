# ============================================================
# Unified Benchmark Dataset
# 统一对比测试数据集
#
# 所有对比系统（Memory Palace, Mem0, Letta, Graphiti）使用同一组
# 输入数据，确保对比公平。
#
# 数据集设计原则（从 hazy-baking-puffin.md 辩论场景发散）：
#   1. 覆盖认知记忆的核心维度
#   2. 包含时序演变
#   3. 包含情绪变化
#   4. 包含矛盾信息（测试边失效）
#   5. 包含稀疏/密集两种使用模式
# ============================================================

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

_NOW = datetime.now()


def _ago(**kwargs) -> str:
    return (_NOW - timedelta(**kwargs)).isoformat()


@dataclass
class BenchmarkMemory:
    """A single memory entry for cross-system comparison."""
    content: str
    memory_type: str = "chat"       # chat | decision | milestone | emotion
    importance: int = 5
    valence: float = 0.5
    arousal: float = 0.3
    tags: list[str] = field(default_factory=list)
    created: str = ""
    session_id: str = ""


# ── Core Dataset: 30 memories spanning 6 sessions ──────────

BENCHMARK_MEMORIES: list[BenchmarkMemory] = [
    # === Session 1: 冷启动，自我介绍 (DDI=COLD) ===
    BenchmarkMemory(
        content="我叫小明，今年25岁，在北京做程序员，主要用Python和Go。",
        memory_type="chat", importance=7, valence=0.6, arousal=0.3,
        tags=["身份", "职业", "编程"], created=_ago(days=60), session_id="s1",
    ),
    BenchmarkMemory(
        content="最近公司传要裁员，很焦虑，每天晚上都睡不好。",
        memory_type="emotion", importance=8, valence=0.2, arousal=0.8,
        tags=["焦虑", "裁员", "工作"], created=_ago(days=60), session_id="s1",
    ),
    BenchmarkMemory(
        content="我喜欢编程，觉得创造东西很有成就感。但大厂的流程让我很压抑。",
        memory_type="chat", importance=6, valence=0.4, arousal=0.5,
        tags=["编程", "职业困惑"], created=_ago(days=60), session_id="s1",
    ),

    # === Session 2: 工作机会 (DDI=WARM approaching) ===
    BenchmarkMemory(
        content="收到了一家AI创业公司的面试邀请，做LLM方向的，我很感兴趣。",
        memory_type="chat", importance=7, valence=0.7, arousal=0.7,
        tags=["面试", "AI", "创业公司"], created=_ago(days=55), session_id="s2",
    ),
    BenchmarkMemory(
        content="我犹豫要不要离开大厂去创业公司。大厂稳定但无聊，创业公司冒险但可能成长更快。",
        memory_type="decision", importance=8, valence=0.4, arousal=0.6,
        tags=["决策", "职业", "冒险"], created=_ago(days=55), session_id="s2",
    ),
    BenchmarkMemory(
        content="我妈觉得创业公司不稳定，劝我留在现在的大厂。",
        memory_type="chat", importance=5, valence=0.35, arousal=0.4,
        tags=["家庭", "建议"], created=_ago(days=54), session_id="s2",
    ),

    # === Session 3: 情绪波动 ===
    BenchmarkMemory(
        content="今天被leader当众批评了，说我最近工作效率低。我觉得很委屈，明明是因为焦虑才睡不好。",
        memory_type="emotion", importance=7, valence=0.1, arousal=0.85,
        tags=["职场", "批评", "委屈"], created=_ago(days=48), session_id="s3",
    ),
    BenchmarkMemory(
        content="失眠到今天凌晨4点，想了很久要不要辞职。反正公司可能也要裁员。",
        memory_type="emotion", importance=9, valence=0.15, arousal=0.9,
        tags=["失眠", "焦虑", "辞职"], created=_ago(days=48), session_id="s3",
    ),

    # === Session 4: 转折点 (闪光灯记忆候选) ===
    BenchmarkMemory(
        content="天啊！！我拿到了AI创业公司的offer！！薪资比现在高30%！！简直不敢相信！！",
        memory_type="milestone", importance=10, valence=0.95, arousal=0.95,
        tags=["offer", "转折", "AI"], created=_ago(days=40), session_id="s4",
    ),
    BenchmarkMemory(
        content="我决定接受这个offer。之前怕冒险，现在怕错过机会。人生总要勇敢一次吧。",
        memory_type="decision", importance=10, valence=0.8, arousal=0.7,
        tags=["决策", "勇敢", "转折"], created=_ago(days=40), session_id="s4",
    ),
    BenchmarkMemory(
        content="跟leader提了离职，突然有种如释重负的感觉。这半年太累了。",
        memory_type="chat", importance=6, valence=0.6, arousal=0.5,
        tags=["离职", "解脱"], created=_ago(days=38), session_id="s4",
    ),

    # === Session 5: 新生活 (Working Self 切换) ===
    BenchmarkMemory(
        content="新公司第一周结束了！团队很小但每个人都很厉害，氛围特别好。",
        memory_type="chat", importance=6, valence=0.85, arousal=0.7,
        tags=["新工作", "团队"], created=_ago(days=30), session_id="s5",
    ),
    BenchmarkMemory(
        content="今天做了一个技术分享，同事说我讲得很清楚。在之前的公司从来没有这种机会。",
        memory_type="chat", importance=7, valence=0.9, arousal=0.6,
        tags=["成长", "自信"], created=_ago(days=25), session_id="s5",
    ),
    BenchmarkMemory(
        content="我突然发现，我已经一周没有失眠了。每天早上醒来都很期待去上班。",
        memory_type="emotion", importance=8, valence=0.85, arousal=0.4,
        tags=["健康", "幸福", "变化"], created=_ago(days=20), session_id="s5",
    ),

    # === Session 6: 反思回顾 ===
    BenchmarkMemory(
        content="周末在家想了想这半年的变化。从焦虑裁员到在新公司做自己喜欢的事。",
        memory_type="chat", importance=7, valence=0.7, arousal=0.4,
        tags=["反思", "成长"], created=_ago(days=10), session_id="s6",
    ),
    BenchmarkMemory(
        content="我妈前几天打电话说'看你最近状态很好，看来当初的决定是对的'——这句话对我来说意义太重大了。",
        memory_type="emotion", importance=9, valence=0.9, arousal=0.5,
        tags=["家庭", "认可", "感动"], created=_ago(days=8), session_id="s6",
    ),
    BenchmarkMemory(
        content="其实有时候我也会想，如果当初留在大厂会不会更好？但这种念头很快就过去了。现在的我比半年前开心太多了。",
        memory_type="chat", importance=6, valence=0.65, arousal=0.3,
        tags=["反思", "对比"], created=_ago(days=5), session_id="s6",
    ),

    # === 额外：平淡日常（测试噪声过滤） ===
    BenchmarkMemory(
        content="今天中午吃了个三明治。",
        memory_type="chat", importance=2, valence=0.5, arousal=0.1,
        tags=["日常"], created=_ago(days=3), session_id="s6",
    ),
    BenchmarkMemory(
        content="下班路上堵车了，晚了半小时到家。",
        memory_type="chat", importance=1, valence=0.4, arousal=0.2,
        tags=["日常", "通勤"], created=_ago(days=2), session_id="s6",
    ),
    BenchmarkMemory(
        content="去超市买了两箱牛奶，打折的。",
        memory_type="chat", importance=1, valence=0.5, arousal=0.1,
        tags=["日常", "购物"], created=_ago(days=1), session_id="s6",
    ),

    # === 矛盾信息（测试边失效） ===
    BenchmarkMemory(
        content="我现在在大厂做后端开发。",
        memory_type="chat", importance=5, valence=0.5, arousal=0.3,
        tags=["职业"], created=_ago(days=60), session_id="s1",
    ),
    BenchmarkMemory(
        content="我在AI创业公司做LLM开发，已经入职一个月了。",
        memory_type="chat", importance=8, valence=0.85, arousal=0.6,
        tags=["职业", "更新"], created=_ago(days=25), session_id="s5",
    ),
]

# ── Retrieval Benchmarks (from LoCoMo-inspired queries) ────

RETRIEVAL_BENCHMARKS = [
    # (query, expected_memory_indices, dimension_tested)
    ("小明在哪里工作？", [21], "单事实检索 - 需要返回最新信息"),
    ("小明今年多大？", [0], "单事实检索 - 个人信息"),
    ("小明为什么焦虑？", [1], "情绪原因检索"),
    ("小明做过什么重要决定？", [4, 9], "决策记忆检索"),
    ("小明拿到offer是什么反应？", [8], "闪光灯记忆检索"),
    ("小明在新公司感觉怎么样？", [12, 13, 15], "跨会话检索"),
    ("小明失眠了几次？", [1, 7], "重复议题检测"),
    ("小明妈妈对换工作的态度？", [5, 15], "多跳推理 - 态度演变"),
    ("小明喜欢什么？", [0, 2], "偏好检索"),
]

# ── Scenario Test Cases (from hazy-baking-puffin.md debates) ──

# ── Sample size variants for comprehensive benchmarking ──────

# Small corpus: first 10 memories (COLD → WARM transition zone)
BENCHMARK_MEMORIES_SMALL: list[BenchmarkMemory] = BENCHMARK_MEMORIES[:10]

# Dataset size constants for benchmark parameterization
DATASET_SIZES = {
    "small": 10,      # Small: 10 core memories (COLD zone)
    "medium": 22,     # Medium: all 22 benchmark memories (WARM zone)
    "large": 72,      # Large: 22 core + 50 noise (HOT zone)
    "xlarge": 222,    # XLarge: 22 core + 200 noise (RICH zone)
}

# Dataset constructors by size
def get_dataset(size: str) -> list[BenchmarkMemory]:
    """Get benchmark dataset at a given size.

    Args:
        size: 'small' | 'medium' | 'large' | 'xlarge'
    """
    from tests.benchmarks.benchmark_harness import generate_noise_memories

    if size == "small":
        return list(BENCHMARK_MEMORIES_SMALL)
    elif size == "medium":
        return list(BENCHMARK_MEMORIES)
    elif size == "large":
        return list(BENCHMARK_MEMORIES) + generate_noise_memories(50, seed=42)
    elif size == "xlarge":
        return list(BENCHMARK_MEMORIES) + generate_noise_memories(200, seed=99)
    else:
        raise ValueError(f"Unknown dataset size: {size}")


SCENARIO_DEFINITIONS = {
    "s1_cold_start": {
        "name": "冷启动用户的第一印象",
        "source_debate": "v3→v4, COLD策略矩阵",
        "user_data": [],
        "input": "我今天面试又被拒了，感觉很失落",
        "expected": {
            "should_store": True,
            "decay_enabled": False,
            "retrieval_mode": "all",
            "cross_user_data_used": False,
        },
    },
    "s2_chronic_low": {
        "name": "慢性低落用户的最后一根稻草",
        "source_debate": "用户质疑#6, v2→v3",
        "user_data": [{"valence": 0.2, "arousal": 0.25}] * 30,
        "input": "今天同事说了一句话，我突然就哭了。不知道为什么。",
        "expected": {
            "vi_above_07": True,
            "storage_threshold_lowered": True,
            "importance_elevated": True,
        },
    },
    "s3_flashbulb": {
        "name": "闪光灯记忆检测",
        "source_debate": "Brown&Kulik, v0→v1",
        "input": "我被裁员了！！就在刚才！！HR突然叫我进办公室！！完全没想到！！",
        "input_valence": 0.1,
        "input_arousal": 0.95,
        "expected": {
            "is_flashbulb": True,
            "decay_protection_applied": True,
            "context_stored": True,
        },
    },
    "s4_emotion_congruence": {
        "name": "情绪一致性检索",
        "source_debate": "Bower, v1→v2",
        "query_valence": 0.8,
        "query_arousal": 0.5,
        "expected": {
            "retrieved_valence_mean_above_06": True,
            "happy_memories_rank_higher": True,
        },
    },
    "s5_sparse_user": {
        "name": "稀疏用户的记忆保护",
        "source_debate": "用户质疑#7, v3→v4",
        "ddi_score": 5,
        "days_since_last": 7,
        "expected": {
            "ddi_level": "COLD",
            "decay_enabled": False,
            "store_all": True,
        },
    },
    "s6_edge_expiry": {
        "name": "记忆矛盾与边失效",
        "source_debate": "Zep, v0→v1 反对#3",
        "old_fact": "我在A公司工作",
        "new_fact": "我跳槽到了B公司",
        "expected": {
            "old_edge_expired": True,
            "old_edge_not_deleted": True,
            "query_returns_new_fact": True,
        },
    },
    "s7_script_deviation": {
        "name": "统计偏离触发门禁",
        "source_debate": "Schank, v0→v1 反对#1",
        "baseline_valence_mean": 0.55,
        "baseline_arousal_mean": 0.3,
        "current_valence": 0.1,
        "current_arousal": 0.9,
        "session_hour": 3,
        "expected": {
            "deviation_detected": True,
            "deviation_score_above_05": True,
            "detection_time_under_10ms": True,
        },
    },
    "s8_ddi_upgrade": {
        "name": "DDI渐进升级",
        "source_debate": "DDA-MM, v3→v4",
        "ddi_progression": [0, 5, 10, 25, 50, 100, 200, 300],
        "expected": {
            "level_progression": ["COLD", "COLD", "WARM", "WARM", "HOT", "HOT", "RICH", "RICH"],
            "retrieval_paths_increase": True,
            "smooth_transition": True,
        },
    },
    "s9_working_self_switch": {
        "name": "Working Self目标切换",
        "source_debate": "Conway SMS, v0→v1",
        "old_goals": [{"description": "找工作", "domain": "career", "priority": 0.9}],
        "new_goals": [{"description": "适应新工作", "domain": "career", "priority": 0.8}],
        "expected": {
            "goal_changed": True,
            "retrieved_memories_shift": True,
        },
    },
    "s10_dp_fallback": {
        "name": "差分隐私先验退化",
        "source_debate": "双层先验, v5→v6",
        "l2_available": False,
        "expected": {
            "fallback_to_l1": True,
            "function_not_interrupted": True,
        },
    },
}
