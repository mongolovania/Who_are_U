# ============================================================
# Comparison QA Dataset — 25 manually annotated questions
# 横向对比QA数据集 — 25道人工标注问题
#
# 6 categories (from LoCoMo + A-MEM + LongMemEval methodology):
#   A. Simple Recall (简单回忆)         — 5 questions
#   B. Multi-hop Reasoning (多跳推理)    — 5 questions
#   C. Temporal Reasoning (时间推理)     — 5 questions
#   D. Emotional Memory (情感记忆)       — 4 questions
#   E. Causal Reasoning (因果推理)       — 3 questions
#   F. Cross-Reference (跨引用)          — 3 questions
#
# All questions use the existing 22-memory XiaoMing dataset
# (benchmark_dataset.py BENCHMARK_MEMORIES) as the unified corpus.
# Every question has manually annotated ground truth with:
#   - expected_answer: the correct answer text
#   - relevant_memory_indices: which memory indices contain the facts
#   - reasoning_chain: for multi-hop, the hop sequence
#   - difficulty: 1(easy) to 5(hard)
# ============================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComparisonQA:
    """A single QA pair for cross-system comparison."""
    id: str                          # e.g. "A1", "B3"
    category: str                    # simple_recall | multi_hop | temporal | emotional | causal | cross_ref
    question: str                    # the query
    expected_answer: str             # manually annotated ground truth
    relevant_memory_indices: list[int]  # indices into BENCHMARK_MEMORIES
    reasoning_chain: list[str] = field(default_factory=list)  # hop descriptions
    difficulty: int = 1              # 1-5
    keywords: list[str] = field(default_factory=list)  # expected keywords in answer
    category_cn: str = ""            # Chinese category name


# ═══════════════════════════════════════════════════════════════
# CATEGORY A: Simple Recall (简单回忆) — 5 questions
# Single-fact retrieval. Does the system find the exact fact?
# ═══════════════════════════════════════════════════════════════

QA_SIMPLE_RECALL = [
    ComparisonQA(
        id="A1",
        category="simple_recall",
        category_cn="简单回忆",
        question="小明叫什么名字？今年多大？做什么工作？",
        expected_answer="小明，25岁，在北京做程序员，主要用Python和Go。",
        relevant_memory_indices=[0],
        keywords=["小明", "25", "北京", "程序员", "Python", "Go"],
        difficulty=1,
    ),
    ComparisonQA(
        id="A2",
        category="simple_recall",
        category_cn="简单回忆",
        question="小明收到的是哪家公司的offer？具体是什么方向的？",
        expected_answer="AI创业公司的offer，做LLM方向的，薪资比现在高30%。",
        relevant_memory_indices=[3, 8],
        keywords=["AI创业公司", "LLM", "30%", "offer"],
        difficulty=1,
    ),
    ComparisonQA(
        id="A3",
        category="simple_recall",
        category_cn="简单回忆",
        question="小明妈妈对他换工作是什么态度？",
        expected_answer="最初劝他留在现有大厂认为创业公司不稳定，后来看到小明状态变好打电话说'看来当初的决定是对的'。",
        relevant_memory_indices=[5, 15],
        keywords=["劝他留", "大厂", "不稳定", "决定是对的"],
        difficulty=2,
    ),
    ComparisonQA(
        id="A4",
        category="simple_recall",
        category_cn="简单回忆",
        question="小明在新公司有什么让他感到自豪的成就？",
        expected_answer="做了一次技术分享，同事说他讲得很清楚。在之前的公司从来没有这种机会。",
        relevant_memory_indices=[12],
        keywords=["技术分享", "讲得清楚", "从来没有", "机会"],
        difficulty=1,
    ),
    ComparisonQA(
        id="A5",
        category="simple_recall",
        category_cn="简单回忆",
        question="小明为什么最开始感到焦虑？",
        expected_answer="公司传要裁员，每天晚上都睡不好。大厂的流程让他很压抑。",
        relevant_memory_indices=[1, 2],
        keywords=["裁员", "焦虑", "睡不好", "压抑", "流程"],
        difficulty=1,
    ),
]

# ═══════════════════════════════════════════════════════════════
# CATEGORY B: Multi-hop Reasoning (多跳推理) — 5 questions
# Requires linking 2+ memories to answer correctly.
# ═══════════════════════════════════════════════════════════════

QA_MULTI_HOP = [
    ComparisonQA(
        id="B1",
        category="multi_hop",
        category_cn="多跳推理",
        question="小明从焦虑到不再失眠，中间经历了哪些关键事件？",
        expected_answer="收到AI创业公司面试邀请→犹豫是否离开大厂→被leader批评+失眠加重→拿到offer→决定接受→入职新公司→状态好转→不再失眠。",
        relevant_memory_indices=[1, 3, 4, 6, 7, 8, 9, 11, 13],
        reasoning_chain=[
            "焦虑原因(0,1): 裁员传言+大厂压抑",
            "转折机会(3): 收到AI创业公司面试",
            "决策纠结(4): 犹豫大厂vs创业公司",
            "低谷(6,7): 被批评+失眠到凌晨",
            "转折(8,9): 拿到offer→接受",
            "恢复(11,13): 新公司→不再失眠",
        ],
        keywords=["面试", "批评", "offer", "入职", "不再失眠"],
        difficulty=4,
    ),
    ComparisonQA(
        id="B2",
        category="multi_hop",
        category_cn="多跳推理",
        question="小明妈妈对小明的职业选择，态度发生了什么变化？",
        expected_answer="最初反对（劝他留在大厂，说创业公司不稳定），后来认可（打电话说'看来当初的决定是对的'）。态度转变是因为看到小明状态变好。",
        relevant_memory_indices=[5, 15],
        reasoning_chain=[
            "初始态度(5): 妈妈劝留大厂",
            "最终态度(15): 妈妈说决定是对的",
            "转变原因(13,15): 看到小明状态好→认可",
        ],
        keywords=["劝", "不稳定", "决定是对的", "状态很好"],
        difficulty=3,
    ),
    ComparisonQA(
        id="B3",
        category="multi_hop",
        category_cn="多跳推理",
        question="小明在新公司获得了哪些在老公司没有的东西？请全面列举。",
        expected_answer="①技术分享机会（在老公司从来没有这种机会）②同事的认可（同事说他讲得清楚）③不再失眠（每天早上醒来都很期待去上班）④更好的氛围（团队很小但每个人都很厉害，氛围特别好）⑤更高的薪资（比现在高30%）。",
        relevant_memory_indices=[2, 12, 13, 8, 11],
        reasoning_chain=[
            "大厂的问题(2): 流程压抑",
            "新公司团队(11): 氛围好",
            "新公司成长(12): 技术分享+认可",
            "新公司健康(13): 不再失眠",
            "薪资对比(8): 高30%",
        ],
        keywords=["技术分享", "认可", "不再失眠", "氛围", "薪资"],
        difficulty=4,
    ),
    ComparisonQA(
        id="B4",
        category="multi_hop",
        category_cn="多跳推理",
        question="小明经历了哪些情绪阶段？从最初到最后，按顺序描述。",
        expected_answer="焦虑压抑（裁员传言）→期待（收到面试）→低落委屈（被批评）→狂喜（拿到offer）→解脱（提离职）→幸福满足（新公司）→平静反思（回顾半年变化）。",
        relevant_memory_indices=[1, 3, 6, 7, 8, 10, 11, 13, 15, 16],
        reasoning_chain=[
            "焦虑(1): 裁员传言 → valence 0.2",
            "期待(3): 面试邀请 → valence 0.7",
            "低落(6,7): 被批评+失眠 → valence 0.1",
            "狂喜(8,9): offer+决定 → valence 0.95",
            "解脱(10): 提离职 → valence 0.6",
            "幸福(11,13): 新公司+不失眠 → valence 0.85",
            "平静(14,15,16): 反思+认可 → valence 0.65",
        ],
        keywords=["焦虑", "低落", "狂喜", "解脱", "幸福", "平静"],
        difficulty=5,
    ),
    ComparisonQA(
        id="B5",
        category="multi_hop",
        category_cn="多跳推理",
        question="从小明最初说'喜欢编程觉得创造东西很有成就感'，到后来'每天早上醒来都很期待去上班'，这中间发生了什么改变了他的工作体验？",
        expected_answer="他从大厂跳到了AI创业公司。大厂流程压抑（'大厂的流程让我很压抑'），新公司团队氛围好（'氛围特别好'），有成长和认可（技术分享被夸），做LLM方向是自己感兴趣的领域。",
        relevant_memory_indices=[2, 4, 8, 9, 11, 12, 13],
        reasoning_chain=[
            "原有热情(2): 喜欢编程+成就感",
            "问题(2,4): 大厂流程压抑",
            "机会(3,8): AI创业+LLM方向",
            "转折(9): 决定接受",
            "结果(11-13): 氛围好+成长+幸福",
        ],
        keywords=["大厂", "创业公司", "流程压抑", "氛围", "LLM"],
        difficulty=4,
    ),
]

# ═══════════════════════════════════════════════════════════════
# CATEGORY C: Temporal Reasoning (时间推理) — 5 questions
# Requires understanding event ordering and time relationships.
# ═══════════════════════════════════════════════════════════════

QA_TEMPORAL = [
    ComparisonQA(
        id="C1",
        category="temporal",
        category_cn="时间推理",
        question="小明什么时候开始失眠的？什么时候好的？持续了多久？",
        expected_answer="大约60天前开始（裁员传言时），持续到至少20天前（入职新公司一周后发现不再失眠），约40天时间跨度。期间最严重是48天前失眠到凌晨4点。",
        relevant_memory_indices=[1, 7, 13],
        keywords=["60天", "40天", "凌晨4点", "不再失眠"],
        difficulty=3,
    ),
    ComparisonQA(
        id="C2",
        category="temporal",
        category_cn="时间推理",
        question="小明拿到offer和提离职之间隔了多久？",
        expected_answer="拿到offer在40天前，提离职在38天前，间隔约2天。拿到offer当天就决定接受，2天后向leader提了离职。",
        relevant_memory_indices=[8, 9, 10],
        keywords=["40天", "38天", "2天"],
        difficulty=2,
    ),
    ComparisonQA(
        id="C3",
        category="temporal",
        category_cn="时间推理",
        question="小明妈妈第一次和第二次提到小明的职业选择，中间隔了多长时间？",
        expected_answer="第一次在54天前（劝留大厂），第二次在8天前（认可决定），间隔约46天。",
        relevant_memory_indices=[5, 15],
        keywords=["54天", "8天", "46天"],
        difficulty=2,
    ),
    ComparisonQA(
        id="C4",
        category="temporal",
        category_cn="时间推理",
        question="小明在新公司已经工作多久了？",
        expected_answer="到目前为止约一个月。第一周在30天前，最近一次提到具体工作内容（技术分享）在25天前。",
        relevant_memory_indices=[11, 12],
        keywords=["一个月", "30天", "25天"],
        difficulty=2,
    ),
    ComparisonQA(
        id="C5",
        category="temporal",
        category_cn="时间推理",
        question="请按时间顺序排列：被leader批评、收到面试邀请、拿到offer、妈妈认可、提离职。",
        expected_answer="正确顺序：收到面试邀请(55天前)→被leader批评(48天前)→拿到offer(40天前)→提离职(38天前)→妈妈认可(8天前)。",
        relevant_memory_indices=[3, 6, 8, 10, 15],
        keywords=["面试邀请", "批评", "offer", "提离职", "认可"],
        difficulty=3,
    ),
]

# ═══════════════════════════════════════════════════════════════
# CATEGORY D: Emotional Memory (情感记忆) — 4 questions
# Tests understanding of emotional states and changes.
# ═══════════════════════════════════════════════════════════════

QA_EMOTIONAL = [
    ComparisonQA(
        id="D1",
        category="emotional",
        category_cn="情感记忆",
        question="小明在整个故事中情绪最低点是什么时候？为什么？",
        expected_answer="48天前被leader当众批评+失眠到凌晨4点，valence只有0.1-0.15，arousal高达0.85-0.9。他觉得'很委屈'，因为'明明是因为焦虑才睡不好'却被说效率低。",
        relevant_memory_indices=[6, 7],
        keywords=["被批评", "凌晨4点", "委屈", "valence 0.1"],
        difficulty=2,
    ),
    ComparisonQA(
        id="D2",
        category="emotional",
        category_cn="情感记忆",
        question="小明情绪最高点是什么时候？为什么会有这么强烈的情感？",
        expected_answer="40天前拿到AI创业公司offer，valence=0.95, arousal=0.95。原因是：①薪资高30% ②是自己感兴趣的LLM方向 ③之前经历了长期焦虑压抑所以反差巨大 ④这是他人生第一次'勇敢'的决定。",
        relevant_memory_indices=[8, 9],
        keywords=["offer", "valence 0.95", "30%", "不敢相信"],
        difficulty=2,
    ),
    ComparisonQA(
        id="D3",
        category="emotional",
        category_cn="情感记忆",
        question="从小明的情绪变化来看，什么因素对他的情绪影响最大？",
        expected_answer="工作状态是影响小明情绪的最大因素。大厂压力→焦虑低落(valence 0.1-0.2)；创业公司成长→幸福满足(valence 0.85-0.9)。其次是他人认可（妈妈认可→valence 0.9，同事认可→valence 0.9）。",
        relevant_memory_indices=[1, 2, 6, 7, 8, 11, 12, 13, 15],
        keywords=["工作状态", "认可", "焦虑", "幸福"],
        difficulty=3,
    ),
    ComparisonQA(
        id="D4",
        category="emotional",
        category_cn="情感记忆",
        question="小明在新公司发现'已经一周没有失眠了'时是什么心情？这说明了什么？",
        expected_answer="惊讶且幸福（valence=0.85, arousal=0.4）。'每天早上醒来都很期待去上班'说明工作环境的变化从根本上解决了他的焦虑源，不仅是表面上的睡眠改善，而是整个人的心理状态恢复了健康。",
        relevant_memory_indices=[13],
        keywords=["一周", "失眠", "期待", "上班", "变化"],
        difficulty=3,
    ),
]

# ═══════════════════════════════════════════════════════════════
# CATEGORY E: Causal Reasoning (因果推理) — 3 questions
# Understanding cause-effect relationships between events.
# ═══════════════════════════════════════════════════════════════

QA_CAUSAL = [
    ComparisonQA(
        id="E1",
        category="causal",
        category_cn="因果推理",
        question="小明失眠的根本原因是什么？后来为什么会好转？",
        expected_answer="根本原因：大厂流程压抑导致的意义感缺失+裁员传言导致的生存焦虑（'每天晚上都睡不好'）。好转原因：换个环境到创业公司后，①做感兴趣的方向（LLM）②团队氛围好③获得了成就感和认可——这些消除了焦虑源，恢复了心理健康。",
        relevant_memory_indices=[1, 2, 7, 11, 12, 13],
        reasoning_chain=[
            "根因: 大厂流程压抑(2) + 裁员传言(1) → 焦虑失眠",
            "中间: 被批评(6) → 加重（失眠到凌晨4点）",
            "转折: 新环境(11-13) → 消除焦虑源 → 不再失眠",
        ],
        keywords=["流程压抑", "裁员", "焦虑源", "氛围", "成就感"],
        difficulty=4,
    ),
    ComparisonQA(
        id="E2",
        category="causal",
        category_cn="因果推理",
        question="什么因素最终促使小明决定离开大厂去创业公司？请列举所有因素。",
        expected_answer="多重因素汇聚：①推因素：大厂流程压抑+裁员传言+被leader当众批评的委屈 ②拉因素：AI创业公司offer的吸引力（LLM方向+薪资高30%）③内驱因素：'人生总要勇敢一次'的自我说服+本来就喜欢编程创造。",
        relevant_memory_indices=[1, 2, 4, 6, 7, 8, 9],
        reasoning_chain=[
            "推因素: 压抑(2) + 裁员(1) + 批评(6,7)",
            "拉因素: offer(8) + 兴趣方向",
            "内驱: 勇敢一次(9) + 喜欢创造(2)",
        ],
        keywords=["压抑", "裁员", "批评", "offer", "LLM", "勇敢"],
        difficulty=4,
    ),
    ComparisonQA(
        id="E3",
        category="causal",
        category_cn="因果推理",
        question="如果小明没有收到那个AI创业公司的offer，根据他的记忆，他最可能怎么做？",
        expected_answer="基于他被批评后失眠到凌晨4点时想的'反正公司可能也要裁员，要不要辞职'——他最可能：①先忍受一段时间（因为妈妈不支持裸辞）②在裁员真的发生时被动离开 ③或者积累到极限后裸辞。不太可能主动去寻找新机会，因为他当时处于抑郁状态，缺乏行动力。",
        relevant_memory_indices=[5, 7],
        keywords=["辞职", "裁员", "失眠", "忍受"],
        difficulty=5,
    ),
]

# ═══════════════════════════════════════════════════════════════
# CATEGORY F: Cross-Reference (跨引用) — 3 questions
# Needs to reference across different memory types.
# ═══════════════════════════════════════════════════════════════

QA_CROSS_REF = [
    ComparisonQA(
        id="F1",
        category="cross_ref",
        category_cn="跨引用",
        question="从小明的整个经历来看，职业环境对他的心理健康产生了怎样的影响？请综合他的情绪记录和决策记录来分析。",
        expected_answer="职业环境直接决定了小明的心理状态，两者高度耦合。证据：①大厂期(emotion)：焦虑失眠valence 0.1-0.2 ②转折(decision)：决定接受创业公司offer ③创业公司期(emotion)：不再失眠valence 0.85-0.9。结论：从压抑环境换到成长型环境后，心理状态从亚健康恢复到健康，职业选择是最关键的治疗因素。",
        relevant_memory_indices=[1, 2, 6, 7, 8, 9, 11, 13],
        reasoning_chain=[
            "大厂情绪数据(1,6,7): valence 0.1-0.2",
            "决策数据(4,9): 犹豫→接受",
            "创业公司情绪数据(11,13): valence 0.85-0.9",
            "综合: 职业环境=心理健康关键变量",
        ],
        keywords=["职业环境", "心理健康", "valence", "决定"],
        difficulty=5,
    ),
    ComparisonQA(
        id="F2",
        category="cross_ref",
        category_cn="跨引用",
        question="在小明的故事中，'他人的认可'出现了几次？分别来自谁？哪一次对他的意义最重大？为什么？",
        expected_answer="出现了3次：①妈妈早期的反对（'创业公司不稳定'）——这不是认可而是质疑 ②同事说'讲得很清楚'——正面认可，让小明感到在老公司从未有过的价值感 ③妈妈后来说'决定是对的'——意义最重大，因为这是从最初反对者到支持者的转变，代表了家人对他人生选择的最终认可。小明说'这句话对我来说意义太重大了'。",
        relevant_memory_indices=[5, 12, 15],
        reasoning_chain=[
            "质疑(5): 妈妈反对 → 压力源",
            "同事认可(12): 讲得清楚 → 价值感",
            "妈妈认可(15): 决定是对的 → 意义最重大",
        ],
        keywords=["3次", "妈妈", "同事", "讲得清楚", "意义太重大"],
        difficulty=4,
    ),
    ComparisonQA(
        id="F3",
        category="cross_ref",
        category_cn="跨引用",
        question="从小明的聊天记录、决策记录和情绪记录中，能否看出他对'冒险'的态度有什么变化？",
        expected_answer="可以看出一条清晰的演变轨迹。①初期(chat): 大厂稳定但压抑，创业公司'冒险但可能成长更快'——态度是'犹豫' ②决策期(decision): '之前怕冒险，现在怕错过机会。人生总要勇敢一次'——态度转向拥抱冒险 ③后期(chat): 偶尔想'如果留在大厂会怎样'但'念头很快就过去了'——对冒险结果满意，不后悔。",
        relevant_memory_indices=[2, 4, 9, 16],
        reasoning_chain=[
            "初期(2,4): 压抑→犹豫是否冒险",
            "决策(9): 从怕冒险到怕错过",
            "后期(16): 不后悔",
        ],
        keywords=["冒险", "犹豫", "勇敢", "不后悔"],
        difficulty=4,
    ),
]

# ═══════════════════════════════════════════════════════════════
# Master QA list — all 25 questions
# ═══════════════════════════════════════════════════════════════

ALL_QA_PAIRS: list[ComparisonQA] = (
    QA_SIMPLE_RECALL + QA_MULTI_HOP + QA_TEMPORAL +
    QA_EMOTIONAL + QA_CAUSAL + QA_CROSS_REF
)

CATEGORY_MAP = {
    "simple_recall": "A. 简单回忆",
    "multi_hop": "B. 多跳推理",
    "temporal": "C. 时间推理",
    "emotional": "D. 情感记忆",
    "causal": "E. 因果推理",
    "cross_ref": "F. 跨引用",
}

CATEGORY_QA_COUNTS = {
    "simple_recall": 5,
    "multi_hop": 5,
    "temporal": 5,
    "emotional": 4,
    "causal": 3,
    "cross_ref": 3,
}

# Verification
assert len(ALL_QA_PAIRS) == 25, f"Expected 25 QA pairs, got {len(ALL_QA_PAIRS)}"
for cat, count in CATEGORY_QA_COUNTS.items():
    actual = sum(1 for q in ALL_QA_PAIRS if q.category == cat)
    assert actual == count, f"Category {cat}: expected {count}, got {actual}"
