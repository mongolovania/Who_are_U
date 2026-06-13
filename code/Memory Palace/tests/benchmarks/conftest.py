# ============================================================
# Benchmark shared fixtures — common test harness for all comparisons
# 对比测试共享 fixtures
# ============================================================

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Check which community libraries are available
_MEM0_AVAILABLE = False
_LETTA_AVAILABLE = False
_GRAPHITI_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_FAISS_AVAILABLE = False
_BM25S_AVAILABLE = False

try:
    import mem0  # noqa: F401
    _MEM0_AVAILABLE = True
except ImportError:
    pass

try:
    import letta  # noqa: F401
    _LETTA_AVAILABLE = True
except ImportError:
    pass

try:
    import graphiti_core  # noqa: F401
    _GRAPHITI_AVAILABLE = True
except ImportError:
    pass

try:
    import sentence_transformers  # noqa: F401
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import faiss  # noqa: F401
    _FAISS_AVAILABLE = True
except ImportError:
    pass

try:
    import bm25s  # noqa: F401
    _BM25S_AVAILABLE = True
except ImportError:
    pass


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "mem0: tests requiring mem0ai library")
    config.addinivalue_line("markers", "letta: tests requiring letta library")
    config.addinivalue_line("markers", "graphiti: tests requiring graphiti-core library")
    config.addinivalue_line("markers", "sentence_transformers: tests requiring sentence-transformers")
    config.addinivalue_line("markers", "faiss: tests requiring faiss-cpu")
    config.addinivalue_line("markers", "bm25s: tests requiring bm25s")
    config.addinivalue_line("markers", "benchmark: benchmark comparison tests")


@pytest.fixture
def mem0_available():
    """Whether mem0ai is installed."""
    return _MEM0_AVAILABLE


@pytest.fixture
def letta_available():
    """Whether letta is installed."""
    return _LETTA_AVAILABLE


@pytest.fixture
def graphiti_available():
    """Whether graphiti-core is installed."""
    return _GRAPHITI_AVAILABLE


@pytest.fixture
def sentence_transformers_available():
    """Whether sentence-transformers is installed."""
    return _SENTENCE_TRANSFORMERS_AVAILABLE


@pytest.fixture
def faiss_available():
    """Whether faiss-cpu is installed."""
    return _FAISS_AVAILABLE


@pytest.fixture
def bm25s_available():
    """Whether bm25s is installed."""
    return _BM25S_AVAILABLE


@pytest.fixture
def benchmark_messages():
    """
    Standard conversation dataset for cross-system comparison.
    Simulates 5 sessions of a user talking about their life.
    All systems process the same input for fair comparison.
    """
    return [
        # Session 1: 初次使用，自我介绍
        {
            "session_id": "s1",
            "messages": [
                {"role": "user", "content": "你好，我今天第一天用这个App。我叫小明，今年25岁，在北京做程序员。"},
                {"role": "assistant", "content": "小明你好！欢迎来到我之山。25岁的程序员在北京，生活节奏应该很快吧？"},
                {"role": "user", "content": "是啊，每天加班到很晚。不过我很喜欢编程，觉得写代码很有成就感。"},
                {"role": "assistant", "content": "能找到自己热爱的工作是很幸运的事。不过也要注意身体。"},
                {"role": "user", "content": "嗯，其实最近有点焦虑，公司要裁员了，不知道我会不会被影响。"},
            ],
        },
        # Session 2: 三天后，继续倾诉
        {
            "session_id": "s2",
            "messages": [
                {"role": "user", "content": "还在加班，但我今天收到了一家公司的面试邀请。"},
                {"role": "assistant", "content": "这听起来是个转机！是哪家公司？你感觉怎么样？"},
                {"role": "user", "content": "是一家AI创业公司，做LLM的。我很想去，但又担心从大厂去创业公司太冒险。"},
                {"role": "assistant", "content": "确实是个需要权衡的决定。你之前聊过，你很喜欢编程——创业公司可能给你更多发挥空间。"},
                {"role": "user", "content": "对，而且他们做的方向正好是我感兴趣的。但我妈觉得创业公司不稳定。"},
            ],
        },
        # Session 3: 一周后，面试经历
        {
            "session_id": "s3",
            "messages": [
                {"role": "user", "content": "我今天面试了！面试官问了我很多关于分布式系统的问题，感觉答得还行。"},
                {"role": "assistant", "content": "听起来不错！你之前为面试做的准备有用上吗？"},
                {"role": "user", "content": "有，我这周刷了几十道题。但我现在很紧张，不知道结果会怎样。"},
                {"role": "assistant", "content": "紧张是正常的，说明你在乎。不管结果如何，至少你认真准备了。"},
                {"role": "user", "content": "你说的对...其实不管拿不拿到offer，我都学到了很多。"},
            ],
        },
        # Session 4: 两周后，拿到offer
        {
            "session_id": "s4",
            "messages": [
                {"role": "user", "content": "我拿到了！！他们给了offer！薪资比现在高30%！"},
                {"role": "assistant", "content": "太棒了小明！这是你努力的结果！你打算接受吗？"},
                {"role": "user", "content": "我想接受，但还在犹豫。毕竟现在的工作虽然累但稳定..."},
                {"role": "assistant", "content": "还记得你第一次提到这个offer时的纠结吗？现在你的担心还是同样的吗？"},
                {"role": "user", "content": "其实不一样了。之前是怕冒险，现在是怕错过机会。我觉得我已经有答案了。"},
            ],
        },
        # Session 5: 一个月后，新生活
        {
            "session_id": "s5",
            "messages": [
                {"role": "user", "content": "我在新公司已经一个月了。虽然很累但真的很开心！团队氛围特别好。"},
                {"role": "assistant", "content": "听起来你的决定是对的。和之前在大厂比，最大的不同是什么？"},
                {"role": "user", "content": "自由度高了很多，而且每个人都很热情。我觉得这才是我想做的工作。"},
                {"role": "assistant", "content": "从大厂到创业公司，从焦虑到充实——你其实完成了一次很大的转变。"},
                {"role": "user", "content": "是啊，回想一个月前，我还在担心裁员。现在完全不一样了。谢谢你陪我走过这段路。"},
            ],
        },
    ]


@pytest.fixture
def benchmark_queries():
    """
    Standard query set for retrieval evaluation.
    Each query has a known answer from the benchmark_messages.
    """
    return [
        {
            "query": "小明在哪里工作？",
            "expected_answer": "北京，AI创业公司做LLM",
            "requires_multihop": True,
            "requires_temporal": True,
        },
        {
            "query": "小明之前担心什么？",
            "expected_answer": "裁员",
            "requires_multihop": False,
            "requires_temporal": False,
        },
        {
            "query": "小明喜欢编程吗？",
            "expected_answer": "喜欢，觉得写代码有成就感",
            "requires_multihop": False,
            "requires_temporal": False,
        },
        {
            "query": "小明为什么决定接受新offer？",
            "expected_answer": "因为觉得从怕冒险变成了怕错过机会，意识到自己想改变",
            "requires_multihop": True,
            "requires_temporal": True,
        },
        {
            "query": "小明在新公司感觉如何？",
            "expected_answer": "很开心，自由度高，团队氛围好",
            "requires_multihop": False,
            "requires_temporal": True,
        },
        {
            "query": "小明妈妈对他的决定有什么看法？",
            "expected_answer": "觉得创业公司不稳定",
            "requires_multihop": True,
            "requires_temporal": True,
        },
    ]


@pytest.fixture
def edge_cases():
    """
    Edge case inputs for testing boundary behavior.
    """
    return {
        "empty_message": "",
        "single_char": "嗯",
        "very_long": "我" * 2000,
        "pure_emoji": "😊😢😡😨🤔",
        "mixed_language": "今天写了个 function 来处理 async/await 的 error handling",
        "rapid_repetition": ["重复", "重复", "重复", "重复", "重复"],
        "extreme_valence": {
            "very_positive": "今天是我人生中最开心的一天！所有事情都完美了！",
            "very_negative": "一切都没有意义了，我再也看不到希望。",
        },
    }
