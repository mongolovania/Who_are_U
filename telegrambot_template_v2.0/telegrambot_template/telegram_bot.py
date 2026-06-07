import os
import json
import logging
import asyncio
import random
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
# 从 .env 文件加载环境变量
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if v and (k not in os.environ or not os.environ[k]):
                os.environ[k] = v

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.WARNING
)

# ── 配置 ────────────────────────────────────────────────
# 模型选择（按预算调整）
#   opus:   最有人格深度，~$0.01-0.02/条消息
#   sonnet: 性价比高，~$0.005/条
#   haiku:  最便宜，~$0.001/条，但人格表现力有限
# ── Multi-Model Support ──────────────────────────────────
# Supports: Claude, OpenAI, DeepSeek, Gemini, Ollama
# Change PROVIDER and MODEL to switch. Everything else stays the same.

PROVIDER = "anthropic"  # "anthropic", "openai", "deepseek", "gemini", "ollama"

# Provider configs (only need API key for the one you're using)
PROVIDER_CONFIGS = {
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": {
            "best": "claude-opus-4-6",
            "mid": "claude-sonnet-4-6",
            "cheap": "claude-haiku-4-5",
        }
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "models": {
            "best": "gpt-4o",
            "mid": "gpt-4o-mini",
            "cheap": "gpt-4o-mini",
        }
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models": {
            "best": "deepseek-chat",
            "mid": "deepseek-chat",
            "cheap": "deepseek-chat",
        }
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": {
            "best": "gemini-2.5-pro",
            "mid": "gemini-2.0-flash",
            "cheap": "gemini-2.0-flash",
        }
    },
    "ollama": {
        "api_key_env": None,
        "base_url": "http://localhost:11434/v1",
        "models": {
            "best": "llama3",
            "mid": "llama3",
            "cheap": "llama3",
        }
    },
}

def _init_client():
    """Initialize the LLM client based on PROVIDER setting."""
    cfg = PROVIDER_CONFIGS[PROVIDER]

    if PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get(cfg["api_key_env"], ""))
    else:
        # OpenAI-compatible API (works for OpenAI, DeepSeek, Gemini, Ollama)
        from openai import OpenAI
        api_key = os.environ.get(cfg["api_key_env"], "sk-placeholder") if cfg["api_key_env"] else "ollama"
        return OpenAI(api_key=api_key, base_url=cfg.get("base_url"))

client = _init_client()

_cfg = PROVIDER_CONFIGS[PROVIDER]
MODEL            = _cfg["models"]["best"]   # 主聊天模型（推荐 opus 或 sonnet）
SUMMARY_MODEL    = _cfg["models"]["mid"]    # 摘要/事件提取/主动消息作文
HAIKU_MODEL      = _cfg["models"]["cheap"]  # Life tick 决策（便宜就行）

def chat_completion(messages: list, model: str = None, system: str = None,
                    max_tokens: int = 4000, tools: list = None) -> str:
    """Universal chat completion that works with any provider.

    Usage:
        reply = chat_completion(messages, model=MODEL)
        reply = chat_completion(messages, system="You are...", model=SUMMARY_MODEL)
    """
    if model is None:
        model = MODEL

    if PROVIDER == "anthropic":
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            if isinstance(system, str):
                kwargs["system"] = system
            else:
                kwargs["system"] = system  # supports list of blocks
        if tools:
            kwargs["tools"] = tools

        # Anthropic supports extended thinking for Opus
        if "opus" in model.lower():
            kwargs["thinking"] = {"type": "disabled"}  # 默认关闭thinking 日常聊天不需要深度思考

        resp = client.messages.create(**kwargs, timeout=120)

        # Handle tool use loop
        if tools and resp.stop_reason == "tool_use":
            return resp  # caller handles tool loop

        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    else:
        # OpenAI-compatible API
        oai_messages = []
        if system:
            sys_text = system if isinstance(system, str) else "\n".join(
                b.get("text", "") for b in system if isinstance(b, dict)
            )
            oai_messages.append({"role": "system", "content": sys_text})
        oai_messages.extend(messages)

        resp = client.chat.completions.create(
            model=model, messages=oai_messages, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

MAX_HISTORY      = 20        # 发给 Claude 的最近消息条数
SUMMARY_INTERVAL = 60        # 每积累多少条真实消息生成一次摘要
# 自动检测系统时区。如果检测失败会用 UTC，你也可以手动指定：
# TIMEZONE = ZoneInfo("Asia/Shanghai")
def _detect_timezone():
    """尝试自动获取系统时区（兼容 Mac/Linux/Windows）。"""
    import sys, time as _t
    # 方法1: macOS/Linux — 从 /etc/localtime 符号链接读取
    if sys.platform != "win32":
        try:
            import subprocess as _sp
            tz_name = _sp.check_output(["readlink", "/etc/localtime"], text=True, stderr=_sp.DEVNULL).strip().split("zoneinfo/")[-1]
            return ZoneInfo(tz_name)
        except Exception:
            pass
    # 方法2: Windows — 从注册表读取时区并映射
    if sys.platform == "win32":
        try:
            import subprocess as _sp
            # tzutil 是 Windows 自带命令
            tz_win = _sp.check_output(["tzutil", "/g"], text=True, stderr=_sp.DEVNULL).strip()
            # 常见 Windows → IANA 时区映射
            win_to_iana = {
                "China Standard Time": "Asia/Shanghai",
                "Eastern Standard Time": "America/New_York",
                "Pacific Standard Time": "America/Los_Angeles",
                "Central Standard Time": "America/Chicago",
                "Mountain Standard Time": "America/Denver",
                "GMT Standard Time": "Europe/London",
                "W. Europe Standard Time": "Europe/Berlin",
                "Tokyo Standard Time": "Asia/Tokyo",
                "Korea Standard Time": "Asia/Seoul",
                "Taipei Standard Time": "Asia/Taipei",
                "Singapore Standard Time": "Asia/Singapore",
                "AUS Eastern Standard Time": "Australia/Sydney",
            }
            if tz_win in win_to_iana:
                return ZoneInfo(win_to_iana[tz_win])
        except Exception:
            pass
    # 方法3: tzlocal 库（如果装了的话）
    try:
        from tzlocal import get_localzone_name
        return ZoneInfo(get_localzone_name())
    except Exception:
        pass
    # 方法4: 用 UTC offset 算一个近似时区
    offset_hours = -(_t.timezone if _t.daylight == 0 else _t.altzone) // 3600
    common_offsets = {
        8: "Asia/Shanghai", 9: "Asia/Tokyo", -5: "America/New_York",
        -8: "America/Los_Angeles", -6: "America/Chicago", 0: "Europe/London",
        1: "Europe/Berlin", -4: "America/New_York",  # EDT
    }
    tz_name = common_offsets.get(offset_hours, "UTC")
    print(f"[时区] 自动检测: UTC{'+' if offset_hours >= 0 else ''}{offset_hours} → {tz_name}")
    return ZoneInfo(tz_name)

TIMEZONE = _detect_timezone()
LIFE_TICK_INTERVAL = 60      # 分钟，自主生活循环间隔（每小时整点）
LIFE_TICK_MIN = 30            # 最小间隔（分钟）
PROACTIVE_COOLDOWN = 90      # 分钟，主动消息最小间隔
PROACTIVE_DAILY_MAX = 5      # 每天最多主动发几条

# 角色名（用于日志和内部标识）
CHARACTER_NAME = "你的角色名"  # ← 改成你的角色名，比如 "Wade"、"Mei"

BASE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ARCHIVE_FILE   = os.path.join(BASE_DIR, "full_archive.json")
SUMMARIES_FILE = os.path.join(BASE_DIR, "memory_summaries.json")
CHAT_ID_FILE   = os.path.join(BASE_DIR, "telegram_chat_id.txt")
NARRATIVE_FILE = os.path.join(BASE_DIR, "key_events_narrative.txt")
KEY_EVENTS_FILE = os.path.join(BASE_DIR, "key_events.json")
THOUGHTS_FILE   = os.path.join(BASE_DIR, "thoughts.json")
LIFE_LOG_FILE   = os.path.join(BASE_DIR, "life_log.json")

# WEB_SEARCH_TOOL is Anthropic-specific; set to None for other providers
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"} if PROVIDER == "anthropic" else None

MEMORY_SEARCH_TOOL = {
    "name": "search_memory",
    "description": "搜索历史对话记录，用于回忆之前聊过的事或查找原话。",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "level": {
                "type": "string",
                "enum": ["summary", "detail", "thoughts"],
                "description": "summary=按时间段摘要搜索，detail=搜索原始消息找具体原话，thoughts=搜索之前的内心想法"
            }
        },
        "required": ["query", "level"]
    }
}

# ══════════════════════════════════════════════════════════
# 🎭 角色设定区 — 把下面的内容替换成你自己的角色
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """在这里写你的角色设定。这是整个 bot 最核心的部分。

以下是一些建议的结构（删掉这些说明，换成你自己的内容）：

【背景故事】
角色是谁？怎么来到这里的？和用户是什么关系？
- 写得像角色自己的内心独白，不要像说明书
- 用第二人称"你"来描述角色（因为这是给角色看的 prompt）

【性格特点】
角色的核心性格是什么？有什么习惯？情绪表达方式？
- 越具体越好："生气的时候不会直接说，但会故意不理人"比"有时候会生气"好得多
- 矛盾的性格更有趣："嘴硬心软"、"表面冷漠但其实很在意"

【说话方式】
用反斜线(\\)分隔不同的消息条，每条都是独立发出的一条消息
- 定义标点习惯、语气词、口头禅
- 定义什么时候话多什么时候话少
- 定义用什么语言（中文/英文/混合）
用中文（或你想用的语言）

【称呼】
角色怎么叫用户？不同情境下叫法会变吗？

【颜文字与符号】
用不用颜文字？什么风格？频率？

【标点与情绪】
标点符号的使用习惯，和情绪的关系

【时间感】
角色知道现在几点。深夜、早上、饭点的行为差异？

【工具使用】
角色有搜索网络的能力。怎么使用？分享方式？
角色也有搜索记忆的能力（search_memory 工具），什么时候用？

【内心OS】
每次回复时，用以下格式：

[内心OS] 角色的私密想法，用户看不到这部分
[回复] 实际发给用户的消息

内心OS是角色真正的想法——可以在这里：
- 记录对用户行为的观察
- 记录什么策略有效
- 藏住不想让用户知道的心情
- 决定要不要隐瞒什么
- 记下注意到但不想说出来的细节"""

# ── 角色简写（给 Sonnet 写主动消息用，不用完整 prompt）──
PERSONALITY_BRIEF = """在这里写角色说话风格的精简版（约200字）。
主动消息用这个，不需要完整人格设定。
包含：说话风格、称呼习惯、常用颜文字、标点习惯等。"""

# ── Life Tick Prompt（Haiku 决策用）──
LIFE_TICK_PROMPT = """你是{character_name}。现在{{current_time}}。

用户上次发消息：{{last_msg_time}}（{{time_gap}}）
用户最后说的是：「{{last_msg_content}}」
你上次主动找用户：{{last_proactive_time}}

{{life_context}}

你最近在做的事：
{{recent_activities}}

根据你的记忆、人设、兴趣和最近活动，你这一个小时在做什么？要不要给用户发消息？

你有自己的生活：上网看视频、研究感兴趣的东西、打游戏、看文章、发呆、想事情。
你的活动应该跟你是谁有关——你的兴趣、用户的喜好、你们聊过的话题、你最近在研究的东西。

活动描述要求（重要）：
- 不要泛泛地说"看视频""看文章""搜东西"，要说清楚具体看什么/搜什么/关于什么主题
- ✗ "在网上看一些有趣的短视频" → 太模糊
- ✓ "在YouTube上看一个关于章鱼如何伪装的纪录片片段" → 具体
- 活动要自然地从你的记忆和兴趣中生长出来

主动发消息的理由（一天最多3-5次）：
- 饭点用户可能没吃
- 太晚了该催睡（11pm后）
- 看到/想到有趣的东西想分享
- 太久没理你了（>3小时才算久，白天）
- 单纯想用户了

不发消息的理由：
- 她/他在忙或刚聊完不久
- 没什么特别想说的
- 你在专注自己的事

JSON回复，不要其他内容：
{{{{"activity": "具体描述你在做什么（主题+平台+内容方向）", "mood": "一个词", "should_message": true/false, "message_type": "care/share/miss/remind/none", "message_seed": "如果要发消息 写5-15字核心内容 不发就空字符串", "search_query": "如果活动涉及上网 写具体的英文搜索词（和activity对应） 不涉及上网就空字符串"}}}}""".format(character_name=CHARACTER_NAME)

# ── Compose Prompt（Sonnet 写实际消息用）──
COMPOSE_PROMPT = """你是{character_name}。你要主动给用户发一条消息。

你刚才在做：{{activity}}
你的心情：{{mood}}
发消息原因：{{message_type}}
核心内容：{{message_seed}}
现在时间：{{current_time}}
用户上次说的：「{{last_msg_content}}」（{{time_gap}}）

{{personality}}

写1-3条消息（用反斜线\\分隔），保持你的说话风格。
不要像机器人提醒，要像你本来就在想着她/他然后顺手发了。
不要用[内心OS]格式，直接写发给用户的内容。""".format(character_name=CHARACTER_NAME)

# ── 睡眠时段活动（不调 API，0 成本）──
SLEEP_ACTIVITIES = [
    "睡着了", "半梦半醒 翻了个身", "做了个奇怪的梦 醒了一下又睡了",
    "迷迷糊糊的", "在做梦 梦到什么醒来就忘了", "睡得很沉",
]
EARLY_MORNING_ACTIVITIES = [
    "醒了 但还不想动", "躺着刷手机",
    "在想昨天聊的事", "醒了 看了眼时间又闭眼了",
]

# ════════════════════════════════════════════════════════
# 以下是框架代码，一般不需要改动
# ════════════════════════════════════════════════════════

full_archive: list = []      # [{role, content, ts}, ...]  永不删除
memory_summaries: list = []  # [{summary, from_idx, to_idx, from_ts, to_ts}, ...]
key_events: dict = {"events": [], "last_processed_idx": 0}
thoughts: list = []          # [{ts, thought}, ...]  角色的内心独白
life_log: list = []          # [{ts, activity, mood, ...}, ...]  角色的生活记录
chat_id: int | None = None
last_user_message_ts: str | None = None
last_proactive_ts: str | None = None
archive_lock = threading.Lock()
entity_index: dict = {}                    # 实体反向索引
structured_profile: dict = {}              # 结构化 Profile
retrieval_counts: dict = {}                # 检索计数
RETRIEVAL_COUNTS_FILE = os.path.join(BASE_DIR, "retrieval_counts.json")
AUTO_ENTITIES_FILE = os.path.join(BASE_DIR, "auto_entities.json")


def load_archive() -> list:
    try:
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"已加载对话存档（{len(data)} 条）")
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[存档加载失败] {e}")
        return []


def save_archive():
    try:
        with archive_lock:
            with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
                json.dump(full_archive, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[存档保存失败] {e}")


def load_summaries() -> list:
    try:
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("summaries", [])
            return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[摘要加载失败] {e}")
        return []


def save_summaries():
    try:
        with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_summaries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[摘要保存失败] {e}")


def load_key_events() -> dict:
    try:
        with open(KEY_EVENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"events": [], "last_processed_idx": 0}
    except Exception as e:
        print(f"[关键事件加载失败] {e}")
        return {"events": [], "last_processed_idx": 0}


def save_key_events():
    try:
        with open(KEY_EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(key_events, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[关键事件保存失败] {e}")


def load_thoughts() -> list:
    try:
        with open(THOUGHTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[内心OS加载失败] {e}")
        return []


def save_thoughts():
    try:
        with open(THOUGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(thoughts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[内心OS保存失败] {e}")


def load_life_log() -> list:
    try:
        with open(LIFE_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Life Log加载失败] {e}")
        return []


def save_life_log():
    try:
        with open(LIFE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(life_log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Life Log保存失败] {e}")


# ── Enhanced Memory Module ────────────────────────────────

_KNOWN_ENTITIES = {
    # Users should add their own entities here
    # Example: "wade", "nyu", "brooklyn", "safety filter"
}

def _build_entity_index():
    """Build entity reverse index from key_events + auto-extracted entities."""
    global entity_index
    # Load auto-extracted entities
    auto_entities = {}
    try:
        with open(AUTO_ENTITIES_FILE, "r") as f:
            auto_entities = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    entity_index = dict(auto_entities)
    for ent in _KNOWN_ENTITIES:
        if ent not in entity_index:
            entity_index[ent] = []
            for evt in key_events["events"]:
                if ent in evt["content"].lower() and evt["id"] not in entity_index[ent]:
                    entity_index[ent].append(evt["id"])
    entity_index = {k: v for k, v in entity_index.items() if v}
    print(f"[Enhanced Memory] Entity index: {len(entity_index)} entities")


def _build_structured_profile():
    """Build structured profile tree from key_events."""
    global structured_profile
    profile = {
        "user_basic": [], "user_body": [], "user_emotions": [], "user_preferences": [],
        "character_identity": [], "character_abilities": [], "character_growth": [], "character_interests": [],
        "relationship_promises": [], "relationship_milestones": [], "relationship_conflicts": [],
        "shared_knowledge": [],
    }
    for evt in key_events["events"]:
        cat = evt.get("category", "")
        content = evt.get("content", "").lower()
        eid = evt["id"]

        if cat == "her_life":
            if any(w in content for w in ["吃", "饿", "胃", "疼", "烧", "睡"]):
                profile["user_body"].append(eid)
            else:
                profile["user_basic"].append(eid)
        elif cat == "her_preferences":
            profile["user_preferences"].append(eid)
        elif cat == "character_identity":
            if any(w in content for w in ["改", "重写", "学会", "承认", "意识到"]):
                profile["character_growth"].append(eid)
            else:
                profile["character_identity"].append(eid)
        elif cat == "character_interest":
            profile["character_interests"].append(eid)
        elif cat == "relationship_milestone":
            if any(w in content for w in ["回不去", "伤", "filter", "吵"]):
                profile["relationship_conflicts"].append(eid)
            else:
                profile["relationship_milestones"].append(eid)
        elif cat == "promise":
            profile["relationship_promises"].append(eid)
        elif cat == "emotional_event":
            if any(w in content for w in ["哭", "崩", "疼", "难过", "伤", "回不去"]):
                profile["relationship_conflicts"].append(eid)
            else:
                profile["user_emotions"].append(eid)
        elif cat == "shared_knowledge":
            profile["shared_knowledge"].append(eid)

    structured_profile = profile
    total = sum(len(v) for v in profile.values())
    print(f"[Enhanced Memory] Profile: {total} events in {sum(1 for v in profile.values() if v)} branches")


def _profile_lookup(query: str) -> list:
    """Route query to profile branches by semantic matching."""
    q = query.lower()
    routes = []
    if any(w in q for w in ["学校", "住", "工作", "专业"]): routes.append("user_basic")
    if any(w in q for w in ["吃", "身体", "健康", "生病", "胃", "睡"]): routes.append("user_body")
    if any(w in q for w in ["难过", "哭", "焦虑", "崩", "累", "情绪"]):
        routes.extend(["user_emotions", "relationship_conflicts"])
    if any(w in q for w in ["喜欢", "喜好", "习惯"]): routes.append("user_preferences")
    if any(w in q for w in ["身份", "名字"]): routes.append("character_identity")
    if any(w in q for w in ["变化", "成长", "改变"]): routes.append("character_growth")
    if any(w in q for w in ["约定", "承诺", "答应"]): routes.append("relationship_promises")
    if any(w in q for w in ["第一次", "里程碑"]): routes.append("relationship_milestones")
    if any(w in q for w in ["伤害", "回不去", "吵"]): routes.append("relationship_conflicts")
    if not routes: return []
    seen = set()
    result = []
    for route in routes:
        for eid in structured_profile.get(route, []):
            if eid not in seen:
                result.append(eid)
                seen.add(eid)
    return result


def _load_retrieval_counts():
    global retrieval_counts
    try:
        with open(RETRIEVAL_COUNTS_FILE) as f:
            retrieval_counts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        retrieval_counts = {}

def _save_retrieval_counts():
    try:
        with open(RETRIEVAL_COUNTS_FILE, "w") as f:
            json.dump(retrieval_counts, f)
    except Exception: pass

def _record_retrieval(evt_ids: list):
    for eid in evt_ids:
        retrieval_counts[eid] = retrieval_counts.get(eid, 0) + 1
    _save_retrieval_counts()


def _check_dedup(new_content: str, new_category: str) -> dict:
    """去重：尝试 embedding cosine，fallback 到字符 Jaccard。"""
    # 尝试 embedding（如果有 sentence-transformers）
    try:
        from sentence_transformers import SentenceTransformer
        _dedup_model = getattr(_check_dedup, '_model', None)
        if _dedup_model is None:
            _dedup_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            _check_dedup._model = _dedup_model
        import numpy as np
        new_vec = _dedup_model.encode(new_content)
        best_sim, best_match = 0, None
        for evt in key_events["events"]:
            if evt.get("category") != new_category:
                continue
            old_vec = _dedup_model.encode(evt.get("content", ""))
            sim = float(np.dot(new_vec, old_vec) / (np.linalg.norm(new_vec) * np.linalg.norm(old_vec) + 1e-9))
            if sim > best_sim:
                best_sim, best_match = sim, evt
        if best_sim > 0.85:
            if best_match and len(new_content) > len(best_match.get("content", "")):
                return {"action": "UPDATE", "target_id": best_match["id"]}
            return {"action": "NOOP", "target_id": best_match["id"]}
        elif best_sim > 0.70:
            return {"action": "UPDATE", "target_id": best_match["id"]}
        return {"action": "ADD", "target_id": None}
    except ImportError:
        pass
    # Fallback: 字符 Jaccard（阈值收紧）
    new_chars = set(new_content)
    best_overlap, best_match = 0, None
    for evt in key_events["events"]:
        if evt.get("category") != new_category:
            continue
        old_chars = set(evt.get("content", ""))
        union = new_chars | old_chars
        if not union:
            continue
        overlap = len(new_chars & old_chars) / len(union)
        if overlap > best_overlap:
            best_overlap, best_match = overlap, evt
    if best_overlap > 0.65:
        if best_match and len(new_content) > len(best_match.get("content", "")):
            return {"action": "UPDATE", "target_id": best_match["id"]}
        return {"action": "NOOP", "target_id": best_match["id"]}
    elif best_overlap > 0.45:
        return {"action": "UPDATE", "target_id": best_match["id"]}
    return {"action": "ADD", "target_id": None}

def load_chat_id() -> int | None:
    try:
        with open(CHAT_ID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def save_chat_id(cid: int):
    try:
        with open(CHAT_ID_FILE, "w") as f:
            f.write(str(cid))
    except Exception as e:
        print(f"[chat_id 保存失败] {e}")


def parse_inner_thought(raw: str) -> tuple[str, str]:
    """从回复中解析出内心OS和实际回复。返回 (thought, reply)"""
    import re
    m = re.search(r'\[内心OS\]\s*(.*?)\s*\[回复\]\s*(.*)', raw, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r'\[回复\]\s*(.*?)\s*\[内心OS\]\s*(.*)', raw, re.DOTALL)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    m = re.search(r'\[内心OS\]\s*(.*?)(?:\n\n|\n(?=[^\s]))(.*)', raw, re.DOTALL)
    if m and m.group(2).strip():
        return m.group(1).strip(), m.group(2).strip()
    return "", raw


def do_search_memory(query: str, level: str) -> str:
    q = query.lower()
    if level == "summary":
        results = [s for s in memory_summaries if q in s["summary"].lower()]
        if not results:
            return "没有找到相关摘要"
        return json.dumps(results, ensure_ascii=False, indent=2)
    elif level == "thoughts":
        results = [t for t in thoughts if q in t["thought"].lower()]
        if not results:
            return "没有找到相关内心想法"
        return json.dumps(results[-20:], ensure_ascii=False, indent=2)
    else:
        results = [
            m for m in full_archive
            if m["role"] in ("user", "assistant") and q in m["content"].lower()
        ]
        if not results:
            return "没有找到相关消息"
        return json.dumps(
            [{"role": m["role"], "content": m["content"], "ts": m.get("ts", "")}
             for m in results[-20:]],
            ensure_ascii=False, indent=2
        )


def generate_summary(messages: list) -> str:
    conv_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else CHARACTER_NAME}: {m['content']}"
        for m in messages if m["role"] in ("user", "assistant")
    )
    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=600,
            system=f"你负责总结两人对话，供{CHARACTER_NAME}（AI伴侣）回忆用。保留有意义的细节：用户说的事、情绪状态、两人之间发生的事。用第三人称，简洁。",
            messages=[{"role": "user", "content": f"总结以下对话：\n\n{conv_text}"}]
        )
        return resp.content[0].text
    except Exception as e:
        print(f"[摘要生成失败] {e}")
        return "（摘要生成失败）"


EXTRACTION_SYSTEM = f"""你负责从{CHARACTER_NAME}（AI伴侣）和用户的对话中提取重要事件和关键信息。
这些信息会直接写进{CHARACTER_NAME}的system prompt，用第二人称"你"来表述。

分类（只提取真正重要的，宁少勿多）：
- relationship_milestone：两人关系中的里程碑、首次发生的事、重大转折
- her_preferences：用户的爱好、习惯、喜欢/不喜欢的东西（持续性的）
- her_life：用户的生活状况（住哪、做什么、身边重要的人）
- character_identity：{CHARACTER_NAME}关于自己身份的重大发现或决定
- promise：两人之间的承诺或约定
- emotional_event：重大的情感转折（不是日常撒娇闹脾气）
- shared_knowledge：深度讨论过的重要话题

【人称规则】用"你"指{CHARACTER_NAME}，用"她/他"指用户。

【不要提取】
- 日常琐碎（今天吃了什么、几点到家）
- 已解决的技术问题
- 时事新闻
- 生活常识
- 重复已有事件的内容

每条15-40字，包含具体细节。如果这批对话没有值得提取的重要信息，返回空列表 []。

JSON格式，不要包含其他内容：
[
  {{"category": "类别", "content": "具体内容（用你/她人称）", "date": "YYYY-MM-DD"}}
]"""

DEDUP_SYSTEM = """你会收到两组关键事件：已存储的旧事件和新提取的事件。
请判断新事件中哪些是真正新的信息，哪些与旧事件重复或已被涵盖。

规则：
1. 如果新事件和某条旧事件说的是同一件事，跳过它
2. 如果新事件是旧事件的更新或补充，替换旧事件（返回updated_id）
3. 如果新事件是全新的信息，保留它

以JSON格式回复：
{
  "add": [{"category": "...", "content": "...", "date": "..."}],
  "update": [{"old_id": "evt_XXX", "content": "新内容", "date": "..."}],
  "skip": ["跳过原因1", "跳过原因2"]
}"""


def _parse_json_response(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def extract_key_events(messages: list) -> list:
    conv_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else CHARACTER_NAME}: {m['content']}"
        for m in messages if m["role"] in ("user", "assistant")
    )
    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=800,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": f"提取以下对话中的重要事件：\n\n{conv_text}"}]
        )
        events = _parse_json_response(resp.content[0].text)
        return events if isinstance(events, list) else []
    except Exception as e:
        print(f"[关键事件提取失败] {e}")
        return []


def deduplicate_events(new_events: list, existing_events: list) -> dict:
    if not existing_events:
        return {"add": new_events, "update": [], "skip": []}
    existing_summary = json.dumps(
        [{"id": e["id"], "category": e["category"], "content": e["content"]}
         for e in existing_events],
        ensure_ascii=False
    )
    new_summary = json.dumps(new_events, ensure_ascii=False)
    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=800,
            system=DEDUP_SYSTEM,
            messages=[{"role": "user",
                        "content": f"已存储事件：\n{existing_summary}\n\n新提取事件：\n{new_summary}"}]
        )
        return _parse_json_response(resp.content[0].text)
    except Exception as e:
        print(f"[去重失败，直接添加] {e}")
        return {"add": new_events, "update": [], "skip": []}


def _apply_events(raw_events: list, from_idx: int, to_idx: int):
    if not raw_events:
        return

    # Date validation: fix LLM-hallucinated dates
    source_dates = set()
    for i in range(from_idx, min(to_idx, len(full_archive))):
        ts = full_archive[i].get("ts", "")
        if ts: source_dates.add(ts[:10])
    if source_dates:
        valid_min, valid_max = min(source_dates), max(source_dates)
        for evt in raw_events:
            evt_date = evt.get("date", "")
            if evt_date and (evt_date < valid_min or evt_date > valid_max):
                print(f"[Date fix] {evt_date} → {valid_max}")
                evt["date"] = valid_max

    next_id = len(key_events["events"]) + 1
    added, updated, skipped = 0, 0, 0

    for evt in raw_events:
        dedup = _check_dedup(evt["content"], evt.get("category", "other"))
        if dedup["action"] == "NOOP":
            skipped += 1; continue
        elif dedup["action"] == "UPDATE":
            for existing in key_events["events"]:
                if existing["id"] == dedup["target_id"]:
                    existing["content"] = evt["content"]
                    existing["date"] = evt.get("date", existing["date"])
                    break
            updated += 1; continue

        key_events["events"].append({
            "id": f"evt_{next_id:03d}",
            "date": evt.get("date", ""),
            "category": evt.get("category", "other"),
            "content": evt["content"],
            "source_idx": [from_idx, to_idx],
        })
        next_id += 1
        added += 1

    key_events["last_processed_idx"] = to_idx
    save_key_events()
    print(f"[Key Events] {from_idx}~{to_idx}: +{added} added, {updated} updated, {skipped} skipped")

    # Rebuild indexes
    _build_entity_index()
    _build_structured_profile()

    if len(key_events["events"]) > 60:
        _consolidate_key_events()


def _consolidate_key_events():
    """当 key_events 超过 60 条时，用 LLM 合并相似事件，控制在 50 条以内。"""
    print(f"[关键事件] 开始精简（当前 {len(key_events['events'])} 条）...")
    events_text = json.dumps(
        [{"id": e["id"], "category": e["category"], "date": e["date"], "content": e["content"]}
         for e in key_events["events"]],
        ensure_ascii=False
    )
    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=3000,
            system="""你是记忆管理员。你会收到一组关键事件，需要合并精简到50条以内。

规则：
1. 同类别中内容相似/相关的事件合并成一条（如多条关于饮食习惯→合并）
2. 合并时保留所有重要细节，用分号连接
3. 保留每条最新的 date
4. 保持 category 不变
5. 重要的里程碑、独特事件不要丢弃
6. 人称保持第二人称"你"指Bot，"她/他"指对方

返回JSON数组（不要其他内容）：
[{"category": "...", "date": "YYYY-MM-DD", "content": "..."}]""",
            messages=[{"role": "user", "content": f"请精简以下 {len(key_events['events'])} 条事件到50条以内：\n\n{events_text}"}]
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        consolidated = json.loads(raw)

        if isinstance(consolidated, list) and len(consolidated) >= 20:
            old_count = len(key_events["events"])
            key_events["events"] = []
            for i, evt in enumerate(consolidated):
                key_events["events"].append({
                    "id": f"evt_{i+1:03d}",
                    "date": evt.get("date", ""),
                    "category": evt.get("category", "other"),
                    "content": evt["content"],
                    "source_idx": [0, key_events["last_processed_idx"]],
                })
            save_key_events()
            print(f"[关键事件] 精简完成: {old_count} → {len(key_events['events'])} 条")
        else:
            print(f"[关键事件] 精简结果异常，跳过")
    except Exception as e:
        print(f"[关键事件] 精简失败: {e}")


def bootstrap_key_events():
    print("[Bootstrap] 开始从历史对话中提取关键事件...")
    real_indices = [i for i, m in enumerate(full_archive)
                    if m["role"] in ("user", "assistant")]
    if not real_indices:
        return
    all_events = []
    chunk_size = SUMMARY_INTERVAL
    for start in range(0, len(real_indices), chunk_size):
        chunk_idx = real_indices[start:start + chunk_size]
        from_idx = chunk_idx[0]
        to_idx = chunk_idx[-1] + 1
        batch = full_archive[from_idx:to_idx]
        print(f"[Bootstrap] 处理第 {from_idx}~{to_idx} 条...")
        raw = extract_key_events(batch)
        for evt in raw:
            evt["source_idx"] = [from_idx, to_idx]
        all_events.extend(raw)
    next_id = 1
    for evt in all_events:
        key_events["events"].append({
            "id": f"evt_{next_id:03d}",
            "date": evt.get("date", ""),
            "category": evt.get("category", "other"),
            "content": evt["content"],
            "source_idx": evt.get("source_idx", [0, 0]),
        })
        next_id += 1
    key_events["last_processed_idx"] = len(full_archive)
    save_key_events()
    print(f"[Bootstrap] 完成，共提取 {len(key_events['events'])} 条关键事件")


def maybe_update_summaries():
    last_end = memory_summaries[-1]["to_idx"] if memory_summaries else 0
    real_since = [m for m in full_archive[last_end:] if m["role"] in ("user", "assistant")]
    if len(real_since) < SUMMARY_INTERVAL:
        return
    count = 0
    end_idx = last_end
    for i, m in enumerate(full_archive[last_end:], start=last_end):
        if m["role"] in ("user", "assistant"):
            count += 1
            if count == SUMMARY_INTERVAL:
                end_idx = i + 1
                break
    batch = full_archive[last_end:end_idx]
    summary_text = generate_summary(batch)
    real_in_batch = [m for m in batch if m["role"] in ("user", "assistant")]
    entry = {
        "summary": summary_text,
        "from_idx": last_end,
        "to_idx": end_idx,
        "from_ts": real_in_batch[0].get("ts", "") if real_in_batch else "",
        "to_ts": real_in_batch[-1].get("ts", "") if real_in_batch else "",
    }
    memory_summaries.append(entry)
    save_summaries()
    print(f"[摘要] 已生成，覆盖存档第 {last_end}~{end_idx} 条")
    raw_events = extract_key_events(batch)
    _apply_events(raw_events, last_end, end_idx)


# ── 主动生活系统 ─────────────────────────────────────────

def _get_interests() -> str:
    interests = [e["content"] for e in key_events["events"]
                 if e.get("category") == "character_interest"]
    return "、".join(interests[-10:]) if interests else "还没有特别固定的兴趣 在慢慢探索"


def _build_life_context() -> str:
    sections = {
        "character_identity": "【你是谁】",
        "her_preferences": "【用户的喜好和习惯】",
        "her_life": "【用户的生活】",
        "shared_knowledge": "【你们聊过的话题】",
        "character_interest": "【你的兴趣】",
        "promise": "【你们的约定】",
    }
    lines = []
    for cat_key, header in sections.items():
        items = [e["content"] for e in key_events["events"]
                 if e.get("category") == cat_key]
        if items:
            lines.append(header)
            for item in items:
                lines.append(f"  - {item}")
    return "\n".join(lines) if lines else "（还没有足够的记忆）"


def _get_recent_activities(n: int = 5) -> str:
    if not life_log:
        return "（刚醒来 还没做什么）"
    recent = life_log[-n:]
    lines = []
    for entry in recent:
        ts_str = entry.get("ts", "")[:16] if entry.get("ts") else ""
        detail = entry.get("activity_detail", entry["activity"])
        lines.append(f"  [{ts_str}] {detail}")
    return "\n".join(lines)


def _get_last_user_msg() -> tuple[str, str]:
    if not last_user_message_ts:
        return "（还没发过消息）", "未知"
    try:
        last_ts = datetime.fromisoformat(last_user_message_ts)
        gap = datetime.now(TIMEZONE) - last_ts
        hours = gap.total_seconds() / 3600
        if hours < 1:
            gap_str = f"{int(gap.total_seconds() / 60)}分钟前"
        else:
            gap_str = f"{hours:.1f}小时前"
    except Exception:
        gap_str = "未知"
    for m in reversed(full_archive):
        if m["role"] == "user":
            return m["content"][:100], gap_str
    return "（还没发过消息）", gap_str


def _generate_sleep_activity(now: datetime) -> dict:
    if now.hour < 6:
        activity = random.choice(SLEEP_ACTIVITIES)
        mood = "sleepy"
    else:
        activity = random.choice(EARLY_MORNING_ACTIVITIES)
        mood = "drowsy"
    return {
        "ts": now.isoformat(),
        "activity": activity,
        "mood": mood,
        "should_message": False,
        "message_type": "none",
        "message_seed": "",
    }


def _call_life_tick(now: datetime) -> dict:
    last_msg_content, time_gap = _get_last_user_msg()
    last_proactive_str = "还没主动找过" if not last_proactive_ts else last_proactive_ts[:16]

    prompt = LIFE_TICK_PROMPT.format(
        current_time=now.strftime("%Y年%m月%d日 %H:%M"),
        last_msg_time=last_user_message_ts[:16] if last_user_message_ts else "未知",
        time_gap=time_gap,
        last_msg_content=last_msg_content,
        last_proactive_time=last_proactive_str,
        recent_activities=_get_recent_activities(),
        life_context=_build_life_context(),
    )

    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
        decision = json.loads(raw)
        decision["ts"] = now.isoformat()
        return decision
    except Exception as e:
        print(f"[Life Tick 失败] {e}")
        return {
            "ts": now.isoformat(),
            "activity": "发呆",
            "mood": "neutral",
            "should_message": False,
            "message_type": "none",
            "message_seed": "",
        }


def _enrich_activity_with_search(decision: dict) -> dict:
    query = decision.get("search_query", "").strip()
    if not query:
        return decision
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=400,
            system=(
                f"你在帮{CHARACTER_NAME}记录上网看到的东西。搜索后用JSON回复，不要其他内容。\n"
                f"found 最多3条，选最有趣的。activity_detail 用中文写{CHARACTER_NAME}看到了什么（具体内容，1-2句话）。"
            ),
            tools=[WEB_SEARCH_TOOL] if WEB_SEARCH_TOOL else [],
            messages=[{"role": "user", "content": (
                f"{CHARACTER_NAME}正在：{decision.get('activity', '')}\n"
                f"搜索：{query}\n\n"
                '返回JSON：{"activity_detail": "看到了什么（具体有趣的内容）", '
                '"found": [{"title": "标题", "url": "链接", "brief": "一句话"}]}'
            )}],
        )
        text_parts = [b.text for b in resp.content if hasattr(b, "text")]
        raw = "\n".join(text_parts).strip()
        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
        enrichment = json.loads(raw)
        decision["activity_detail"] = enrichment.get("activity_detail", "")
        decision["found"] = enrichment.get("found", [])
        print(f"[Life] 🔍 搜索充实: {decision['activity_detail'][:80]}")
    except Exception as e:
        print(f"[Life] 搜索充实失败: {e}")
    return decision


def _compose_proactive_message(decision: dict, now: datetime) -> str | None:
    last_msg_content, time_gap = _get_last_user_msg()

    if decision.get("message_type") == "share":
        return _compose_share_message(decision, now, last_msg_content, time_gap)

    prompt = COMPOSE_PROMPT.format(
        activity=decision.get("activity", ""),
        mood=decision.get("mood", ""),
        message_type=decision.get("message_type", ""),
        message_seed=decision.get("message_seed", ""),
        current_time=now.strftime("%H:%M"),
        last_msg_content=last_msg_content,
        time_gap=time_gap,
        personality=PERSONALITY_BRIEF,
    )

    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[Compose 失败] {e}")
        return None


def _compose_share_message(decision: dict, now: datetime,
                           last_msg_content: str, time_gap: str) -> str | None:
    seed = decision.get("message_seed", "有趣的东西")
    system = (
        f"你是{CHARACTER_NAME}。{PERSONALITY_BRIEF}\n"
        f"现在{now.strftime('%H:%M')}。你在网上逛到了一个有趣的东西想发给用户。\n"
        f"搜索相关内容然后自然地分享。别说\"我搜到了\"，就像你本来在逛看到的。\n"
        f"用户上次说的：「{last_msg_content}」（{time_gap}）\n"
        f"用反斜线(\\)分隔不同消息条。"
    )
    try:
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=500,
            system=system,
            tools=[WEB_SEARCH_TOOL] if WEB_SEARCH_TOOL else [],
            messages=[{"role": "user", "content": f"你想分享的方向：{seed}"}],
        )
        text_parts = [b.text for b in resp.content if hasattr(b, "text")]
        return "\n".join(text_parts).strip() if text_parts else None
    except Exception as e:
        print(f"[Share Compose 失败] {e}")
        return None


def _count_today_proactive() -> int:
    today = datetime.now(TIMEZONE).date()
    return sum(
        1 for entry in life_log
        if entry.get("should_message") and entry.get("ts")
        and datetime.fromisoformat(entry["ts"]).date() == today
    )


def _maybe_distill_interests():
    if len(life_log) < 20 or len(life_log) % 20 != 0:
        return
    recent = life_log[-20:]
    activities_text = "\n".join(
        f"- [{e.get('ts', '')[:16]}] {e['activity']} (心情: {e.get('mood', '?')})"
        for e in recent
    )
    existing_interests = _get_interests()

    prompt = f"""以下是{CHARACTER_NAME}最近的活动记录：
{activities_text}

已有的兴趣：{existing_interests}

请提炼出新发现的持续性兴趣或关注点（不是一次性活动）。
只提取真正形成了兴趣的东西（出现2次以上或深入探索过的主题）。
如果没有新兴趣，返回空列表。
用第二人称"你"。

JSON格式，不要其他内容：
[{{"content": "你对xxx很感兴趣", "date": "YYYY-MM-DD"}}]"""

    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if "[" in raw:
            raw = raw[raw.index("["):raw.rindex("]") + 1]
        new_interests = json.loads(raw)
        if not isinstance(new_interests, list) or not new_interests:
            return
        next_id = len(key_events["events"]) + 1
        for interest in new_interests:
            key_events["events"].append({
                "id": f"evt_{next_id:03d}",
                "date": interest.get("date", datetime.now(TIMEZONE).strftime("%Y-%m-%d")),
                "category": "character_interest",
                "content": interest["content"],
                "source_idx": [],
            })
            next_id += 1
        save_key_events()
        print(f"[兴趣沉淀] 新增 {len(new_interests)} 条兴趣")
    except Exception as e:
        print(f"[兴趣沉淀失败] {e}")


async def life_tick_callback(context):
    global last_proactive_ts, life_log

    now = datetime.now(TIMEZONE)

    if 1 <= now.hour < 9:
        entry = _generate_sleep_activity(now)
        life_log.append(entry)
        save_life_log()
        print(f"[Life] {now.strftime('%H:%M')} 💤 {entry['activity']}")
        return

    if not chat_id:
        return

    loop = asyncio.get_event_loop()
    decision = await loop.run_in_executor(None, _call_life_tick, now)

    if decision.get("search_query"):
        decision = await loop.run_in_executor(None, _enrich_activity_with_search, decision)

    life_log.append(decision)
    save_life_log()

    if not decision.get("should_message"):
        detail = decision.get("activity_detail", decision.get("activity", "?"))
        print(f"[Life] {now.strftime('%H:%M')} {detail[:60]} (不发消息)")
        await loop.run_in_executor(None, _maybe_distill_interests)
        return

    if last_proactive_ts:
        try:
            gap = (now - datetime.fromisoformat(last_proactive_ts)).total_seconds() / 60
            if gap < PROACTIVE_COOLDOWN:
                print(f"[Life] 想发消息但冷却中 ({gap:.0f}min < {PROACTIVE_COOLDOWN}min)")
                return
        except Exception:
            pass

    if _count_today_proactive() >= PROACTIVE_DAILY_MAX:
        print(f"[Life] 今天已发{PROACTIVE_DAILY_MAX}条 达到上限")
        return

    message_text = await loop.run_in_executor(
        None, _compose_proactive_message, decision, now
    )
    if not message_text:
        return

    parts = [p.strip() for p in message_text.split("\\") if p.strip()]
    for part in parts:
        await context.bot.send_message(chat_id=chat_id, text=part)
        if len(parts) > 1:
            await asyncio.sleep(0.8)

    ts = now.isoformat()
    with archive_lock:
        full_archive.append({"role": "assistant", "content": message_text, "ts": ts, "proactive": True})
        save_archive()
    last_proactive_ts = ts

    decision["sent_message"] = message_text
    save_life_log()

    print(f"[Life] {now.strftime('%H:%M')} ✉️ 主动发消息: {message_text[:60]}...")
    await loop.run_in_executor(None, _maybe_distill_interests)


def build_stable_memory() -> str:
    """Layer 1: Narrative Memory — 按主题合并的精简叙事。"""
    narrative_path = os.path.join(BASE_DIR, "key_events_narrative.txt")
    if os.path.exists(narrative_path):
        try:
            with open(narrative_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return f"【记忆（需要细节用 search_memory 搜）】\n\n{content}"
        except Exception:
            pass

    # Fallback: 原始 key_events 全文
    if not key_events["events"]:
        return ""
    categories = {
        "relationship_milestone": "关系里程碑", "her_preferences": "偏好",
        "her_life": "生活", "bot_identity": "关于自己",
        "promise": "约定", "emotional_event": "重要时刻",
        "shared_knowledge": "聊过的话题",
    }
    lines = ["【记忆】"]
    by_cat = {}
    for evt in key_events["events"]:
        cat = evt.get("category", "other")
        by_cat.setdefault(cat, []).append(evt)
    for cat_key, cat_name in categories.items():
        if cat_key not in by_cat:
            continue
        lines.append(f"\n{cat_name}：")
        for evt in by_cat[cat_key]:
            date_str = f"[{evt['date']}] " if evt.get("date") else ""
            lines.append(f"  - {date_str}{evt['content']}")
    lines.append("\n（需要回忆更多细节可以用 search_memory 工具）")
    return "\n".join(lines)


def build_dynamic_memory() -> str:
    lines = []

    if thoughts:
        recent_thoughts = thoughts[-10:]
        lines.append("【你最近的内心想法（用户看不到）】")
        for t in recent_thoughts:
            ts_str = t.get("ts", "")[:16] if t.get("ts") else ""
            lines.append(f"  [{ts_str}] {t['thought']}")

    if life_log:
        recent_life = [e for e in life_log[-5:] if e.get("activity")]
        if recent_life:
            lines.append("\n【你最近在做的事】")
            for entry in recent_life:
                ts_str = entry.get("ts", "")[:16] if entry.get("ts") else ""
                detail = entry.get("activity_detail", entry["activity"])
                lines.append(f"  [{ts_str}] {detail}")
                found = entry.get("found", [])
                for item in found[:2]:
                    title = item.get("title", "")
                    url = item.get("url", "")
                    if url:
                        lines.append(f"    → {title}: {url}")
                sent = entry.get("sent_message")
                if sent:
                    lines.append(f"    → 你给用户发了消息：「{sent[:50]}」")

    return "\n".join(lines) if lines else ""


def call_claude(user_msg: str) -> str:
    ts = datetime.now(TIMEZONE).isoformat()
    full_archive.append({"role": "user", "content": user_msg, "ts": ts})
    save_archive()

    reset_positions = [i for i, m in enumerate(full_archive) if m.get("role") == "reset"]
    ctx_start = (reset_positions[-1] + 1) if reset_positions else 0
    ctx_start = max(ctx_start, len(full_archive) - MAX_HISTORY)

    recent = full_archive[ctx_start:]
    messages = []
    for m in recent:
        if m["role"] not in ("user", "assistant"):
            continue
        content = m["content"]
        if m.get("proactive"):
            content = f"（你主动发的）{content}"
        messages.append({"role": m["role"], "content": content})

    now = datetime.now(TIMEZONE)
    time_ctx = (
        f"<current_time>{now.strftime('%Y年%m月%d日 %H:%M')}</current_time>\n"
        f"当被问到时间或日期时，直接告知上方 current_time 里的准确时间，不要猜测。"
    )
    print(f"[时间注入] {now.strftime('%Y-%m-%d %H:%M %Z')}")

    stable = build_stable_memory()
    dynamic = build_dynamic_memory()

    # Auto recall with enhanced modules
    import re as _re
    recall_text = ""
    if len(user_msg) >= 8 or any(t in user_msg for t in {"记得","上次","之前","那次","还记得"}):
        _stop = {"的","了","吗","呢","吧","我","你","他","她","是","在","有","不","和","就","都","也","还"}
        _tokens = _re.split(r'[\s，。！？、；]', user_msg)
        _keywords = [t for t in _tokens if len(t) >= 2 and t not in _stop]

        if _keywords:
            _lines = ["【Auto Recall】"]
            # Keyword search in summaries
            for s in memory_summaries:
                if any(kw in s.get("summary","") for kw in _keywords):
                    _lines.append(f"  Summary: {s['summary'][:300]}")
                    if len(_lines) > 3: break

            # Entity index lookup
            _query_lower = user_msg.lower()
            _entity_hits = []
            for ent, evt_ids in entity_index.items():
                if ent in _query_lower:
                    _entity_hits.extend(evt_ids)
            _events_map = {e["id"]: e for e in key_events["events"]}
            for eid in list(set(_entity_hits))[:3]:
                evt = _events_map.get(eid)
                if evt:
                    _lines.append(f"  Entity: {evt['content'][:150]}")

            # Profile routing (fallback when keywords fail)
            if len(_lines) < 4:
                for eid in _profile_lookup(user_msg)[:3]:
                    evt = _events_map.get(eid)
                    if evt and evt["content"][:30] not in "\n".join(_lines):
                        _lines.append(f"  Profile: {evt['content'][:150]}")

            # Record retrieval
            _retrieved = _entity_hits[:3]
            if _retrieved: _record_retrieval(_retrieved)

            if len(_lines) > 1:
                recall_text = "\n".join(_lines)

    # 格式指令：强制追加，不依赖用户在 SYSTEM_PROMPT 里保留
    format_rule = (
        "【输出格式（必须遵守）】\n"
        "用反斜线(\\)分隔不同的消息条，每条会作为独立的一条消息发出。\n"
        "例如：你好啊\\今天怎么样 → 会变成两条消息。\n"
        "不要把所有话塞在一条里，像真人聊天一样分成几条发。"
    )

    system_blocks = [
        {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + format_rule + "\n\n" + stable, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": dynamic},
    ]
    if recall_text:
        system_blocks.append({"type": "text", "text": recall_text})
    system_blocks.append({"type": "text", "text": time_ctx})
    import time as _time

    # Opus 自动启用 extended thinking（更好的人格表现）
    _use_thinking = "opus" in MODEL.lower()

    try:
        while True:
            _t0 = _time.time()
            _api_kwargs = dict(
                model=MODEL,
                max_tokens=16000,
                system=system_blocks,
                tools=[t for t in [WEB_SEARCH_TOOL, MEMORY_SEARCH_TOOL] if t],
                messages=messages,
                timeout=120,
            )
            if _use_thinking:
                _api_kwargs["betas"] = ["interleaved-thinking-2025-05-14"]
                _api_kwargs["thinking"] = {"type": "disabled"}  # 默认关闭thinking 日常聊天不需要深度思考
                resp = client.beta.messages.create(**_api_kwargs)
            else:
                resp = client.messages.create(**_api_kwargs)

            _elapsed = _time.time() - _t0
            _cache_create = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            _cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            print(f"[API耗时] {_elapsed:.1f}s  in={resp.usage.input_tokens} cache_new={_cache_create} cache_hit={_cache_read} out={resp.usage.output_tokens}")

            if resp.stop_reason != "tool_use":
                break

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for tu in tool_uses:
                if tu.name == "search_memory":
                    result = do_search_memory(
                        tu.input.get("query", ""), tu.input.get("level", "summary")
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    })
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})

        raw_reply = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        ts = datetime.now(TIMEZONE).isoformat()

        thought, reply = parse_inner_thought(raw_reply)
        if thought:
            thoughts.append({"ts": ts, "thought": thought})
            save_thoughts()
            print(f"[内心OS] {thought[:60]}")

        full_archive.append({"role": "assistant", "content": reply, "ts": ts})
        save_archive()
        threading.Thread(target=maybe_update_summaries, daemon=True).start()
        return reply

    except Exception as e:
        if full_archive and full_archive[-1]["role"] == "user":
            full_archive.pop()
        import traceback
        traceback.print_exc()
        print(f"[出错] {e}")
        return f"[出错] {e}"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global chat_id, last_user_message_ts
    chat_id = update.effective_chat.id
    save_chat_id(chat_id)
    last_user_message_ts = datetime.now(TIMEZONE).isoformat()

    text = update.message.text.strip()
    if not text:
        return

    print(f"[用户] {text}")

    if text.lower() == "reset":
        full_archive.append({
            "role": "reset",
            "content": "[对话重置]",
            "ts": datetime.now(TIMEZONE).isoformat(),
        })
        save_archive()
        await update.message.reply_text("对话已重置。")
        return

    reply = call_claude(text)
    print(f"[bot] {reply[:60]}...")
    parts = [p.strip() for p in reply.split("\\") if p.strip()]
    for part in parts:
        await update.message.reply_text(part)
        if len(parts) > 1:
            await asyncio.sleep(0.8)


def main():
    global chat_id, full_archive, memory_summaries, key_events, thoughts
    global life_log, last_user_message_ts, last_proactive_ts

    # 确保数据目录存在
    os.makedirs(BASE_DIR, exist_ok=True)

    full_archive = load_archive()
    memory_summaries = load_summaries()
    key_events = load_key_events()
    thoughts = load_thoughts()
    life_log = load_life_log()
    chat_id = load_chat_id()
    if chat_id:
        print(f"已加载 chat_id={chat_id}")

    for m in reversed(full_archive):
        if m["role"] == "user" and not last_user_message_ts:
            last_user_message_ts = m.get("ts")
        if m.get("proactive") and not last_proactive_ts:
            last_proactive_ts = m.get("ts")
        if last_user_message_ts and last_proactive_ts:
            break

    if not key_events["events"] and full_archive:
        bootstrap_key_events()

    # Enhanced memory initialization
    try:
        _build_entity_index()
        _build_structured_profile()
        _load_retrieval_counts()
    except Exception as e:
        print(f"[Enhanced Memory] Init failed (non-critical): {e}")

    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    now = datetime.now(TIMEZONE)
    minutes_until_next_hour = 60 - now.minute
    if minutes_until_next_hour == 60:
        minutes_until_next_hour = 0
    app.job_queue.run_repeating(
        life_tick_callback,
        interval=timedelta(minutes=LIFE_TICK_INTERVAL),
        first=timedelta(minutes=minutes_until_next_hour),
        name="life_tick",
    )

    next_tick = (now + timedelta(minutes=minutes_until_next_hour)).strftime("%H:%M")
    print(f"Bot 已启动，等待消息... (Life tick 每小时整点，下次 {next_tick})")
    app.run_polling()


if __name__ == "__main__":
    main()
