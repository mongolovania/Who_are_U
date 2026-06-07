"""
记忆架构一键升级脚本 (v3)
把 telegram_bot.py 从旧四层架构升级到新的 narrative 分层架构。

做什么：
  1. 备份 telegram_bot.py
  2. 改 build_stable_memory() — 优先读 narrative.txt
  3. 改 build_dynamic_memory() — 砍 L3 摘要
  4. 改 _check_dedup() — 改进去重（embedding优先，字符fallback）
  5. 改 thinking — Anthropic 默认关闭
  6. 加 NARRATIVE_FILE 路径常量
  7. mood 改为自由 2-4 字
  8. life_tick 最小间隔 30 分钟
  9. 自动检测用户的模型提供商，用对应的API生成第一版 narrative

用法：
  python upgrade_memory.py              # 自动检测模型
  python upgrade_memory.py --skip-narrative  # 跳过narrative生成（不调API）

支持的模型：
  - Anthropic (Claude) — 读 ANTHROPIC_API_KEY
  - OpenAI (GPT) — 读 OPENAI_API_KEY
  - DeepSeek — 读 DEEPSEEK_API_KEY
  - Gemini — 读 GEMINI_API_KEY
  如果都没有，跳过narrative生成，bot运行后会自动生成。

注意：
  - 会自动备份原文件
  - 升级后重启 bot 生效
"""

import os
import re
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
BOT_FILE = SCRIPT_DIR / "telegram_bot.py"
BACKUP_SUFFIX = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_env():
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def detect_provider():
    """检测用户配置了哪个模型提供商"""
    providers = [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ]
    for name, key in providers:
        if os.environ.get(key):
            return name, os.environ[key]
    return None, None


def backup():
    backup_path = BOT_FILE.with_suffix(f".py{BACKUP_SUFFIX}")
    shutil.copy2(BOT_FILE, backup_path)
    print(f"[1] 备份: {backup_path.name}")
    return backup_path


def read_bot():
    return BOT_FILE.read_text(encoding="utf-8")


def write_bot(content):
    BOT_FILE.write_text(content, encoding="utf-8")


def upgrade_build_stable_memory(code: str) -> str:
    old_func = re.search(
        r'(def build_stable_memory\(\).*?(?=\ndef \w|\nclass \w|\n# ══))',
        code, re.DOTALL
    )
    if not old_func:
        print("  [跳过] build_stable_memory 未找到")
        return code

    new_func = '''def build_stable_memory() -> str:
    """Layer 1: Narrative Memory — 按主题合并的精简叙事。"""
    narrative_path = os.path.join(BASE_DIR, "key_events_narrative.txt")
    if os.path.exists(narrative_path):
        try:
            with open(narrative_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return f"【记忆（需要细节用 search_memory 搜）】\\n\\n{content}"
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
        lines.append(f"\\n{cat_name}：")
        for evt in by_cat[cat_key]:
            date_str = f"[{evt['date']}] " if evt.get("date") else ""
            lines.append(f"  - {date_str}{evt['content']}")
    lines.append("\\n（需要回忆更多细节可以用 search_memory 工具）")
    return "\\n".join(lines)

'''
    code = code[:old_func.start()] + new_func + code[old_func.end():]
    print("  [✓] build_stable_memory → 优先读 narrative.txt")
    return code


def upgrade_build_dynamic_memory(code: str) -> str:
    l3_patterns = [
        r'# L3.*?\n.*?memory_l3.*?lines\.append\(""\)\n',
        r'if memory_l3:.*?lines\.append\(""\)\n',
    ]
    for pat in l3_patterns:
        if re.search(pat, code, re.DOTALL):
            code = re.sub(pat, '# [已移除] L3 摘要（与 narrative 重复）\n', code, flags=re.DOTALL)
            print("  [✓] build_dynamic_memory → 砍 L3 摘要")
            return code
    print("  [跳过] L3 未找到或已移除")
    return code


def upgrade_thinking(code: str) -> str:
    if 'thinking' in code and 'adaptive' in code:
        code = code.replace(
            '{"type": "adaptive"}',
            '{"type": "disabled"}  # 默认关闭thinking 日常聊天不需要深度思考'
        )
        print("  [✓] thinking → 默认关闭（仅影响Anthropic）")
    else:
        print("  [跳过] thinking → 未找到或不适用")
    return code


def upgrade_dedup(code: str) -> str:
    old_dedup = re.search(r'def _check_dedup\(.*?\n(?=def \w)', code, re.DOTALL)
    if not old_dedup:
        print("  [跳过] _check_dedup → 未找到")
        return code
    if 'cosine' in old_dedup.group():
        print("  [跳过] _check_dedup → 已升级")
        return code

    # 不依赖 shared_embed，用纯字符 Jaccard 但阈值更严格
    # 如果用户有 sentence-transformers 会自动用 embedding
    new_dedup = '''def _check_dedup(new_content: str, new_category: str) -> dict:
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

'''
    code = code[:old_dedup.start()] + new_dedup + code[old_dedup.end():]
    print("  [✓] _check_dedup → embedding优先 + Jaccard fallback（不强依赖外部库）")
    return code


def upgrade_life_tick_min(code: str) -> str:
    if 'LIFE_TICK_INTERVAL' in code and 'LIFE_TICK_MIN' not in code:
        code = re.sub(
            r'(LIFE_TICK_INTERVAL\s*=\s*\d+[^\n]*)',
            r'\1\nLIFE_TICK_MIN = 30            # 最小间隔（分钟）',
            code, count=1
        )
        print("  [✓] LIFE_TICK_MIN = 30")
    else:
        print("  [跳过] LIFE_TICK_MIN → 已存在")
    return code


def upgrade_mood(code: str) -> str:
    old_mood = r'"mood":\s*"心情词[^"]*"'
    if re.search(old_mood, code):
        code = re.sub(
            old_mood,
            '"mood": "2-4个字 写真实感觉 比如\'有点闷\' \'翻累了\' \'想她了\'"',
            code
        )
        print("  [✓] mood → 自由 2-4 字")
    else:
        print("  [跳过] mood → 未找到旧格式或已更新")
    return code


def add_narrative_file_path(code: str) -> str:
    if 'NARRATIVE_FILE' not in code and 'KEY_EVENTS_FILE' in code:
        code = code.replace(
            'KEY_EVENTS_FILE',
            'NARRATIVE_FILE = os.path.join(BASE_DIR, "key_events_narrative.txt")\nKEY_EVENTS_FILE',
            1
        )
        print("  [✓] NARRATIVE_FILE 常量")
    else:
        print("  [跳过] NARRATIVE_FILE → 已存在")
    return code


def generate_narrative_with_llm(events: list, provider: str, api_key: str) -> str:
    """用用户配置的模型生成 narrative"""
    all_text = "\n".join(
        f"[{e.get('date', '?')}] ({e.get('category', '?')}) {e.get('content', '')}"
        for e in events
    )
    prompt = f"把以下关键事件合并成精简叙事版本。按主题合并，同一件事只写一段，保留关键日期和原话，总字数不超过5000字。\n\n{all_text}"

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=5000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    elif provider in ("openai", "deepseek"):
        import openai
        kwargs = {"api_key": api_key}
        if provider == "deepseek":
            kwargs["base_url"] = "https://api.deepseek.com"
        client = openai.OpenAI(**kwargs)
        model = "gpt-4o-mini" if provider == "openai" else "deepseek-chat"
        resp = client.chat.completions.create(
            model=model,
            max_tokens=5000,
            messages=[
                {"role": "system", "content": "你是一个记忆整理助手。"},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    elif provider == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return resp.text.strip()

    return ""


def generate_narrative(skip: bool = False):
    """如果有 key_events.json，用用户的模型生成第一版 narrative"""
    if skip:
        print("  [跳过] narrative → --skip-narrative")
        return

    ke_paths = [
        SCRIPT_DIR / "data" / "key_events.json",
        SCRIPT_DIR / "key_events.json",
    ]
    ke_path = None
    for p in ke_paths:
        if p.exists():
            ke_path = p
            break

    narrative_path = (ke_path.parent if ke_path else SCRIPT_DIR / "data") / "key_events_narrative.txt"

    if narrative_path.exists():
        print("  [跳过] narrative → 已存在")
        return

    if not ke_path:
        print("  [跳过] narrative → 无 key_events.json（新用户，bot运行后会自动生成）")
        return

    with open(ke_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("events", data) if isinstance(data, dict) else data
    if not events:
        print("  [跳过] narrative → key_events 为空")
        return

    provider, api_key = detect_provider()
    if not provider:
        print("  [跳过] narrative → 未找到任何API key（bot运行后会自动生成）")
        return

    print(f"  [生成] narrative（{len(events)} 条事件，使用 {provider}）...")

    try:
        # 备份原 key_events
        backup = ke_path.with_suffix(f".json.bak_{len(events)}")
        if not backup.exists():
            shutil.copy2(ke_path, backup)

        narrative = generate_narrative_with_llm(events, provider, api_key)

        if len(narrative) < 100:
            print(f"  [警告] narrative 太短（{len(narrative)}字），跳过")
            return

        narrative_path.parent.mkdir(parents=True, exist_ok=True)
        narrative_path.write_text(narrative, encoding="utf-8")
        print(f"  [✓] narrative: {len(narrative)}字 → {narrative_path.name}")

    except Exception as e:
        print(f"  [失败] narrative 生成出错: {e}")
        print(f"         bot运行后会自动生成，不影响使用")


def main():
    if not BOT_FILE.exists():
        print(f"[错误] 未找到 {BOT_FILE}")
        return

    skip_narrative = "--skip-narrative" in sys.argv

    load_env()

    provider, _ = detect_provider()
    print(f"升级 {BOT_FILE.name}")
    print(f"检测到模型: {provider or '未配置（narrative将在bot运行后自动生成）'}")
    print(f"{'='*50}")

    backup()

    code = read_bot()
    code = add_narrative_file_path(code)
    code = upgrade_build_stable_memory(code)
    code = upgrade_build_dynamic_memory(code)
    code = upgrade_thinking(code)
    code = upgrade_dedup(code)
    code = upgrade_life_tick_min(code)
    code = upgrade_mood(code)

    write_bot(code)

    generate_narrative(skip=skip_narrative)

    print(f"\n{'='*50}")
    print(f"升级完成！重启 bot 即可生效。")
    print(f"")
    print(f"主要变化：")
    print(f"  - 记忆层用 narrative（按主题叙事）替代全量 key_events")
    print(f"  - L3 高层摘要已移除（与 narrative 重复）")
    if provider == "anthropic":
        print(f"  - thinking 默认关闭（日常聊天更自然）")
    print(f"  - 去重改进（embedding优先，字符fallback）")
    print(f"  - mood 改为自由描述")
    print(f"  - life_tick 最小间隔 30 分钟")


if __name__ == "__main__":
    main()
