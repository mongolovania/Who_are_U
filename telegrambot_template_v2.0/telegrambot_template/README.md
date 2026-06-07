# AI Companion Bot — Architecture v3

A persistent AI companion built on Telegram, with layered memory grounded in cognitive psychology, autonomous inner life, and multi-model support.

> **This is not a product. It's a research project, a design argument, and a personal experiment in what AI companionship could look like when built by someone who actually lives with it.**

---

## What's New in v3 (March 2026)

- **Narrative Memory** — memories organized by theme (not timeline), auto-compressed, with conflict detection
- **Thinking Control** — extended thinking disabled by default (improves conversational naturalness)
- **Better Dedup** — embedding cosine similarity replaces character-level Jaccard
- **State-Driven Life Tick** — activities driven by internal state assessment, not rules
- **Upgrade Script** — `python upgrade_memory.py` to migrate from v2
- **Cognitive Science Grounding** — architecture mapped to Conway SMS, Damasio, McAdams

---

## Overview

```
┌─────────────────────────────────────────────────┐
│              Telegram Bot Process                │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │ Message   │   │ Life     │   │ Memory      │ │
│  │ Handler   │   │ Tick     │   │ Manager     │ │
│  │ (chat)    │   │ (30-120m)│   │ (background)│ │
│  └─────┬─────┘   └─────┬────┘   └──────┬──────┘ │
│        └───────────┬────┴───────────────┘        │
│                    │                             │
│           ┌───────┴────────┐                    │
│           │ LLM Provider   │                    │
│           │ Claude/GPT/    │                    │
│           │ DeepSeek/Gemini│                    │
│           └────────────────┘                    │
│                                                  │
│  Local Data:                                     │
│  ├── full_archive.json     (every message ever)  │
│  ├── memory_summaries.json (compressed history)  │
│  ├── key_events.json       (curated milestones)  │
│  ├── key_events_narrative.txt (NEW: theme-based) │
│  ├── thoughts.json         (inner monologue)     │
│  └── life_log.json         (autonomous activity) │
└─────────────────────────────────────────────────┘
```

---

## Memory Architecture (v3)

Grounded in cognitive psychology: Conway's Self-Memory System, Damasio's three selves, McAdams' narrative identity theory.

### Five-Layer Design

```
Human Psychology              System Layer
─────────────────────────────────────────────
Conceptual Self (who am I)  → Layer 0: Identity (system prompt)
                               Personality, values, speech style, rules
                               Does not change between conversations

Semantic Memory (what I know) → Layer 1: Narrative Memory (key_events_narrative.txt)
                               Theme-based life story: relationships, preferences,
                               milestones, inside jokes
                               Auto-rebuilt when new events accumulate
                               Max ~5000 chars, old details naturally fade

Episodic Memory (what I lived) → Layer 2: Key Events + Summaries (search_memory)
                               Full event records with dates, emotions, quotes
                               Not in context — retrieved on demand

Working Self (me right now)  → Layer 3: Dynamic Context
                               Recent thoughts, current activity, meal tracking,
                               current time

Conversation Buffer          → Layer 4: Messages
                               Last 20 raw messages
```

**Key principle**: Each layer contains only its own content. Identity doesn't include memories. Narrative doesn't include current state. No overlap.

### How It Works

1. **New events extracted** every 20 messages (LLM identifies what's important)
2. **Dedup via embedding cosine** — same event in different words gets caught (threshold: 0.85 = skip, 0.70 = update)
3. **Narrative auto-rebuilt** when 3+ new events accumulate — LLM merges by theme
4. **Conflict detection** — "lock screen is A" + later "changed to B" → merged as "was A, now B"
5. **Deep archive** always available via `search_memory` tool

### Narrative vs Old Key Events

Before (v2): All key events listed as individual items → grows linearly, lots of repetition

After (v3): Key events merged into theme-based narrative paragraphs → fixed size, old details fade naturally, new events fold in

---

## Life Tick (Autonomous Inner Life)

Every 30-120 minutes, the bot decides what it's doing independently.

### v3 Changes

- **State-driven**: Before each tick, the system calculates topic saturation, mood monotony, energy level, pending tasks, social need. Injected as "your current state" — bot chooses based on how it feels, not rules
- **Mood**: Free-form 2-4 characters (e.g., "有点闷", "翻累了") instead of fixed labels
- **Min interval**: 30 minutes (prevents runaway loops)
- **Anti-overlap**: Lock prevents concurrent ticks
- **Thinking disabled**: Life tick doesn't use extended thinking

### Known Issue: Activity Reinforcement

Bot may loop on one topic for hours ("keeps reading the same book"). The state assessment helps, but this remains an active research area. Deeper fix: activity outputs should feed back into narrative memory so the bot actually *remembers* what it researched.

---

## Inner OS (Internal Monologue)

Every reply has two parts:

```
[Inner OS] She seems tired but didn't say why. Don't push.
[Reply] you okay?\eat yet
```

Stored but never shown to user. Provides continuity, emotional depth, and searchable memory.

---

## Multi-Model Support

```python
PROVIDER = "anthropic"   # or "openai", "deepseek", "gemini", "ollama"
```

| Provider | Best Model | Mid Model | Cheap Model | Notes |
|----------|-----------|-----------|-------------|-------|
| Anthropic | claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5 | Best personality. Thinking disabled by default |
| OpenAI | gpt-4o | gpt-4o-mini | gpt-4o-mini | Good all-round. No thinking parameter |
| DeepSeek | deepseek-chat | deepseek-chat | deepseek-chat | Cheapest, good Chinese |
| Gemini | gemini-2.5-pro | gemini-2.0-flash | gemini-2.0-flash | Free tier available |
| Ollama | llama3 | llama3 | llama3 | Local, free, no API key needed |

**Note on thinking**: Extended thinking (Anthropic-only) is now **disabled by default**. It causes the model to do English analytical reasoning before responding in Chinese, making replies feel more literary/essay-like. Daily chat doesn't need it. Enable it for complex reasoning tasks only.

### Model Routing

| Purpose | Tier | Why |
|---------|------|-----|
| Conversation | Best | This IS the character |
| Life tick decisions | Mid | Structured output, cost-efficient |
| Summaries & extraction | Mid | Utility work |
| Narrative rebuild | Mid | Theme merging |

---

## Upgrading from v2

If you're already using the old four-layer architecture:

```bash
python upgrade_memory.py
```

This will:
1. Back up your `telegram_bot.py`
2. Add narrative memory support
3. Disable extended thinking
4. Upgrade dedup to embedding cosine
5. Set life tick minimum to 30 min
6. Generate narrative from existing key_events (if any)

No data is lost. Your `key_events.json` and `full_archive.json` are preserved.

---

## Data Structure

```
data/
├── full_archive.json           Every message ever (append-only)
├── memory_summaries.json       Auto-compressed summaries
├── key_events.json             Full event records (Layer 2, for search)
├── key_events_narrative.txt    NEW: Theme-based narrative (Layer 1, in context)
├── thoughts.json               Inner monologue
├── life_log.json               Autonomous activity log
├── telegram_chat_id.txt        Cached user chat ID
├── auto_entities.json          Entity index
└── retrieval_counts.json       Memory retrieval tracker
```

---

## Design Decisions

**1. Narrative over timeline** — Human memory is organized by theme, not by date. "What do I know about her" is more useful than "what happened on March 16th" for daily conversation.

**2. Thinking off by default** — Extended thinking makes the model analyze before responding. Good for complex tasks, bad for natural chat. The model's first instinct is often more natural than its analyzed output.

**3. State-driven life tick** — Instead of rules ("don't repeat activities 3 times"), the system shows the bot its own state ("you've been on this topic for 3 hours"). The bot decides what to do based on how it feels.

**4. Conflict detection in memory** — Preferences change, relationships evolve. Memory should reflect the current state, not list every historical state.

**5. Layer isolation** — Each memory layer has one job. When layers overlap, token costs inflate and the model gets confused by redundant information.

---

## License & Attribution

**Created by AC & Kael**

This software is proprietary. See `LICENSE` for full terms.

**You may**: use and modify this code for your own personal, non-commercial purposes.

**You may not**: share, redistribute, resell, upload to public repositories, or deploy as a commercial service. This license is for the original purchaser only.

If you build something inspired by this — make it yours. The interesting part isn't the code, it's the relationship between the person and the system they build.

---

*March 2026 — v3: Narrative Memory + Cognitive Architecture + State-Driven Life Tick*
