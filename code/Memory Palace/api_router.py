# ============================================================
# Module: API Router (api_router.py)
# REST API endpoints for Memory Palace.
# ============================================================

from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from app import (
    auth_service, namespace_mgr, dehydrator, llm_gateway, config,
    _make_components, _make_orchestrator,
)

from memory_orchestrator import DUYING_SYSTEM_PROMPT

from models import (
    ChatRequest, ChatResponse,
    HoldRequest, GrowRequest, TraceRequest, BreathRequest,
    RegisterRequest, LoginRequest, TokenResponse,
    PulseResponse, MemoryStats,
)

logger = logging.getLogger("memory_palace.api")
router = APIRouter()


# ── Auth dependency ─────────────────────────────────────
async def require_auth(request: Request) -> str:
    """FastAPI dependency: extract user_id from JWT."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header[7:]
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]


# ══════════════════════════════════════════════════════════
# Auth endpoints
# ══════════════════════════════════════════════════════════

@router.post("/auth/register", response_model=TokenResponse)
async def auth_register(req: RegisterRequest):
    try:
        user = auth_service.register(req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    tokens = auth_service.login(req.email, req.password)
    if not tokens:
        raise HTTPException(status_code=500, detail="Registration failed")
    return TokenResponse(**tokens)


@router.post("/auth/login", response_model=TokenResponse)
async def auth_login(req: LoginRequest):
    tokens = auth_service.login(req.email, req.password)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(**tokens)


@router.post("/auth/refresh")
async def auth_refresh(request: Request):
    body = await request.json()
    refresh_token = body.get("refresh_token", "")
    tokens = auth_service.refresh(refresh_token)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return tokens


@router.post("/auth/logout")
async def auth_logout(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        auth_service.revoke(auth_header[7:])
    return {"ok": True}


# ══════════════════════════════════════════════════════════
# Chat endpoint (core pipeline)
# ══════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_id: str = Depends(require_auth)):
    """Main chat endpoint: breath -> inject -> LLM -> reply."""
    orch = _make_orchestrator(user_id)
    result = await orch.chat(
        user_message=req.user_message,
        conversation_id=req.conversation_id,
        context_window=req.context_window,
    )
    return ChatResponse(**result)


# ══════════════════════════════════════════════════════════
# Chat SSE streaming endpoint
# ══════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, user_id: str = Depends(require_auth)):
    """
    Streaming chat endpoint with Server-Sent Events.

    Client receives:
      data: {"token": "..."}
      data: {"emotion_tags": {...}}
      data: [DONE]
    """
    from fastapi.responses import StreamingResponse

    orch = _make_orchestrator(user_id)

    async def event_stream():
        import json as _json

        # Step 1: breath
        memories = await orch._breath(query=req.user_message)

        # Step 2: build system prompt
        memory_text = orch._build_memory_injection(memories)
        from datetime import datetime as _dt
        current_time = _dt.now().strftime("%Y年%m月%d日 %H:%M")
        system_prompt = DUYING_SYSTEM_PROMPT.format(
            injected_memories=memory_text,
            current_time=current_time,
        )

        messages = list(req.context_window or [])
        messages.append({"role": "user", "content": req.user_message})

        try:
            # Stream LLM tokens
            async for token in llm_gateway.chat_stream(
                messages=messages,
                system=system_prompt,
            ):
                if token == "[STREAM_ERROR]":
                    yield f"data: {_json.dumps({'error': 'Stream interrupted'})}\n\n"
                    break
                yield f"data: {_json.dumps({'token': token})}\n\n"

            # Send emotion tags after stream
            emotion = orch._extract_emotion_signals(req.user_message)
            yield f"data: {_json.dumps({'emotion_tags': {'valence': emotion.valence, 'arousal': emotion.arousal}})}\n\n"

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ══════════════════════════════════════════════════════════
# Memory CRUD endpoints
# ══════════════════════════════════════════════════════════

@router.post("/memories/breath")
async def memories_breath(req: BreathRequest, user_id: str = Depends(require_auth)):
    """Retrieve/surface memories."""
    comps = _make_components(user_id)
    bucket_mgr = comps["bucket_mgr"]
    decay_engine = comps["decay_engine"]

    await decay_engine.ensure_started()

    from utils import strip_wikilinks, count_tokens_approx

    # No query: surfacing mode
    if not req.query or not req.query.strip():
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        pinned = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
        ]
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved")
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned")
            and not b["metadata"].get("protected")
        ]
        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )[:req.max_results]

        results = []
        for b in pinned + scored:
            clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "summary": summary,
                "valence": b["metadata"].get("valence", 0.5),
                "arousal": b["metadata"].get("arousal", 0.3),
                "type": b["metadata"].get("type", "dynamic"),
                "pinned": b["metadata"].get("pinned", False),
            })
        return {"memories": results}

    # With query: search mode
    matches = await bucket_mgr.search(
        req.query, limit=req.max_results
    )
    results = []
    for bucket in matches:
        clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
        summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
        results.append({
            "id": bucket["id"],
            "name": bucket["metadata"].get("name", ""),
            "summary": summary,
            "score": bucket.get("score", 0),
            "valence": bucket["metadata"].get("valence", 0.5),
            "arousal": bucket["metadata"].get("arousal", 0.3),
        })
    return {"memories": results, "query": req.query}


@router.post("/memories/hold")
async def memories_hold(req: HoldRequest, user_id: str = Depends(require_auth)):
    """Store a single memory."""
    comps = _make_components(user_id)
    bucket_mgr = comps["bucket_mgr"]
    decay_engine = comps["decay_engine"]
    embedding_engine = comps["embedding_engine"]

    await decay_engine.ensure_started()

    if not req.content.strip():
        return {"error": "Content cannot be empty"}

    importance = max(1, min(10, req.importance))
    extra_tags = [t.strip() for t in req.tags.split(",") if t.strip()]

    # Feel mode
    if req.feel:
        feel_valence = req.valence if 0 <= req.valence <= 1 else 0.5
        feel_arousal = req.arousal if 0 <= req.arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=req.content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, req.content)
        except Exception:
            pass
        if req.source_bucket:
            try:
                await bucket_mgr.update(req.source_bucket, digested=True)
            except Exception:
                pass
        return {"action": "feel", "bucket_id": bucket_id}

    # Auto-tagging
    try:
        analysis = await dehydrator.analyze(req.content)
    except Exception:
        analysis = {"domain": ["未分类"], "valence": 0.5, "arousal": 0.3, "tags": [], "suggested_name": ""}

    domain = analysis.get("domain", ["未分类"])
    auto_valence = analysis.get("valence", 0.5)
    auto_arousal = analysis.get("arousal", 0.3)
    auto_tags = analysis.get("tags", [])
    suggested_name = analysis.get("suggested_name", "")

    final_valence = req.valence if 0 <= req.valence <= 1 else auto_valence
    final_arousal = req.arousal if 0 <= req.arousal <= 1 else auto_arousal

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # Pinned: direct create
    if req.pinned:
        bucket_id = await bucket_mgr.create(
            content=req.content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=final_valence,
            arousal=final_arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, req.content)
        except Exception:
            pass
        return {"action": "pinned", "bucket_id": bucket_id, "domain": domain}

    # Merge or create
    result_name, is_merged = await _merge_or_create(
        bucket_mgr=bucket_mgr,
        dehydrator=dehydrator,
        embedding_engine=embedding_engine,
        config=config,
        content=req.content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=final_valence,
        arousal=final_arousal,
        name=suggested_name,
    )

    return {
        "action": "merged" if is_merged else "created",
        "result": result_name,
        "domain": domain,
    }


@router.post("/memories/grow")
async def memories_grow(req: GrowRequest, user_id: str = Depends(require_auth)):
    """Diary digest - split and archive."""
    comps = _make_components(user_id)
    bucket_mgr = comps["bucket_mgr"]
    decay_engine = comps["decay_engine"]
    embedding_engine = comps["embedding_engine"]

    await decay_engine.ensure_started()

    if not req.content.strip():
        return {"error": "Content cannot be empty"}

    # Short content: fast path
    if len(req.content.strip()) < 30:
        try:
            analysis = await dehydrator.analyze(req.content)
        except Exception:
            analysis = {"domain": ["未分类"], "valence": 0.5, "arousal": 0.3, "tags": [], "suggested_name": ""}

        result_name, is_merged = await _merge_or_create(
            bucket_mgr, dehydrator, embedding_engine, config,
            content=req.content.strip(),
            tags=analysis.get("tags", []),
            importance=5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
        )
        return {"action": "fast_path", "result": result_name}

    # Full digest
    try:
        items = await dehydrator.digest(req.content)
    except Exception as e:
        return {"error": f"Digest failed: {e}"}

    results = []
    created, merged = 0, 0
    for item in items:
        try:
            result_name, is_merged = await _merge_or_create(
                bucket_mgr, dehydrator, embedding_engine, config,
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
            )
            if is_merged:
                merged += 1
                results.append(f"Merged: {result_name}")
            else:
                created += 1
                results.append(f"Created: {item.get('name', result_name)}")
        except Exception as e:
            logger.warning(f"grow item failed: {e}")

    return {
        "total": len(items),
        "created": created,
        "merged": merged,
        "results": results,
    }


@router.patch("/memories/{bucket_id}")
async def memories_trace(bucket_id: str, req: TraceRequest, user_id: str = Depends(require_auth)):
    """Edit/delete a memory."""
    comps = _make_components(user_id)
    bucket_mgr = comps["bucket_mgr"]
    embedding_engine = comps["embedding_engine"]

    if req.delete:
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
        return {"ok": success}

    updates = {}
    if req.name:
        updates["name"] = req.name
    if req.domain:
        updates["domain"] = [d.strip() for d in req.domain.split(",") if d.strip()]
    if 0 <= req.valence <= 1:
        updates["valence"] = req.valence
    if 0 <= req.arousal <= 1:
        updates["arousal"] = req.arousal
    if 1 <= req.importance <= 10:
        updates["importance"] = req.importance
    if req.tags:
        updates["tags"] = [t.strip() for t in req.tags.split(",") if t.strip()]
    if req.resolved in (0, 1):
        updates["resolved"] = bool(req.resolved)
    if req.pinned in (0, 1):
        updates["pinned"] = bool(req.pinned)
        if req.pinned == 1:
            updates["importance"] = 10
    if req.digested in (0, 1):
        updates["digested"] = bool(req.digested)
    if req.content:
        updates["content"] = req.content

    if not updates:
        return {"ok": False, "error": "No fields to update"}

    success = await bucket_mgr.update(bucket_id, **updates)
    if success and "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    return {"ok": success, "updated_fields": list(updates.keys())}


@router.delete("/memories/{bucket_id}")
async def memories_delete(bucket_id: str, user_id: str = Depends(require_auth)):
    """Delete a memory."""
    comps = _make_components(user_id)
    success = await comps["bucket_mgr"].delete(bucket_id)
    if success:
        comps["embedding_engine"].delete_embedding(bucket_id)
    return {"ok": success}


# ══════════════════════════════════════════════════════════
# System & Stats
# ══════════════════════════════════════════════════════════

@router.get("/pulse", response_model=PulseResponse)
async def pulse(user_id: str = Depends(require_auth)):
    """System status and memory stats."""
    comps = _make_components(user_id)
    stats = await comps["bucket_mgr"].get_stats(user_id=user_id)
    return PulseResponse(
        status="ok",
        decay_engine="running" if comps["decay_engine"].is_running else "stopped",
        stats=MemoryStats(
            total_memories=sum(stats[k] for k in ["permanent_count", "dynamic_count", "feel_count"]),
            dynamic_count=stats.get("dynamic_count", 0),
            permanent_count=stats.get("permanent_count", 0),
            feel_count=stats.get("feel_count", 0),
            archive_count=stats.get("archive_count", 0),
            decision_count=0,
            emotion_curve=[],
        ),
        ddi_level="COLD",
    )


@router.post("/dream")
async def dream(user_id: str = Depends(require_auth)):
    """Trigger self-reflection / dreaming."""
    comps = _make_components(user_id)
    bucket_mgr = comps["bucket_mgr"]
    decay_engine = comps["decay_engine"]

    await decay_engine.ensure_started()
    all_buckets = await bucket_mgr.list_all(include_archive=False)

    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned")
        and not b["metadata"].get("protected")
    ]
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[:10]

    parts = []
    for b in recent:
        meta = b["metadata"]
        parts.append({
            "id": b["id"],
            "name": meta.get("name", ""),
            "domain": ",".join(meta.get("domain", [])),
            "valence": meta.get("valence", 0.5),
            "arousal": meta.get("arousal", 0.3),
            "resolved": meta.get("resolved", False),
            "content_preview": b["content"][:300],
            "created": meta.get("created", ""),
        })

    return {"recent_memories": parts, "message": "消化最近的记忆——有沉淀就写feel，能放下的就resolve。"}


@router.get("/stats/memory-count")
async def stats_memory_count(user_id: str = Depends(require_auth)):
    """Memory count for achievement system."""
    comps = _make_components(user_id)
    stats = await comps["bucket_mgr"].get_stats(user_id=user_id)
    return {
        "total": sum(stats[k] for k in ["permanent_count", "dynamic_count", "feel_count"]),
        "dynamic": stats.get("dynamic_count", 0),
        "permanent": stats.get("permanent_count", 0),
        "feel": stats.get("feel_count", 0),
    }


@router.get("/stats/emotion-curve")
async def stats_emotion_curve(period: str = "month", user_id: str = Depends(require_auth)):
    """Emotion curve data."""
    comps = _make_components(user_id)
    all_buckets = await comps["bucket_mgr"].list_all(include_archive=False)
    points = []
    for b in all_buckets:
        meta = b["metadata"]
        if meta.get("valence") is not None and meta.get("created"):
            points.append({
                "date": meta["created"][:10],
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
            })
    points.sort(key=lambda p: p["date"])
    return {"period": period, "points": points[-90:]}  # Last 90 data points


# ══════════════════════════════════════════════════════════
# Helper: merge_or_create
# ══════════════════════════════════════════════════════════

async def _merge_or_create(
    bucket_mgr, dehydrator, embedding_engine, config,
    content, tags, importance, domain, valence, arousal, name, user_id,
) -> tuple:
    """Check for similar bucket; merge or create. Returns (name/id, is_merged)."""
    try:
        existing = await bucket_mgr.search(content, limit=1, domain_filter=domain or None)
    except Exception:
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
        bucket = existing[0]
        if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=round((old_v + valence) / 2, 2),
                    arousal=round((old_a + arousal) / 2, 2),
                )
                try:
                    await embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                return bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed: {e}")

    bucket_id = await bucket_mgr.create(
        content=content, tags=tags, importance=importance,
        domain=domain, valence=valence, arousal=arousal,
        name=name or None,
    )
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return bucket_id, False
