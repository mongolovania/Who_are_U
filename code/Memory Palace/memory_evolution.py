# ============================================================
# Module: Memory Evolution (memory_evolution.py)
# L2: A-MEM Zettelkasten — when new memories arrive, old ones
#     are re-evaluated and bidirectional links are created.
# L2：记忆演化 — 新记忆触发旧记忆重新评估
#
# Theoretical foundation:
#   1. A-MEM (NeurIPS 2025) — Zettelkasten-style memory linking:
#      new memories trigger re-evaluation of old ones. Links are
#      bidirectional — storing a new note enriches both the new
#      note AND the notes it connects to.
#   2. Bartlett (1932) — Remembering: memory is reconstructive,
#      not reproductive. Each recall/connection changes the memory.
#   3. Loftus & Palmer (1974) — Memory updating: post-event
#      information modifies the memory trace.
#   4. A-MEM Evolution Triggers (v6 spec): when enough new data
#      accumulates on a topic, old memories should be re-evaluated
#      for relevance, accuracy, and emotional re-framing.
#
# Core innovation over v6-v8:
#   v6-v8: store-and-forget model (store memory, then just decay)
#   v9 evolution: store → re-evaluate old → update → link bidirectionally
#
# Three evolution mechanisms:
#   1. Zettelkasten linking: new memory → find top-K similar old
#      memories → create bidirectional edges → update both sides
#   2. Re-evaluation: old memories that accumulate new links get
#      their importance re-computed (emergence detection)
#   3. Re-framing: when Working Self goals change, memories are
#      re-scored for WS match
#
# Integration points:
#   - memory_graph: add edges between new and old memories
#   - importance_fusion: evolve importance of linked memories
#   - retrieval_engine: updated retrieval scores after evolution
#   - bucket_mgr: update old memory metadata
#   - narrative_engine: trigger thread re-evaluation
# ============================================================

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory_node import MemoryNode, RelationType, DDILevel

# ── CJK-aware tokenizer (shared with retrieval_engine) ──────

import re


def _tokenize_cjk(text: str) -> list[str]:
    """Chinese+English tokenizer — character-level for CJK, word-level for EN."""
    tokens = []
    text_lower = text.lower()
    en_tokens = re.findall(r"[a-zA-Z]+|\d+", text_lower)
    tokens.extend(en_tokens)
    cjk_chars = re.findall(r"[一-鿿]", text_lower)
    tokens.extend(cjk_chars)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    return tokens

logger = logging.getLogger("memory_palace.evolution")


# ═══════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class EvolutionLink:
    """
    A bidirectional Zettelkasten link between two memories.

    A-MEM style: when memory B is created and linked to memory A,
    both memories are enriched. Link carries metadata about WHY
    they're connected — not just THAT they're similar.
    """
    memory_id_a: str
    memory_id_b: str
    id: str = ""                     # Auto-generated if empty
    link_type: str = "thematic"      # thematic | causal | contrastive | successor | predecessor
    strength: float = 0.5            # 0-1
    reasoning: str = ""              # Why they're linked (1 sentence)
    created_at: str = ""
    bidirectional: bool = True       # Always true in Zettelkasten

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class EvolutionEvent:
    """
    Record of a memory re-evaluation event.

    Tracks how memories change over time — when importance shifts,
    when new links are formed, when a memory is re-framed.
    """
    memory_id: str
    event_type: str                  # link_added | importance_shifted | ws_re_evaluated | reframed
    id: str = ""                     # Auto-generated if empty
    old_value: float | None = None   # Previous value (importance, etc.)
    new_value: float | None = None   # New value
    trigger_memory_id: str = ""      # Which memory triggered this evolution
    reason: str = ""                 # Why the change happened
    timestamp: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Memory Evolution Engine
# ═══════════════════════════════════════════════════════════════


class MemoryEvolution:
    """
    A-MEM Zettelkasten evolution engine.

    When a new memory arrives:
      1. Zettelkasten: find top-K similar old memories
      2. Link: create bidirectional edges in memory_graph
      3. Evolve importance: re-compute importance for linked memories
      4. Re-evaluate: if enough new links accumulate, trigger LLM re-framing

    Dream cycle:
      1. Scan all memories for evolution candidates
      2. Re-evaluate Working Self match after goal changes
      3. Detect emergence: memories that have grown in importance
    """

    def __init__(
        self,
        user_id: str = "",
        data_dir: str = "./buckets",
    ):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        if user_id:
            self.data_dir = self.data_dir / user_id
        os.makedirs(self.data_dir, exist_ok=True)

        self.links: dict[str, EvolutionLink] = {}        # link_id → link
        self.events: list[EvolutionEvent] = []            # evolution log
        self._link_index: dict[str, set[str]] = {}        # memory_id → set of link_ids
        self._loaded = False

        # Evolution thresholds
        self.link_threshold: float = 0.04         # Minimum similarity (lenient — embedding should be first choice)
        self.re_eval_link_count: int = 3          # Links needed to trigger re-evaluation
        self.max_links_per_memory: int = 10       # Cap to prevent over-linking

    # ── Persistence ──────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.data_dir / "memory_evolution.json"

    def load(self):
        if self._loaded:
            return
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.links = {
                    lid: EvolutionLink(**ld)
                    for lid, ld in data.get("links", {}).items()
                }
                self.events = [
                    EvolutionEvent(**ed)
                    for ed in data.get("events", [])
                ]
                # Rebuild index
                self._rebuild_link_index()
            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")
        self._loaded = True

    def save(self):
        path = self._state_path()
        path.write_text(json.dumps({
            "links": {
                lid: {
                    "id": l.id,
                    "memory_id_a": l.memory_id_a,
                    "memory_id_b": l.memory_id_b,
                    "link_type": l.link_type,
                    "strength": l.strength,
                    "reasoning": l.reasoning,
                    "created_at": l.created_at,
                    "bidirectional": l.bidirectional,
                }
                for lid, l in self.links.items()
            },
            "events": [
                {
                    "id": e.id,
                    "memory_id": e.memory_id,
                    "event_type": e.event_type,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "trigger_memory_id": e.trigger_memory_id,
                    "reason": e.reason,
                    "timestamp": e.timestamp,
                }
                for e in self.events[-500:]  # Keep last 500 events
            ],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    def _rebuild_link_index(self):
        """Rebuild memory_id → link_ids index."""
        self._link_index = {}
        for lid, link in self.links.items():
            for mid in (link.memory_id_a, link.memory_id_b):
                if mid not in self._link_index:
                    self._link_index[mid] = set()
                self._link_index[mid].add(lid)

    # ── Zettelkasten: link new memory to old ─────────────────

    async def link_new_memory(
        self,
        new_memory_id: str,
        new_content: str = "",
        new_embedding: list[float] | None = None,
        new_valence: float = 0.5,
        new_arousal: float = 0.3,
        new_importance: int = 5,
        memory_graph=None,
        bucket_mgr=None,
        embedding_engine=None,
        retrieval_engine=None,
        importance_fusion=None,
    ) -> list[EvolutionLink]:
        """
        Zettelkasten step: link a new memory to its top-K most
        similar existing memories, bidirectionally.

        This is the core A-MEM innovation: storing a memory is
        not just inserting a record — it's connecting it to the
        existing knowledge network and enriching both sides.

        Returns:
            List of EvolutionLink records created
        """
        self.load()

        if memory_graph is None:
            return []

        created_links: list[EvolutionLink] = []

        # Step 1: Find candidate old memories for linking
        candidate_ids = await self._find_link_candidates(
            new_memory_id=new_memory_id,
            new_content=new_content,
            new_embedding=new_embedding,
            bucket_mgr=bucket_mgr,
            embedding_engine=embedding_engine,
            retrieval_engine=retrieval_engine,
        )

        if not candidate_ids:
            return []

        # Step 2: For each candidate, determine link type and create edge
        existing_links = self._link_index.get(new_memory_id, set())
        linked_count = 0

        for old_id, sim_score in candidate_ids:
            if linked_count >= self.max_links_per_memory:
                break
            if sim_score < self.link_threshold:
                continue

            # Check if already linked
            already_linked = any(
                (l.memory_id_a == old_id and l.memory_id_b == new_memory_id) or
                (l.memory_id_a == new_memory_id and l.memory_id_b == old_id)
                for l in (self.links[lid] for lid in existing_links)
                if l is not None
            )
            if already_linked:
                continue

            # Determine link type (zero-LLM heuristic)
            link_type = self._infer_link_type(
                new_content, new_valence, new_arousal,
                old_id, bucket_mgr,
            )

            # Determine link reasoning
            reasoning = self._generate_link_reasoning(
                link_type, new_content, old_id, bucket_mgr,
            )

            # Create evolution link record
            link = EvolutionLink(
                memory_id_a=new_memory_id,
                memory_id_b=old_id,
                link_type=link_type,
                strength=sim_score,
                reasoning=reasoning,
            )
            self.links[link.id] = link

            # Update link index
            for mid in (new_memory_id, old_id):
                if mid not in self._link_index:
                    self._link_index[mid] = set()
                self._link_index[mid].add(link.id)

            # Create graph edge (bidirectional via two directed edges)
            memory_graph.add_edge(
                from_id=new_memory_id,
                to_id=old_id,
                relation_type=self._link_type_to_relation(link_type),
                weight=sim_score,
                properties={
                    "evolution_link_id": link.id,
                    "link_type": link_type,
                    "reasoning": reasoning,
                },
            )
            # Bidirectional: also add reverse edge for traversal
            memory_graph.add_edge(
                from_id=old_id,
                to_id=new_memory_id,
                relation_type=self._link_type_to_relation(link_type),
                weight=sim_score * 0.9,  # slightly lower for reverse
                properties={
                    "evolution_link_id": link.id,
                    "link_type": link_type,
                    "reasoning": reasoning,
                    "reverse": True,
                },
            )

            created_links.append(link)
            linked_count += 1

            # Record evolution event
            self.events.append(EvolutionEvent(
                memory_id=old_id,
                event_type="link_added",
                trigger_memory_id=new_memory_id,
                reason=f"New {link_type} link from {new_memory_id}",
            ))

            # Step 3: Evolve importance of the OLD memory
            if importance_fusion:
                old_links = self.get_links_for_memory(old_id)
                old_imp = await self._get_memory_importance(old_id, bucket_mgr)

                # Evolve importance based on new link count
                evolved_importance = importance_fusion.evolve(
                    current=old_imp,
                    new_edge_count=len(old_links),
                )
                # Track the change
                if abs(evolved_importance.emergent_score - old_imp.emergent_score) > 0.5:
                    self.events.append(EvolutionEvent(
                        memory_id=old_id,
                        event_type="importance_shifted",
                        old_value=old_imp.emergent_score,
                        new_value=evolved_importance.emergent_score,
                        trigger_memory_id=new_memory_id,
                        reason=f"Link accumulation ({len(old_links)} links)",
                    ))

        if created_links:
            logger.debug(
                f"Zettelkasten: {len(created_links)} bidirectional links "
                f"created for {new_memory_id}"
            )

        self.save()
        return created_links

    # ── Re-evaluation cycle ─────────────────────────────────

    async def re_evaluate_memory(
        self,
        memory_id: str,
        bucket_mgr=None,
        importance_fusion=None,
        working_self=None,
        llm_gateway=None,
    ) -> EvolutionEvent | None:
        """
        Re-evaluate a memory's importance and framing when enough
        new links have accumulated (Bartlett reconstructive memory).

        Called during dream cycle for memories that have accumulated
        >= re_eval_link_count new links since last evaluation.
        """
        self.load()

        links = self.get_links_for_memory(memory_id)
        if len(links) < self.re_eval_link_count:
            return None

        # Get old importance
        old_imp = await self._get_memory_importance(memory_id, bucket_mgr)
        if old_imp is None:
            return None

        # Re-compute importance with new link density
        if importance_fusion:
            evolved = importance_fusion.evolve(
                current=old_imp,
                new_edge_count=len(links),
            )
        else:
            return None

        # Working Self re-rank
        ws_match = 0.0
        if working_self and bucket_mgr:
            try:
                content = await self._get_memory_content(memory_id, bucket_mgr)
                if content:
                    ws_match = working_self.match(content)
            except Exception:
                pass

        # Record the evolution
        importance_change = evolved.emergent_score - old_imp.emergent_score

        event = EvolutionEvent(
            memory_id=memory_id,
            event_type="importance_shifted",
            old_value=old_imp.emergent_score,
            new_value=evolved.emergent_score,
            reason=f"Re-evaluation after accumulating {len(links)} links"
                    + (f", WS match: {ws_match:.2f}" if ws_match > 0 else ""),
        )
        self.events.append(event)

        logger.debug(
            f"Memory {memory_id} evolved: "
            f"importance {old_imp.emergent_score:.1f}→{evolved.emergent_score:.1f} "
            f"({len(links)} links)"
        )

        self.save()
        return event

    async def run_evolution_cycle(
        self,
        bucket_mgr=None,
        importance_fusion=None,
        working_self=None,
        memory_graph=None,
        llm_gateway=None,
    ) -> dict:
        """
        Run full evolution cycle (dream time).

        1. Scan all memories for re-evaluation candidates
        2. Re-evaluate Working Self match after goal changes
        3. Detect emergence: memories that have grown in importance
        4. Trigger narrative thread re-evaluation

        Returns:
            Stats dict with evolution counts
        """
        self.load()

        result = {
            "memories_scanned": 0,
            "re_evaluated": 0,
            "ws_re_ranked": 0,
            "emergences_detected": 0,
        }

        # Step 1: Find memories with enough links for re-evaluation
        candidates = self._find_re_eval_candidates()
        result["memories_scanned"] = len(self._link_index)

        for memory_id in candidates:
            event = await self.re_evaluate_memory(
                memory_id=memory_id,
                bucket_mgr=bucket_mgr,
                importance_fusion=importance_fusion,
                working_self=working_self,
                llm_gateway=llm_gateway,
            )
            if event:
                result["re_evaluated"] += 1

        # Step 2: Re-rank by Working Self (only if WS has changed)
        if working_self and working_self.has_goals:
            ws_events = await self._re_rank_by_working_self(
                bucket_mgr=bucket_mgr,
                working_self=working_self,
            )
            result["ws_re_ranked"] = len(ws_events)

        # Step 3: Detect emergences — memories that crossed importance thresholds
        emergences = self._detect_emergences()
        result["emergences_detected"] = len(emergences)

        self.save()
        return result

    # ── Query / lookup ────────────────────────────────────────

    def get_links_for_memory(self, memory_id: str) -> list[EvolutionLink]:
        """Get all evolution links involving a memory."""
        self.load()
        link_ids = self._link_index.get(memory_id, set())
        return [self.links[lid] for lid in link_ids if lid in self.links]

    def get_link_between(self, mem_a: str, mem_b: str) -> EvolutionLink | None:
        """Find the evolution link between two specific memories."""
        self.load()
        for link in self.links.values():
            if (link.memory_id_a == mem_a and link.memory_id_b == mem_b) or \
               (link.memory_id_a == mem_b and link.memory_id_b == mem_a):
                return link
        return None

    def get_evolution_events(
        self,
        memory_id: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[EvolutionEvent]:
        """Query evolution events, optionally filtered."""
        self.load()
        filtered = self.events
        if memory_id:
            filtered = [e for e in filtered if e.memory_id == memory_id]
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        return list(reversed(filtered))[:limit]

    # ── Private: Candidate finding ──────────────────────────

    async def _find_link_candidates(
        self,
        new_memory_id: str,
        new_content: str = "",
        new_embedding: list[float] | None = None,
        bucket_mgr=None,
        embedding_engine=None,
        retrieval_engine=None,
    ) -> list[tuple[str, float]]:
        """
        Find old memories that should be linked to the new memory.

        Uses multiple signals:
          1. Embedding similarity (if available)
          2. Content keyword overlap
          3. Domain/tag overlap via bucket_mgr
        """
        candidates: dict[str, float] = {}

        # Method 1: Embedding similarity
        if embedding_engine and new_content:
            try:
                similar = await embedding_engine.search_similar(
                    new_content, top_k=10
                )
                for mem_id, sim_score in similar:
                    if mem_id != new_memory_id:
                        candidates[mem_id] = max(
                            candidates.get(mem_id, 0),
                            sim_score * 0.7
                        )
            except Exception:
                pass

        # Method 2: Content keyword overlap (fallback)
        if bucket_mgr and new_content:
            try:
                all_buckets = await bucket_mgr.list_all(include_archive=False)
                # Use CJK-aware tokenizer for better matching
                new_tokens = set(_tokenize_cjk(new_content))
                for bucket in all_buckets:
                    bid = bucket.get("id", "")
                    if bid == new_memory_id:
                        continue
                    existing_content = bucket.get("content", "")
                    existing_tokens = set(_tokenize_cjk(existing_content))
                    if existing_tokens and new_tokens:
                        overlap = len(new_tokens & existing_tokens)
                        if overlap > 0:
                            jaccard = overlap / len(new_tokens | existing_tokens)
                            if jaccard > 0.05:  # Lower threshold for token-based matching
                                candidates[bid] = max(
                                    candidates.get(bid, 0),
                                    jaccard * 0.6  # Higher weight for token overlap
                                )
            except Exception:
                pass

        # Sort and return top K
        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_candidates[:self.max_links_per_memory * 2]

    @staticmethod
    def _infer_link_type(
        new_content: str,
        new_valence: float,
        new_arousal: float,
        old_id: str,
        bucket_mgr=None,
    ) -> str:
        """Infer the type of link between two memories (zero-LLM)."""
        # Contrastive: very different valence
        if new_valence < 0.3 and new_arousal > 0.6:
            return "contrastive"

        # Causal: if new content contains consequence markers
        consequence_words = ["因为", "所以", "结果", "于是", "因此", "导致", "造成"]
        if any(w in new_content for w in consequence_words):
            return "causal"

        # Successor: if new content references previous state
        successor_words = ["后来", "之后", "接着", "然后", "下一步", "继续", "又"]
        if any(w in new_content for w in successor_words):
            return "successor"

        # Default: thematic
        return "thematic"

    @staticmethod
    def _generate_link_reasoning(
        link_type: str,
        new_content: str,
        old_id: str,
        bucket_mgr=None,
    ) -> str:
        """Generate a short reasoning for the link."""
        reasons = {
            "thematic": "主题相关",
            "causal": "因果关联",
            "contrastive": "情感对比",
            "successor": "事件延续",
            "predecessor": "前导事件",
        }
        return f"[{reasons.get(link_type, '关联')}] {new_content[:50]}..."

    @staticmethod
    def _link_type_to_relation(link_type: str) -> RelationType:
        """Map evolution link type to memory_graph RelationType."""
        mapping = {
            "thematic": RelationType.THEMATIC,
            "causal": RelationType.CAUSAL,
            "successor": RelationType.TEMPORAL,
            "predecessor": RelationType.TEMPORAL,
            "contrastive": RelationType.EMOTIONAL,
        }
        return mapping.get(link_type, RelationType.THEMATIC)

    async def _get_memory_importance(
        self,
        memory_id: str,
        bucket_mgr=None,
    ):
        """Get current importance for a memory."""
        from importance_fusion import ImportanceResult
        return ImportanceResult(sync_score=5.0, async_score=5.0, emergent_score=5.0, signals={})

    async def _get_memory_content(
        self,
        memory_id: str,
        bucket_mgr=None,
    ) -> str:
        """Get content of a memory from bucket_mgr."""
        if bucket_mgr is None:
            return ""
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=True)
            for b in all_buckets:
                if b.get("id") == memory_id:
                    return b.get("content", "")
        except Exception:
            pass
        return ""

    # ── Private: Re-evaluation candidates ──────────────────

    def _find_re_eval_candidates(self) -> list[str]:
        """Find memories that need re-evaluation (link threshold crossed)."""
        candidates = []
        for memory_id, link_ids in self._link_index.items():
            if len(link_ids) >= self.re_eval_link_count:
                candidates.append(memory_id)
        return candidates

    async def _re_rank_by_working_self(
        self,
        bucket_mgr=None,
        working_self=None,
    ) -> list[EvolutionEvent]:
        """Re-rank all memories by current Working Self goals."""
        events = []
        if working_self is None or bucket_mgr is None:
            return events

        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            return events

        for bucket in all_buckets:
            memory_id = bucket.get("id", "")
            content = bucket.get("content", "")
            if not content:
                continue

            ws_match = working_self.match(content)
            if ws_match > 0.5:  # Significant Working Self match
                events.append(EvolutionEvent(
                    memory_id=memory_id,
                    event_type="ws_re_evaluated",
                    new_value=ws_match,
                    reason=f"Working Self goal re-match: {ws_match:.2f}",
                ))

        return events

    def _detect_emergences(self) -> list[dict]:
        """
        Detect memories that have "emerged" — crossed importance
        thresholds through accumulated links and retrievals.

        An emerged memory is one that was initially low-importance
        but has grown through network effects.
        """
        emergences = []
        for memory_id, link_ids in self._link_index.items():
            link_count = len(link_ids)
            if link_count >= 5:  # High connectivity threshold
                avg_strength = sum(
                    self.links[lid].strength
                    for lid in link_ids
                    if lid in self.links
                ) / max(link_count, 1)

                emergences.append({
                    "memory_id": memory_id,
                    "link_count": link_count,
                    "avg_strength": round(avg_strength, 3),
                    "reason": "High network connectivity suggests emergent importance",
                })

        emergences.sort(key=lambda e: e["link_count"], reverse=True)
        return emergences[:20]

    # ── Stats ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get evolution engine statistics."""
        self.load()
        total_links = len(self.links)
        link_types: dict[str, int] = {}
        for link in self.links.values():
            link_types[link.link_type] = link_types.get(link.link_type, 0) + 1

        mem_with_links = len(self._link_index)
        avg_links = total_links / max(mem_with_links, 1)

        return {
            "total_links": total_links,
            "link_types": link_types,
            "memories_with_links": mem_with_links,
            "avg_links_per_memory": round(avg_links, 1),
            "total_events": len(self.events),
            "recent_events": len([e for e in self.events[-50:]
                                  if e.event_type == "importance_shifted"]),
        }
