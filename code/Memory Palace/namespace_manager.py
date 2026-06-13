# ============================================================
# Module: Namespace Manager (namespace_manager.py)
# Multi-user directory isolation.
#
# Each user gets:
#   buckets/{user_id}/
#     permanent/  dynamic/  feel/  archive/
#     embeddings.db  memory_edges.db
#     retrieval_counts.json  entity_index.json
#     structured_profile.json  flashbulb_index.json
#     decay_state.json  life_tick.log
# ============================================================

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger("memory_palace.namespace")


class NamespaceManager:
    """Manages per-user storage isolation."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        os.makedirs(self.base_dir, exist_ok=True)

    def user_dir(self, user_id: str) -> Path:
        """Get the root directory for a user."""
        return self.base_dir / user_id

    def user_buckets_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id)

    def ensure_user_namespace(self, user_id: str) -> Path:
        """Create all subdirectories for a user. Returns user root dir."""
        root = self.user_dir(user_id)
        for sub in ["permanent", "dynamic", "feel", "archive"]:
            os.makedirs(root / sub, exist_ok=True)
        return root

    def user_db_path(self, user_id: str, db_name: str = "embeddings.db") -> str:
        """Path to a user-specific SQLite database."""
        self.ensure_user_namespace(user_id)
        return str(self.user_dir(user_id) / db_name)

    def user_retrieval_counts_path(self, user_id: str) -> str:
        return str(self.user_dir(user_id) / "retrieval_counts.json")

    def user_profile_path(self, user_id: str) -> str:
        return str(self.user_dir(user_id) / "structured_profile.json")

    def user_entity_index_path(self, user_id: str) -> str:
        return str(self.user_dir(user_id) / "entity_index.json")

    def user_flashbulb_path(self, user_id: str) -> str:
        return str(self.user_dir(user_id) / "flashbulb_index.json")

    def user_decay_state_path(self, user_id: str) -> str:
        return str(self.user_dir(user_id) / "decay_state.json")

    def user_edges_db_path(self, user_id: str) -> str:
        return str(self.user_dir(user_id) / "memory_edges.db")

    def resolve(self, user_id: str) -> dict:
        """Resolve all paths for a user."""
        self.ensure_user_namespace(user_id)
        root = str(self.user_dir(user_id))
        return {
            "root": root,
            "buckets_dir": root,
            "permanent_dir": os.path.join(root, "permanent"),
            "dynamic_dir": os.path.join(root, "dynamic"),
            "feel_dir": os.path.join(root, "feel"),
            "archive_dir": os.path.join(root, "archive"),
            "db_path": self.user_db_path(user_id),
            "edges_db_path": self.user_edges_db_path(user_id),
            "retrieval_counts_path": self.user_retrieval_counts_path(user_id),
            "profile_path": self.user_profile_path(user_id),
            "entity_index_path": self.user_entity_index_path(user_id),
            "flashbulb_path": self.user_flashbulb_path(user_id),
            "decay_state_path": self.user_decay_state_path(user_id),
        }
