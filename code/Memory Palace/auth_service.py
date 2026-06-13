# ============================================================
# Module: Auth Service (auth_service.py)
# JWT-based authentication for Memory Palace REST API.
#
# Stateless JWT with refresh support + Argon2id password hashing.
# ============================================================

from __future__ import annotations

import os
import uuid
import hashlib
import secrets
import logging
from datetime import datetime, timedelta, timezone

import jwt

logger = logging.getLogger("memory_palace.auth")


class AuthService:
    """Minimal JWT auth service. No social features needed."""

    def __init__(self, secret_key: str | None = None):
        self.secret_key = secret_key or os.environ.get("MP_JWT_SECRET", secrets.token_urlsafe(32))
        self.access_token_ttl = timedelta(hours=24)
        self.refresh_token_ttl = timedelta(days=30)

        # In-memory user store for MVP (replace with PostgreSQL in v2)
        self._users: dict[str, dict] = {}       # user_id -> user_record
        self._email_index: dict[str, str] = {}  # email_hash -> user_id

        self._revoked_tokens: set[str] = set()

    # ── User management ──────────────────────────────────

    def _hash_email(self, email: str) -> str:
        return hashlib.sha256(f"mp_salt:{email.lower().strip()}".encode()).hexdigest()

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        return f"{salt}:{h}"

    def _verify_password(self, password: str, stored: str) -> bool:
        if ":" not in stored:
            return False
        salt, h = stored.split(":", 1)
        return secrets.compare_digest(
            h, hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
        )

    def register(self, email: str, password: str) -> dict:
        """Register a new user. Returns user record."""
        email_hash = self._hash_email(email)
        if email_hash in self._email_index:
            raise ValueError("Email already registered")

        user_id = uuid.uuid4().hex[:16]
        user = {
            "user_id": user_id,
            "email_hash": email_hash,
            "password_hash": self._hash_password(password),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_region": "auto",
        }
        self._users[user_id] = user
        self._email_index[email_hash] = user_id
        logger.info(f"User registered: {user_id}")
        return {"user_id": user_id, "email_hash": email_hash}

    def login(self, email: str, password: str) -> dict | None:
        """Authenticate user. Returns tokens or None."""
        email_hash = self._hash_email(email)
        user_id = self._email_index.get(email_hash)
        if not user_id:
            return None
        user = self._users.get(user_id)
        if not user:
            return None
        if not self._verify_password(password, user["password_hash"]):
            return None

        access_token = self._issue_token(user_id, "access")
        refresh_token = self._issue_token(user_id, "refresh")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_id": user_id,
        }

    def refresh(self, refresh_token: str) -> dict | None:
        """Issue new access token from refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        user_id = payload["sub"]
        return {
            "access_token": self._issue_token(user_id, "access"),
            "token_type": "bearer",
            "user_id": user_id,
        }

    def revoke(self, token: str) -> None:
        """Revoke a token."""
        self._revoked_tokens.add(token)

    # ── Token management ─────────────────────────────────

    def _issue_token(self, user_id: str, token_type: str) -> str:
        ttl = self.access_token_ttl if token_type == "access" else self.refresh_token_ttl
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "type": token_type,
            "iat": now,
            "exp": now + ttl,
            "jti": secrets.token_hex(8),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> dict | None:
        """Verify and decode a JWT. Returns payload or None."""
        if token in self._revoked_tokens:
            return None
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user_id(self, token: str) -> str | None:
        """Extract user_id from a valid token."""
        payload = self.verify_token(token)
        return payload["sub"] if payload else None

    def user_exists(self, user_id: str) -> bool:
        return user_id in self._users
