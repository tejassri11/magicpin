"""
state.py — Thread-safe in-memory state store.

Stores four families of context (category / merchant / customer / trigger)
and provides atomic versioned replace, suppression tracking, and opt-out lists.

All public methods acquire a lock before reading or writing — safe for
concurrent FastAPI async handlers running in a threadpool.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from logger import log


# ---------------------------------------------------------------------------
# Versioned context entry
# ---------------------------------------------------------------------------
@dataclass
class VersionedContext:
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    stored_at: str


# ---------------------------------------------------------------------------
# Suppression record
# ---------------------------------------------------------------------------
@dataclass
class SuppressionRecord:
    key: str
    fired_at: str
    conversation_id: str


# ---------------------------------------------------------------------------
# Merchant opt-out record
# ---------------------------------------------------------------------------
@dataclass
class OptOutRecord:
    merchant_id: str
    opted_out_at: float   # unix timestamp
    cooldown_days: int    # how long to honour the opt-out


# ---------------------------------------------------------------------------
# Thread-safe state store
# ---------------------------------------------------------------------------
class StateStore:
    """
    Single global store for all bot state.

    Layout:
        _contexts[(scope, context_id)] = VersionedContext
        _suppressions[suppression_key] = SuppressionRecord
        _optouts[merchant_id] = OptOutRecord
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._contexts: dict[tuple[str, str], VersionedContext] = {}
        self._suppressions: dict[str, SuppressionRecord] = {}
        self._optouts: dict[str, OptOutRecord] = {}

    # ------------------------------------------------------------------
    # Context storage
    # ------------------------------------------------------------------

    def put_context(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict[str, Any],
    ) -> tuple[bool, Optional[int]]:
        """
        Store or upgrade a versioned context.

        Returns:
            (True, None)      — accepted, stored
            (False, current)  — rejected; current_version returned
        """
        key = (scope, context_id)
        stored_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing = self._contexts.get(key)
            if existing is not None:
                if version < existing.version:
                    log.debug(
                        "context rejected (stale) scope=%s id=%s incoming_v=%s current_v=%s",
                        scope, context_id, version, existing.version,
                    )
                    return False, existing.version
                elif version == existing.version:
                    log.debug(
                        "context idempotent accept scope=%s id=%s version=%s",
                        scope, context_id, version,
                    )
                    return True, existing.version

            self._contexts[key] = VersionedContext(
                scope=scope,
                context_id=context_id,
                version=version,
                payload=payload,
                stored_at=stored_at,
            )
            log.info(
                "context stored scope=%s id=%s version=%s", scope, context_id, version
            )
            return True, None

    def get_context(self, scope: str, context_id: str) -> Optional[dict[str, Any]]:
        """Return the payload dict for (scope, context_id) or None."""
        with self._lock:
            entry = self._contexts.get((scope, context_id))
            return entry.payload if entry else None

    def get_context_entry(
        self, scope: str, context_id: str
    ) -> Optional[VersionedContext]:
        """Return the full VersionedContext entry or None."""
        with self._lock:
            return self._contexts.get((scope, context_id))

    def count_by_scope(self) -> dict[str, int]:
        """Return {scope: count} for the /healthz endpoint."""
        counts: dict[str, int] = {
            "category": 0,
            "merchant": 0,
            "customer": 0,
            "trigger": 0,
        }
        with self._lock:
            for (scope, _) in self._contexts:
                if scope in counts:
                    counts[scope] += 1
        return counts

    def list_context_ids(self, scope: str) -> list[str]:
        """Return all context_ids for a given scope."""
        with self._lock:
            return [cid for (s, cid) in self._contexts if s == scope]

    # ------------------------------------------------------------------
    # Suppression
    # ------------------------------------------------------------------

    def fire_suppression(self, key: str, conversation_id: str) -> None:
        """Mark a suppression_key as fired."""
        with self._lock:
            self._suppressions[key] = SuppressionRecord(
                key=key,
                fired_at=datetime.now(timezone.utc).isoformat(),
                conversation_id=conversation_id,
            )
        log.debug("suppression fired key=%s conv=%s", key, conversation_id)

    def is_suppressed(self, key: str) -> bool:
        """Return True if suppression_key has already been fired."""
        with self._lock:
            return key in self._suppressions

    def get_suppression(self, key: str) -> Optional[SuppressionRecord]:
        with self._lock:
            return self._suppressions.get(key)

    # ------------------------------------------------------------------
    # Merchant opt-out
    # ------------------------------------------------------------------

    def add_optout(self, merchant_id: str, cooldown_days: int = 30) -> None:
        """Record that a merchant opted out; suppress for cooldown_days."""
        with self._lock:
            self._optouts[merchant_id] = OptOutRecord(
                merchant_id=merchant_id,
                opted_out_at=time.time(),
                cooldown_days=cooldown_days,
            )
        log.info("merchant opted-out merchant_id=%s cooldown_days=%s", merchant_id, cooldown_days)

    def is_opted_out(self, merchant_id: str) -> bool:
        """
        Return True if merchant is in the opt-out cooldown window.
        Automatically expires old opt-outs.
        """
        with self._lock:
            record = self._optouts.get(merchant_id)
            if record is None:
                return False
            elapsed_days = (time.time() - record.opted_out_at) / 86400
            if elapsed_days > record.cooldown_days:
                del self._optouts[merchant_id]
                return False
            return True

    def remove_optout(self, merchant_id: str) -> None:
        """Clear opt-out status when a merchant sends a positive human response."""
        with self._lock:
            if self._optouts.pop(merchant_id, None) is not None:
                log.info("merchant opt-out removed merchant_id=%s", merchant_id)

    def optout_days_remaining(self, merchant_id: str) -> float:
        """Return how many days remain in the opt-out period (0 if not opted-out)."""
        with self._lock:
            record = self._optouts.get(merchant_id)
            if record is None:
                return 0.0
            elapsed_days = (time.time() - record.opted_out_at) / 86400
            remaining = record.cooldown_days - elapsed_days
            return max(0.0, remaining)

    # ------------------------------------------------------------------
    # Bulk reset (for teardown / testing)
    # ------------------------------------------------------------------

    def wipe(self) -> None:
        """Clear all state — called on POST /v1/teardown."""
        with self._lock:
            self._contexts.clear()
            self._suppressions.clear()
            self._optouts.clear()
        log.info("state store wiped")


# Singleton instance — import this everywhere
store = StateStore()
