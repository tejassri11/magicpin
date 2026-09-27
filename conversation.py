"""
conversation.py — Per-conversation state management.

Each conversation is started by the bot in /v1/tick and updated via /v1/reply.
The state tracks turn history, auto-reply counts, and lifecycle status.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from logger import log


# ---------------------------------------------------------------------------
# Turn record
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    from_role: str        # "bot" | "merchant" | "customer"
    body: str
    ts: str               # ISO timestamp
    action: Optional[str] = None   # "send" | "wait" | "end" (bot turns only)


# ---------------------------------------------------------------------------
# Conversation lifecycle
# ---------------------------------------------------------------------------
CONV_STATUS_ACTIVE = "active"
CONV_STATUS_WAITING = "waiting"
CONV_STATUS_AUTO_REPLY_WAIT = "auto_reply_wait"
CONV_STATUS_ENDED = "ended"


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    trigger_id: str
    trigger_kind: str
    send_as: str                      # "vera" | "merchant_on_behalf"
    status: str = CONV_STATUS_ACTIVE
    turns: list[Turn] = field(default_factory=list)
    auto_reply_count: int = 0         # consecutive identical auto-replies
    auto_reply_wait_until: float = 0.0
    last_bot_body: str = ""           # for anti-repetition checks
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_activity_at: float = field(default_factory=time.time)

    # -----------------------------------------------------------------
    # Turn helpers
    # -----------------------------------------------------------------

    def add_bot_turn(self, body: str, action: str = "send") -> None:
        self.turns.append(
            Turn(
                from_role="bot",
                body=body,
                ts=datetime.now(timezone.utc).isoformat(),
                action=action,
            )
        )
        self.last_bot_body = body
        self.last_activity_at = time.time()

    def add_merchant_turn(self, body: str) -> None:
        self.turns.append(
            Turn(
                from_role="merchant",
                body=body,
                ts=datetime.now(timezone.utc).isoformat(),
            )
        )
        self.last_activity_at = time.time()

    def merchant_messages(self) -> list[str]:
        return [t.body for t in self.turns if t.from_role == "merchant"]

    def bot_messages(self) -> list[str]:
        return [t.body for t in self.turns if t.from_role == "bot"]

    def total_turns(self) -> int:
        return len(self.turns)

    def end(self) -> None:
        self.status = CONV_STATUS_ENDED
        self.last_activity_at = time.time()

    def wait(self) -> None:
        self.status = CONV_STATUS_WAITING
        self.last_activity_at = time.time()

    def set_auto_reply_wait(self, wait_seconds: int = 86400) -> None:
        self.status = CONV_STATUS_AUTO_REPLY_WAIT
        self.auto_reply_wait_until = time.time() + wait_seconds
        self.last_activity_at = time.time()

    def resume(self) -> None:
        self.status = CONV_STATUS_ACTIVE
        self.auto_reply_count = 0
        self.auto_reply_wait_until = 0.0
        self.last_activity_at = time.time()

    def is_active(self) -> bool:
        return self.status == CONV_STATUS_ACTIVE

    def is_ended(self) -> bool:
        return self.status == CONV_STATUS_ENDED

    def is_waiting(self) -> bool:
        if self.status in (CONV_STATUS_WAITING, CONV_STATUS_AUTO_REPLY_WAIT):
            if self.auto_reply_wait_until > 0 and time.time() >= self.auto_reply_wait_until:
                return False
            return True
        return False

    def recent_context(self, n: int = 4) -> list[dict[str, Any]]:
        """Return last N turns as plain dicts for LLM prompt injection."""
        return [
            {"from": t.from_role, "body": t.body, "ts": t.ts}
            for t in self.turns[-n:]
        ]


# ---------------------------------------------------------------------------
# Conversation registry
# ---------------------------------------------------------------------------
class ConversationStore:
    """Thread-safe registry of all conversations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._convs: dict[str, ConversationState] = {}
        self._merchant_wait_until: dict[str, float] = {}
        self._merchant_auto_reply_counts: dict[str, int] = {}

    def start(
        self,
        conversation_id: str,
        merchant_id: str,
        trigger_id: str,
        trigger_kind: str,
        send_as: str,
        bot_body: str,
        customer_id: Optional[str] = None,
    ) -> ConversationState:
        """Create and register a new conversation, recording the opening bot turn."""
        conv = ConversationState(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            trigger_id=trigger_id,
            trigger_kind=trigger_kind,
            send_as=send_as,
        )
        conv.add_bot_turn(bot_body, action="send")

        with self._lock:
            self._convs[conversation_id] = conv

        log.info(
            "conversation started id=%s merchant=%s trigger=%s",
            conversation_id, merchant_id, trigger_id,
        )
        return conv

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        with self._lock:
            return self._convs.get(conversation_id)

    def increment_merchant_auto_reply_count(self, merchant_id: str) -> int:
        with self._lock:
            cnt = self._merchant_auto_reply_counts.get(merchant_id, 0) + 1
            self._merchant_auto_reply_counts[merchant_id] = cnt
            return cnt

    def get_merchant_auto_reply_count(self, merchant_id: str) -> int:
        with self._lock:
            return self._merchant_auto_reply_counts.get(merchant_id, 0)

    def set_merchant_auto_reply_wait(self, merchant_id: str, wait_seconds: int = 86400) -> None:
        with self._lock:
            self._merchant_wait_until[merchant_id] = time.time() + wait_seconds

    def is_merchant_waiting(self, merchant_id: str) -> bool:
        with self._lock:
            wait_until = self._merchant_wait_until.get(merchant_id, 0.0)
            if wait_until > 0:
                if time.time() < wait_until:
                    return True
                else:
                    # Wait expired
                    self._merchant_wait_until.pop(merchant_id, None)

            for conv in self._convs.values():
                if conv.merchant_id == merchant_id and not conv.is_ended():
                    if conv.is_waiting():
                        return True
            return False

    def clear_merchant_wait(self, merchant_id: str) -> None:
        with self._lock:
            self._merchant_wait_until.pop(merchant_id, None)
            self._merchant_auto_reply_counts.pop(merchant_id, None)
            for conv in self._convs.values():
                if conv.merchant_id == merchant_id and not conv.is_ended():
                    conv.resume()

    def exists_for_trigger(self, merchant_id: str, trigger_id: str) -> bool:
        """Return True if a non-ended conversation already exists for this (merchant, trigger)."""
        with self._lock:
            for conv in self._convs.values():
                if (
                    conv.merchant_id == merchant_id
                    and conv.trigger_id == trigger_id
                    and not conv.is_ended()
                ):
                    return True
        return False

    def has_active_conversation_for_merchant(self, merchant_id: str) -> bool:
        """Return True if any non-ended conversation exists for this merchant."""
        with self._lock:
            for conv in self._convs.values():
                if conv.merchant_id == merchant_id and not conv.is_ended():
                    return True
        return False

    def list_active_for_merchant(self, merchant_id: str) -> list[ConversationState]:
        with self._lock:
            return [
                c for c in self._convs.values()
                if c.merchant_id == merchant_id and c.is_active()
            ]

    def end(self, conversation_id: str) -> None:
        with self._lock:
            conv = self._convs.get(conversation_id)
            if conv:
                conv.end()
        log.info("conversation ended id=%s", conversation_id)

    def wipe(self) -> None:
        with self._lock:
            self._convs.clear()
            self._merchant_wait_until.clear()
            self._merchant_auto_reply_counts.clear()
        log.info("conversation store wiped")

    def count(self) -> int:
        with self._lock:
            return len(self._convs)

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            result: dict[str, int] = {
                CONV_STATUS_ACTIVE: 0,
                CONV_STATUS_WAITING: 0,
                CONV_STATUS_AUTO_REPLY_WAIT: 0,
                CONV_STATUS_ENDED: 0,
            }
            for c in self._convs.values():
                result[c.status] = result.get(c.status, 0) + 1
            return result


# Singleton
conversations = ConversationStore()

