"""
models.py — All Pydantic request/response models for the 5 endpoints.

Every field matches the challenge-testing-brief.md API contract exactly.
Extra unknown fields in incoming JSON are silently ignored (model_config).
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared extras config
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    model_config = {"extra": "ignore"}


# ===========================================================================
# POST /v1/context
# ===========================================================================

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class ContextPushRequest(_Base):
    scope: str
    context_id: str
    version: int = Field(ge=1)
    payload: dict[str, Any]
    delivered_at: str

    @model_validator(mode="after")
    def _validate_scope(self) -> "ContextPushRequest":
        if self.scope not in VALID_SCOPES:
            raise ValueError(
                f"Invalid scope '{self.scope}'. Must be one of: {sorted(VALID_SCOPES)}"
            )
        return self


class ContextAcceptedResponse(_Base):
    accepted: bool = True
    ack_id: str
    stored_at: str


class ContextRejectedResponse(_Base):
    accepted: bool = False
    reason: str
    current_version: Optional[int] = None
    details: Optional[str] = None


# ===========================================================================
# POST /v1/tick
# ===========================================================================

class TickRequest(_Base):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class TickAction(_Base):
    """A single proactive send action returned in /v1/tick."""
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str                           # "vera" | "merchant_on_behalf"
    trigger_id: str
    template_name: str
    template_params: list[str] = Field(default_factory=list)
    body: str
    cta: str                               # "binary_yes_no" | "open_ended" | "multi_choice_slot" | "none"
    suppression_key: str
    rationale: str


class TickResponse(_Base):
    actions: list[TickAction] = Field(default_factory=list)


# ===========================================================================
# POST /v1/reply
# ===========================================================================

class ReplyRequest(_Base):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str                         # "merchant" | "customer"
    message: str
    received_at: str
    turn_number: int = Field(ge=1)


class ReplyActionSend(_Base):
    action: str = "send"
    body: str
    cta: str
    rationale: str


class ReplyActionWait(_Base):
    action: str = "wait"
    wait_seconds: int
    rationale: str


class ReplyActionEnd(_Base):
    action: str = "end"
    rationale: str


# Union type hint — the actual response is one of these three dicts
ReplyResponse = dict[str, Any]


# ===========================================================================
# GET /v1/healthz
# ===========================================================================

class HealthzContextCounts(_Base):
    category: int = 0
    merchant: int = 0
    customer: int = 0
    trigger: int = 0


class HealthzResponse(_Base):
    status: str = "ok"
    uptime_seconds: int
    contexts_loaded: HealthzContextCounts


# ===========================================================================
# GET /v1/metadata
# ===========================================================================

class MetadataResponse(_Base):
    team_name: str
    team_members: list[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str
