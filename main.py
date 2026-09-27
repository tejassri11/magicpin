"""
main.py — FastAPI application with all 5 required endpoints.

Endpoints (matching challenge-testing-brief.md exactly):
  POST /v1/context   — receive context push from judge
  POST /v1/tick      — periodic wake-up; bot decides what to send
  POST /v1/reply     — receive a merchant/customer reply
  GET  /v1/healthz   — liveness probe
  GET  /v1/metadata  — bot identity

Optional:
  POST /v1/teardown  — wipe all state (end-of-test cleanup)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import composer as composer_module
from config import settings
from conversation import conversations
from decision import evaluate_trigger, Decision
from logger import log
from models import (
    ContextPushRequest,
    HealthzContextCounts,
    HealthzResponse,
    MetadataResponse,
    ReplyRequest,
    TickAction,
    TickRequest,
    TickResponse,
)
from reply_handler import handle_reply
from state import store

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VERA Bot — magicpin AI Challenge",
    version=settings.bot_version,
    description="Merchant AI assistant backend implementing the 5-endpoint judge contract.",
)

START_TIME = time.time()


# ---------------------------------------------------------------------------
# Global exception handler — never crash on bad input
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    log.warning("validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"accepted": False, "reason": "invalid_request", "details": str(exc)},
    )


# ===========================================================================
# GET /v1/healthz
# ===========================================================================

@app.get("/v1/healthz", response_model=HealthzResponse)
async def healthz() -> dict[str, Any]:
    counts = store.count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {
            "category": counts.get("category", 0),
            "merchant": counts.get("merchant", 0),
            "customer": counts.get("customer", 0),
            "trigger": counts.get("trigger", 0),
        },
    }


# ===========================================================================
# GET /v1/metadata
# ===========================================================================

@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata() -> dict[str, Any]:
    return {
        "team_name": settings.team_name,
        "team_members": settings.team_members,
        "model": settings.llm_model,
        "approach": (
            "Trigger-kind-routed composer with grounded prompts, "
            "post-LLM validation, and multi-turn conversation state."
        ),
        "contact_email": settings.contact_email,
        "version": settings.bot_version,
        "submitted_at": settings.submitted_at,
    }


# ===========================================================================
# POST /v1/context
# ===========================================================================

@app.post("/v1/context")
async def push_context(request: Request) -> JSONResponse:
    """
    Receive a context push from the judge.
    Idempotent by (context_id, version).
    Higher version replaces lower atomically.
    Returns HTTP 200 always (judge checks 'accepted' field in JSON body).
    """
    try:
        raw_body = await request.json()
    except Exception as exc:
        log.warning("malformed JSON in /v1/context: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_json", "details": str(exc)},
        )

    try:
        body = ContextPushRequest.model_validate(raw_body)
    except ValidationError as exc:
        log.warning("validation error in /v1/context: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": str(exc)},
        )

    accepted, current_version = store.put_context(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
    )

    if not accepted:
        return JSONResponse(
            status_code=200,          # challenge spec uses 200 with accepted=false
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )

    stored_at = datetime.now(timezone.utc).isoformat()
    return JSONResponse(
        status_code=200,
        content={
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": stored_at,
        },
    )


# ===========================================================================
# POST /v1/tick
# ===========================================================================

@app.post("/v1/tick", response_model=TickResponse)
async def tick(request: Request) -> dict[str, Any]:
    """
    Periodic wake-up. Bot inspects available triggers and decides what to send.
    Must respond within 30s. Returns up to 20 actions.
    """
    try:
        raw_body = await request.json()
    except Exception as exc:
        log.warning("malformed JSON in /v1/tick: %s", exc)
        return {"actions": []}

    try:
        body = TickRequest.model_validate(raw_body)
    except ValidationError as exc:
        log.warning("validation error in /v1/tick: %s", exc)
        return {"actions": []}

    now_iso = body.now
    available_triggers = body.available_triggers

    log.info("tick received now=%s triggers=%d", now_iso, len(available_triggers))

    if not available_triggers:
        return {"actions": []}

    # Evaluate all triggers in parallel (within asyncio event loop)
    actions: list[dict[str, Any]] = []
    try:
        tasks = [
            _process_trigger(trigger_id, now_iso)
            for trigger_id in available_triggers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r and not isinstance(r, Exception):
                actions.append(r)
    except Exception as exc:
        log.error("tick processing error: %s", exc, exc_info=True)

    # Cap at 20 (challenge requirement)
    actions = actions[: settings.max_actions_per_tick]
    log.info("tick returning %d actions", len(actions))
    return {"actions": actions}


async def _process_trigger(trigger_id: str, now_iso: str) -> dict[str, Any] | None:
    """
    Evaluate a single trigger and compose a message if SEND decision.
    Returns a TickAction-shaped dict or None.
    """
    try:
        result = evaluate_trigger(trigger_id, now_iso)
        if not result.should_send:
            log.debug("SKIP trigger=%s reason=%s", trigger_id, result.skip_reason)
            return None

        trg = result.trigger_payload
        merchant = result.merchant_payload
        category = result.category_payload
        customer = result.customer_payload

        # Run compose in a thread (LLM is blocking I/O)
        loop = asyncio.get_event_loop()
        composed = await loop.run_in_executor(
            None,
            composer_module.compose,
            category,
            merchant,
            trg,
            customer,
        )

        if composed is None:
            log.warning("composer returned None for trigger=%s", trigger_id)
            return None

        # Generate conversation ID
        merchant_id = trg.get("merchant_id", "")
        customer_id = trg.get("customer_id")
        conv_id = f"conv_{merchant_id}_{trigger_id}_{_short_uid()}"

        # Register suppression
        store.fire_suppression(trg.get("suppression_key", ""), conv_id)

        # Register conversation
        conversations.start(
            conversation_id=conv_id,
            merchant_id=merchant_id,
            trigger_id=trigger_id,
            trigger_kind=trg.get("kind", ""),
            send_as=composed.send_as,
            bot_body=composed.body,
            customer_id=customer_id,
        )

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.send_as,
            "trigger_id": trigger_id,
            "template_name": composed.template_name,
            "template_params": composed.template_params,
            "body": composed.body,
            "cta": composed.cta,
            "suppression_key": composed.suppression_key,
            "rationale": composed.rationale,
        }

        log.info(
            "action composed trigger=%s merchant=%s conv=%s",
            trigger_id, merchant_id, conv_id,
        )
        return action

    except Exception as exc:
        log.error("error processing trigger=%s: %s", trigger_id, exc, exc_info=True)
        return None


# ===========================================================================
# POST /v1/reply
# ===========================================================================

@app.post("/v1/reply")
async def reply(request: Request) -> JSONResponse:
    """
    Receive a merchant/customer reply. Bot responds synchronously.
    Must respond within 30s.
    """
    try:
        raw_body = await request.json()
    except Exception as exc:
        log.warning("malformed JSON in /v1/reply: %s", exc)
        return JSONResponse(
            status_code=200,
            content={"action": "end", "rationale": "Malformed request body."},
        )

    try:
        body = ReplyRequest.model_validate(raw_body)
    except ValidationError as exc:
        log.warning("validation error in /v1/reply: %s", exc)
        return JSONResponse(
            status_code=200,
            content={"action": "end", "rationale": "Invalid request schema."},
        )

    log.info(
        "reply received conv=%s merchant=%s turn=%d",
        body.conversation_id, body.merchant_id, body.turn_number,
    )

    # Run in executor (reply handler may do blocking LLM call)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        handle_reply,
        body.conversation_id,
        body.merchant_id,
        body.message,
        body.turn_number,
        body.customer_id,
    )

    return JSONResponse(status_code=200, content=result)


# ===========================================================================
# POST /v1/teardown (optional — wipes all state)
# ===========================================================================

@app.post("/v1/teardown")
async def teardown() -> dict[str, Any]:
    """Wipe all state at end of test. Called optionally by the judge."""
    store.wipe()
    conversations.wipe()
    log.info("teardown complete")
    return {"status": "ok", "message": "All state wiped."}


# ===========================================================================
# Utility
# ===========================================================================

def _short_uid() -> str:
    return uuid.uuid4().hex[:8]


# ===========================================================================
# Dev server entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    log.info(
        "starting VERA bot server host=%s port=%d",
        settings.host, settings.port,
    )
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
