"""
reply_handler.py — Multi-turn conversation handling.

Handles the full /v1/reply logic:
  1. Auto-reply detection + escalation
  2. Intent classification (commit / reject / hostile / wait / curveball)
  3. Reply composition (delegates to composer or returns canned responses)
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Optional

from config import settings
from conversation import ConversationState, conversations
from logger import log
from prompts.reply_prompts import (
    REPLY_ACTION_MODE,
    REPLY_AUTO_PROBE,
    REPLY_CURVEBALL,
    REPLY_GRACEFUL_EXIT,
    REPLY_NORMAL,
)
from prompts.system_prompt import get_system_prompt
from state import store

# ---------------------------------------------------------------------------
# Auto-reply detection
# ---------------------------------------------------------------------------

AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond shortly",
    "i am an automated",
    "auto-reply",
    "will get back to you",
    "automated assistant",
    "team tak pahuncha deti hoon",
    "main ek automated",
    "shukriya, lekin main ek automated",
    "hamari team pahuncha",
    "we will respond",
    "we'll get back",
    "thanks for reaching out",
    "out of office",
    "automated response",
    "automatic reply",
    "auto-responder",
    "auto responder",
    "this is an automated message",
]


def is_auto_reply(message: str, conv: ConversationState) -> bool:
    """Return True if the message is a WhatsApp Business auto-reply."""
    msg_lower = message.lower().strip()

    # Phrase match
    for phrase in AUTO_REPLY_PHRASES:
        if phrase in msg_lower:
            log.debug("auto-reply detected (phrase match): '%s'", phrase)
            return True

    # Header / indicator match
    if msg_lower.startswith("[auto-reply]") or msg_lower.startswith("[automated]"):
        return True

    # Verbatim repeat of the PRIOR merchant message (excluding current message)
    prior = conv.merchant_messages()[:-1]
    if prior and (message.strip() == prior[-1].strip() or _jaccard(message, prior[-1]) > 0.85):
        log.debug("auto-reply detected (verbatim/similarity repeat)")
        return True

    return False


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

COMMIT_PHRASES = [
    "let's do it", "lets do it", "ok go ahead", "go ahead",
    "confirm", "proceed", "chaliye", "haan", "theek hai",
    "ok lets", "sure go", "sounds good", "please proceed",
    "please go", "please do", "bilkul", "what's next", "whats next",
    "what next", "how to proceed", "how do we proceed", "let's proceed",
    "lets proceed", "let's start", "lets start", "do it", "sure",
    "okay", "ok", "yes", "yep", "yeah", "agree", "approved", "go for it",
]

REJECT_PHRASES = [
    "not interested", "band karo", "mat bhejo", "nahi chahiye",
    "unsubscribe", "remove me", "please stop", "stop messaging",
    "don't message", "dont message", "no thanks", "stop",
]

HOSTILE_PHRASES = [
    "useless", "spam", "why are you bothering",
    "bothering me", "leave me alone", "bakwas", "faltu",
]

WAIT_PHRASES = [
    "later", "busy", "call back", "baad mein", "abhi nahi",
    "not now", "give me time", "thodi der",
]

INFO_PHRASES = [
    "tell me more", "more info", "details", "how does it work",
    "kaise karna hai", "explain", "what is this", "jankari",
    "more details", "what do you mean",
]

CURVEBALL_TOPICS = [
    "gst", "income tax", "legal", "court", "visa", "salary",
    "insurance", "loan", "property", "police",
]


def classify_intent(message: str) -> str:
    """
    Returns one of:
        "hard_reject" | "hostile" | "explicit_commit" | "wait_request"
        | "request_info" | "curveball" | "ambiguous" | "normal_engaged"

    NOTE: reject/hostile checks run BEFORE commit checks to ensure
    'band karo' or 'no' never accidentally maps to 'explicit_commit'.
    """
    msg_lower = message.lower().strip()
    msg_clean = re.sub(r"[^\w\s]", " ", msg_lower).strip()

    # 1. Reject / hostile (checked first — STOP/reject beats commit)
    if msg_clean in ("no", "nahi", "stop", "na", "dont", "don t"):
        return "hard_reject"
    if any(p in msg_lower for p in HOSTILE_PHRASES):
        return "hostile"
    if any(p in msg_lower for p in REJECT_PHRASES):
        return "hard_reject"

    # 2. Wait request
    if any(p in msg_lower for p in WAIT_PHRASES):
        return "wait_request"

    # 3. Explicit commitment / Positive action intent
    words = msg_clean.split()
    if words and words[0] in ("yes", "haan", "ha", "ok", "okay", "sure", "yep", "yeah", "lets", "let"):
        return "explicit_commit"
    if any(p in msg_lower for p in COMMIT_PHRASES):
        return "explicit_commit"

    # 4. Request for info
    if any(p in msg_lower for p in INFO_PHRASES):
        return "request_info"

    # 5. Curveball
    if any(t in msg_lower for t in CURVEBALL_TOPICS):
        return "curveball"

    # Ambiguous short single word
    if len(words) == 1 and len(words[0]) < 4:
        return "ambiguous"

    return "normal_engaged"


# ---------------------------------------------------------------------------
# Repetition check
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def body_is_repeat(conv: ConversationState, new_body: str) -> bool:
    if "drafting the content" in new_body or "I'm on it" in new_body:
        return False
    for past in conv.bot_messages():
        if new_body.strip() == past.strip():
            return True
        if _jaccard(new_body, past) > 0.85:
            return True
    return False


# ---------------------------------------------------------------------------
# Main reply handler
# ---------------------------------------------------------------------------

def handle_reply(
    conv_id: str,
    merchant_id: str,
    message: str,
    turn_number: int,
    customer_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Process a merchant/customer reply and return the bot's next action dict.
    """
    conv = conversations.get(conv_id)

    # Re-initialize or auto-recover if conv is missing or re-used from an earlier test run
    if conv is None or (turn_number <= 2 and len(conv.turns) >= 2):
        log.info("initializing conversation state for conv_id=%s merchant=%s turn=%d", conv_id, merchant_id, turn_number)
        conv = conversations.start(
            conversation_id=conv_id,
            merchant_id=merchant_id,
            trigger_id="trg_recovered",
            trigger_kind="generic",
            send_as="vera",
            bot_body="Opening message",
            customer_id=customer_id,
        )

    # ----------------------------------------------------------------
    # 1. Intent classification — run FIRST on incoming message
    # ----------------------------------------------------------------
    intent = classify_intent(message)
    log.info("[DEBUG] reply received conv_id=%s merchant_id=%s intent=%s msg='%s'", conv_id, merchant_id, intent, message)

    # 2. Permanent opt-out / explicit STOP / hostile in THIS turn
    if intent in ("hard_reject", "hostile"):
        conv.end()
        if merchant_id:
            store.add_optout(merchant_id, cooldown_days=settings.optout_cooldown_days)
            conversations.clear_merchant_wait(merchant_id)
        return {
            "action": "end",
            "rationale": (
                f"Merchant expressed disinterest (intent={intent}); "
                "closing and suppressing for 30 days."
            ),
        }

    # 3. Genuine human commitment or intent -> clear wait/optout and resume conversation
    if intent in ("explicit_commit", "wait_request", "request_info", "curveball"):
        if merchant_id:
            store.remove_optout(merchant_id)
            conversations.clear_merchant_wait(merchant_id)
        conv.resume()

    # 4. Auto-reply detection (only if NOT an explicit human commit/intent)
    if is_auto_reply(message, conv) and intent not in ("explicit_commit", "wait_request", "request_info", "curveball"):
        conv.auto_reply_count += 1
        merchant_count = conversations.increment_merchant_auto_reply_count(conv.merchant_id)
        effective_count = max(conv.auto_reply_count, merchant_count)

        log.info(
            "[DEBUG] auto-reply detected conv=%s count=%d effective=%d",
            conv_id, conv.auto_reply_count, effective_count
        )

        # 3rd+ consecutive auto-reply or turn limit reached -> END
        if effective_count >= 3 or conv.total_turns() >= 6:
            conv.end()
            return {
                "action": "end",
                "rationale": (
                    f"Auto-reply pattern detected {effective_count} times in a row; "
                    "ending conversation to prevent auto-reply loop."
                ),
            }

        # Transition to AUTO_REPLY_WAIT and return WAIT action (DO NOT send new message)
        conv.set_auto_reply_wait(settings.auto_reply_wait_seconds)
        conversations.set_merchant_auto_reply_wait(conv.merchant_id, settings.auto_reply_wait_seconds)
        return {
            "action": "wait",
            "wait_seconds": settings.auto_reply_wait_seconds,
            "rationale": (
                f"Auto-reply detected (count={effective_count}); "
                f"transitioning to AUTO_REPLY_WAIT ({settings.auto_reply_wait_seconds // 3600}h) for owner."
            ),
        }

    # 4. Genuine human response (not auto-reply, not STOP)
    # Re-open / resume conversation even if previously ended or in auto-reply wait
    if merchant_id:
        store.remove_optout(merchant_id)
        conversations.clear_merchant_wait(merchant_id)
    conv.resume()

    # Record inbound turn
    conv.add_merchant_turn(message)

    # 5. Hard turn-limit guard
    if conv.total_turns() > settings.hard_turn_limit * 2:
        conv.end()
        log.info("hard turn limit reached: %s", conv_id)
        return {
            "action": "end",
            "rationale": f"Turn limit reached after {conv.total_turns()} turns; ending gracefully.",
        }

    # Wait request
    if intent == "wait_request":
        conv.wait()
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Merchant asked for time; backing off 30 minutes.",
        }

    # Soft turn limit — graceful exit
    if conv.total_turns() >= settings.soft_turn_limit * 2:
        body = _compose_graceful_exit(conv)
        if body:
            conv.add_bot_turn(body, action="send")
            conv.end()
            return {
                "action": "send",
                "body": body,
                "cta": "none",
                "rationale": "Soft turn limit reached; closing gracefully.",
            }
        conv.end()
        return {
            "action": "end",
            "rationale": "Soft turn limit reached; closing gracefully.",
        }

    # Compose reply based on intent
    if intent == "explicit_commit":
        variant_prompt = REPLY_ACTION_MODE
    elif intent == "curveball":
        variant_prompt = REPLY_CURVEBALL
    else:
        variant_prompt = REPLY_NORMAL

    body, cta = _compose_reply(conv, message, variant_prompt)

    # Repetition guard
    if body and body_is_repeat(conv, body):
        log.warning("reply body is a repeat — skipping send: %s", conv_id)
        conv.end()
        return {
            "action": "end",
            "rationale": "Composed reply was a repeat of a previous message; ending to avoid spam.",
        }

    if body:
        conv.add_bot_turn(body, action="send")
        return {
            "action": "send",
            "body": body,
            "cta": cta,
            "rationale": f"Responding to merchant intent={intent}; advancing conversation.",
        }

    # Fallback — can't compose, end gracefully
    conv.end()
    return {
        "action": "end",
        "rationale": "Could not compose a reply; ending gracefully.",
    }


# ---------------------------------------------------------------------------
# Reply composition helpers
# ---------------------------------------------------------------------------

def _compose_reply(
    conv: ConversationState,
    latest_message: str,
    variant_prompt: str,
) -> tuple[str, str]:
    """Return (body, cta) for the reply. Returns ("", "none") on failure."""
    if not settings.llm_api_key:
        return _stub_reply(conv, latest_message, variant_prompt)

    # Resolve contexts for the reply
    merchant = store.get_context("merchant", conv.merchant_id)
    trigger = store.get_context("trigger", conv.trigger_id)
    category_slug = (merchant or {}).get("category_slug", "")
    category = store.get_context("category", category_slug) if category_slug else {}

    prompt_parts = [
        "=== CONVERSATION HISTORY ===",
        json.dumps(conv.recent_context(6), ensure_ascii=False),
        f"\n=== MERCHANT LATEST MESSAGE ===\n{latest_message}",
        f"\n=== MERCHANT CONTEXT (summary) ===",
        f"Name: {(merchant or {}).get('identity', {}).get('name', '')}",
        f"Owner: {(merchant or {}).get('identity', {}).get('owner_first_name', '')}",
        f"Languages: {(merchant or {}).get('identity', {}).get('languages', ['en'])}",
        f"Trigger kind: {conv.trigger_kind}",
        f"\n=== TASK ===\n{variant_prompt}",
    ]
    prompt = "\n".join(prompt_parts)
    system = get_system_prompt()

    raw = _call_llm_reply(system, prompt)
    if not raw:
        return _stub_reply(conv, latest_message, variant_prompt)

    body, cta = _parse_reply_output(raw)
    if not body:
        return _stub_reply(conv, latest_message, variant_prompt)

    return body, cta


def _parse_reply_output(raw: str) -> tuple[str, str]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return "", "none"
    try:
        data = json.loads(match.group())
        body = data.get("body", "").strip()
        cta = data.get("cta", "open_ended")
        # Strip any URLs
        body = re.sub(r"https?://\S+", "[link removed]", body)
        return body, cta
    except (json.JSONDecodeError, KeyError):
        return "", "none"


def _stub_reply(
    conv: ConversationState,
    latest_message: str,
    variant_prompt: str,
) -> tuple[str, str]:
    """Stub reply for when LLM is not configured."""
    if "REPLY_ACTION_MODE" in variant_prompt or "committed" in variant_prompt.lower():
        merchant = store.get_context("merchant", conv.merchant_id) or {}
        name = merchant.get("identity", {}).get("owner_first_name", "there")
        return (
            f"Great, {name}! I'm on it — drafting the content now. "
            "I'll have it ready in under 2 minutes. Please stand by.",
            "none",
        )
    if "CURVEBALL" in variant_prompt:
        return (
            "That's outside what I can help with directly — "
            "coming back to where we left off. Want me to proceed with the earlier step?",
            "binary_yes_no",
        )
    if "GRACEFUL_EXIT" in variant_prompt:
        return (
            "No worries! If anything changes, just reply 'Hi Vera'. 🙏",
            "none",
        )
    # Normal reply stub
    return (
        "Got it! Let me take care of that for you. "
        "Want me to proceed with the next step?",
        "binary_yes_no",
    )


def _compose_auto_probe(conv: ConversationState) -> str:
    merchant = store.get_context("merchant", conv.merchant_id) or {}
    name = merchant.get("identity", {}).get("owner_first_name", "")

    if settings.llm_api_key:
        prompt = (
            f"=== CONVERSATION HISTORY ===\n{json.dumps(conv.recent_context(4))}\n\n"
            f"=== TASK ===\n{REPLY_AUTO_PROBE}"
        )
        raw = _call_llm_reply(get_system_prompt(), prompt)
        if raw:
            body, _ = _parse_reply_output(raw)
            if body:
                return body

    # Stub
    greeting = f"Hi {name}, " if name else ""
    return (
        f"{greeting}looks like this might be an auto-reply 😊 "
        "When you see this, just reply YES to continue. No action needed from the bot!"
    )


def _compose_graceful_exit(conv: ConversationState) -> str:
    if settings.llm_api_key:
        prompt = (
            f"=== CONVERSATION HISTORY ===\n{json.dumps(conv.recent_context(4))}\n\n"
            f"=== TASK ===\n{REPLY_GRACEFUL_EXIT}"
        )
        raw = _call_llm_reply(get_system_prompt(), prompt)
        if raw:
            body, _ = _parse_reply_output(raw)
            if body:
                return body
    return "No worries! If anything changes, just reply 'Hi Vera'. Have a great day! 🙏"


def _call_llm_reply(system: str, prompt: str) -> Optional[str]:
    """Thin wrapper that reuses composer's LLM logic."""
    try:
        from composer import _call_llm
        return _call_llm(system, prompt)
    except Exception as exc:
        log.error("LLM reply call failed: %s", exc)
        return None
