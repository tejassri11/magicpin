"""
composer.py — Context assembler + LLM call + output parser.

This module is the core intelligence plug-in point. It:
  1. Assembles all 4 contexts into a structured prompt
  2. Selects the right prompt variant via the trigger-kind router
  3. Calls the LLM (or returns a stub if no API key is configured)
  4. Parses and validates the LLM output

The stub mode is intentional for Phase 1 — it returns a templated message
with real context values so the endpoints work end-to-end before the full
LLM integration is added in Phase 4.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from config import settings
from logger import log
from prompts.system_prompt import get_system_prompt, build_voice_section
from prompts.trigger_prompts import get_trigger_variant


# ---------------------------------------------------------------------------
# Composed message dataclass
# ---------------------------------------------------------------------------
@dataclass
class ComposedMessage:
    body: str
    cta: str                      # "binary_yes_no" | "open_ended" | "multi_choice_slot" | "none"
    send_as: str                  # "vera" | "merchant_on_behalf"
    suppression_key: str
    rationale: str
    template_name: str
    template_params: list[str]


VALID_CTAS = {"binary_yes_no", "open_ended", "multi_choice_slot", "binary_confirm_cancel", "none"}


# ---------------------------------------------------------------------------
from validator import validate_message


# Public compose entry point
# ---------------------------------------------------------------------------

def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]] = None,
) -> Optional[ComposedMessage]:
    """
    Compose a message from 4 contexts. Returns None on unrecoverable failure.

    In stub mode (no LLM_API_KEY), returns a context-grounded template
    message so all endpoints work out-of-the-box without an LLM key.
    """
    trigger_kind = trigger.get("kind", "unknown")
    merchant_id = merchant.get("merchant_id", "unknown")

    log.info("composing for merchant=%s trigger_kind=%s", merchant_id, trigger_kind)

    if not settings.llm_api_key:
        log.warning("LLM_API_KEY not set — using stub composer")
        return _stub_compose(category, merchant, trigger, customer)

    # Build the full prompt
    prompt = _build_prompt(category, merchant, trigger, customer)
    system = get_system_prompt()

    # Call LLM
    raw_response = _call_llm(system, prompt)
    if raw_response is None:
        log.error("LLM returned None for merchant=%s trigger=%s", merchant_id, trigger_kind)
        return _stub_compose(category, merchant, trigger, customer)

    # Parse + validate
    parsed = _parse_llm_output(raw_response, trigger)
    if parsed is None:
        log.warning("LLM output parse failed; falling back to stub for merchant=%s", merchant_id)
        return _stub_compose(category, merchant, trigger, customer)

    v_result = validate_message(
        body=parsed.body,
        cta=parsed.cta,
        send_as=parsed.send_as,
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
    )

    if not v_result.is_valid:
        log.warning(
            "LLM output validation failed (%s); falling back to stub for merchant=%s",
            v_result.violations, merchant_id,
        )
        return _stub_compose(category, merchant, trigger, customer)

    parsed.body = v_result.cleaned_body
    return parsed


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
) -> str:
    trigger_kind = trigger.get("kind", "unknown")
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    customer_agg = merchant.get("customer_aggregate", {})
    conv_history = merchant.get("conversation_history", [])[-2:]
    review_themes = merchant.get("review_themes", [])

    peer_stats = category.get("peer_stats", {})
    digest_items = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])

    sections = []

    # --- Voice section ---
    sections.append("=== CATEGORY CONTEXT ===")
    sections.append(build_voice_section(category))
    if digest_items:
        sections.append(f"Digest items ({len(digest_items)}):")
        for d in digest_items[:5]:
            sections.append(
                f"  - [{d.get('kind', 'research')}] {d.get('title', '')} "
                f"| source: {d.get('source', '')} "
                f"| trial_n: {d.get('trial_n', 'N/A')} "
                f"| segment: {d.get('patient_segment', 'N/A')}"
            )
    if seasonal:
        sections.append(f"Seasonal beats: {json.dumps(seasonal)}")
    if trends:
        sections.append(f"Trend signals: {json.dumps(trends[:3])}")

    # --- Merchant section ---
    sections.append("\n=== MERCHANT CONTEXT ===")
    sections.append(f"Name: {identity.get('name', '')}")
    sections.append(f"Owner first name: {identity.get('owner_first_name', '')}")
    sections.append(f"City: {identity.get('city', '')} | Locality: {identity.get('locality', '')}")
    sections.append(f"Languages: {identity.get('languages', ['en'])}")
    sections.append(f"Verified: {identity.get('verified', False)}")
    sections.append(
        f"Subscription: status={sub.get('status', 'unknown')} "
        f"days_remaining={sub.get('days_remaining', '?')} "
        f"plan={sub.get('plan', '?')}"
    )
    delta = perf.get("delta_7d", {})
    sections.append(
        f"Performance (30d): views={perf.get('views', '?')} calls={perf.get('calls', '?')} "
        f"directions={perf.get('directions', '?')} ctr={perf.get('ctr', '?')} leads={perf.get('leads', '?')}"
    )
    sections.append(
        f"7d delta: views_pct={delta.get('views_pct', '?')} calls_pct={delta.get('calls_pct', '?')}"
    )
    sections.append(f"Peer avg_ctr: {peer_stats.get('avg_ctr', '?')}")
    if active_offers:
        sections.append(f"Active offers: {[o.get('title') for o in active_offers]}")
    else:
        sections.append("Active offers: none")
    if signals:
        sections.append(f"Signals: {signals}")
    if customer_agg:
        sections.append(f"Customer aggregate: {json.dumps(customer_agg)}")
    if review_themes:
        sections.append(f"Review themes: {json.dumps(review_themes[:3])}")
    if conv_history:
        sections.append("Recent conversation (last 2 turns):")
        for turn in conv_history:
            sections.append(f"  [{turn.get('from', '?')}] {turn.get('body', '')[:120]}")

    # --- Trigger section ---
    sections.append("\n=== TRIGGER CONTEXT ===")
    sections.append(f"Kind: {trigger_kind}")
    sections.append(f"Source: {trigger.get('source', '?')} | Scope: {trigger.get('scope', '?')}")
    sections.append(f"Urgency: {trigger.get('urgency', '?')}/5")
    sections.append(f"Suppression key: {trigger.get('suppression_key', '')}")
    sections.append(f"Expires at: {trigger.get('expires_at', '')}")
    payload = trigger.get("payload", {})
    if payload:
        sections.append(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

    # --- Customer section ---
    if customer:
        sections.append("\n=== CUSTOMER CONTEXT ===")
        cid = customer.get("identity", {})
        rel = customer.get("relationship", {})
        prefs = customer.get("preferences", {})
        consent = customer.get("consent", {})
        sections.append(f"Name: {cid.get('name', '')}")
        sections.append(f"Language pref: {cid.get('language_pref', 'en')}")
        sections.append(f"Age band: {cid.get('age_band', '?')}")
        sections.append(f"State: {customer.get('state', '?')}")
        sections.append(
            f"Relationship: first_visit={rel.get('first_visit', '?')} "
            f"last_visit={rel.get('last_visit', '?')} "
            f"visits_total={rel.get('visits_total', '?')}"
        )
        services = rel.get("services_received", [])
        if services:
            sections.append(f"Services received: {services}")
        sections.append(f"Preferences: {json.dumps(prefs)}")
        sections.append(f"Consent scope: {consent.get('scope', [])}")

    # --- Task ---
    sections.append("\n=== TASK ===")
    sections.append(get_trigger_variant(trigger_kind))
    sections.append(
        "\nNow compose the message. Remember: output ONLY a valid JSON object, "
        "no markdown, no preamble."
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(system: str, prompt: str) -> Optional[str]:
    """Call the configured LLM provider. Returns the raw text response or None."""
    provider = settings.llm_provider.lower()

    try:
        if provider == "anthropic":
            return _call_anthropic(system, prompt)
        elif provider == "openai":
            return _call_openai(system, prompt)
        elif provider == "gemini":
            return _call_gemini(system, prompt)
        elif provider == "deepseek":
            return _call_deepseek(system, prompt)
        else:
            log.error("unknown LLM provider: %s", provider)
            return None
    except Exception as exc:
        log.error("LLM call failed provider=%s error=%s", provider, exc)
        return None


def _call_anthropic(system: str, prompt: str) -> Optional[str]:
    import urllib.request, json as _json
    body = _json.dumps({
        "model": settings.llm_model,
        "max_tokens": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": settings.llm_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


def _call_openai(system: str, prompt: str) -> Optional[str]:
    import urllib.request, json as _json
    body = _json.dumps({
        "model": settings.llm_model or "gpt-4o",
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_gemini(system: str, prompt: str) -> Optional[str]:
    import urllib.request, json as _json
    full_prompt = f"{system}\n\n{prompt}"
    body = _json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": settings.llm_temperature,
            "maxOutputTokens": settings.llm_max_tokens,
        },
    }).encode("utf-8")
    model = settings.llm_model or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.llm_api_key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_deepseek(system: str, prompt: str) -> Optional[str]:
    import urllib.request, json as _json
    body = _json.dumps({
        "model": settings.llm_model or "deepseek-chat",
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Output parser + validator
# ---------------------------------------------------------------------------

def _parse_llm_output(
    raw: str, trigger: dict[str, Any]
) -> Optional[ComposedMessage]:
    """Extract and validate the JSON from LLM output."""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        log.warning("no JSON found in LLM response: %s", raw[:200])
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.warning("JSON parse error in LLM response: %s | raw: %s", exc, raw[:200])
        return None

    body = data.get("body", "").strip()
    if not body:
        log.warning("LLM returned empty body")
        return None

    # URL check — hard violation
    if re.search(r"https?://", body):
        log.warning("URL detected in LLM output — stripping URL")
        body = re.sub(r"https?://\S+", "[link removed]", body)

    cta = data.get("cta", "open_ended")
    if cta not in VALID_CTAS:
        log.warning("invalid cta '%s' — defaulting to open_ended", cta)
        cta = "open_ended"

    send_as = data.get("send_as", "vera")
    if send_as not in {"vera", "merchant_on_behalf"}:
        send_as = "vera"

    suppression_key = data.get("suppression_key", trigger.get("suppression_key", ""))
    rationale = data.get("rationale", "").strip()

    return ComposedMessage(
        body=body,
        cta=cta,
        send_as=send_as,
        suppression_key=suppression_key,
        rationale=rationale,
        template_name=_infer_template_name(trigger.get("kind", "generic"), send_as),
        template_params=_extract_template_params(body),
    )


# ---------------------------------------------------------------------------
# Stub composer (used when LLM_API_KEY not set)
# ---------------------------------------------------------------------------

def _stub_compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
) -> ComposedMessage:
    """
    Returns a context-grounded stub message.
    Every field is populated with real values from the contexts so that
    the judge simulator can score it (even if imperfectly).
    """
    trigger_kind = trigger.get("kind", "unknown")
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", "there"))
    merchant_name = identity.get("name", "your business")
    perf = merchant.get("performance", {})
    peer_stats = category.get("peer_stats", {})
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    suppression_key = trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id', '')}")

    if customer:
        c_name = customer.get("identity", {}).get("name", "there")
        c_state = customer.get("state", "lapsed_soft")
        rel = customer.get("relationship", {})
        last_visit = rel.get("last_visit", "a while ago")
        payload = trigger.get("payload", {})
        slots = payload.get("available_slots", [])
        slot_text = ""
        if slots:
            slot_text = " | ".join(s.get("label", "") for s in slots[:2])
        offer_text = active_offers[0].get("title", "our services") if active_offers else "our services"
        body = (
            f"Hi {c_name}, {merchant_name} here. It's been a while since your last visit "
            f"({last_visit}). We'd love to see you back — {offer_text}. "
            f"{'Slots available: ' + slot_text + '. ' if slot_text else ''}"
            f"Reply YES to book."
        )
        send_as = "merchant_on_behalf"
        cta = "binary_yes_no"
    else:
        ctr = perf.get("ctr", 0)
        peer_ctr = peer_stats.get("avg_ctr", 0)
        ctr_note = (
            f"Your CTR is {ctr:.1%} vs peer median {peer_ctr:.1%}. "
            if peer_ctr and ctr else ""
        )
        offer_note = (
            f"Active offer: {active_offers[0].get('title')}. "
            if active_offers else ""
        )
        body = (
            f"Hi {owner}! Quick update from Vera — "
            f"{ctr_note}{offer_note}"
            f"Trigger: {trigger_kind}. Want me to help with the next step? Reply YES."
        )
        send_as = "vera"
        cta = "binary_yes_no"

    return ComposedMessage(
        body=body,
        cta=cta,
        send_as=send_as,
        suppression_key=suppression_key,
        rationale=f"Stub compose for trigger_kind={trigger_kind} — LLM not configured",
        template_name=_infer_template_name(trigger_kind, send_as),
        template_params=[owner, merchant_name, trigger_kind],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_template_name(trigger_kind: str, send_as: str) -> str:
    prefix = "merchant" if send_as == "merchant_on_behalf" else "vera"
    kind_slug = trigger_kind.replace("_", "-")
    return f"{prefix}_{kind_slug}_v1"


def _extract_template_params(body: str) -> list[str]:
    """Extract first 3 sentences as rough template params."""
    sentences = [s.strip() for s in re.split(r"[.!?]", body) if s.strip()]
    return sentences[:3]
