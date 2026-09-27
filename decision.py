"""
decision.py — Deterministic SEND/SKIP decision engine.

Architecture
============
The engine runs in two phases:

PHASE 1 — ELIGIBILITY GATES (fail-fast, ordered):
    Gate  1: Trigger context loaded in store?
    Gate  2: Trigger not expired?
    Gate  3: Trigger suppression_key not already fired?
    Gate  4: No active conversation already running for (merchant, trigger)?
    Gate  5: Merchant context loaded?
    Gate  6: Category context loaded?
    Gate  7: Merchant not opted-out (with renewal_due override)?
    Gate  8: Merchant subscription not expired (unless trigger is renewal_due)?
    Gate  9: Customer context loaded (for customer-scope triggers)?
    Gate 10: Customer consent covers this trigger kind?
    Gate 11: Customer not opted-out of reminders?

PHASE 2 — SIGNAL EVALUATION (for triggers that pass all gates):
    Step A: Extract merchant signals, performance delta, and peer gap.
    Step B: Extract trigger payload facts.
    Step C: Score actionability: is there a useful CTA to offer?
    Step D: Choose the single strongest signal (primary_signal).
    Step E: Build supporting_facts list (max 3, all grounded in context).
    Step F: Determine send_now urgency framing.
    Step G: Recommend CTA type based on trigger kind.
    Step H: Assign priority (1–5; higher = send first).

Returns a StructuredDecision object. The composer reads this directly.

Design principles
=================
- Pure function: no LLM, no network, no side effects.
- Deterministic: same inputs → same output always.
- Explainable: every field explains the reasoning chain.
- Conservative: skip unless there is a clear actionable signal.
- Grounded: supporting_facts only contain values from context payloads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from config import settings
from conversation import conversations
from logger import log
from state import store


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Decision(Enum):
    SEND = "send"
    SKIP = "skip"


class SkipReason(str, Enum):
    TRIGGER_NOT_LOADED = "trigger_not_loaded"
    TRIGGER_EXPIRED = "trigger_expired"
    SUPPRESSED = "suppressed"
    CONVERSATION_ACTIVE = "conversation_already_active"
    MERCHANT_NOT_LOADED = "merchant_not_loaded"
    CATEGORY_NOT_LOADED = "category_not_loaded"
    MERCHANT_OPTED_OUT = "merchant_opted_out"
    SUBSCRIPTION_EXPIRED = "subscription_expired"
    CUSTOMER_NOT_LOADED = "customer_not_loaded"
    NO_CONSENT = "customer_no_consent"
    REMINDER_OPT_OUT = "customer_reminder_opt_out"
    ANONYMOUS_CUSTOMER = "anonymous_customer"
    NO_ACTIONABLE_CTA = "no_actionable_cta"
    TRIGGER_KIND_IRRELEVANT = "trigger_kind_irrelevant_for_context"
    NO_STRONG_SIGNAL = "no_strong_signal"


class CTAType(str, Enum):
    BINARY_YES_NO = "binary_yes_no"
    MULTI_CHOICE_SLOT = "multi_choice_slot"
    OPEN_ENDED = "open_ended"
    BINARY_CONFIRM_CANCEL = "binary_confirm_cancel"
    NONE = "none"


# ---------------------------------------------------------------------------
# Trigger kind → category relevance mapping
# Prevents category mismatch (e.g., "recall_due" for a restaurant merchant)
# ---------------------------------------------------------------------------

# Kinds that only make sense for specific categories
KIND_CATEGORY_RESTRICTIONS: dict[str, set[str]] = {
    "recall_due": {"dentists"},
    "chronic_refill_due": {"pharmacies"},
    "supply_alert": {"pharmacies"},
    "research_digest": {"dentists", "pharmacies", "gyms"},
    "regulation_change": {"dentists", "pharmacies"},
    "cde_opportunity": {"dentists"},
    "gbp_unverified": {"dentists", "salons", "restaurants", "gyms", "pharmacies"},  # all
    "ipl_match_today": {"restaurants"},
    "wedding_package_followup": {"salons"},
    "bridal_followup": {"salons"},
    "seasonal_perf_dip": {"gyms", "restaurants", "salons"},
    "category_seasonal": {"pharmacies", "restaurants"},
}

# Kinds that are valid for any category (no restriction)
_UNRESTRICTED_KINDS = {
    "perf_dip", "perf_spike", "milestone_reached", "dormant_with_vera",
    "curious_ask_due", "review_theme_emerged", "renewal_due", "winback_eligible",
    "festival_upcoming", "competitor_opened", "category_trend_movement",
    "local_news_event", "customer_lapsed_soft", "customer_lapsed_hard",
    "appointment_tomorrow", "trial_followup", "active_planning_intent",
    "weather_heatwave", "gbp_unverified",
}

# Trigger kinds that fire CTA = binary_yes_no by default
_BINARY_CTA_KINDS = {
    "perf_dip", "perf_spike", "renewal_due", "winback_eligible",
    "review_theme_emerged", "competitor_opened", "festival_upcoming",
    "ipl_match_today", "weather_heatwave", "gbp_unverified",
    "supply_alert", "regulation_change", "category_seasonal",
    "customer_lapsed_soft", "customer_lapsed_hard",
}

_SLOT_CTA_KINDS = {
    "recall_due", "appointment_tomorrow", "trial_followup",
    "wedding_package_followup", "bridal_followup",
}

_OPEN_CTA_KINDS = {
    "research_digest", "milestone_reached", "dormant_with_vera",
    "curious_ask_due", "active_planning_intent", "seasonal_perf_dip",
    "cde_opportunity", "perf_spike",
}

_CONFIRM_CTA_KINDS = {
    "chronic_refill_due", "active_planning_intent",
}


# ---------------------------------------------------------------------------
# Structured decision object
# ---------------------------------------------------------------------------

@dataclass
class StructuredDecision:
    """
    The complete, explainable output of the decision engine.
    Passed directly to the composer so it never needs to re-derive context.
    """
    # ---- Core decision ----
    should_send: bool
    decision: Decision

    # ---- Identifiers ----
    trigger_id: str
    merchant_id: str
    customer_id: Optional[str]
    trigger_kind: str

    # ---- Skip details (populated when should_send=False) ----
    skip_reason: Optional[SkipReason] = None
    skip_detail: Optional[str] = None

    # ---- Signal analysis (populated when should_send=True) ----
    primary_signal: Optional[str] = None        # single strongest signal
    supporting_facts: list[str] = field(default_factory=list)  # ≤3 grounded facts
    send_now_reason: Optional[str] = None       # WHY now (urgency framing)
    recommended_cta_type: Optional[CTAType] = None

    # ---- Prioritisation ----
    priority: int = 3                           # 1 (low) – 5 (critical)

    # ---- Resolved contexts (passed through to composer) ----
    trigger_payload: Optional[dict[str, Any]] = None
    merchant_payload: Optional[dict[str, Any]] = None
    category_payload: Optional[dict[str, Any]] = None
    customer_payload: Optional[dict[str, Any]] = None

    # ---- For composer / action assembly ----
    suppression_key: str = ""
    send_as: str = "vera"                       # "vera" | "merchant_on_behalf"

    # ---- Explainability ----
    reasoning_chain: list[str] = field(default_factory=list)

    def gate_passed(self, gate: str) -> None:
        self.reasoning_chain.append(f"GATE OK: {gate}")

    def gate_failed(self, gate: str, detail: str) -> None:
        self.reasoning_chain.append(f"GATE FAIL [{gate}]: {detail}")

    def note(self, msg: str) -> None:
        self.reasoning_chain.append(f"  NOTE: {msg}")

    @property
    def reason(self) -> str:
        """Single-line reason — used for logging and rationale fields."""
        if self.should_send:
            return self.primary_signal or "trigger is actionable"
        return f"{self.skip_reason}: {self.skip_detail}" if self.skip_detail else str(self.skip_reason)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_trigger(trigger_id: str, now_iso: str) -> StructuredDecision:
    """
    Full SEND/SKIP evaluation for a single trigger.
    Returns a StructuredDecision — never raises exceptions.
    """
    result = StructuredDecision(
        should_send=False,
        decision=Decision.SKIP,
        trigger_id=trigger_id,
        merchant_id="",
        customer_id=None,
        trigger_kind="",
    )

    try:
        _run_gates(result, trigger_id, now_iso)
        if result.should_send:
            _run_signal_analysis(result)
    except Exception as exc:
        log.error("decision engine error for trigger=%s: %s", trigger_id, exc, exc_info=True)
        result.should_send = False
        result.decision = Decision.SKIP
        result.skip_reason = SkipReason.NO_STRONG_SIGNAL
        result.skip_detail = f"internal error: {exc}"

    _log_decision(result)
    return result


# ---------------------------------------------------------------------------
# PHASE 1 — Eligibility gates
# ---------------------------------------------------------------------------

def _run_gates(result: StructuredDecision, trigger_id: str, now_iso: str) -> None:
    """Mutates result. Returns early on first gate failure."""

    # ── Gate 1: Trigger context loaded ────────────────────────────────────
    trg = store.get_context("trigger", trigger_id)
    if trg is None:
        return _fail(result, SkipReason.TRIGGER_NOT_LOADED,
                     f"trigger '{trigger_id}' not found in store")
    result.trigger_id = trigger_id
    result.trigger_kind = trg.get("kind", "unknown")
    result.suppression_key = trg.get("suppression_key", "")
    result.merchant_id = trg.get("merchant_id", "")
    result.customer_id = trg.get("customer_id")
    result.trigger_payload = trg
    result.gate_passed("trigger_loaded")

    # ── Gate 2: Trigger not expired ───────────────────────────────────────
    expires_at_str: str = trg.get("expires_at", "")
    if expires_at_str:
        now_dt = _parse_iso(now_iso)
        exp_dt = _parse_iso(expires_at_str)
        if now_dt and exp_dt and now_dt > exp_dt:
            return _fail(result, SkipReason.TRIGGER_EXPIRED,
                         f"expired at {expires_at_str}, now={now_iso}")
    result.gate_passed("not_expired")

    # ── Gate 3: Suppression key not fired ────────────────────────────────
    if result.suppression_key and store.is_suppressed(result.suppression_key):
        return _fail(result, SkipReason.SUPPRESSED,
                     f"suppression_key '{result.suppression_key}' already fired")
    result.gate_passed("not_suppressed")

    # ── Gate 4: No active conversation or waiting state for merchant ───────
    if not result.merchant_id:
        return _fail(result, SkipReason.MERCHANT_NOT_LOADED,
                     "trigger has no merchant_id")
    if conversations.has_active_conversation_for_merchant(result.merchant_id):
        return _fail(result, SkipReason.CONVERSATION_ACTIVE,
                     f"active conversation already exists for merchant={result.merchant_id}")
    if conversations.is_merchant_waiting(result.merchant_id):
        return _fail(result, SkipReason.CONVERSATION_ACTIVE,
                     f"merchant is currently in waiting/auto-reply wait state merchant={result.merchant_id}")
    result.gate_passed("no_active_conversation")

    # ── Gate 5: Merchant context loaded ──────────────────────────────────
    merchant = store.get_context("merchant", result.merchant_id)
    if merchant is None:
        return _fail(result, SkipReason.MERCHANT_NOT_LOADED,
                     f"merchant '{result.merchant_id}' not loaded")
    result.merchant_payload = merchant
    result.gate_passed("merchant_loaded")

    # ── Gate 6: Category context loaded ──────────────────────────────────
    category_slug: str = merchant.get("category_slug", "")
    category = store.get_context("category", category_slug) if category_slug else None
    if category is None:
        return _fail(result, SkipReason.CATEGORY_NOT_LOADED,
                     f"category '{category_slug}' not loaded for merchant={result.merchant_id}")
    result.category_payload = category
    result.gate_passed("category_loaded")

    # ── Gate 7: Trigger kind is relevant for this category ────────────────
    restrictions = KIND_CATEGORY_RESTRICTIONS.get(result.trigger_kind)
    if restrictions and category_slug not in restrictions:
        return _fail(result, SkipReason.TRIGGER_KIND_IRRELEVANT,
                     f"trigger kind '{result.trigger_kind}' not relevant for category '{category_slug}'")
    result.gate_passed("category_kind_match")

    # ── Gate 8: Merchant opt-out ──────────────────────────────────────────
    if store.is_opted_out(result.merchant_id):
        # renewal_due with urgency ≥ 4 can override after RENEWAL_OPTOUT_OVERRIDE_DAYS
        if result.trigger_kind == "renewal_due":
            urgency = int(trg.get("urgency", 0))
            days_remaining = store.optout_days_remaining(result.merchant_id)
            days_elapsed = settings.optout_cooldown_days - days_remaining
            if urgency >= 4 and days_elapsed >= settings.renewal_optout_override_days:
                result.note(f"opt-out override: renewal_due urgency={urgency}")
            else:
                return _fail(result, SkipReason.MERCHANT_OPTED_OUT,
                             f"merchant opted-out, {days_remaining:.0f}d remaining")
        else:
            return _fail(result, SkipReason.MERCHANT_OPTED_OUT,
                         f"merchant opted-out from all messaging")
    result.gate_passed("merchant_not_opted_out")

    # ── Gate 9: Subscription gate ─────────────────────────────────────────
    sub = merchant.get("subscription", {})
    sub_status: str = sub.get("status", "active")
    if sub_status == "expired" and result.trigger_kind not in {
        "renewal_due", "winback_eligible", "dormant_with_vera"
    }:
        return _fail(result, SkipReason.SUBSCRIPTION_EXPIRED,
                     f"subscription expired; only renewal/winback triggers fire")
    result.gate_passed("subscription_ok")

    # ── Gates 10-12: Customer-scope checks ────────────────────────────────
    if result.customer_id:
        # Gate 10: Customer loaded
        customer = store.get_context("customer", result.customer_id)
        if customer is None:
            return _fail(result, SkipReason.CUSTOMER_NOT_LOADED,
                         f"customer '{result.customer_id}' not loaded")
        result.customer_payload = customer

        # Guard: anonymous customer with no phone (cannot send WhatsApp)
        identity = customer.get("identity", {})
        phone = identity.get("phone_redacted")
        if not phone:
            return _fail(result, SkipReason.ANONYMOUS_CUSTOMER,
                         "customer has no phone (walk-in anonymous record)")

        # Gate 11: Customer consent
        consent = customer.get("consent", {})
        consent_scope: list[str] = consent.get("scope", [])
        has_bool_consent = any(v is True for k, v in consent.items() if k not in {"scope", "opted_in_at"})

        if not consent_scope and not has_bool_consent:
            return _fail(result, SkipReason.NO_CONSENT,
                         f"customer '{result.customer_id}' has no consent")

        # Validate consent covers this trigger kind family if scope is specified
        if consent_scope and not _consent_covers_kind(result.trigger_kind, consent_scope):
            return _fail(result, SkipReason.NO_CONSENT,
                         f"consent scope {consent_scope} doesn't cover trigger kind '{result.trigger_kind}'")

        # Gate 12: Reminder opt-in
        prefs = customer.get("preferences", {})
        if not prefs.get("reminder_opt_in", True):
            return _fail(result, SkipReason.REMINDER_OPT_OUT,
                         f"customer reminder_opt_in=False")

        result.gate_passed("customer_consent_ok")

        # Determine send_as for customer-scope messages
        channel: str = prefs.get("channel", "whatsapp")
        if channel in {"whatsapp_via_parent", "whatsapp_via_son"}:
            result.send_as = "merchant_on_behalf"
        elif channel in {"none_recorded", "none"}:
            return _fail(result, SkipReason.NO_CONSENT,
                         "customer channel=none_recorded; cannot send")
        else:
            result.send_as = "merchant_on_behalf"  # customer messages always via merchant
    else:
        # Merchant-scope: Vera speaks directly
        result.send_as = "vera"

    # All gates passed → eligible for SEND
    result.should_send = True
    result.decision = Decision.SEND
    result.gate_passed("ALL_GATES_PASSED")


# ---------------------------------------------------------------------------
# PHASE 2 — Signal analysis
# ---------------------------------------------------------------------------

def _run_signal_analysis(result: StructuredDecision) -> None:
    """
    Find the single strongest signal and supporting facts.
    Mutates result.primary_signal, supporting_facts, send_now_reason, priority.
    If no actionable signal is found, flips result to SKIP.
    """
    trg = result.trigger_payload or {}
    merchant = result.merchant_payload or {}
    category = result.category_payload or {}
    customer = result.customer_payload

    kind = result.trigger_kind
    urgency = int(trg.get("urgency", 2))
    payload = trg.get("payload", {})

    facts: list[str] = []
    primary: str = ""
    now_reason: str = ""

    # ── A. Trigger-kind specific primary signal extraction ─────────────────
    primary, now_reason, facts = _extract_primary_signal(
        kind=kind,
        payload=payload,
        merchant=merchant,
        category=category,
        customer=customer,
        urgency=urgency,
    )

    # ── B. No primary signal → check if we should SKIP ────────────────────
    if not primary:
        result.note("no strong primary signal found — SKIP")
        _fail(result, SkipReason.NO_STRONG_SIGNAL,
              f"no actionable signal for kind={kind}")
        return

    # ── C. Supporting facts — grounded only ───────────────────────────────
    extra_facts = _build_supporting_facts(merchant, category, trg, customer)
    all_facts = list(dict.fromkeys(facts + extra_facts))  # deduplicate, preserve order
    result.supporting_facts = all_facts[:3]

    # ── D. CTA type ────────────────────────────────────────────────────────
    result.recommended_cta_type = _recommend_cta(kind, payload, customer)

    # ── E. Check there's a useful CTA (not just facts with no next step) ──
    has_slot = bool(payload.get("available_slots") or payload.get("next_session_options"))
    has_offer = any(o.get("status") == "active" for o in merchant.get("offers", []))
    has_payload_action = _payload_has_action(payload, kind)

    if result.recommended_cta_type == CTAType.NONE and not has_payload_action:
        result.note("no actionable CTA available — SKIP")
        _fail(result, SkipReason.NO_ACTIONABLE_CTA,
              f"kind={kind} has no useful next step in current context")
        return

    # ── F. Priority from urgency + kind ────────────────────────────────────
    result.priority = _compute_priority(urgency, kind, merchant, trg)

    # ── G. Populate final fields ────────────────────────────────────────────
    result.primary_signal = primary
    result.send_now_reason = now_reason
    result.note(f"primary_signal={primary!r}")
    result.note(f"priority={result.priority} urgency={urgency}")


# ---------------------------------------------------------------------------
# Signal extraction per trigger kind
# ---------------------------------------------------------------------------

def _extract_primary_signal(
    kind: str,
    payload: dict[str, Any],
    merchant: dict[str, Any],
    category: dict[str, Any],
    customer: Optional[dict[str, Any]],
    urgency: int,
) -> tuple[str, str, list[str]]:
    """
    Returns (primary_signal, send_now_reason, facts_list).
    primary_signal="" means no actionable signal was found.
    """
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    sub = merchant.get("subscription", {})
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", ""))
    city = identity.get("city", "")
    peer_ctr = peer.get("avg_ctr", 0.0)
    merchant_ctr = perf.get("ctr", 0.0)
    calls = perf.get("calls", 0)
    views = perf.get("views", 0)
    delta = perf.get("delta_7d", {})
    facts: list[str] = []

    # ── research_digest / regulation_change ────────────────────────────────
    if kind in {"research_digest", "regulation_change", "cde_opportunity"}:
        # Find the relevant digest item from category context
        digest_item_id = payload.get("top_item_id") or payload.get("digest_item_id", "")
        digest = category.get("digest", [])
        item = next((d for d in digest if d.get("id") == digest_item_id), None)
        if item:
            source = item.get("source", "")
            trial_n = item.get("trial_n")
            segment = item.get("patient_segment", "")
            primary = f"new {item.get('kind', 'research')} from {source}" if source else f"new clinical digest: {item.get('title', 'research update')}"
            now_reason = f"just published this week; relevant to {segment or 'patients'}"
            if trial_n:
                facts.append(f"{trial_n:,}-participant study")
            if segment:
                facts.append(f"relevant to: {segment}")
            if source:
                facts.append(f"source: {source}")
        elif kind == "cde_opportunity":
            primary = "free CDE webinar opportunity"
            now_reason = f"registration window open; {payload.get('credits', 0)} credits"
            facts.append(f"credits: {payload.get('credits', 0)}")
        elif kind == "research_digest":
            cat_slug = payload.get("category") or category.get("slug", "your category")
            primary = f"weekly clinical research digest for {cat_slug}"
            now_reason = f"weekly research digest update available for {cat_slug}"
            facts.append(f"category: {cat_slug}")
        elif kind == "regulation_change":
            cat_slug = payload.get("category") or category.get("slug", "your category")
            primary = f"regulatory update for {cat_slug}"
            now_reason = f"regulatory update published for {cat_slug}"
            facts.append(f"category: {cat_slug}")
        else:
            return "", "", []

    # ── perf_dip / seasonal_perf_dip ──────────────────────────────────────
    elif kind in {"perf_dip", "seasonal_perf_dip"}:
        metric = payload.get("metric", "views")
        delta_pct = payload.get("delta_pct", 0.0)
        baseline = payload.get("vs_baseline", 0)
        is_seasonal = payload.get("is_expected_seasonal", False)

        if delta_pct == 0:
            return "", "", []

        pct_str = f"{abs(delta_pct):.0%}"
        if is_seasonal:
            primary = f"{metric} down {pct_str} — normal seasonal dip, not an alarm"
            now_reason = "reframe the dip, save ad spend for recovery window"
        else:
            primary = f"{metric} dropped {pct_str} vs {baseline} baseline"
            now_reason = "dip is this week; early action prevents lead loss"
        facts.append(f"{metric} delta: {delta_pct:+.0%}")
        if peer_ctr and merchant_ctr and kind == "perf_dip":
            if merchant_ctr < peer_ctr:
                facts.append(f"CTR {merchant_ctr:.1%} below peer median {peer_ctr:.1%}")

    # ── perf_spike ─────────────────────────────────────────────────────────
    elif kind == "perf_spike":
        metric = payload.get("metric", "calls")
        delta_pct = payload.get("delta_pct", 0.0)
        driver = payload.get("likely_driver", "")
        if delta_pct <= 0:
            return "", "", []
        primary = f"{metric} up {delta_pct:.0%} — capitalize on the momentum"
        now_reason = f"spike happening now; {driver or 'window to lock in new customers'}"
        facts.append(f"{metric} +{delta_pct:.0%} vs baseline {payload.get('vs_baseline', 0)}")
        if driver:
            facts.append(f"likely driver: {driver}")

    # ── milestone_reached ──────────────────────────────────────────────────
    elif kind == "milestone_reached":
        metric = payload.get("metric", "")
        value_now = payload.get("value_now", 0)
        milestone = payload.get("milestone_value", 0)
        is_imminent = payload.get("is_imminent", False)
        if not metric or not value_now:
            return "", "", []
        gap = milestone - value_now
        if is_imminent and gap > 0:
            primary = f"{gap} more {metric} to {milestone:,} milestone"
            now_reason = f"merchant is {gap} away — quick push can close it this week"
        else:
            primary = f"crossed {milestone:,} {metric} milestone"
            now_reason = "achievement worth amplifying for social proof"
        facts.append(f"current {metric}: {value_now}")
        if milestone:
            facts.append(f"milestone target: {milestone:,}")

    # ── renewal_due / winback_eligible ────────────────────────────────────
    elif kind in {"renewal_due", "winback_eligible"}:
        days_remaining = sub.get("days_remaining")
        days_since_expiry = (
            sub.get("days_since_expiry") or
            payload.get("days_since_expiry")
        )
        plan = sub.get("plan", "Pro")
        renewal_amount = payload.get("renewal_amount", 0)
        lapsed = payload.get("lapsed_customers_added_since_expiry", 0)

        if kind == "renewal_due":
            if days_remaining is not None:
                primary = f"subscription expires in {days_remaining}d"
                now_reason = f"only {days_remaining} days left on {plan} plan"
                facts.append(f"plan: {plan}")
                if renewal_amount:
                    facts.append(f"renewal: ₹{renewal_amount:,}")
            else:
                return "", "", []
        else:  # winback_eligible
            if days_since_expiry:
                primary = f"subscription lapsed {days_since_expiry}d ago"
                now_reason = "merchant has been without visibility since expiry"
                facts.append(f"lapsed: {days_since_expiry}d ago")
                if lapsed:
                    facts.append(f"{lapsed} new customers missed since expiry")
                perf_dip = payload.get("perf_dip_pct", 0)
                if perf_dip:
                    facts.append(f"perf drop: {perf_dip:.0%} since expiry")
            else:
                return "", "", []

    # ── dormant_with_vera / curious_ask_due ───────────────────────────────
    elif kind in {"dormant_with_vera", "curious_ask_due"}:
        days = payload.get("days_since_last_merchant_message", 0)
        last_topic = payload.get("last_topic", "")
        ask_template = payload.get("ask_template", "what_is_working_this_week")

        if kind == "dormant_with_vera" and not days:
            return "", "", []
        primary = f"weekly curiosity check-in — low-stakes question to re-engage"
        now_reason = (
            f"{days}d since last merchant message" if days
            else "weekly cadence to gather business intel"
        )
        if last_topic:
            facts.append(f"last topic: {last_topic}")
        if days:
            facts.append(f"dormant for {days}d")

    # ── review_theme_emerged ──────────────────────────────────────────────
    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "")
        occurrences = payload.get("occurrences_30d", 0)
        trend = payload.get("trend", "")
        quote = payload.get("common_quote", "")
        sentiment = payload.get("sentiment", "neg")

        if not theme or not occurrences:
            # Also check merchant.review_themes
            review_themes = merchant.get("review_themes", [])
            if review_themes:
                top = review_themes[0]
                theme = top.get("theme", "")
                occurrences = top.get("occurrences_30d", 0)
                quote = top.get("common_quote", "")
                sentiment = top.get("sentiment", "neg")

        if not theme:
            return "", "", []

        primary = f"{'negative' if sentiment == 'neg' else 'positive'} review pattern: {theme}"
        now_reason = f"{occurrences} mentions in 30d{'  — ' + trend if trend else ''}"
        facts.append(f"theme: {theme} ({occurrences}x in 30d)")
        if quote:
            facts.append(f"common quote: \"{quote[:80]}\"")
        if trend:
            facts.append(f"trend: {trend}")

    # ── festival_upcoming / ipl_match_today / weather_heatwave ───────────
    elif kind in {"festival_upcoming", "ipl_match_today",
                  "weather_heatwave", "local_news_event"}:
        event_name = (
            payload.get("festival") or
            payload.get("match") or
            payload.get("event_name", "upcoming event")
        )
        days_until = payload.get("days_until", 0)
        match_time = payload.get("match_time_iso", "")
        is_weeknight = payload.get("is_weeknight", True)

        if not event_name:
            return "", "", []

        if kind == "ipl_match_today":
            venue = payload.get("venue", "")
            primary = f"{event_name} today at {venue}" if venue else f"{event_name} today"
            now_reason = (
                "contrarian play: Saturday match = delivery > dine-in"
                if not is_weeknight else
                "match night = dine-in + delivery upsell opportunity"
            )
            facts.append(f"match: {event_name}")
            facts.append(f"weeknight: {is_weeknight}")
            if match_time:
                facts.append(f"time: {_format_time(match_time)}")
        else:
            primary = f"{event_name} in {days_until}d" if days_until else event_name
            now_reason = (
                f"{days_until}d to prepare" if days_until
                else "event approaching; time to prepare campaign"
            )
            if days_until:
                facts.append(f"days until: {days_until}")

    # ── competitor_opened ─────────────────────────────────────────────────
    elif kind == "competitor_opened":
        comp_name = payload.get("competitor_name", "a competitor")
        distance = payload.get("distance_km", 0)
        their_offer = payload.get("their_offer", "")
        opened = payload.get("opened_date", "")

        primary = f"{comp_name} opened {distance:.1f}km away"
        now_reason = "new competitor = urgency to sharpen offer + differentiation"
        facts.append(f"distance: {distance:.1f}km")
        if their_offer:
            facts.append(f"their offer: {their_offer}")
        if opened:
            facts.append(f"opened: {opened}")

    # ── category_trend_movement / category_seasonal ───────────────────────
    elif kind in {"category_trend_movement", "category_seasonal"}:
        trends = payload.get("trends", [])
        trend_signals = category.get("trend_signals", [])

        if trends:
            top_trend = trends[0] if isinstance(trends[0], str) else str(trends[0])
            primary = f"seasonal demand shift: {top_trend}"
            now_reason = "summer demand trends active now — shelf action recommended"
            for t in trends[:2]:
                facts.append(str(t))
        elif trend_signals:
            top = trend_signals[0]
            primary = f"search trend: '{top.get('query')}' up {top.get('delta_yoy', 0):.0%}"
            now_reason = "trending search = organic visibility opportunity"
            facts.append(f"trend: {top.get('query')} +{top.get('delta_yoy', 0):.0%}")
        else:
            return "", "", []

    # ── recall_due / appointment_tomorrow ─────────────────────────────────
    elif kind in {"recall_due", "appointment_tomorrow"}:
        if not customer:
            return "", "", []
        c_name = customer.get("identity", {}).get("name", "customer")
        c_state = customer.get("state", "")
        last_visit = customer.get("relationship", {}).get("last_visit", "")
        slots = payload.get("available_slots", [])
        service_due = payload.get("service_due", "recall")
        due_date = payload.get("due_date", "")

        if not slots and kind == "recall_due":
            return "", "", []  # no slots → can't book → no CTA

        primary = f"{c_name}'s {service_due.replace('_', ' ')} is due"
        now_reason = f"recall window just opened{'; ' + due_date if due_date else ''}"
        if last_visit:
            facts.append(f"last visit: {last_visit}")
        if slots:
            facts.append(f"{len(slots)} slot(s) available: {slots[0].get('label', '')}")
        if c_state:
            facts.append(f"customer state: {c_state}")

    # ── customer_lapsed_soft / customer_lapsed_hard ───────────────────────
    elif kind in {"customer_lapsed_soft", "customer_lapsed_hard"}:
        if not customer:
            return "", "", []
        c_name = customer.get("identity", {}).get("name", "customer")
        c_state = customer.get("state", "")
        days_since = payload.get("days_since_last_visit", 0)
        prev_focus = payload.get("previous_focus", "")
        services = customer.get("relationship", {}).get("services_received", [])

        if not days_since and not c_state:
            return "", "", []

        primary = f"{c_name} lapsed — {days_since}d since last visit"
        now_reason = "winback window: approach before 90d hard-churn threshold"
        if days_since:
            facts.append(f"days since visit: {days_since}")
        focus = prev_focus or (services[-1] if services else "")
        if focus:
            facts.append(f"previous focus: {focus}")
        if c_state:
            facts.append(f"customer state: {c_state}")

    # ── chronic_refill_due ────────────────────────────────────────────────
    elif kind == "chronic_refill_due":
        if not customer:
            return "", "", []
        c_name = customer.get("identity", {}).get("name", "customer")
        molecules = payload.get("molecule_list", [])
        run_out = payload.get("stock_runs_out_iso", "")
        delivery_saved = payload.get("delivery_address_saved", False)

        if not molecules:
            return "", "", []

        primary = f"{c_name}'s chronic medicines run out {_format_date(run_out)}"
        now_reason = "refill window — act before stock runs out"
        facts.append(f"medicines: {', '.join(molecules)}")
        if run_out:
            facts.append(f"runs out: {_format_date(run_out)}")
        if delivery_saved:
            facts.append("home delivery address: saved")

    # ── supply_alert ──────────────────────────────────────────────────────
    elif kind == "supply_alert":
        molecule = payload.get("molecule", "")
        batches = payload.get("affected_batches", [])
        manufacturer = payload.get("manufacturer", "")

        if not molecule and not batches:
            return "", "", []

        primary = f"voluntary recall: {molecule} batches {', '.join(batches[:2])}"
        now_reason = "recall issued; merchant must inform affected patients"
        facts.append(f"molecule: {molecule}")
        if batches:
            facts.append(f"batches: {', '.join(batches)}")
        if manufacturer:
            facts.append(f"manufacturer: {manufacturer}")

    # ── trial_followup / wedding_package_followup ─────────────────────────
    elif kind in {"trial_followup", "wedding_package_followup", "bridal_followup"}:
        if not customer:
            return "", "", []
        c_name = customer.get("identity", {}).get("name", "customer")
        trial_date = payload.get("trial_completed") or payload.get("trial_date", "")
        wedding_date = (
            payload.get("wedding_date") or
            customer.get("preferences", {}).get("wedding_date", "")
        )
        days_to_wedding = payload.get("days_to_wedding", 0)
        next_step = payload.get("next_step_window_open", "")
        next_sessions = payload.get("next_session_options", [])

        primary = f"{c_name} trial complete — follow-up window open"
        now_reason = (
            f"{days_to_wedding}d to wedding — skin-prep window now"
            if days_to_wedding else
            "follow-up window: convert trial to booking"
        )
        if trial_date:
            facts.append(f"trial date: {trial_date}")
        if wedding_date:
            facts.append(f"wedding: {wedding_date}")
        if next_step:
            facts.append(f"next step: {next_step.replace('_', ' ')}")
        if next_sessions:
            facts.append(f"slot: {next_sessions[0].get('label', '')}")

    # ── active_planning_intent ────────────────────────────────────────────
    elif kind == "active_planning_intent":
        topic = payload.get("intent_topic", "")
        last_msg = payload.get("merchant_last_message", "")

        if not topic:
            return "", "", []

        primary = f"merchant actively planning: {topic.replace('_', ' ')}"
        now_reason = "merchant explicitly asked — highest engagement window"
        if last_msg:
            facts.append(f"merchant said: \"{last_msg[:80]}\"")
        facts.append(f"topic: {topic.replace('_', ' ')}")

    # ── gbp_unverified ────────────────────────────────────────────────────
    elif kind == "gbp_unverified":
        estimated_uplift = payload.get("estimated_uplift_pct", 0)
        verification_path = payload.get("verification_path", "")

        primary = "Google Business Profile is unverified"
        now_reason = "unverified GBP = invisible to local search; quick win available"
        if estimated_uplift:
            facts.append(f"estimated uplift: {estimated_uplift:.0%} more visibility")
        if verification_path:
            facts.append(f"path: {verification_path.replace('_', ' ')}")
        facts.append("verified profiles get 2.7x more calls on average")

    # ── Unknown kind → no signal ───────────────────────────────────────────
    else:
        primary = f"trigger kind '{kind}' fired"
        now_reason = "trigger is active and unexpired"
        # Minimal — only if urgency is high enough
        if urgency < 4:
            return "", "", []

    return primary, now_reason, facts


# ---------------------------------------------------------------------------
# Supporting facts (additional grounded context)
# ---------------------------------------------------------------------------

def _build_supporting_facts(
    merchant: dict,
    category: dict,
    trg: dict,
    customer: Optional[dict],
) -> list[str]:
    """Return up to 3 grounded supporting facts not already in primary facts."""
    facts: list[str] = []
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    sub = merchant.get("subscription", {})
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    agg = merchant.get("customer_aggregate", {})
    identity = merchant.get("identity", {})

    # Peer gap fact
    ctr = perf.get("ctr", 0)
    peer_ctr = peer.get("avg_ctr", 0)
    if ctr and peer_ctr and abs(ctr - peer_ctr) / peer_ctr > 0.15:
        direction = "below" if ctr < peer_ctr else "above"
        facts.append(f"CTR {ctr:.1%} vs peer {peer_ctr:.1%} ({direction})")

    # Active offer fact
    if active_offers:
        facts.append(f"active offer: {active_offers[0].get('title', '')}")

    # Customer aggregate fact (for merchant-scope messages)
    if not customer:
        lapsed = agg.get("lapsed_180d_plus") or agg.get("lapsed_90d_plus")
        if lapsed:
            facts.append(f"{lapsed} lapsed customers in roster")

    return facts


# ---------------------------------------------------------------------------
# CTA recommendation
# ---------------------------------------------------------------------------

def _recommend_cta(
    kind: str,
    payload: dict,
    customer: Optional[dict],
) -> CTAType:
    if kind in _SLOT_CTA_KINDS:
        slots = payload.get("available_slots") or payload.get("next_session_options")
        if slots and len(slots) >= 2:
            return CTAType.MULTI_CHOICE_SLOT
        return CTAType.BINARY_YES_NO
    if kind in _CONFIRM_CTA_KINDS:
        return CTAType.BINARY_CONFIRM_CANCEL
    if kind in _OPEN_CTA_KINDS:
        return CTAType.OPEN_ENDED
    if kind in _BINARY_CTA_KINDS:
        return CTAType.BINARY_YES_NO
    return CTAType.OPEN_ENDED


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

def _compute_priority(
    urgency: int,
    kind: str,
    merchant: dict,
    trg: dict,
) -> int:
    """Return priority 1–5. Deterministic function of context."""
    base = urgency  # 1–5 directly from trigger

    # Boost: subscription about to expire
    sub = merchant.get("subscription", {})
    days_rem = sub.get("days_remaining", 999)
    if sub.get("status") == "active" and days_rem <= 7:
        base = max(base, 5)

    # Boost: customer will lose access today
    payload = trg.get("payload", {})
    run_out = payload.get("stock_runs_out_iso", "")
    if run_out and kind == "chronic_refill_due":
        base = max(base, 4)

    # Boost: supply/compliance alerts
    if kind in {"supply_alert", "regulation_change"}:
        base = max(base, 5)

    # Boost: merchant actively engaged (planning intent)
    if kind == "active_planning_intent":
        base = max(base, 4)

    # Dampen: low-urgency curiosity check-ins
    if kind in {"curious_ask_due", "dormant_with_vera"} and urgency <= 2:
        base = min(base, 2)

    return min(max(base, 1), 5)


# ---------------------------------------------------------------------------
# Consent mapping
# ---------------------------------------------------------------------------

_KIND_CONSENT_MAP: dict[str, list[str]] = {
    "recall_due":            ["recall_reminders", "appointment_reminders"],
    "appointment_tomorrow":  ["appointment_reminders"],
    "chronic_refill_due":    ["refill_reminders", "recall_reminders"],
    "trial_followup":        ["appointment_reminders", "bridal_package_followup",
                              "kids_program_updates", "program_updates"],
    "wedding_package_followup": ["bridal_package_followup", "appointment_reminders"],
    "bridal_followup":       ["bridal_package_followup", "appointment_reminders"],
    "customer_lapsed_soft":  ["promotional_offers", "winback_offers", "recall_reminders"],
    "customer_lapsed_hard":  ["promotional_offers", "winback_offers", "renewal_reminders"],
    "supply_alert":          ["recall_alerts", "refill_reminders"],
}

_DEFAULT_CONSENT_BUCKETS = {
    "promotional_offers", "appointment_reminders", "recall_reminders",
    "program_updates", "renewal_reminders", "health_content",
    "program_updates", "lunch_thali_updates", "match_night_specials",
}


def _consent_covers_kind(kind: str, consent_scope: list[str]) -> bool:
    """Return True if the customer's consent scope covers this trigger kind."""
    required = _KIND_CONSENT_MAP.get(kind)
    if not required:
        # For unlisted kinds, any consent bucket is fine
        return bool(consent_scope)
    scope_set = set(consent_scope)
    return bool(scope_set & set(required))


# ---------------------------------------------------------------------------
# Actionability check
# ---------------------------------------------------------------------------

def _payload_has_action(payload: dict, kind: str) -> bool:
    """Return True if the payload contains enough data to compose a useful message."""
    checks: dict[str, bool] = {
        "recall_due": bool(payload.get("available_slots")),
        "appointment_tomorrow": True,  # always actionable
        "chronic_refill_due": bool(payload.get("molecule_list")),
        "supply_alert": bool(payload.get("affected_batches") or payload.get("molecule")),
        "perf_dip": bool(payload.get("delta_pct")),
        "perf_spike": bool(payload.get("delta_pct")),
        "renewal_due": True,
        "winback_eligible": True,
        "active_planning_intent": bool(payload.get("intent_topic")),
        "competitor_opened": bool(payload.get("competitor_name")),
        "research_digest": True,  # checked via digest item in signal analysis
        "customer_lapsed_soft": True,
        "customer_lapsed_hard": True,
    }
    return checks.get(kind, True)  # unknown kinds get benefit of the doubt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(
    result: StructuredDecision,
    reason: SkipReason,
    detail: str,
) -> None:
    result.should_send = False
    result.decision = Decision.SKIP
    result.skip_reason = reason
    result.skip_detail = detail
    result.gate_failed(reason.value, detail)


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse ISO datetime string; return None on failure."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _format_date(iso: str) -> str:
    """Return human-readable date from ISO string."""
    dt = _parse_iso(iso)
    if not dt:
        return iso
    return dt.strftime("%-d %b") if hasattr(dt, "strftime") else iso


def _format_time(iso: str) -> str:
    dt = _parse_iso(iso)
    if not dt:
        return iso
    return dt.strftime("%-I:%M%p IST").lower() if hasattr(dt, "strftime") else iso


def _log_decision(result: StructuredDecision) -> None:
    if result.should_send:
        log.info(
            "SEND trigger=%s merchant=%s kind=%s priority=%d signal=%r",
            result.trigger_id, result.merchant_id, result.trigger_kind,
            result.priority, result.primary_signal,
        )
    else:
        log.debug(
            "SKIP trigger=%s merchant=%s reason=%s detail=%s",
            result.trigger_id, result.merchant_id,
            result.skip_reason, result.skip_detail,
        )
