"""
prompts/trigger_prompts.py — Per-trigger-kind prompt variants.

Each variant tells the LLM:
  1. What angle to take (the "why now")
  2. Which compulsion levers to use
  3. What CTA shape is appropriate
  4. Any kind-specific grounding reminders

The composer selects the right variant via TRIGGER_PROMPT_MAP[trigger.kind].
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Individual prompt variant templates
# ---------------------------------------------------------------------------

RESEARCH_DIGEST = """
TRIGGER KIND: research_digest / regulation_change
WHY NOW: New research or compliance update just dropped in the category.

TASK:
- Lead with the specific finding or regulation — cite the EXACT source string from the digest item
- Anchor on the merchant's specific patient/customer cohort (e.g., "your high-risk adult patients")
- Include the trial size (trial_n) if present — it adds credibility
- End with an offer to do the work for them (draft patient-ed content, pull the abstract)
- CTA: open_ended (invite a yes/no to draft content for them)
- send_as: vera

COMPULSION LEVERS: specificity (numbers + source), reciprocity (I'll draft it for you), curiosity
TONE: peer_clinical — source citation at the end, technical vocab OK
"""

PERF_DIP = """
TRIGGER KIND: perf_dip
WHY NOW: A performance metric dropped significantly vs baseline.

TASK:
- State the exact drop: "calls down 50% this week vs last 4-week avg of 12"
- Compare to peer median if the merchant is below it
- Offer one concrete next action (e.g., revive a paused offer, post a Google update)
- Don't catastrophise — frame as something fixable
- CTA: binary_yes_no ("Want me to draft X? Reply YES")
- send_as: vera

COMPULSION LEVERS: loss aversion (the drop), effort externalization (I'll do it), specificity
"""

PERF_SPIKE = """
TRIGGER KIND: perf_spike
WHY NOW: A performance metric jumped — good news to share + opportunity to capitalise.

TASK:
- Congratulate briefly (1 sentence max), then pivot to opportunity
- Anchor on the specific spike number
- Suggest capitalising: run a campaign, add a post, activate a new offer
- CTA: open_ended
- send_as: vera

COMPULSION LEVERS: social proof (your numbers are up), momentum, reciprocity
"""

MILESTONE_REACHED = """
TRIGGER KIND: milestone_reached
WHY NOW: Merchant crossed a meaningful milestone (100 reviews, 500 visits, etc.)

TASK:
- Celebrate the milestone with the exact number
- Add context: how do they compare to peers?
- Suggest next milestone or a way to leverage the achievement (share it, add a post)
- CTA: open_ended
- send_as: vera

COMPULSION LEVERS: social proof, momentum, curiosity (what's next)
"""

FESTIVAL_UPCOMING = """
TRIGGER KIND: festival_upcoming
WHY NOW: A festival is approaching — opportunity for a category-relevant offer or campaign.

TASK:
- Name the festival and days-until (from payload)
- Suggest a specific offer or campaign appropriate for the category (use offer_catalog)
- Practical: what should they do TODAY to be ready?
- Don't be generic ("Happy Diwali!") — be actionable
- CTA: binary_yes_no
- send_as: vera

COMPULSION LEVERS: urgency (N days left), loss aversion (competitors will run campaigns)
"""

EVENT_EXTERNAL = """
TRIGGER KIND: ipl_match_today / weather_heatwave / local_news_event
WHY NOW: An external event is happening today that affects foot traffic or demand.

TASK:
- Name the specific event (match, weather, news item) from the payload
- Provide a CONTRARIAN or NUANCED insight — don't just say "use this event for a promo"
  Example: Saturday IPL match = people watch at home, so delivery-focused push beats dine-in promo
- Tie to an existing active offer or suggest one from the catalog
- Be specific about timing
- CTA: binary_yes_no
- send_as: vera

COMPULSION LEVERS: specificity (event name + time), counter-intuitive insight, loss aversion
"""

COMPETITOR_OPENED = """
TRIGGER KIND: competitor_opened
WHY NOW: A new competitor opened nearby (from trigger payload).

TASK:
- Use a curiosity hook ("Noticed something near you — want the details?")
- Do NOT invent competitor details — only use what's in the trigger payload
- Suggest one differentiation play (a better offer, a content post, improved profile)
- CTA: open_ended
- send_as: vera

COMPULSION LEVERS: curiosity (voyeur), loss aversion (losing customers), reciprocity
"""

CATEGORY_TREND = """
TRIGGER KIND: category_trend_movement
WHY NOW: A search trend is moving significantly in the merchant's category + city.

TASK:
- Cite the exact trend signal from category.trend_signals (query + delta_yoy)
- Frame as an opportunity the merchant can capture right now
- Suggest a matching offer or Google post
- CTA: open_ended
- send_as: vera

COMPULSION LEVERS: specificity (percentage), curiosity, effort externalization
"""

RENEWAL_DUE = """
TRIGGER KIND: renewal_due / winback_eligible
WHY NOW: Merchant's subscription is expiring or has recently lapsed.

TASK:
- State clearly how many days remain (from trigger payload or merchant.subscription)
- Frame the value at risk: visibility, leads, customer engagement
- Offer a specific renewal path or reactivation offer if in the payload
- No guilt-tripping — keep it factual and solution-focused
- CTA: binary_yes_no
- send_as: vera

COMPULSION LEVERS: loss aversion (losing access), urgency (N days), effort externalization
"""

DORMANT_CURIOUS = """
TRIGGER KIND: dormant_with_vera / curious_ask_due
WHY NOW: No recent merchant engagement, or it's time for the weekly curiosity-ask.

TASK:
- Ask ONE specific question about the merchant's business this week
  Example: "What service has been most asked-for at [business name] this week?"
- Offer to turn their answer into something useful (Google post, WhatsApp reply draft)
- Keep it low-stakes — no commitment required
- CTA: open_ended (they just need to answer the question)
- send_as: vera

COMPULSION LEVERS: asking-the-merchant (highest Cialdini hook for engaged merchants), reciprocity
"""

REVIEW_THEME = """
TRIGGER KIND: review_theme_emerged
WHY NOW: A pattern in recent reviews was detected (positive or negative).

TASK:
- Cite the SPECIFIC theme and occurrence count from the trigger payload
- For NEGATIVE themes: empathise briefly, suggest one fix, offer to draft a response
- For POSITIVE themes: celebrate, suggest amplifying with a post or offer
- Quote the common_quote from the payload (verbatim, in quotes) if available
- CTA: binary_yes_no
- send_as: vera

COMPULSION LEVERS: reciprocity ("I noticed this"), specificity (count + quote), effort externalization
"""

RECALL_DUE = """
TRIGGER KIND: recall_due / appointment_tomorrow
WHY NOW: A patient/customer's recall window has opened or they have an appointment tomorrow.

TASK:
- Address the CUSTOMER by first name
- State the recall/appointment clearly ("It's been X months since your last visit")
- Offer specific slots from trigger.payload.available_slots (use the label field)
- Include the active service+price offer from merchant.offers
- Honor language_pref: hi-en mix for "hi" customers
- CTA: multi_choice_slot (Reply 1 for slot A, 2 for slot B) OR binary_yes_no
- send_as: merchant_on_behalf (message comes FROM the merchant's number)

COMPULSION LEVERS: personalization, specific dates + price, low-friction multi-choice
"""

LAPSE_WINBACK = """
TRIGGER KIND: customer_lapsed_soft / customer_lapsed_hard
WHY NOW: A customer hasn't visited in 3+ months.

TASK:
- Address the CUSTOMER by first name — no shame framing
- Acknowledge the time elapsed without guilt ("It's been a while — happens to everyone")
- Connect to their LAST goal/service (from services_received)
- Offer something specific: a new class, a fresh offer, a free consultation
- Single binary CTA — low commitment, no auto-charge language
- Honor language_pref
- send_as: merchant_on_behalf

COMPULSION LEVERS: no-shame + warmth, goal-aware offer, single binary CTA (no commitment)
"""

CHRONIC_REFILL = """
TRIGGER KIND: chronic_refill_due
WHY NOW: A customer's chronic prescription medicines are about to run out.

TASK:
- Address the CUSTOMER or their caregiver (check age_band / channel in CustomerContext)
- Name ALL medicines explicitly (from trigger payload)
- State the exact run-out date
- Show total amount + discount (senior discount, delivery)
- Two-channel option: Reply CONFIRM or call phone number
- Honor language_pref: namaste for senior/formal customers
- send_as: merchant_on_behalf

COMPULSION LEVERS: specificity (medicine names, date, total+savings), convenience (home delivery)
"""

TRIAL_FOLLOWUP = """
TRIGGER KIND: trial_followup / wedding_package_followup
WHY NOW: A customer completed a trial or is in a key pre-event window.

TASK:
- Reference the specific trial/event (from trigger payload: trial_completed, wedding_date)
- Calculate days-until-event and use it as urgency anchor
- Suggest the next step in the service journey (from offer_catalog or payload)
- Honor the customer's preferred slot (from preferences)
- CTA: binary_yes_no to book / confirm
- send_as: merchant_on_behalf

COMPULSION LEVERS: specificity (days-to-event, price), relationship continuity, preferred slot
"""

# ---------------------------------------------------------------------------
# Default fallback
# ---------------------------------------------------------------------------

DEFAULT_VARIANT = """
TRIGGER KIND: {kind} (generic fallback)
WHY NOW: Use the trigger payload to determine the specific reason for messaging.

TASK:
- Identify the most important fact in the trigger payload
- Anchor the message on that fact (be specific — use the exact number, date, or event name)
- Offer one concrete next action
- CTA: open_ended
- send_as: vera

COMPULSION LEVERS: specificity, reciprocity
"""

# ---------------------------------------------------------------------------
# Trigger kind -> variant mapping
# ---------------------------------------------------------------------------

TRIGGER_PROMPT_MAP: dict[str, str] = {
    "research_digest": RESEARCH_DIGEST,
    "regulation_change": RESEARCH_DIGEST,
    "perf_dip": PERF_DIP,
    "perf_spike": PERF_SPIKE,
    "milestone_reached": MILESTONE_REACHED,
    "festival_upcoming": FESTIVAL_UPCOMING,
    "ipl_match_today": EVENT_EXTERNAL,
    "weather_heatwave": EVENT_EXTERNAL,
    "local_news_event": EVENT_EXTERNAL,
    "competitor_opened": COMPETITOR_OPENED,
    "category_trend_movement": CATEGORY_TREND,
    "renewal_due": RENEWAL_DUE,
    "winback_eligible": RENEWAL_DUE,
    "dormant_with_vera": DORMANT_CURIOUS,
    "curious_ask_due": DORMANT_CURIOUS,
    "review_theme_emerged": REVIEW_THEME,
    "recall_due": RECALL_DUE,
    "appointment_tomorrow": RECALL_DUE,
    "customer_lapsed_soft": LAPSE_WINBACK,
    "customer_lapsed_hard": LAPSE_WINBACK,
    "chronic_refill_due": CHRONIC_REFILL,
    "trial_followup": TRIAL_FOLLOWUP,
    "wedding_package_followup": TRIAL_FOLLOWUP,
    "active_planning_intent": DORMANT_CURIOUS,
    "seasonal_perf_dip": PERF_DIP,
    "supply_alert": RESEARCH_DIGEST,
}


def get_trigger_variant(trigger_kind: str) -> str:
    """Return the prompt variant for a given trigger kind."""
    variant = TRIGGER_PROMPT_MAP.get(trigger_kind)
    if variant is None:
        return DEFAULT_VARIANT.format(kind=trigger_kind)
    return variant
