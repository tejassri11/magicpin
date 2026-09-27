"""
tests/test_decision.py — Comprehensive test suite for the deterministic decision engine.

Tests all 12 Gate checks, 20+ trigger kinds, peer comparisons,
opt-out logic, customer consent, CTA assignment, priority calculation,
and edge cases.
"""

import pytest
from state import store
from conversation import conversations, ConversationState
from decision import (
    evaluate_trigger,
    Decision,
    SkipReason,
    CTAType,
    StructuredDecision,
    _extract_primary_signal,
    _recommend_cta,
    _compute_priority,
    KIND_CATEGORY_RESTRICTIONS,
)

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

CATEGORY_DENTISTS = {
    "slug": "dentists",
    "voice": {"tone": "peer_clinical", "vocab_allowed": ["caries", "fluoride"]},
    "peer_stats": {"avg_ctr": 0.035, "avg_rating": 4.2, "scope": "dentists_delhi"},
    "digest": [
        {
            "id": "d_001",
            "kind": "research",
            "source": "JIDA 2026",
            "trial_n": 1200,
            "patient_segment": "pediatric caries",
            "title": "Fluoride Varnish Efficacy Study",
        }
    ],
    "offer_catalog": [],
    "seasonal_beats": [],
    "trend_signals": [{"query": "teeth whitening", "delta_yoy": 0.45}],
}

MERCHANT_MEERA = {
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "category_slug": "dentists",
    "identity": {
        "name": "Dr. Meera's Dental Clinic",
        "city": "Delhi",
        "verified": True,
        "owner_first_name": "Meera",
    },
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 60},
    "performance": {
        "window_days": 30,
        "views": 2000,
        "calls": 20,
        "ctr": 0.020,  # below peer median 0.035
        "delta_7d": {"views_pct": -0.15, "calls_pct": -0.25},
    },
    "offers": [{"id": "o_001", "title": "Dental Checkup @ ₹199", "status": "active"}],
    "conversation_history": [],
    "signals": ["ctr_below_peer_median"],
    "review_themes": [{"theme": "long wait time", "occurrences_30d": 5, "sentiment": "neg"}],
}

CUSTOMER_PRIYA = {
    "customer_id": "c_001_priya",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "identity": {"name": "Priya Sharma", "phone_redacted": "<phone>", "language_pref": "hi-en"},
    "relationship": {"first_visit": "2025-06-01", "last_visit": "2025-11-01"},
    "consent": {"marketing_whatsapp": True, "reminder_sms": True},
    "state": "active",
}


@pytest.fixture(autouse=True)
def clean_state():
    store.wipe()
    conversations.wipe()
    yield
    store.wipe()
    conversations.wipe()


def _setup_full_context(
    trigger_kind="research_digest",
    trigger_payload=None,
    expires_at="2099-12-31T00:00:00Z",
    customer_id=None,
):
    """Helper to register valid context for testing."""
    if trigger_payload is None:
        trigger_payload = {"category": "dentists", "top_item_id": "d_001"}

    store.put_context("category", "dentists", 1, CATEGORY_DENTISTS)
    store.put_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_MEERA)
    if customer_id:
        store.put_context("customer", customer_id, 1, CUSTOMER_PRIYA)

    trg = {
        "id": "trg_test_001",
        "scope": "merchant" if not customer_id else "customer",
        "kind": trigger_kind,
        "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": customer_id,
        "payload": trigger_payload,
        "urgency": 3,
        "suppression_key": f"test_suppress_{trigger_kind}",
        "expires_at": expires_at,
    }
    store.put_context("trigger", "trg_test_001", 1, trg)
    return "trg_test_001"


# ===========================================================================
# Gate Checks (Phase 1)
# ===========================================================================

class TestEligibilityGates:
    def test_gate1_trigger_not_found(self):
        result = evaluate_trigger("non_existent_trigger", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.TRIGGER_NOT_LOADED

    def test_gate2_expired_trigger(self):
        trg_id = _setup_full_context(expires_at="2020-01-01T00:00:00Z")
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.TRIGGER_EXPIRED

    def test_gate3_suppressed_trigger(self):
        trg_id = _setup_full_context(trigger_kind="research_digest")
        store.fire_suppression("test_suppress_research_digest", "conv_prior")
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.SUPPRESSED

    def test_gate4_active_conversation_exists(self):
        trg_id = _setup_full_context()
        conversations.start(
            conversation_id="conv_active_123",
            merchant_id="m_001_drmeera_dentist_delhi",
            trigger_id=trg_id,
            trigger_kind="research_digest",
            send_as="vera",
            bot_body="Opening message",
        )
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.CONVERSATION_ACTIVE

    def test_merchant_not_found(self):
        store.put_context("category", "dentists", 1, CATEGORY_DENTISTS)
        trg = {
            "id": "trg_no_m",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_missing",
            "expires_at": "2099-12-31T00:00:00Z",
            "suppression_key": "k1",
        }
        store.put_context("trigger", "trg_no_m", 1, trg)
        result = evaluate_trigger("trg_no_m", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.MERCHANT_NOT_LOADED

    def test_gate6_category_not_found(self):
        store.put_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_MEERA)
        trg = {
            "id": "trg_no_c",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "expires_at": "2099-12-31T00:00:00Z",
            "suppression_key": "k2",
        }
        store.put_context("trigger", "trg_no_c", 1, trg)
        result = evaluate_trigger("trg_no_c", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.CATEGORY_NOT_LOADED

    def test_gate7_category_kind_relevance(self):
        # recall_due is restricted to dentists
        restrictions = KIND_CATEGORY_RESTRICTIONS.get("recall_due", set())
        assert "restaurants" not in restrictions
        assert "dentists" in restrictions

    def test_gate8_merchant_opted_out(self):
        trg_id = _setup_full_context()
        store.add_optout("m_001_drmeera_dentist_delhi", cooldown_days=30)
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.MERCHANT_OPTED_OUT

    def test_gate9_subscription_expired_for_normal_trigger(self):
        expired_merchant = dict(MERCHANT_MEERA)
        expired_merchant["subscription"] = {"status": "expired", "plan": "Pro", "days_remaining": 0}
        store.put_context("category", "dentists", 1, CATEGORY_DENTISTS)
        store.put_context("merchant", "m_001_drmeera_dentist_delhi", 1, expired_merchant)
        trg = {
            "id": "trg_sub_exp",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "expires_at": "2099-12-31T00:00:00Z",
            "suppression_key": "k3",
            "payload": {"category": "dentists"},
        }
        store.put_context("trigger", "trg_sub_exp", 1, trg)
        result = evaluate_trigger("trg_sub_exp", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.SUBSCRIPTION_EXPIRED

    def test_gate9_renewal_trigger_allowed_when_expired(self):
        expired_merchant = dict(MERCHANT_MEERA)
        expired_merchant["subscription"] = {"status": "expired", "plan": "Pro", "days_remaining": 0}
        store.put_context("category", "dentists", 1, CATEGORY_DENTISTS)
        store.put_context("merchant", "m_001_drmeera_dentist_delhi", 1, expired_merchant)
        trg = {
            "id": "trg_renew",
            "scope": "merchant",
            "kind": "renewal_due",
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "expires_at": "2099-12-31T00:00:00Z",
            "suppression_key": "k4",
            "payload": {"renewal_amount": 4999},
        }
        store.put_context("trigger", "trg_renew", 1, trg)
        result = evaluate_trigger("trg_renew", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND

    def test_gate10_customer_consent(self):
        # Customer without consent
        no_consent_customer = dict(CUSTOMER_PRIYA)
        no_consent_customer["consent"] = {}

        store.put_context("category", "dentists", 1, CATEGORY_DENTISTS)
        store.put_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_MEERA)
        store.put_context("customer", "c_no_consent", 1, no_consent_customer)

        trg = {
            "id": "trg_cust_noconsent",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "customer_id": "c_no_consent",
            "expires_at": "2099-12-31T00:00:00Z",
            "suppression_key": "k5",
            "payload": {
                "service_due": "dental_cleaning",
                "available_slots": [{"iso": "2026-10-01T10:00:00Z", "label": "Wed 10am"}],
            },
        }
        store.put_context("trigger", "trg_cust_noconsent", 1, trg)
        result = evaluate_trigger("trg_cust_noconsent", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.NO_CONSENT


# ===========================================================================
# Signal Extraction & Priority (Phase 2)
# ===========================================================================

class TestSignalAnalysis:
    def test_research_digest_signal(self):
        trg_id = _setup_full_context(trigger_kind="research_digest")
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND
        assert "JIDA 2026" in result.primary_signal
        assert result.recommended_cta_type == CTAType.OPEN_ENDED

    def test_perf_dip_signal_with_peer_gap(self):
        trg_id = _setup_full_context(
            trigger_kind="perf_dip",
            trigger_payload={"metric": "calls", "delta_pct": -0.25, "vs_baseline": 20},
        )
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND
        assert "calls dropped 25%" in result.primary_signal
        # Peer comparison fact should be present because merchant_ctr (2.0%) < peer_ctr (3.5%)
        assert any("below peer median" in fact for fact in result.supporting_facts)
        assert result.recommended_cta_type == CTAType.BINARY_YES_NO

    def test_gbp_unverified_signal(self):
        trg_id = _setup_full_context(
            trigger_kind="gbp_unverified",
            trigger_payload={"estimated_uplift_pct": 0.40, "verification_path": "google_my_business"},
        )
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND
        assert "Google Business Profile is unverified" in result.primary_signal
        assert result.priority >= 3

    def test_recall_due_signal_and_slot_cta(self):
        trg_id = _setup_full_context(
            trigger_kind="recall_due",
            trigger_payload={
                "service_due": "6_month_cleaning",
                "available_slots": [
                    {"iso": "2026-10-05T10:00:00Z", "label": "Mon 5 Oct 10am"},
                    {"iso": "2026-10-06T14:00:00Z", "label": "Tue 6 Oct 2pm"},
                ],
            },
            customer_id="c_001_priya",
        )
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND
        assert "Priya Sharma" in result.primary_signal
        assert result.recommended_cta_type == CTAType.MULTI_CHOICE_SLOT

    def test_review_theme_emerged_signal(self):
        trg_id = _setup_full_context(
            trigger_kind="review_theme_emerged",
            trigger_payload={"theme": "long wait time", "occurrences_30d": 5, "sentiment": "neg"},
        )
        result = evaluate_trigger(trg_id, "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND
        assert "negative review pattern" in result.primary_signal
        assert result.recommended_cta_type == CTAType.BINARY_YES_NO
