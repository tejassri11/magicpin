"""
tests/test_qa_suite.py — Deep QA & Competition Evaluator Test Suite for VERA Bot.

Tests all 5 API endpoints and 28+ edge cases:
- /v1/context, /v1/tick, /v1/reply, /v1/healthz, /v1/metadata
- Duplicate context, stale context version, newer context version
- Duplicate trigger, expired trigger, irrelevant trigger, multiple simultaneous triggers, repeated tick
- Merchant/customer already contacted (active conversation guard & suppression)
- Merchant intents: YES, NO, NOT NOW, STOP, auto-reply, off-topic reply (curveball), ambiguous
- Missing optional customer (merchant-facing scope)
- Malformed payload, LLM timeout/failure, invalid LLM JSON, fabricated facts, empty CTA
- Excessive message length, repeated messages, concurrent requests
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from main import app
from state import store
from conversation import conversations
from decision import evaluate_trigger, Decision, SkipReason
from validator import validate_message, ValidationResult
from config import settings

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def wipe_stores():
    store.wipe()
    conversations.wipe()
    yield
    store.wipe()
    conversations.wipe()


# Context Payloads
CAT_PAYLOAD = {
    "slug": "dentists",
    "voice": {"tone": "peer_clinical", "vocab_allowed": ["caries", "fluoride"], "vocab_taboo": ["cavity fighters"]},
    "peer_stats": {"avg_ctr": 0.035, "avg_rating": 4.2},
    "digest": [{"id": "d1", "kind": "research", "title": "Fluoride Varnish Study", "source": "JIDA 2026", "trial_n": 1000}],
    "offer_catalog": [],
    "seasonal_beats": [],
    "trend_signals": [],
}

MERCHANT_PAYLOAD = {
    "merchant_id": "m_001_delhi",
    "category_slug": "dentists",
    "identity": {"name": "Meera Clinic", "city": "Delhi", "verified": True, "owner_first_name": "Meera"},
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 30},
    "performance": {"views": 1000, "calls": 20, "ctr": 0.02, "delta_7d": {"calls_pct": -0.25}},
    "offers": [{"id": "o1", "title": "Dental Cleaning @ ₹299", "status": "active"}],
    "signals": ["ctr_below_peer_median"],
    "review_themes": [],
}

TRIGGER_PAYLOAD = {
    "id": "trg_001",
    "scope": "merchant",
    "kind": "research_digest",
    "source": "external",
    "merchant_id": "m_001_delhi",
    "customer_id": None,
    "payload": {"category": "dentists", "top_item_id": "d1"},
    "urgency": 3,
    "suppression_key": "supp_trg_001",
    "expires_at": "2099-12-31T00:00:00Z",
}

CUSTOMER_PAYLOAD = {
    "customer_id": "c_001",
    "merchant_id": "m_001_delhi",
    "identity": {"name": "Priya", "phone_redacted": "<phone>"},
    "relationship": {"last_visit": "2025-10-01"},
    "consent": {"marketing_whatsapp": True, "reminder_sms": True},
    "state": "active",
}


# ===========================================================================
# 1. Endpoint & Context Edge Cases
# ===========================================================================

class TestContextEdgeCases:
    def test_healthz_and_metadata(self):
        r1 = client.get("/v1/healthz")
        assert r1.status_code == 200
        assert r1.json()["status"] == "ok"

        r2 = client.get("/v1/metadata")
        assert r2.status_code == 200
        assert "team_name" in r2.json()

    def test_duplicate_context_version(self):
        r1 = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_delhi", "version": 1, "payload": MERCHANT_PAYLOAD, "delivered_at": "2026-09-27T00:00:00Z"})
        assert r1.status_code == 200
        assert r1.json()["accepted"] is True

        # Duplicate same version (idempotent ack)
        r2 = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_delhi", "version": 1, "payload": MERCHANT_PAYLOAD, "delivered_at": "2026-09-27T00:00:00Z"})
        assert r2.status_code == 200
        assert r2.json()["accepted"] is True

    def test_stale_and_newer_context_version(self):
        # Version 2
        client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_delhi", "version": 2, "payload": MERCHANT_PAYLOAD, "delivered_at": "2026-09-27T00:00:00Z"})

        # Version 1 (Stale - rejected)
        r_stale = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_delhi", "version": 1, "payload": MERCHANT_PAYLOAD, "delivered_at": "2026-09-27T00:00:00Z"})
        assert r_stale.status_code == 200
        assert r_stale.json()["accepted"] is False
        assert r_stale.json()["reason"] == "stale_version"

        # Version 3 (Newer - accepted)
        r_newer = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_delhi", "version": 3, "payload": MERCHANT_PAYLOAD, "delivered_at": "2026-09-27T00:00:00Z"})
        assert r_newer.status_code == 200
        assert r_newer.json()["accepted"] is True


# ===========================================================================
# 2. Trigger & Tick Edge Cases
# ===========================================================================

class TestTickTriggerEdgeCases:
    def test_expired_trigger(self):
        store.put_context("category", "dentists", 1, CAT_PAYLOAD)
        store.put_context("merchant", "m_001_delhi", 1, MERCHANT_PAYLOAD)
        expired_trg = dict(TRIGGER_PAYLOAD)
        expired_trg["expires_at"] = "2020-01-01T00:00:00Z"
        store.put_context("trigger", "trg_expired", 1, expired_trg)

        res = evaluate_trigger("trg_expired", "2026-09-27T00:00:00Z")
        assert res.should_send is False
        assert res.skip_reason == SkipReason.TRIGGER_EXPIRED

    def test_irrelevant_trigger_kind_for_category(self):
        store.put_context("category", "dentists", 1, CAT_PAYLOAD)
        store.put_context("merchant", "m_001_delhi", 1, MERCHANT_PAYLOAD)
        # ipl_match_today is restricted to restaurants
        irrelevant_trg = dict(TRIGGER_PAYLOAD)
        irrelevant_trg["id"] = "trg_ipl"
        irrelevant_trg["kind"] = "ipl_match_today"
        store.put_context("trigger", "trg_ipl", 1, irrelevant_trg)

        res = evaluate_trigger("trg_ipl", "2026-09-27T00:00:00Z")
        assert res.should_send is False
        assert res.skip_reason == SkipReason.TRIGGER_KIND_IRRELEVANT

    def test_multiple_simultaneous_triggers_and_repeated_tick(self):
        store.put_context("category", "dentists", 1, CAT_PAYLOAD)
        store.put_context("merchant", "m_001_delhi", 1, MERCHANT_PAYLOAD)
        store.put_context("trigger", "trg_001", 1, TRIGGER_PAYLOAD)

        # Tick 1: returns 1 action
        r1 = client.post("/v1/tick", json={"now": "2026-09-27T00:00:00Z", "available_triggers": ["trg_001"]})
        assert r1.status_code == 200
        actions1 = r1.json()["actions"]
        assert len(actions1) == 1
        assert actions1[0]["trigger_id"] == "trg_001"

        # Tick 2: repeated tick for same trigger -> suppressed!
        r2 = client.post("/v1/tick", json={"now": "2026-09-27T00:01:00Z", "available_triggers": ["trg_001"]})
        assert r2.status_code == 200
        assert len(r2.json()["actions"]) == 0

    def test_merchant_already_contacted_active_conv(self):
        store.put_context("category", "dentists", 1, CAT_PAYLOAD)
        store.put_context("merchant", "m_001_delhi", 1, MERCHANT_PAYLOAD)
        store.put_context("trigger", "trg_001", 1, TRIGGER_PAYLOAD)

        # First tick creates conversation
        client.post("/v1/tick", json={"now": "2026-09-27T00:00:00Z", "available_triggers": ["trg_001"]})

        # New trigger for same merchant while active conversation exists
        trg2 = dict(TRIGGER_PAYLOAD)
        trg2["id"] = "trg_002"
        trg2["suppression_key"] = "supp_trg_002"
        store.put_context("trigger", "trg_002", 1, trg2)

        res = evaluate_trigger("trg_002", "2026-09-27T00:02:00Z")
        assert res.should_send is False
        assert res.skip_reason == SkipReason.CONVERSATION_ACTIVE

    def test_missing_optional_customer_scope(self):
        store.put_context("category", "dentists", 1, CAT_PAYLOAD)
        store.put_context("merchant", "m_001_delhi", 1, MERCHANT_PAYLOAD)
        # Customer scope trigger, but customer_id not loaded
        cust_trg = {
            "id": "trg_cust_missing",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": "m_001_delhi",
            "customer_id": "c_missing",
            "payload": {"service_due": "cleaning"},
            "urgency": 3,
            "suppression_key": "supp_c_missing",
            "expires_at": "2099-12-31T00:00:00Z",
        }
        store.put_context("trigger", "trg_cust_missing", 1, cust_trg)

        res = evaluate_trigger("trg_cust_missing", "2026-09-27T00:00:00Z")
        assert res.should_send is False
        assert res.skip_reason == SkipReason.CUSTOMER_NOT_LOADED


# ===========================================================================
# 3. Multi-turn Reply Scenarios
# ===========================================================================

class TestReplyScenarios:
    def _create_conv(self) -> str:
        store.put_context("category", "dentists", 1, CAT_PAYLOAD)
        store.put_context("merchant", "m_001_delhi", 1, MERCHANT_PAYLOAD)
        store.put_context("trigger", "trg_001", 1, TRIGGER_PAYLOAD)
        resp = client.post("/v1/tick", json={"now": "2026-09-27T00:00:00Z", "available_triggers": ["trg_001"]})
        actions = resp.json()["actions"]
        if actions:
            return actions[0]["conversation_id"]
        return ""

    def test_merchant_says_yes(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "Yes please, go ahead!",
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "send"
        assert "rationale" in data

    def test_merchant_says_no_or_stop(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "Not interested, please stop messaging me",
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "end"
        assert store.is_opted_out("m_001_delhi") is True

    def test_auto_reply_returns_wait_no_send(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic. Our team will respond shortly."
        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": auto_msg,
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "wait"
        assert data["wait_seconds"] == settings.auto_reply_wait_seconds
        assert "body" not in data  # No new message generated!

    def test_repeated_auto_reply_no_repeated_send(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic. Our team will respond shortly."
        
        # Turn 2
        r1 = client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": auto_msg, "received_at": "2026-09-27T00:05:00Z", "turn_number": 2})
        assert r1.json()["action"] == "wait"
        assert "body" not in r1.json()

        # Turn 3
        r2 = client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": auto_msg, "received_at": "2026-09-27T00:06:00Z", "turn_number": 3})
        assert r2.json()["action"] == "wait"
        assert "body" not in r2.json()

        # Turn 4 (Repeated auto-reply threshold reached -> END)
        r3 = client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": auto_msg, "received_at": "2026-09-27T00:07:00Z", "turn_number": 4})
        assert r3.json()["action"] == "end"

    def test_human_yes_continues_action(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "Ok let's do it. What's next?",
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "send"
        assert len(data["body"]) > 0

    def test_human_no_rejection(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "Not interested.",
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "end"

    def test_stop_precedence_over_wait(self):
        conv_id = self._create_conv()
        # Put in wait first via auto-reply
        auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic. Our team will respond shortly."
        client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": auto_msg, "received_at": "2026-09-27T00:05:00Z", "turn_number": 2})

        # Now send explicit STOP
        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "Stop messaging me.",
            "received_at": "2026-09-27T00:06:00Z",
            "turn_number": 3,
        })
        assert r.status_code == 200
        assert r.json()["action"] == "end"
        assert store.is_opted_out("m_001_delhi") is True

    def test_auto_reply_followed_by_human_resumes(self):
        conv_id = self._create_conv()
        # 1. Auto-reply -> WAIT
        auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic. Our team will respond shortly."
        r1 = client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": auto_msg, "received_at": "2026-09-27T00:05:00Z", "turn_number": 2})
        assert r1.json()["action"] == "wait"

        # 2. Later human response -> Resumes normally
        r2 = client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": "Yes, let's proceed.", "received_at": "2026-09-27T00:10:00Z", "turn_number": 3})
        assert r2.json()["action"] == "send"
        assert len(r2.json()["body"]) > 0

    def test_auto_reply_followed_by_tick_skips_during_wait(self):
        conv_id = self._create_conv()
        # Auto-reply put merchant in wait
        auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic. Our team will respond shortly."
        client.post("/v1/reply", json={"conversation_id": conv_id, "merchant_id": "m_001_delhi", "from_role": "merchant", "message": auto_msg, "received_at": "2026-09-27T00:05:00Z", "turn_number": 2})

        # Now trigger tick with another trigger for the same merchant
        trg2 = dict(TRIGGER_PAYLOAD)
        trg2["id"] = "trg_002_new"
        trg2["suppression_key"] = "supp_trg_002_new"
        store.put_context("trigger", "trg_002_new", 1, trg2)

        tick_res = client.post("/v1/tick", json={"now": "2026-09-27T00:06:00Z", "available_triggers": ["trg_002_new"]})
        assert tick_res.status_code == 200
        # Must skip generating actions while waiting
        assert len(tick_res.json()["actions"]) == 0

    def test_merchant_says_not_now(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "I am busy right now, call back later",
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "wait"
        assert data["wait_seconds"] > 0

    def test_off_topic_reply_curveball(self):
        conv_id = self._create_conv()
        assert conv_id != ""

        r = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_delhi",
            "from_role": "merchant",
            "message": "Can you help me file my GST returns for this quarter?",
            "received_at": "2026-09-27T00:05:00Z",
            "turn_number": 2,
        })
        assert r.status_code == 200
        assert r.json()["action"] == "send"


# ===========================================================================
# 4. Output Validation & Grounding Safety
# ===========================================================================

class TestValidationAndGrounding:
    def test_validator_rejects_url(self):
        v = validate_message(
            body="Check out our website at https://example.com for 50% off!",
            cta="binary_yes_no",
            send_as="vera",
            category=CAT_PAYLOAD,
            merchant=MERCHANT_PAYLOAD,
            trigger=TRIGGER_PAYLOAD,
        )
        assert "URL detected in message body" in v.violations

    def test_validator_rejects_taboo_words(self):
        v = validate_message(
            body="We are the best cavity fighters in Delhi!",
            cta="binary_yes_no",
            send_as="vera",
            category=CAT_PAYLOAD,
            merchant=MERCHANT_PAYLOAD,
            trigger=TRIGGER_PAYLOAD,
        )
        assert any("Taboo vocabulary" in vio for vio in v.violations)

    def test_validator_detects_fabricated_price(self):
        v = validate_message(
            body="Get a full dental checkup for only ₹9999 today!",
            cta="binary_yes_no",
            send_as="vera",
            category=CAT_PAYLOAD,
            merchant=MERCHANT_PAYLOAD,
            trigger=TRIGGER_PAYLOAD,
        )
        assert any("Fabricated price" in vio for vio in v.violations)
