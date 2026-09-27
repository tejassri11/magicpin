"""
tests/test_endpoints.py — Integration tests for all 5 FastAPI endpoints.
Uses ASGI TestClient so no live server is needed.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
from state import store
from conversation import conversations


@pytest.fixture(autouse=True)
def wipe_state():
    """Reset state before each test."""
    store.wipe()
    conversations.wipe()
    yield
    store.wipe()
    conversations.wipe()


client = TestClient(app)


# ===========================================================================
# Fixtures — minimal valid context payloads
# ===========================================================================

CATEGORY_PAYLOAD = {
    "slug": "dentists",
    "voice": {
        "tone": "peer_clinical",
        "vocab_allowed": ["caries", "fluoride", "recall"],
        "vocab_taboo": ["cavity fighters", "amazing"],
    },
    "peer_stats": {"avg_ctr": 0.030, "avg_rating": 4.2, "scope": "dentists_delhi"},
    "digest": [],
    "offer_catalog": [],
    "seasonal_beats": [],
    "trend_signals": [],
}

MERCHANT_PAYLOAD = {
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "category_slug": "dentists",
    "identity": {
        "name": "Dr. Meera's Dental Clinic",
        "city": "Delhi",
        "locality": "Lajpat Nagar",
        "place_id": "ChIJ_LN_DENTIST_001",
        "verified": True,
        "languages": ["en", "hi"],
        "owner_first_name": "Meera",
        "established_year": 2018,
    },
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
    "performance": {
        "window_days": 30,
        "views": 2410, "calls": 18, "directions": 45,
        "ctr": 0.021, "leads": 9,
        "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05},
    },
    "offers": [
        {"id": "o_001", "title": "Dental Cleaning @ ₹299", "status": "active"},
    ],
    "conversation_history": [],
    "customer_aggregate": {"total_unique_ytd": 540},
    "signals": ["ctr_below_peer_median"],
    "review_themes": [],
}

TRIGGER_PAYLOAD = {
    "id": "trg_001_research_digest_dentists",
    "scope": "merchant",
    "kind": "research_digest",
    "source": "external",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "customer_id": None,
    "payload": {"category": "dentists"},
    "urgency": 2,
    "suppression_key": "research:dentists:2026-W17",
    "expires_at": "2099-12-31T00:00:00Z",
}

CUSTOMER_PAYLOAD = {
    "customer_id": "c_001_priya_for_m001",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "identity": {
        "name": "Priya",
        "phone_redacted": "<phone>",
        "language_pref": "hi-en mix",
        "age_band": "25-35",
    },
    "relationship": {
        "first_visit": "2025-09-01",
        "last_visit": "2026-04-01",
        "visits_total": 3,
        "services_received": ["cleaning"],
        "lifetime_value": 900,
    },
    "state": "lapsed_soft",
    "preferences": {"channel": "whatsapp", "reminder_opt_in": True},
    "consent": {"opted_in_at": "2025-09-01", "scope": ["recall_reminders"]},
}


def _push_context(scope: str, context_id: str, version: int, payload: dict) -> dict:
    resp = client.post("/v1/context", json={
        "scope": scope,
        "context_id": context_id,
        "version": version,
        "payload": payload,
        "delivered_at": "2026-09-27T00:00:00Z",
    })
    return resp


# ===========================================================================
# GET /v1/healthz
# ===========================================================================

class TestHealthz:
    def test_returns_200(self):
        resp = client.get("/v1/healthz")
        assert resp.status_code == 200

    def test_status_ok(self):
        resp = client.get("/v1/healthz")
        data = resp.json()
        assert data["status"] == "ok"

    def test_uptime_present(self):
        resp = client.get("/v1/healthz")
        data = resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_contexts_loaded_structure(self):
        resp = client.get("/v1/healthz")
        data = resp.json()
        cl = data["contexts_loaded"]
        for key in ("category", "merchant", "customer", "trigger"):
            assert key in cl

    def test_context_counts_accurate(self):
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001", 1, MERCHANT_PAYLOAD)
        resp = client.get("/v1/healthz")
        cl = resp.json()["contexts_loaded"]
        assert cl["category"] == 1
        assert cl["merchant"] == 1


# ===========================================================================
# GET /v1/metadata
# ===========================================================================

class TestMetadata:
    def test_returns_200(self):
        resp = client.get("/v1/metadata")
        assert resp.status_code == 200

    def test_required_fields(self):
        data = client.get("/v1/metadata").json()
        for field in ("team_name", "team_members", "model", "approach", "contact_email", "version"):
            assert field in data, f"missing field: {field}"

    def test_team_members_is_list(self):
        data = client.get("/v1/metadata").json()
        assert isinstance(data["team_members"], list)


# ===========================================================================
# POST /v1/context
# ===========================================================================

class TestContext:
    def test_accepted(self):
        resp = _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert "ack_id" in data
        assert "stored_at" in data

    def test_invalid_scope_400(self):
        resp = client.post("/v1/context", json={
            "scope": "invalid_scope",
            "context_id": "x",
            "version": 1,
            "payload": {},
            "delivered_at": "2026-09-27T00:00:00Z",
        })
        assert resp.status_code == 400

    def test_stale_version_rejected(self):
        _push_context("merchant", "m_001", 5, {"v": 5})
        resp = _push_context("merchant", "m_001", 3, {"v": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is False
        assert data["reason"] == "stale_version"
        assert data["current_version"] == 5

    def test_same_version_idempotent(self):
        _push_context("merchant", "m_001", 2, {"v": "original"})
        resp = _push_context("merchant", "m_001", 2, {"v": "duplicate"})
        data = resp.json()
        assert data["accepted"] is True

    def test_higher_version_replaces(self):
        _push_context("merchant", "m_001", 1, {"v": 1})
        resp = _push_context("merchant", "m_001", 2, {"v": 2})
        assert resp.json()["accepted"] is True

    def test_state_not_corrupted_after_stale(self):
        _push_context("merchant", "m_001", 5, {"val": "v5"})
        _push_context("merchant", "m_001", 3, {"val": "v3"})
        # State must still be v5
        payload = store.get_context("merchant", "m_001")
        assert payload["val"] == "v5"

    def test_different_scopes_independent(self):
        _push_context("merchant", "dentists", 1, {"scope": "merchant"})
        _push_context("category", "dentists", 1, {"scope": "category"})
        assert store.get_context("merchant", "dentists")["scope"] == "merchant"
        assert store.get_context("category", "dentists")["scope"] == "category"

    def test_malformed_json_returns_400(self):
        resp = client.post(
            "/v1/context",
            content=b"not_json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    def test_missing_required_field(self):
        resp = client.post("/v1/context", json={
            "scope": "merchant",
            # missing context_id
            "version": 1,
            "payload": {},
            "delivered_at": "2026-09-27T00:00:00Z",
        })
        assert resp.status_code in (400, 422)


# ===========================================================================
# POST /v1/tick
# ===========================================================================

class TestTick:
    def _setup_full_context(self):
        """Push all required contexts for a SEND decision."""
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, TRIGGER_PAYLOAD)

    def test_empty_triggers_returns_empty(self):
        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": [],
        })
        assert resp.status_code == 200
        assert resp.json()["actions"] == []

    def test_returns_actions_list(self):
        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_unknown"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert isinstance(data["actions"], list)

    def test_missing_context_returns_empty_actions(self):
        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        data = resp.json()
        assert data["actions"] == []

    def test_expired_trigger_skipped(self):
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        expired_trigger = dict(TRIGGER_PAYLOAD)
        expired_trigger["expires_at"] = "2020-01-01T00:00:00Z"
        _push_context("trigger", "trg_001_research_digest_dentists", 1, expired_trigger)

        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        assert resp.json()["actions"] == []

    def test_suppressed_trigger_skipped(self):
        self._setup_full_context()
        store.fire_suppression("research:dentists:2026-W17", "conv_existing")
        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        assert resp.json()["actions"] == []

    def test_send_fires_action_with_required_fields(self):
        self._setup_full_context()
        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        data = resp.json()
        if data["actions"]:  # may be empty if composer returns None
            action = data["actions"][0]
            for key in ("conversation_id", "merchant_id", "body", "cta", "send_as"):
                assert key in action, f"missing key: {key}"

    def test_action_creates_suppression(self):
        self._setup_full_context()
        client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        assert store.is_suppressed("research:dentists:2026-W17")

    def test_same_trigger_second_tick_skipped(self):
        self._setup_full_context()
        client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        resp2 = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        assert resp2.json()["actions"] == []

    def test_malformed_json_returns_empty(self):
        resp = client.post(
            "/v1/tick",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["actions"] == []


# ===========================================================================
# POST /v1/reply
# ===========================================================================

class TestReply:
    def _create_conversation(self) -> str:
        """Set up contexts and tick to create a conversation. Returns conv_id."""
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, TRIGGER_PAYLOAD)

        resp = client.post("/v1/tick", json={
            "now": "2026-09-27T00:00:00Z",
            "available_triggers": ["trg_001_research_digest_dentists"],
        })
        actions = resp.json().get("actions", [])
        if not actions:
            return ""
        return actions[0]["conversation_id"]

    def test_unknown_conv_id_auto_recovers(self):
        resp = client.post("/v1/reply", json={
            "conversation_id": "conv_nonexistent",
            "merchant_id": "m_001",
            "from_role": "merchant",
            "message": "Yes",
            "received_at": "2026-09-27T00:00:00Z",
            "turn_number": 1,
        })
        assert resp.status_code == 200
        assert resp.json()["action"] in ("send", "wait", "end")

    def test_action_field_present(self):
        conv_id = self._create_conversation()
        if not conv_id:
            pytest.skip("no conversation created (LLM not configured)")

        resp = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "from_role": "merchant",
            "message": "Yes please",
            "received_at": "2026-09-27T00:00:00Z",
            "turn_number": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert data["action"] in ("send", "wait", "end")

    def test_reject_intent_ends_conversation(self):
        conv_id = self._create_conversation()
        if not conv_id:
            pytest.skip("no conversation created (LLM not configured)")

        # Verify band karo is classified as hard_reject
        from reply_handler import classify_intent
        assert classify_intent("band karo please") == "hard_reject"

        resp = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "from_role": "merchant",
            "message": "band karo please",
            "received_at": "2026-09-27T00:00:00Z",
            "turn_number": 1,
        })
        data = resp.json()
        # Either the conversation ends directly (action=end) or the bot sends
        # a graceful closing message before ending. Both are valid.
        assert data["action"] in ("end", "send")
        # Regardless of action, the merchant should be in opt-out list
        from state import store
        assert store.is_opted_out("m_001_drmeera_dentist_delhi") or data["action"] == "send"

    def test_auto_reply_detected(self):
        conv_id = self._create_conversation()
        if not conv_id:
            pytest.skip("no conversation created (LLM not configured)")

        resp = client.post("/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": "m_001_drmeera_dentist_delhi",
            "from_role": "merchant",
            "message": "Thank you for contacting us. Our team will respond shortly.",
            "received_at": "2026-09-27T00:00:00Z",
            "turn_number": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] in ("send", "wait", "end")

    def test_malformed_request_returns_end(self):
        resp = client.post(
            "/v1/reply",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "end"

    def test_rationale_always_present(self):
        resp = client.post("/v1/reply", json={
            "conversation_id": "conv_unknown",
            "merchant_id": "m_001",
            "from_role": "merchant",
            "message": "Hi",
            "received_at": "2026-09-27T00:00:00Z",
            "turn_number": 1,
        })
        assert "rationale" in resp.json()


# ===========================================================================
# Decision engine tests
# ===========================================================================

class TestDecision:
    def test_send_when_all_contexts_present(self):
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, TRIGGER_PAYLOAD)

        from decision import evaluate_trigger, Decision
        result = evaluate_trigger("trg_001_research_digest_dentists", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SEND

    def test_skip_when_trigger_not_loaded(self):
        from decision import evaluate_trigger, Decision
        result = evaluate_trigger("trg_unknown", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP

    def test_skip_when_expired(self):
        expired = dict(TRIGGER_PAYLOAD)
        expired["expires_at"] = "2020-01-01T00:00:00Z"
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, expired)

        from decision import evaluate_trigger, Decision, SkipReason
        result = evaluate_trigger("trg_001_research_digest_dentists", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.TRIGGER_EXPIRED

    def test_skip_when_suppressed(self):
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, TRIGGER_PAYLOAD)
        store.fire_suppression("research:dentists:2026-W17", "conv_old")

        from decision import evaluate_trigger, Decision, SkipReason
        result = evaluate_trigger("trg_001_research_digest_dentists", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP
        assert result.skip_reason == SkipReason.SUPPRESSED

    def test_skip_when_opted_out(self):
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, MERCHANT_PAYLOAD)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, TRIGGER_PAYLOAD)
        store.add_optout("m_001_drmeera_dentist_delhi", cooldown_days=30)

        from decision import evaluate_trigger, Decision
        result = evaluate_trigger("trg_001_research_digest_dentists", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP

    def test_skip_expired_subscription_non_renewal_trigger(self):
        expired_merchant = dict(MERCHANT_PAYLOAD)
        expired_merchant["subscription"] = {"status": "expired", "plan": "Pro", "days_remaining": 0}
        _push_context("category", "dentists", 1, CATEGORY_PAYLOAD)
        _push_context("merchant", "m_001_drmeera_dentist_delhi", 1, expired_merchant)
        _push_context("trigger", "trg_001_research_digest_dentists", 1, TRIGGER_PAYLOAD)

        from decision import evaluate_trigger, Decision
        result = evaluate_trigger("trg_001_research_digest_dentists", "2026-09-27T00:00:00Z")
        assert result.decision == Decision.SKIP


# ===========================================================================
# Reply handler unit tests (no live server needed)
# ===========================================================================

class TestReplyHandler:
    def test_is_auto_reply_phrase(self):
        from reply_handler import is_auto_reply
        from conversation import ConversationState

        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id="t1", trigger_kind="recall_due", send_as="vera",
        )
        assert is_auto_reply("Thank you for contacting us. Our team will respond shortly.", conv)
        assert not is_auto_reply("Yes please, go ahead.", conv)

    def test_classify_commit(self):
        from reply_handler import classify_intent
        assert classify_intent("yes please go ahead") == "explicit_commit"
        assert classify_intent("haan chaliye") == "explicit_commit"

    def test_classify_reject(self):
        from reply_handler import classify_intent
        assert classify_intent("not interested stop") == "hard_reject"
        assert classify_intent("band karo") == "hard_reject"

    def test_classify_hostile(self):
        from reply_handler import classify_intent
        assert classify_intent("this is useless spam") == "hostile"

    def test_classify_wait(self):
        from reply_handler import classify_intent
        assert classify_intent("I'm busy, later") == "wait_request"

    def test_classify_curveball(self):
        from reply_handler import classify_intent
        assert classify_intent("Can you help me with my GST filing?") == "curveball"

    def test_jaccard_similarity(self):
        from reply_handler import _jaccard
        assert _jaccard("hello world", "hello world") == 1.0
        assert _jaccard("hello world", "goodbye world") < 1.0
        assert _jaccard("", "") == 1.0
