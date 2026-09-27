"""
tests/test_state.py — Unit tests for state store, suppression, and opt-out logic.
"""
import time
import pytest
import sys
import os

# Make sure the parent directory (vera-bot) is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import StateStore


@pytest.fixture
def fresh_store():
    s = StateStore()
    return s


# ---------------------------------------------------------------------------
# Context storage
# ---------------------------------------------------------------------------

class TestContextStorage:
    def test_put_and_get(self, fresh_store):
        ok, _ = fresh_store.put_context("merchant", "m_001", 1, {"name": "Test"})
        assert ok is True
        payload = fresh_store.get_context("merchant", "m_001")
        assert payload == {"name": "Test"}

    def test_newer_version_replaces(self, fresh_store):
        fresh_store.put_context("merchant", "m_001", 1, {"name": "v1"})
        ok, _ = fresh_store.put_context("merchant", "m_001", 2, {"name": "v2"})
        assert ok is True
        assert fresh_store.get_context("merchant", "m_001") == {"name": "v2"}

    def test_older_version_rejected(self, fresh_store):
        fresh_store.put_context("merchant", "m_001", 5, {"name": "v5"})
        ok, current = fresh_store.put_context("merchant", "m_001", 3, {"name": "v3"})
        assert ok is False
        assert current == 5
        # State should not be corrupted
        assert fresh_store.get_context("merchant", "m_001") == {"name": "v5"}

    def test_same_version_idempotent(self, fresh_store):
        fresh_store.put_context("merchant", "m_001", 2, {"name": "original"})
        ok, current = fresh_store.put_context("merchant", "m_001", 2, {"name": "duplicate"})
        assert ok is True

    def test_missing_context_returns_none(self, fresh_store):
        assert fresh_store.get_context("merchant", "nonexistent") is None

    def test_count_by_scope(self, fresh_store):
        fresh_store.put_context("category", "dentists", 1, {})
        fresh_store.put_context("merchant", "m_001", 1, {})
        fresh_store.put_context("merchant", "m_002", 1, {})
        counts = fresh_store.count_by_scope()
        assert counts["category"] == 1
        assert counts["merchant"] == 2
        assert counts["customer"] == 0

    def test_different_scopes_independent(self, fresh_store):
        fresh_store.put_context("merchant", "dentists", 1, {"scope": "merchant"})
        fresh_store.put_context("category", "dentists", 1, {"scope": "category"})
        assert fresh_store.get_context("merchant", "dentists") == {"scope": "merchant"}
        assert fresh_store.get_context("category", "dentists") == {"scope": "category"}


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

class TestSuppression:
    def test_fire_and_check(self, fresh_store):
        fresh_store.fire_suppression("recall:m_001:6mo", "conv_001")
        assert fresh_store.is_suppressed("recall:m_001:6mo") is True

    def test_unfired_key_not_suppressed(self, fresh_store):
        assert fresh_store.is_suppressed("recall:m_001:6mo") is False

    def test_suppression_record_contents(self, fresh_store):
        fresh_store.fire_suppression("key_abc", "conv_xyz")
        record = fresh_store.get_suppression("key_abc")
        assert record is not None
        assert record.key == "key_abc"
        assert record.conversation_id == "conv_xyz"


# ---------------------------------------------------------------------------
# Opt-out
# ---------------------------------------------------------------------------

class TestOptOut:
    def test_add_and_check_optout(self, fresh_store):
        fresh_store.add_optout("m_001", cooldown_days=30)
        assert fresh_store.is_opted_out("m_001") is True

    def test_not_opted_out_by_default(self, fresh_store):
        assert fresh_store.is_opted_out("m_999") is False

    def test_expired_optout_cleared(self, fresh_store):
        # Manually insert an expired opt-out record
        from state import OptOutRecord
        fresh_store._optouts["m_002"] = OptOutRecord(
            merchant_id="m_002",
            opted_out_at=time.time() - 86400 * 31,  # 31 days ago
            cooldown_days=30,
        )
        assert fresh_store.is_opted_out("m_002") is False

    def test_optout_days_remaining(self, fresh_store):
        fresh_store.add_optout("m_003", cooldown_days=10)
        days = fresh_store.optout_days_remaining("m_003")
        assert 9.9 < days <= 10.0


# ---------------------------------------------------------------------------
# Wipe
# ---------------------------------------------------------------------------

class TestWipe:
    def test_wipe_clears_everything(self, fresh_store):
        fresh_store.put_context("merchant", "m_001", 1, {})
        fresh_store.fire_suppression("key_abc", "conv_001")
        fresh_store.add_optout("m_001")
        fresh_store.wipe()
        assert fresh_store.get_context("merchant", "m_001") is None
        assert fresh_store.is_suppressed("key_abc") is False
        assert fresh_store.is_opted_out("m_001") is False
