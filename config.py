"""
config.py — All configuration loaded from environment variables.
No secrets in source code. Load via python-dotenv at startup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env if present (dev convenience)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv is optional; env vars can be set externally


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Bot identity (returned by GET /v1/metadata)                          #
    # ------------------------------------------------------------------ #
    team_name: str = field(default_factory=lambda: os.getenv("TEAM_NAME", "Team Vera"))
    team_members: list[str] = field(
        default_factory=lambda: os.getenv("TEAM_MEMBERS", "Builder").split(",")
    )
    contact_email: str = field(
        default_factory=lambda: os.getenv("CONTACT_EMAIL", "team@example.com")
    )
    bot_version: str = field(default_factory=lambda: os.getenv("BOT_VERSION", "1.0.0"))
    submitted_at: str = field(
        default_factory=lambda: os.getenv("SUBMITTED_AT", "2026-09-27T00:00:00Z")
    )

    # ------------------------------------------------------------------ #
    # LLM configuration                                                    #
    # ------------------------------------------------------------------ #
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "600"))
    )
    # temperature=0 is required by the challenge spec for determinism
    llm_temperature: float = 0.0

    # ------------------------------------------------------------------ #
    # Bot behaviour                                                         #
    # ------------------------------------------------------------------ #
    # Max actions returned per /v1/tick (challenge limit = 20)
    max_actions_per_tick: int = field(
        default_factory=lambda: int(os.getenv("MAX_ACTIONS_PER_TICK", "20"))
    )
    # After a merchant opts-out, suppress all triggers for N days
    optout_cooldown_days: int = field(
        default_factory=lambda: int(os.getenv("OPTOUT_COOLDOWN_DAYS", "30"))
    )
    # renewal_due can fire again after shorter cooldown even if opted-out
    renewal_optout_override_days: int = field(
        default_factory=lambda: int(os.getenv("RENEWAL_OPTOUT_OVERRIDE_DAYS", "14"))
    )
    # Conversation turn limits
    soft_turn_limit: int = field(
        default_factory=lambda: int(os.getenv("SOFT_TURN_LIMIT", "5"))
    )
    hard_turn_limit: int = field(
        default_factory=lambda: int(os.getenv("HARD_TURN_LIMIT", "7"))
    )
    # After N consecutive auto-replies: 1=probe, 2=wait, 3+=end
    auto_reply_probe_threshold: int = 1
    auto_reply_wait_threshold: int = 2
    auto_reply_end_threshold: int = 3
    auto_reply_wait_seconds: int = field(
        default_factory=lambda: int(os.getenv("AUTO_REPLY_WAIT_SECONDS", "86400"))
    )

    # ------------------------------------------------------------------ #
    # Server                                                               #
    # ------------------------------------------------------------------ #
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )


# Singleton — import this everywhere
settings = Config()
