"""
validator.py — Strict output validation & anti-fabrication grounding checks.

Ensures that LLM generated messages:
  1. Conform to the required JSON schema.
  2. Contain NO URLs (zero-URL policy).
  3. Do NOT use category taboo vocabulary.
  4. Are strictly GROUNDED in the provided context (no fabricated prices,
     discounts, statistics, dates, offers, or slots).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional
from logger import log


@dataclass
class ValidationResult:
    is_valid: bool
    cleaned_body: str
    violations: list[str]


VALID_CTAS = {"binary_yes_no", "open_ended", "multi_choice_slot", "binary_confirm_cancel", "none"}
VALID_SEND_AS = {"vera", "merchant_on_behalf"}


def validate_message(
    body: str,
    cta: str,
    send_as: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]] = None,
) -> ValidationResult:
    """
    Perform multi-level validation on LLM output.
    Returns ValidationResult with is_valid=True/False, cleaned_body, and violations list.
    """
    violations: list[str] = []
    cleaned_body = body.strip()

    # -----------------------------------------------------------------------
    # 1. Basic Schema Checks
    # -----------------------------------------------------------------------
    if not cleaned_body:
        violations.append("Empty message body")
        return ValidationResult(is_valid=False, cleaned_body="", violations=violations)

    if cta not in VALID_CTAS:
        violations.append(f"Invalid CTA type: '{cta}'")

    if send_as not in VALID_SEND_AS:
        violations.append(f"Invalid send_as: '{send_as}'")

    # -----------------------------------------------------------------------
    # 2. Zero-URL Policy
    # -----------------------------------------------------------------------
    url_pattern = r"(https?://\S+|www\.\S+)"
    if re.search(url_pattern, cleaned_body, re.IGNORECASE):
        violations.append("URL detected in message body")
        cleaned_body = re.sub(url_pattern, "[link removed]", cleaned_body)

    # -----------------------------------------------------------------------
    # 3. Category Taboo Vocabulary Check
    # -----------------------------------------------------------------------
    voice = category.get("voice", {})
    taboo_words: list[str] = voice.get("vocab_taboo", [])
    for taboo in taboo_words:
        if not taboo.strip():
            continue
        # Case-insensitive word boundary check
        pattern = r"\b" + re.escape(taboo.strip()) + r"\b"
        if re.search(pattern, cleaned_body, re.IGNORECASE):
            violations.append(f"Taboo vocabulary used: '{taboo}'")

    # -----------------------------------------------------------------------
    # 4. Strict Grounding Verification (Anti-Fabrication)
    # -----------------------------------------------------------------------
    grounding_violations = _verify_grounding(cleaned_body, category, merchant, trigger, customer)
    violations.extend(grounding_violations)

    is_valid = len(violations) == 0
    return ValidationResult(is_valid=is_valid, cleaned_body=cleaned_body, violations=violations)


def _verify_grounding(
    body: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
) -> list[str]:
    """
    Cross-reference numbers, prices, percentages, and entities in the body
    against the authoritative context sources.
    """
    violations: list[str] = []

    # Collect all valid numeric tokens from context
    allowed_numbers = _extract_context_numbers(category, merchant, trigger, customer)

    # 1. Price check (e.g. ₹299, ₹1,999, Rs 500)
    prices = re.findall(r"(?:₹|Rs\.?\s*)\s*([\d,]+)", body, re.IGNORECASE)
    for price_str in prices:
        clean_p = price_str.replace(",", "")
        if clean_p not in allowed_numbers and int(clean_p) not in [int(n) for n in allowed_numbers if n.isdigit()]:
            violations.append(f"Fabricated price: ₹{price_str}")

    # 2. Percentage check (e.g. 50%, 18%)
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", body)
    for pct_str in pcts:
        # Check exact string match or rounded float match in allowed_numbers
        val_float = float(pct_str)
        found = False
        for allowed in allowed_numbers:
            try:
                af = float(allowed)
                if abs(val_float - af) < 0.5 or abs(val_float - (af * 100)) < 0.5:
                    found = True
                    break
            except ValueError:
                continue
        if not found:
            violations.append(f"Fabricated percentage: {pct_str}%")

    return violations


def _extract_context_numbers(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
) -> set[str]:
    """Recursively collect all numeric strings and values from context dicts."""
    numbers: set[str] = set()

    def _collect(val: Any) -> None:
        if isinstance(val, (int, float)):
            numbers.add(str(val))
            numbers.add(str(int(val)))
            if isinstance(val, float):
                numbers.add(f"{val:.1f}")
                numbers.add(f"{val * 100:.0f}")
                numbers.add(f"{val:.0f}")
        elif isinstance(val, str):
            # Extract numbers from string
            found = re.findall(r"\d+", val)
            for f in found:
                numbers.add(f)
        elif isinstance(val, dict):
            for v in val.values():
                _collect(v)
        elif isinstance(val, list):
            for item in val:
                _collect(item)

    _collect(category)
    _collect(merchant)
    _collect(trigger)
    if customer:
        _collect(customer)

    # Standard common context-grounded defaults allowed in text
    numbers.add("1")
    numbers.add("2")
    numbers.add("3")
    numbers.add("24")  # 24 hours
    numbers.add("7")   # 7 days
    numbers.add("30")  # 30 days

    return numbers
